from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from colorama import Fore, Style

from src.features.load_features import load_and_prepare_data
from src.models.train import train_final_model_inner_cv
from src.selection.shared import TARGET_MAPPING, compute_cv_folds, save_json
from src.util.config_loader import load_deployment_config
from src.util.deployment_outputs import sync_latest_artifacts, write_latest_run_marker


def _load_settings(run_root: Path) -> dict:
    settings_path = run_root / "settings.json"
    if not settings_path.exists():
        return {}
    with settings_path.open("r") as file_handle:
        return json.load(file_handle)


def _resolve_config_path(run_root: Path, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path)
    return run_root / "run_config.yaml"


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _select_top_features(feature_frequency_path: Path, top_n: int) -> list[str]:
    if not feature_frequency_path.exists():
        raise FileNotFoundError(f"Feature frequency file not found: {feature_frequency_path}")

    freq_df = pd.read_csv(feature_frequency_path)
    if "feature" not in freq_df.columns:
        raise ValueError("feature_frequency.csv must contain a 'feature' column.")

    sort_columns = [col for col in ["selection_count", "mean_abs_coefficient"] if col in freq_df.columns]
    if sort_columns:
        freq_df = freq_df.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    return freq_df["feature"].astype(str).dropna().head(max(1, int(top_n))).tolist()


def main(*, run_folder: str, config_path: str | None = None) -> dict:
    run_root = Path(run_folder)
    if not run_root.exists():
        raise FileNotFoundError(f"Run folder not found: {run_root}")

    resolved_config_path = _resolve_config_path(run_root, config_path)
    config = load_deployment_config(resolved_config_path)

    target_arg = config["target"]
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")
    target_column = TARGET_MAPPING[target_arg]

    full_feature_path = run_root / "00_features" / "all_sequences_features.csv"
    if not full_feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {full_feature_path}")

    feature_frequency_path = run_root / "01_feature_selection" / f"feature_frequency.csv"
    selected_features = _select_top_features(feature_frequency_path, config["top_n_features"])

    X, y, _, data_df = load_and_prepare_data(str(full_feature_path), target_column)

    missing = [feature for feature in selected_features if feature not in X.columns]
    if missing:
        raise ValueError("Selected features are missing in feature table: " + ", ".join(missing))

    X_selected = X[selected_features].copy()
    inner_cv = compute_cv_folds(len(y))[1]

    models_root = run_root / "02_models"
    models_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feature": selected_features}).to_csv(models_root / "top_features.csv", index=False)
    save_json(models_root / "top_features.json", selected_features)

    ompp_ids = data_df["OMPP nr"].tolist() if "OMPP nr" in data_df.columns else ["row" for _ in range(len(y))]

    model_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    model_paths: dict[str, str] = {}
    deployment_artifacts: dict[str, Path] = {
        "models/top_features.csv": models_root / "top_features.csv",
        "models/top_features.json": models_root / "top_features.json",
    }

    for model_spec in config["final_model_settings"]:
        model_name = model_spec.model_name
        n_trials = int(model_spec.n_trials)

        model_root = models_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        final_model, final_details, y_pred_train = train_final_model_inner_cv(
            X=X_selected,
            y=y,
            model_name=model_name,
            scoring_scheme=config["scoring_scheme"],
            inner_cv=inner_cv,
            random_state=config["random_state"],
            n_trials=n_trials,
            use_default_params=config["use_default_params"],
            n_jobs=config["n_jobs"],
            scale_target=config["scale_target"],
            show_progress_bar=False,
        )

        model_path = model_root / f"{model_name}_{target_arg}_{len(selected_features)}features.joblib"
        joblib.dump(final_model, model_path)
        model_paths[model_name] = str(model_path)
        deployment_artifacts[f"models/{model_name}.joblib"] = model_path

        save_json(model_root / f"{model_path.stem}.features.json", selected_features)

        signed_coefficients = final_details.get("coefficients")
        if signed_coefficients is not None:
            coefficient_array = np.asarray(signed_coefficients, dtype=float).reshape(-1)
            importance_df = pd.DataFrame(
                {
                    "feature": selected_features,
                    "coefficient": coefficient_array,
                    "abs_coefficient": np.abs(coefficient_array),
                }
            ).sort_values("abs_coefficient", ascending=False)
        else:
            importance_values = list(final_details.get("feature_importance", []))
            importance_df = pd.DataFrame(
                {
                    "feature": selected_features,
                    "importance": importance_values,
                }
            ).sort_values("importance", ascending=False)
        importance_path = model_root / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        deployment_artifacts[f"models/{model_name}_feature_importance.csv"] = importance_path

        prediction_df = pd.DataFrame(
            {
                "OMPP nr": [str(ompp_id) for ompp_id in ompp_ids],
                "y_true": y.to_numpy(dtype=float),
                "y_pred": np.asarray(y_pred_train, dtype=float),
                "split": "train",
                "model_name": model_name,
                "target_arg": target_arg,
                "feature_option": config["feature_option"],
            }
        )
        prediction_df.to_csv(model_root / "training_predictions.csv", index=False)
        prediction_frames.append(prediction_df)

        metrics_train = _calc_metrics(y.to_numpy(dtype=float), np.asarray(y_pred_train, dtype=float))
        model_rows.append(
            {
                "model_name": model_name,
                "n_trials": int(n_trials),
                "model_path": str(model_path),
                "rmse_train": metrics_train["rmse"],
                "mae_train": metrics_train["mae"],
                "r2_train": metrics_train["r2"],
                "inner_cv_score": float(final_details.get("inner_cv_score", np.nan)),
            }
        )

        save_json(
            model_root / "final_model_details.json",
            {
                "target_arg": target_arg,
                "model_name": model_name,
                "selected_features": selected_features,
                "selected_feature_count": len(selected_features),
                "metrics_train": metrics_train,
                "final_model_details": final_details,
                "model_path": str(model_path),
            },
        )

    predictions_long_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_long_df.to_csv(models_root / "training_predictions_long.csv", index=False)

    models_summary_df = pd.DataFrame(model_rows).sort_values(
        ["rmse_train", "mae_train", "r2_train"],
        ascending=[True, True, False],
    )
    models_summary_path = models_root / "models_summary.csv"
    models_summary_df.to_csv(models_summary_path, index=False)
    deployment_artifacts["models/models_summary.csv"] = models_summary_path

    settings_payload = _load_settings(run_root)
    settings_payload.update(
        {
            "target_arg": target_arg,
            "target_column": target_column,
            "model_paths": model_paths,
            "models_summary_path": str(models_summary_path),
            "top_n_features": int(config["top_n_features"]),
        }
    )
    save_json(run_root / "settings.json", settings_payload)

    write_latest_run_marker(target_arg=target_arg, run_root=run_root)
    sync_latest_artifacts(
        target_arg=target_arg,
        artifacts=deployment_artifacts,
    )

    return {
        "run_folder": str(run_root),
        "models_summary_path": str(models_summary_path),
        "model_count": int(len(model_rows)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final deployment models on top-N features")
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Override config path (default: run_config.yaml)")
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nTraining deployment models at {datetime.now()}{Style.RESET_ALL}")

    result = main(
        run_folder=args.run_folder,
        config_path=args.config,
    )

    print(f"{Fore.GREEN}Run folder: {result['run_folder']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Models summary: {result['models_summary_path']}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Model training completed at {datetime.now()}{Style.RESET_ALL}")
