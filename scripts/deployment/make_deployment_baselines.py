from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import json

import numpy as np
import pandas as pd
from colorama import Fore, Style
from scipy.stats import spearmanr

from src.features.load_features import load_and_prepare_data
from src.models.train import train_final_model_inner_cv
from src.selection.shared import TARGET_MAPPING, compute_cv_folds
from src.util.config_loader import load_deployment_config
from src.features.compute import compute_features_for_sequences

BASELINE_FEATURES = ["Discrimination_Factor", "Hydrophobic_Moment"]

REQUIRED_COLUMNS = [
    "target",
    "model",
    "r2",
    "r2_std",
    "rmse",
    "rmse_std",
    "mae",
    "mae_std",
    "spearman",
    "spearman_std",
    "allfeats_r2",
    "allfeats_r2_std",
    "allfeats_rmse",
    "allfeats_rmse_std",
    "allfeats_mae",
    "allfeats_mae_std",
    "allfeats_spearman",
    "allfeats_spearman_std",
    "baseline_r2",
    "baseline_r2_std",
    "baseline_rmse",
    "baseline_rmse_std",
    "baseline_mae",
    "baseline_mae_std",
    "baseline_spearman",
    "baseline_spearman_std",
    "dummy_r2",
    "dummy_r2_std",
    "dummy_rmse",
    "dummy_rmse_std",
    "dummy_mae",
    "dummy_mae_std",
    "dummy_spearman",
    "dummy_spearman_std",
]

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")

    spearman = float("nan")
    if y_true.size >= 2 and y_pred.size >= 2:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(valid_mask) >= 2:
            spearman_value, _ = spearmanr(y_true[valid_mask], y_pred[valid_mask])
            if np.isfinite(spearman_value):
                spearman = float(spearman_value)

    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}


def _metrics_with_std(metrics: dict[str, float] | None) -> dict[str, float]:
    if not metrics:
        return {
            "r2": float("nan"),
            "r2_std": float("nan"),
            "rmse": float("nan"),
            "rmse_std": float("nan"),
            "mae": float("nan"),
            "mae_std": float("nan"),
            "spearman": float("nan"),
            "spearman_std": float("nan"),
        }
    return {
        "r2": float(metrics.get("r2", np.nan)),
        "r2_std": float("nan"),
        "rmse": float(metrics.get("rmse", np.nan)),
        "rmse_std": float("nan"),
        "mae": float(metrics.get("mae", np.nan)),
        "mae_std": float("nan"),
        "spearman": float(metrics.get("spearman", np.nan)),
        "spearman_std": float("nan"),
    }


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _train_and_score(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    scoring_scheme: str,
    random_state: int,
    n_trials: int,
    n_jobs: int,
    scale_target: bool,
    use_default_params: bool,
) -> dict[str, float]:
    inner_cv = compute_cv_folds(len(y))[1]
    model, _, y_pred = train_final_model_inner_cv(
        X=X,
        y=y,
        model_name=model_name,
        scoring_scheme=scoring_scheme,
        inner_cv=inner_cv,
        random_state=random_state,
        n_trials=n_trials,
        use_default_params=use_default_params,
        n_jobs=n_jobs,
        scale_target=scale_target,
        show_progress_bar=False,
    )
    return model, _calc_metrics(y.to_numpy(dtype=float), np.asarray(y_pred, dtype=float))


def main(*, config_path: str | None = None) -> Path:

    config = load_deployment_config(config_path)
    target_arg = config["target"]
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")
    target_column = TARGET_MAPPING[target_arg]

    output_dir = Path("data") / "deployment_metrics" / "baselines" / str(target_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = config["input_file"]
    source_df = pd.read_csv(input_file)
    source_df = source_df.dropna(subset=[target_column]).reset_index(drop=True)

    full_df = compute_features_for_sequences(df=source_df)
    feature_path = output_dir / "all_sequences_features.csv"
    full_df.to_csv(feature_path, index=False)

    X, y, _, _ = load_and_prepare_data(str(feature_path), target_column)
    y = pd.to_numeric(y, errors="coerce")
    valid_mask = y.notna()
    if not valid_mask.all():
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

    missing_features = [feature for feature in BASELINE_FEATURES if feature not in X.columns]
    if missing_features:
        raise ValueError(
            "Missing baseline features in input data: "
            f"{missing_features}."
        )

    X_baseline = X[BASELINE_FEATURES].copy()

    model_names = [str(model_name) for model_name in config["final_model_names"]]
    n_trials_by_model = {str(key).lower(): int(value) for key, value in config["final_model_trials"].items()}
    default_trials = max(n_trials_by_model.values()) if n_trials_by_model else 100

    allfeats_metrics: dict[str, dict[str, float]] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}

    deployment_models_dir = Path("outputs") / "deployment" / str(target_arg) / "models"
    deployment_models_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        model_key = str(model_name).lower()
        model_trials = int(n_trials_by_model.get(model_key, default_trials))

        all_feats_model,allfeats_metrics[model_name] = _train_and_score(
            X=X,
            y=y,
            model_name=model_name,
            scoring_scheme=config["scoring_scheme"],
            random_state=config["random_state"],
            n_trials=model_trials,
            n_jobs=config["n_jobs"],
            scale_target=config["scale_target"],
            use_default_params=config["use_default_params"],
        )
        baseline_model, baseline_metrics[model_name] = _train_and_score(
            X=X_baseline,
            y=y,
            model_name=model_name,
            scoring_scheme=config["scoring_scheme"],
            random_state=config["random_state"],
            n_trials=model_trials,
            n_jobs=config["n_jobs"],
            scale_target=config["scale_target"],
            use_default_params=config["use_default_params"],
        )

        model_path = deployment_models_dir / f"{model_name}_baseline.joblib"
        joblib.dump(baseline_model, model_path)
        model_path = deployment_models_dir / f"{model_name}_allfeats.joblib"
        joblib.dump(all_feats_model, model_path)

    dummy_mean = float(np.mean(y.to_numpy(dtype=float))) if len(y) else float("nan")
    dummy_pred = np.full(len(y), dummy_mean, dtype=float) if len(y) else np.asarray([], dtype=float)
    dummy_metrics = _metrics_with_std(_calc_metrics(y.to_numpy(dtype=float), dummy_pred))

    rows: list[dict] = []
    for model_name in model_names:
        allfeats_summary = _metrics_with_std(allfeats_metrics.get(model_name))
        baseline_summary = _metrics_with_std(baseline_metrics.get(model_name))
        rows.append(
            {
                "target": target_arg,
                "model": model_name,
                "r2": float("nan"),
                "r2_std": float("nan"),
                "rmse": float("nan"),
                "rmse_std": float("nan"),
                "mae": float("nan"),
                "mae_std": float("nan"),
                "spearman": float("nan"),
                "spearman_std": float("nan"),
                **_prefix_metrics(allfeats_summary, "allfeats"),
                **_prefix_metrics(baseline_summary, "baseline"),
                **_prefix_metrics(dummy_metrics, "dummy"),
            }
        )

    metrics_df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)

    
    output_path = output_dir / f"baseline_metrics_{target_arg}.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"{Fore.GREEN}Saved deployment baseline metrics to: {output_path}{Style.RESET_ALL}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute deployment baselines on full data and save metrics")
    parser.add_argument(
        "--config",
        type=str,
        default="workflow_deployment/deployment_config.yaml",
    )
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nBuilding deployment baselines...{Style.RESET_ALL}")

    main(config_path=args.config)
