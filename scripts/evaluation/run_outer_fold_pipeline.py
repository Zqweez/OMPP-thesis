from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.features.clustering import run_cluster_pipeline_with_amp
from src.features.load_features import load_and_prepare_data
from src.models.train import train_final_model_inner_cv
from src.util.config_loader import load_evaluation_config
from src.selection.double_pyramid import (
    PyramidConfig,
    run_double_pyramid_selection,
    run_single_pyramid_selection,
)
from src.selection.shared import TARGET_MAPPING, compute_cv_folds, generate_seed_list, save_json


def _resolve_split_from_manifest(
    *,
    manifest_path: Path,
    outer_seed: int | None,
    outer_fold: int | None,
) -> dict:
    """Resolve one split job from manifest by (outer_seed, outer_fold)."""
    manifest_df = pd.read_csv(manifest_path)

    if outer_seed is None or outer_fold is None:
        raise ValueError("Provide both --outer_seed and --outer_fold.")

    hit = manifest_df[(manifest_df["outer_seed"] == outer_seed) & (manifest_df["outer_fold"] == outer_fold)]
    if hit.empty:
        raise ValueError(f"No manifest row found for outer_seed={outer_seed}, outer_fold={outer_fold}.")

    return hit.iloc[0].to_dict()


def _load_manifest_tasks(manifest_path: Path) -> list[tuple[int, int]]:
    """Load all (outer_seed, outer_fold) pairs from a manifest file."""
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    manifest_df = manifest_df.sort_values(["outer_seed", "outer_fold"]).reset_index(drop=True)
    return [
        (int(row["outer_seed"]), int(row["outer_fold"]))
        for _, row in manifest_df.iterrows()
    ]


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for one prediction subset."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _build_prediction_frame(
    *,
    ompp_ids: list,
    y_true: pd.Series,
    y_pred: np.ndarray,
    split: str,
    model_name: str,
    outer_seed: int,
    outer_fold: int,
    target_arg: str,
    feature_option: str,
) -> pd.DataFrame:
    """Build long-format per-sample prediction table for one fold/model/split."""
    return pd.DataFrame(
        {
            "OMPP nr": [str(ompp_id) for ompp_id in ompp_ids],
            "y_true": y_true.to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "split": str(split),
            "model_name": str(model_name),
            "outer_seed": int(outer_seed),
            "outer_fold": int(outer_fold),
            "target_arg": str(target_arg),
            "feature_option": str(feature_option),
        }
    )


def _save_prediction_frame(path: Path, prediction_df: pd.DataFrame) -> None:
    """Persist one prediction dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(path, index=False)


def _save_scaled_test_feature_matrix(
    path: Path,
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    test_ompp_ids: list,
    intercept: bool = False,
) -> None:
    """Save train-scaled test feature values with features as rows and test OMPP ids as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    scaled_df = pd.DataFrame(
        X_test_scaled.T,
        index=[str(feature) for feature in X_train_selected.columns],
        columns=[str(ompp_id) for ompp_id in test_ompp_ids],
    )
    if intercept:
        scaled_df.loc["intercept"] = intercept
    scaled_df.index.name = "feature"
    scaled_df.to_csv(path)

def _extract_model_intercept(fitted_model) -> float | None:
    """Extract scalar intercept from final model when available."""
    inner_estimator = fitted_model.regressor_ if hasattr(fitted_model, "regressor_") else fitted_model
    base_model = inner_estimator.named_steps["model"] if hasattr(inner_estimator, "named_steps") else inner_estimator

    if not hasattr(base_model, "intercept_"):
        return None

    intercept_array = np.asarray(base_model.intercept_, dtype=float).reshape(-1)
    if intercept_array.size == 0:
        return None
    return float(intercept_array[0])

def run_one_fold(
    *,
    config_path: str,
    manifest_path: str,
    outer_seed: int | None,
    outer_fold: int | None,
    run_folder: str | None = None,
) -> dict:
    """Execute one full outer-fold run from split loading to artifacts and summary.

    The pipeline performs:
    1. Split resolution and fold data loading
    2. Feature-selection strategy execution (single or double)
    3. Inner-CV model tuning on train fold
    4. Prediction on train/test and artifact persistence
    """
    config = load_evaluation_config(config_path)
    target_arg = config["target"]
    pyramid_model_name = config["pyramid_model_name"]
    final_model_settings = [
        {
            "model_name": str(model_cfg["model_name"]),
            "n_trials": int(model_cfg["n_trials"]),
        }
        for model_cfg in config["final_model_settings"]
    ]
    final_model_names = [str(model_cfg["model_name"]) for model_cfg in final_model_settings]
    scoring_scheme = config["scoring_scheme"]
    top_percentage = config["top_percentage"]
    double_top_percent = config["double_top_percent"]
    set_size = config["set_size"]
    scale_target = config["scale_target"]
    feature_option = config["feature_option"]
    num_pyramid_seeds = config["num_pyramid_seeds"]
    start_pyramid_seed = config["start_pyramid_seed"]
    cluster_threshold = config["cluster_threshold"]
    cluster_method = config["cluster_method"]
    max_cluster_size = config["max_cluster_size"]
    variance_threshold = config["variance_threshold"]
    perfect_corr_threshold = config["perfect_corr_threshold"]

    amp_sequence_path = config["amp_sequence_path"]
    amp_feature_path = config["amp_feature_path"]

    # Validate top-level arguments and resolve one fold from manifest.
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")
    if not (0 < double_top_percent <= 1):
        raise ValueError("double_top_percent must be in (0, 1].")
    if not final_model_names:
        raise ValueError("final_model_names must contain at least one model name.")

    manifest = _resolve_split_from_manifest(
        manifest_path=Path(manifest_path),
        outer_seed=outer_seed,
        outer_fold=outer_fold,
    )

    # print(f"{Fore.CYAN}Running outer fold pipeline for seed {outer_seed}, fold {outer_fold}...{Style.RESET_ALL}")

    # Build run folder for this fold. Prefer explicit run_folder to avoid dependency on manifest staging mode.
    run_root = Path(run_folder).resolve() if run_folder else Path(manifest_path).resolve().parent.parent

    fold_root = Path(run_root) / "01_evaluation" / f"outer_seed_{outer_seed}" / f"outer_fold_{outer_fold}"
    fold_root.mkdir(parents=True, exist_ok=True)

    # Load train and test fold data.
    target_column = TARGET_MAPPING[target_arg]
    train_data_path = Path(str(manifest["train_data_csv"]))
    test_data_path = Path(str(manifest["test_data_csv"]))

    X_train, y_train, _, train_df = load_and_prepare_data(str(train_data_path), target_column)
    X_test, y_test, _, test_df = load_and_prepare_data(str(test_data_path), target_column)

    # Remove split bookkeeping columns from modeling features.

    train_ompp_ids = train_df["OMPP nr"].tolist()
    test_ompp_ids = test_df["OMPP nr"].tolist()
    amp_feature_path = amp_feature_path
    amp_sequence_path = amp_sequence_path
    # Run feature clustering inside this outer fold using train data only.
    cluster_result = run_cluster_pipeline_with_amp(
        ompp_feature_path=train_data_path,
        ompp_features_df=train_df,
        amp_sequence_path=amp_sequence_path,
        amp_feature_path=amp_feature_path,
        output_dir=fold_root / "clustering",
        threshold=cluster_threshold,
        method=cluster_method,
        max_cluster_size=max_cluster_size,
        variance_threshold=variance_threshold,
        perfect_corr_threshold=perfect_corr_threshold,
        make_plots=False,
        save_clustering_csv=False,
    )
    cluster_df = cluster_result["cluster_df_reduced"]

    # Prepare clustered feature list and selection configuration.
    available_features = [feature for feature in cluster_df["feature_name"].astype(str).tolist() if feature in X_train.columns]
    if not available_features:
        raise ValueError("No clustering features available in outer-train feature matrix.")

    fold_seed = int(outer_seed) # Could add override with --seed or something if desired
    base_config = PyramidConfig(
        model_name=pyramid_model_name,
        scoring_scheme=scoring_scheme,
        top_percentage=top_percentage,
        set_size=set_size,
        scale_target=scale_target,
        random_state=fold_seed,
        num_pyramid_seeds=num_pyramid_seeds,
        start_seed=start_pyramid_seed,
    )
    pyramid_seed_list = generate_seed_list(start_pyramid_seed, num_pyramid_seeds)

    # Step 4: Run selected feature-selection strategy and store selected feature artifacts.
    if feature_option == "single":
        selected_features, ranking_df, selection_meta = run_single_pyramid_selection(
            X=X_train,
            y=y_train,
            available_features=available_features,
            config=base_config,
            seed_list=pyramid_seed_list,
            save_root=None,
        )
    elif feature_option == "double":
        selected_features, ranking_df, _, selection_meta = run_double_pyramid_selection(
            X=X_train,
            y=y_train,
            available_features=available_features,
            config=base_config,
            seed_list=pyramid_seed_list,
            double_top_percent=double_top_percent,
            save_root=None,
        )
    else:
        raise ValueError("feature_option must be 'single' or 'double'.")

    ranking_df.to_csv(fold_root / "feature_frequency.csv", index=False)
    selected_payload = {
        "selection_meta": selection_meta,
        "selected_features": [str(feature) for feature in selected_features],
    }
    save_json(fold_root / "selected_features.json", selected_payload)

    # Step 5: Train each configured final model sequentially on shared selected features.
    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    inner_cv = compute_cv_folds(len(y_train))[1]

    models_root = fold_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    model_results: dict[str, dict] = {}
    all_train_prediction_frames: list[pd.DataFrame] = []
    all_test_prediction_frames: list[pd.DataFrame] = []

    for model_cfg in final_model_settings:
        final_model_name = str(model_cfg["model_name"])
        model_n_trials = int(model_cfg["n_trials"])
        final_model, final_details, y_pred_train = train_final_model_inner_cv(
            X=X_train_selected,
            y=y_train,
            model_name=final_model_name,
            scoring_scheme=scoring_scheme,
            inner_cv=inner_cv,
            random_state=fold_seed,
            n_trials=model_n_trials,
            use_default_params=False,
            n_jobs=1,
            scale_target=scale_target,
            show_progress_bar=False,
        )
        y_pred_test = np.asarray(final_model.predict(X_test_selected), dtype=float)

        model_path = models_root / f"{final_model_name}_{outer_seed}_{outer_fold}.joblib"
        joblib.dump(final_model, model_path)

        train_prediction_df = _build_prediction_frame(
            ompp_ids=train_ompp_ids,
            y_true=y_train,
            y_pred=np.asarray(y_pred_train, dtype=float),
            split="train",
            model_name=final_model_name,
            outer_seed=int(outer_seed),
            outer_fold=int(outer_fold),
            target_arg=target_arg,
            feature_option=feature_option,
        )
        test_prediction_df = _build_prediction_frame(
            ompp_ids=test_ompp_ids,
            y_true=y_test,
            y_pred=y_pred_test,
            split="test",
            model_name=final_model_name,
            outer_seed=int(outer_seed),
            outer_fold=int(outer_fold),
            target_arg=target_arg,
            feature_option=feature_option,
        )

        all_train_prediction_frames.append(train_prediction_df)
        all_test_prediction_frames.append(test_prediction_df)

        signed_coefficients = final_details.get("coefficients")
        test_feature_values_scaled_path: str | None = None
        model_intercept: float | None = None

        if signed_coefficients is not None:
            coefficient_array = np.asarray(signed_coefficients, dtype=float).reshape(-1)
            coefficient_df = pd.DataFrame(
                {
                    "feature": selected_features,
                    "coefficient": coefficient_array,
                    "abs_coefficient": np.abs(coefficient_array),
                }
            ).sort_values("abs_coefficient", ascending=False)

            model_intercept = _extract_model_intercept(final_model)
            test_scaled_features_path = models_root / f"{final_model_name}_test_feature_values_scaled.csv"
            _save_scaled_test_feature_matrix(
                test_scaled_features_path,
                X_train_selected,
                X_test_selected,
                test_ompp_ids,
                intercept=model_intercept,
            )
            test_feature_values_scaled_path = str(test_scaled_features_path)
            feature_importance_path = models_root / f"{final_model_name}_final_feature_importance.csv"
            coefficient_df.to_csv(feature_importance_path, index=False)
        else:
            importance_values = list(final_details.get("feature_importance", []))
            coefficient_df = pd.DataFrame(
                {
                    "feature": selected_features,
                    "importance": importance_values,
                }
            ).sort_values("importance", ascending=False)
            feature_importance_path = models_root / f"{final_model_name}_final_feature_importance.csv"
            coefficient_df.to_csv(feature_importance_path, index=False)

        train_metrics = _calc_metrics(y_train.to_numpy(dtype=float), np.asarray(y_pred_train, dtype=float))
        test_metrics = _calc_metrics(y_test.to_numpy(dtype=float), y_pred_test)

        model_results[str(final_model_name)] = {
            "model_name": str(final_model_name),
            "n_trials": int(model_n_trials),
            "final_feature_importance_path": str(feature_importance_path),
            "test_feature_values_scaled_path": test_feature_values_scaled_path,
            "model_path": str(model_path),
            "metrics_train": train_metrics,
            "metrics_test": test_metrics,
            "inner_cv_score": float(final_details.get("inner_cv_score", np.nan)),
            "model_intercept": model_intercept,
            "best_params": final_details.get("best_params", {}),
        }

    predictions_train_long_df = pd.concat(all_train_prediction_frames, ignore_index=True)
    predictions_test_long_df = pd.concat(all_test_prediction_frames, ignore_index=True)

    predictions_train_long_path = fold_root / "predictions_train_long.csv"
    predictions_test_long_path = fold_root / "predictions_test_long.csv"
    _save_prediction_frame(predictions_train_long_path, predictions_train_long_df)
    _save_prediction_frame(predictions_test_long_path, predictions_test_long_df)

    # Step 6: Persist fold summary in a stable schema for downstream aggregation.
    fold_summary = {
        "outer_seed": int(outer_seed),
        "outer_fold": int(outer_fold),
        "target_arg": target_arg,
        "target_column": target_column,
        "pyramid_model_name": pyramid_model_name,
        "final_model_names": [str(model_name) for model_name in final_model_names],
        "final_model_settings": final_model_settings,
        "final_model_trials": {
            str(model_cfg["model_name"]): int(model_cfg["n_trials"])
            for model_cfg in final_model_settings
        },
        "feature_option": feature_option,
        "selected_features_path": str(fold_root / "selected_features.json"),
        "selection_summary_path": str(fold_root / "feature_frequency.csv"),
        "predictions_train_long_path": str(predictions_train_long_path),
        "predictions_test_long_path": str(predictions_test_long_path),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "models": model_results,
        "config": {
            "scoring_scheme": scoring_scheme,
            "final_model_trials": {
                str(model_cfg["model_name"]): int(model_cfg["n_trials"])
                for model_cfg in final_model_settings
            },
            "final_model_settings": final_model_settings,
            "top_percentage": float(top_percentage),
            "double_top_percent": float(double_top_percent),
            "set_size": int(set_size),
            "scale_target": bool(scale_target),
            "cluster_threshold": float(cluster_threshold),
            "cluster_method": cluster_method,
            "max_cluster_size": int(max_cluster_size),
            "variance_threshold": float(variance_threshold),
            "perfect_corr_threshold": float(perfect_corr_threshold),
            "num_pyramid_seeds": int(num_pyramid_seeds),
            "start_pyramid_seed": int(start_pyramid_seed),
            "pyramid_seed_list": [int(seed) for seed in pyramid_seed_list],
            "final_model_names": [str(model_name) for model_name in final_model_names],
        },
        "selection_meta": selection_meta,
    }
    save_json(fold_root / "fold_summary.json", fold_summary)

    model_metrics_test = {
        str(model_name): {
            "rmse": float(model_payload["metrics_test"].get("rmse", np.nan)),
            "r2": float(model_payload["metrics_test"].get("r2", np.nan)),
        }
        for model_name, model_payload in model_results.items()
    }

    return {
        "fold_root": str(fold_root),
        "selected_feature_count": int(len(selected_features)),
        "model_metrics_test": model_metrics_test,
    }

def _run_fold_task(task: tuple[int, int], common_args: dict) -> dict:
    """Worker entrypoint for one (outer_seed, outer_fold) job."""
    outer_seed, outer_fold = task
    result = run_one_fold(
        config_path=common_args["config_path"],
        manifest_path=common_args["manifest_path"],
        outer_seed=int(outer_seed),
        outer_fold=int(outer_fold),
        run_folder=common_args.get("run_folder"),
    )
    result["outer_seed"] = int(outer_seed)
    result["outer_fold"] = int(outer_fold)
    return result


def run_all_folds(
    *,
    config_path: str,
    manifest_path: str,
    run_folder: str | None = None,
) -> dict:
    """Execute all folds defined in a manifest using multiprocessing."""
    config = load_evaluation_config(config_path)
    n_jobs = max(1, int(config["n_jobs"]))

    tasks = _load_manifest_tasks(Path(manifest_path))

    common_args = {
        "config_path": str(Path(config_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "run_folder": run_folder,
    }

    fold_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_task = {
            executor.submit(_run_fold_task, task, common_args): task
            for task in tasks
        }
        with tqdm(total=len(future_to_task), desc="Outer folds", unit="fold") as progress:
            for future in as_completed(future_to_task):
                result = future.result()
                fold_results.append(result)
                progress.update(1)

    return {
        "run_folder": str(run_folder) if run_folder else None,
        "manifest_path": str(Path(manifest_path).resolve()),
        "num_folds": int(len(tasks)),
        "fold_results": fold_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run outer-fold evaluation pipeline from split manifest")
    parser.add_argument("--config", type=str, default="workflow_evaluation/evaluation_config.yaml")

    ## Flags for loading the fold and seed
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--run_folder", type=str, default=None)
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nStarting outer-fold batch at {datetime.now()}{Style.RESET_ALL}")
    run_all_folds(
        config_path=args.config,
        manifest_path=args.manifest,
        run_folder=args.run_folder,
    )
    print(f"{Fore.LIGHTGREEN_EX}Outer-fold batch completed at {datetime.now()}{Style.RESET_ALL}")
