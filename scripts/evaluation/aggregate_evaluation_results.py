from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from src.selection.shared import save_json, REQUIRED_COLUMNS
from src.visualization.plot_evaluation_pipeline import run_evaluation_plots, collect_final_selection_counts, load_run_settings

OUTLIER_R2_THRESHOLD: float = -1000.0

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for aggregated prediction tables."""
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    spearman = float("nan")
    if y_true.size >= 2 and y_pred.size >= 2:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(valid_mask) >= 2:
            spearman_value, _ = spearmanr(y_true[valid_mask], y_pred[valid_mask])
            if np.isfinite(spearman_value):
                spearman = float(spearman_value)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}


def _load_fold_summaries(evaluation_root: Path) -> pd.DataFrame:
    """Load all fold summaries under 01_evaluation into one dataframe."""
    # Collect one JSON summary per finished fold run.
    summary_files = sorted(evaluation_root.rglob("fold_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No fold_summary.json files found under {evaluation_root}")

    rows: list[dict] = []
    for path in summary_files:
        with path.open("r") as file_handle:
            payload = json.load(file_handle)
        payload["fold_summary_path"] = str(path)
        rows.append(payload)

    return pd.DataFrame(rows)


def _safe_load_predictions(path: str | Path) -> pd.DataFrame:
    """Load prediction CSV and raise an explicit error if it is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def _load_selected_features_long(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Load selected features from each fold into one long-format dataframe."""
    rows: list[dict] = []

    for _, row in summary_df.iterrows():
        selected_path_raw = row.get("selected_features_path")
        if selected_path_raw is None:
            continue

        selected_path = Path(selected_path_raw)
        if not selected_path.exists():
            continue

        with selected_path.open("r") as file_handle:
            selected_payload = json.load(file_handle)

        if isinstance(selected_payload, dict):
            selected_features = selected_payload.get("selected_features", [])
        elif isinstance(selected_payload, list):
            selected_features = selected_payload
        else:
            selected_features = []

        selected_features = [str(feature) for feature in selected_features]
        config_payload = row.get("config")
        num_pyramid_seeds = config_payload.get("num_pyramid_seeds", np.nan)

        for rank, feature in enumerate(selected_features, start=1):
            rows.append(
                {
                    "outer_seed": int(row.get("outer_seed")),
                    "outer_fold": int(row.get("outer_fold")),
                    "target_arg": row.get("target_arg"),
                    "feature_option": row.get("feature_option"),
                    "feature": str(feature),
                    "selected_feature_rank": int(rank),
                    "selected_feature_count": int(len(selected_features)),
                    "num_pyramid_seeds": num_pyramid_seeds,
                    "fold_summary_path": row.get("fold_summary_path"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "outer_seed",
            "outer_fold",
            "target_arg",
            "feature_option",
            "feature",
            "selected_feature_rank",
            "selected_feature_count",
            "num_pyramid_seeds",
            "fold_summary_path",
        ],
    )


def _trimmed_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    """Compute symmetric trimmed mean on a 1D numeric array."""
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return float("nan")

    sorted_values = np.sort(clean_values)
    trim_count = int(np.floor(sorted_values.size * float(trim_fraction)))

    if trim_count <= 0 or (2 * trim_count) >= sorted_values.size:
        return float(np.mean(sorted_values))

    trimmed_values = sorted_values[trim_count:-trim_count]
    if trimmed_values.size == 0:
        return float(np.mean(sorted_values))
    return float(np.mean(trimmed_values))


def _build_seed_metric_summary(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build robust summary stats over seed-level metrics."""
    metric_columns = ["mae", "mse", "rmse", "r2", "spearman"]
    rows: list[dict] = []

    for metric in metric_columns:
        values = pd.to_numeric(seed_metrics_df.get(metric), errors="coerce").to_numpy(dtype=float)

        if values.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "trimmed_mean_10pct": float("nan"),
                    "std": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "trimmed_mean_10pct": _trimmed_mean(values, trim_fraction=0.10),
                "std": float(np.std(values)),
            }
        )

    return pd.DataFrame(rows, columns=["metric", "mean", "median", "trimmed_mean_10pct", "std"])


def _summarize_seed_metrics_for_combined(seed_metrics_df: pd.DataFrame) -> dict[str, float]:
    """Summarize seed-level metrics into mean/std values for combined plots."""
    if seed_metrics_df.empty:
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

    r2_values = pd.to_numeric(seed_metrics_df.get("r2"), errors="coerce")
    rmse_values = pd.to_numeric(seed_metrics_df.get("rmse"), errors="coerce")
    mae_values = pd.to_numeric(seed_metrics_df.get("mae"), errors="coerce")
    spearman_values = pd.to_numeric(seed_metrics_df.get("spearman"), errors="coerce")

    return {
        "r2": float(r2_values.mean()),
        "r2_std": float(r2_values.std()),
        "rmse": float(rmse_values.mean()),
        "rmse_std": float(rmse_values.std()),
        "mae": float(mae_values.mean()),
        "mae_std": float(mae_values.std()),
        "spearman": float(spearman_values.mean()),
        "spearman_std": float(spearman_values.std()),
    }


def _split_by_outlier_folds(df: pd.DataFrame, outlier_keys_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into non-outlier and outlier fold rows by (outer_seed, outer_fold)."""
    if df.empty or outlier_keys_df.empty:
        return df.copy(), df.iloc[0:0].copy()

    flagged = df.merge(outlier_keys_df, on=["outer_seed", "outer_fold"], how="inner")
    cleaned = (
        df.merge(outlier_keys_df, on=["outer_seed", "outer_fold"], how="left", indicator=True)
        .loc[lambda frame: frame["_merge"] == "left_only"]
        .drop(columns=["_merge"])
    )
    return cleaned.reset_index(drop=True), flagged.reset_index(drop=True)


def _filter_summary_by_outlier_folds(summary_df: pd.DataFrame, outlier_keys_df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier folds from fold-level summary dataframe."""
    if summary_df.empty or outlier_keys_df.empty:
        return summary_df.copy()

    filtered = (
        summary_df.merge(outlier_keys_df, on=["outer_seed", "outer_fold"], how="left", indicator=True)
        .loc[lambda frame: frame["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .reset_index(drop=True)
    )
    return filtered


def _resolve_model_trials_for_plot(model_fold_metrics_df: pd.DataFrame) -> int | str | None:
    """Resolve one display value for model Optuna trial count across folds."""
    if "n_trials" not in model_fold_metrics_df.columns:
        return None

    trial_values = (
        pd.to_numeric(model_fold_metrics_df["n_trials"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    unique_values = sorted(set(trial_values))

    if not unique_values:
        return None
    if len(unique_values) == 1:
        return int(unique_values[0])
    return ",".join(str(value) for value in unique_values)


def main(*, run_folder: str) -> dict:
    """Aggregate fold-level outputs into model-specific analysis folders and summaries."""
    # Step 1: Resolve output folders and load all fold summary contracts.
    run_root = Path(run_folder)
    evaluation_root = run_root / "01_evaluation"
    analysis_root = run_root / "02_analysis"

    analysis_root.mkdir(parents=True, exist_ok=True)

    summary_df = _load_fold_summaries(evaluation_root)
    summary_df = summary_df.sort_values(["outer_seed", "outer_fold"]).reset_index(drop=True)

    fold_metric_rows: list[dict] = []
    all_test_frames: list[pd.DataFrame] = []
    all_train_frames: list[pd.DataFrame] = []

    # Step 2: Build fold-metric table and collect all train/test prediction frames for each model.
    for _, row in summary_df.iterrows():
        models_payload = row.get("models", {})
        if not isinstance(models_payload, dict):
            continue

        for model_name, model_payload in models_payload.items():
            metrics_test = model_payload.get("metrics_test", {})
            model_trials = model_payload.get("n_trials")
            if model_trials is None:
                raise ValueError(
                    f"Fold summary is missing per-model n_trials for model '{model_name}' "
                    f"(outer_seed={row.get('outer_seed')}, outer_fold={row.get('outer_fold')})."
                )
            fold_metric_rows.append(
                {
                    "outer_seed": int(row["outer_seed"]),
                    "outer_fold": int(row["outer_fold"]),
                    "target_arg": row.get("target_arg"),
                    "model_name": str(model_name),
                    "feature_option": row.get("feature_option"),
                    "n_trials": int(model_trials),
                    "mse": float(metrics_test.get("mse", np.nan)),
                    "rmse": float(metrics_test.get("rmse", np.nan)),
                    "mae": float(metrics_test.get("mae", np.nan)),
                    "r2": float(metrics_test.get("r2", np.nan)),
                    "n_test": int(row.get("n_test", 0)),
                    "fold_summary_path": row.get("fold_summary_path"),
                }
            )

        test_pred_path = row.get("predictions_test_long_path")
        train_pred_path = row.get("predictions_train_long_path")

        test_df = _safe_load_predictions(test_pred_path)
        all_test_frames.append(test_df)

        train_df = _safe_load_predictions(train_pred_path)
        all_train_frames.append(train_df)

    if not fold_metric_rows:
        raise ValueError(f"No model entries found in fold summaries under {evaluation_root}.")
    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values(["model_name", "outer_seed", "outer_fold"]).reset_index(drop=True)
    fold_metrics_all_models_path = analysis_root / "fold_metrics_all_models.csv"
    fold_metrics_df.to_csv(fold_metrics_all_models_path, index=False)

    # Get target for later use
    target = fold_metrics_df["target_arg"].iloc[0]

    all_test_df = pd.concat(all_test_frames, ignore_index=True)
    all_train_df = pd.concat(all_train_frames, ignore_index=True)
    all_test_predictions_path = analysis_root / "all_test_predictions_long.csv"
    all_train_predictions_path = analysis_root / "all_train_predictions_long.csv"
    selected_features_long_df = _load_selected_features_long(summary_df)
    selected_features_long_path = analysis_root / "selected_features_all_folds.csv"

    all_test_df.to_csv(all_test_predictions_path, index=False)
    all_train_df.to_csv(all_train_predictions_path, index=False)
    selected_features_long_df.to_csv(selected_features_long_path, index=False)

    # Step 3: Run full aggregation separately for each final model.
    comparison_rows: list[dict] = []
    combined_metric_rows: list[dict] = []

    model_names = sorted(str(name) for name in fold_metrics_df["model_name"].dropna().unique().tolist())
    for model_name in model_names:
        model_analysis_root = analysis_root / model_name
        model_plots_root = model_analysis_root / "plots"
        model_analysis_root.mkdir(parents=True, exist_ok=True)
        model_plots_root.mkdir(parents=True, exist_ok=True)

        model_fold_metrics_df = (
            fold_metrics_df[fold_metrics_df["model_name"] == model_name]
            .sort_values(["outer_seed", "outer_fold"])
            .reset_index(drop=True)
        )
        model_all_test_df = all_test_df[all_test_df["model_name"] == model_name].reset_index(drop=True)
        model_all_train_df = all_train_df[all_train_df["model_name"] == model_name].reset_index(drop=True)

        model_fold_metrics_raw_path = model_analysis_root / "fold_metrics_raw.csv"
        model_all_test_raw_path = model_analysis_root / "all_test_predictions_raw.csv"
        model_all_train_raw_path = model_analysis_root / "all_train_predictions_raw.csv"
        model_fold_metrics_df.to_csv(model_fold_metrics_raw_path, index=False)
        model_all_test_df.to_csv(model_all_test_raw_path, index=False)
        model_all_train_df.to_csv(model_all_train_raw_path, index=False)

        outlier_folds_df = (
            model_fold_metrics_df[model_fold_metrics_df["r2"] < float(OUTLIER_R2_THRESHOLD)]
            .sort_values(["outer_seed", "outer_fold"])
            .reset_index(drop=True)
        )
        outlier_keys_df = outlier_folds_df[["outer_seed", "outer_fold"]].drop_duplicates().reset_index(drop=True)

        model_fold_metrics_filtered_df, _ = _split_by_outlier_folds(model_fold_metrics_df, outlier_keys_df)
        model_all_test_filtered_df, model_all_test_outlier_df = _split_by_outlier_folds(model_all_test_df, outlier_keys_df)
        model_all_train_filtered_df, model_all_train_outlier_df = _split_by_outlier_folds(model_all_train_df, outlier_keys_df)
        model_summary_filtered_df = _filter_summary_by_outlier_folds(summary_df, outlier_keys_df)

        model_outlier_predictions_df = pd.concat([model_all_train_outlier_df, model_all_test_outlier_df], ignore_index=True)

        model_fold_metrics_path = model_analysis_root / "fold_metrics.csv"
        model_all_test_path = model_analysis_root / "all_test_predictions.csv"
        model_all_train_path = model_analysis_root / "all_train_predictions.csv"
        model_outlier_folds_path = model_analysis_root / "outlier_folds.csv"
        model_outlier_predictions_path = model_analysis_root / "outlier_predictions.csv"

        model_fold_metrics_filtered_df.to_csv(model_fold_metrics_path, index=False)
        model_all_test_filtered_df.to_csv(model_all_test_path, index=False)
        model_all_train_filtered_df.to_csv(model_all_train_path, index=False)
        outlier_folds_df.to_csv(model_outlier_folds_path, index=False)
        model_outlier_predictions_df.to_csv(model_outlier_predictions_path, index=False)

        seed_rows: list[dict] = []
        for seed, seed_df in model_all_test_filtered_df.groupby("outer_seed"):
            seed_metrics = _calc_metrics(seed_df["y_true"].to_numpy(dtype=float), seed_df["y_pred"].to_numpy(dtype=float))
            seed_rows.append(
                {
                    "outer_seed": int(seed),
                    "n_samples": int(len(seed_df)),
                    "mse": seed_metrics["mse"],
                    "rmse": seed_metrics["rmse"],
                    "mae": seed_metrics["mae"],
                    "r2": seed_metrics["r2"],
                    "spearman": seed_metrics["spearman"],
                }
            )

        model_seed_metrics_df = pd.DataFrame(seed_rows).sort_values("outer_seed").reset_index(drop=True)
        model_seed_metrics_path = model_analysis_root / "seed_aggregated_metrics.csv"
        model_seed_metrics_df.to_csv(model_seed_metrics_path, index=False)

        model_seed_metric_summary_df = _build_seed_metric_summary(model_seed_metrics_df)
        model_seed_metric_summary_path = model_analysis_root / "seed_metric_summary.csv"
        model_seed_metric_summary_df.to_csv(model_seed_metric_summary_path, index=False)

        combined_seed_summary = _summarize_seed_metrics_for_combined(model_seed_metrics_df)
        combined_metric_rows.append(
            {
                "target": target,
                "model": model_name,
                "r2": combined_seed_summary["r2"],
                "r2_std": combined_seed_summary["r2_std"],
                "rmse": combined_seed_summary["rmse"],
                "rmse_std": combined_seed_summary["rmse_std"],
                "mae": combined_seed_summary["mae"],
                "mae_std": combined_seed_summary["mae_std"],
                "spearman": combined_seed_summary["spearman"],
                "spearman_std": combined_seed_summary["spearman_std"],
                "allfeats_r2": float("nan"),
                "allfeats_r2_std": float("nan"),
                "allfeats_rmse": float("nan"),
                "allfeats_rmse_std": float("nan"),
                "allfeats_mae": float("nan"),
                "allfeats_mae_std": float("nan"),
                "allfeats_spearman": float("nan"),
                "allfeats_spearman_std": float("nan"),
                "baseline_r2": float("nan"),
                "baseline_r2_std": float("nan"),
                "baseline_rmse": float("nan"),
                "baseline_rmse_std": float("nan"),
                "baseline_mae": float("nan"),
                "baseline_mae_std": float("nan"),
                "baseline_spearman": float("nan"),
                "baseline_spearman_std": float("nan"),
                "dummy_r2": float("nan"),
                "dummy_r2_std": float("nan"),
                "dummy_rmse": float("nan"),
                "dummy_rmse_std": float("nan"),
                "dummy_mae": float("nan"),
                "dummy_mae_std": float("nan"),
                "dummy_spearman": float("nan"),
                "dummy_spearman_std": float("nan"),
            }
        )

        global_test_metrics = _calc_metrics(
            model_all_test_filtered_df["y_true"].to_numpy(dtype=float),
            model_all_test_filtered_df["y_pred"].to_numpy(dtype=float),
        )
        global_train_metrics = _calc_metrics(
            model_all_train_filtered_df["y_true"].to_numpy(dtype=float),
            model_all_train_filtered_df["y_pred"].to_numpy(dtype=float),
        )
        model_optuna_trials = _resolve_model_trials_for_plot(model_fold_metrics_df)

        run_evaluation_plots(
            run_folder=run_root,
            output_dir=model_plots_root,
            summary_df=model_summary_filtered_df,
            seed_metrics_df=model_seed_metrics_df,
            all_train_df=model_all_train_filtered_df,
            all_test_df=model_all_test_filtered_df,
            model_name=model_name,
            model_optuna_trials=model_optuna_trials,
        )

        comparison_rows.append(
            {
                "model_name": model_name,
                "target": target,
                "n_trials": model_optuna_trials,
                "fold_count": int(len(model_fold_metrics_filtered_df)),
                "outlier_fold_count": int(len(outlier_folds_df)),
                "seed_count": int(model_seed_metrics_df["outer_seed"].nunique()),
                "test_rmse": float(global_test_metrics["rmse"]),
                "test_mae": float(global_test_metrics["mae"]),
                "test_r2": float(global_test_metrics["r2"]),
                "train_rmse": float(global_train_metrics["rmse"]),
                "train_mae": float(global_train_metrics["mae"]),
                "train_r2": float(global_train_metrics["r2"]),
            }
        )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(["test_rmse", "test_mae", "test_r2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    model_comparison_path = analysis_root / "model_comparison.csv"
    comparison_df.to_csv(model_comparison_path, index=False)

    best_model_name = str(comparison_df.iloc[0]["model_name"]) if not comparison_df.empty else ""
    best_rmse = float(comparison_df.iloc[0]["test_rmse"]) if not comparison_df.empty else float("nan")
    best_r2 = float(comparison_df.iloc[0]["test_r2"]) if not comparison_df.empty else float("nan")

    # Save selection counts for the final model
    settings = load_run_settings(run_root)
    if Path(summary_df["selected_features_path"].iloc[0]).is_absolute():
        settings["run_folder"] = None
    else:
        settings["run_folder"] = Path(run_folder).resolve()
    selection_df, _ = collect_final_selection_counts(
        summary_df=summary_df,
        settings=settings
    )
    selection_df.to_csv(analysis_root / "final_selection_counts.csv", index=False)

    combined_metrics_df = pd.DataFrame(combined_metric_rows, columns=REQUIRED_COLUMNS)
    combined_metrics_path = analysis_root / f"evaluation_metrics_{target}.csv"
    combined_metrics_df.to_csv(combined_metrics_path, index=False)

    metrics_root = Path("data") / "evaluation_metrics" / str(target)
    metrics_root.mkdir(parents=True, exist_ok=True)
    shared_metrics_path = metrics_root / f"evaluation_metrics_{target}.csv"
    combined_metrics_df.to_csv(shared_metrics_path, index=False)

    return {
        "run_folder": str(run_root),
        "analysis_folder": str(analysis_root),
        "selected_features_long_path": str(selected_features_long_path),
        "model_count": int(len(model_names)),
        "best_model_name": best_model_name,
        "best_global_test_rmse": best_rmse,
        "best_global_test_r2": best_r2,
        "evaluation_metrics_path": str(combined_metrics_path),
    }


if __name__ == "__main__":
    """CLI entry point for evaluation aggregation."""
    parser = argparse.ArgumentParser(description="Aggregate evaluation fold outputs into seed/global metrics and plots")
    parser.add_argument("--run_folder", type=str, required=True, help="Evaluation run root containing 01_evaluation/")
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nAggregating evaluation outputs at {datetime.now()}{Style.RESET_ALL}")

    result = main(
        run_folder=args.run_folder,
    )

    print(f"{Fore.GREEN}Analysis folder: {result['analysis_folder']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Evaluation metrics CSV: {result['evaluation_metrics_path']}{Style.RESET_ALL}")

    print(f"{Fore.LIGHTGREEN_EX}Aggregation completed at {datetime.now()}{Style.RESET_ALL}")
