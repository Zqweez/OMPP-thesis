from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from src.features.compute import compute_features_for_sequences
from src.features.load_features import load_and_prepare_data
from src.selection.shared import TARGET_MAPPING, generate_seed_list, REQUIRED_COLUMNS
from src.util.config_loader import load_evaluation_config
from scripts.baseline.baseline_functions import run_baseline_1_parallel, run_baseline_4

BASELINE_FEATURES = ["Discrimination_Factor", "Hydrophobic_Moment"]
OUTLIER_R2_THRESHOLD: float = -1000.0


def _find_repo_root() -> Path:
    start = Path.cwd().resolve()
    for path in (start, *start.parents):
        if (path / "data").exists() and (path / "scripts").exists():
            return path
    raise FileNotFoundError("Could not locate repo root containing 'data' and 'scripts' folders.")


def _summarize_seed_metrics(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics_df.empty:
        return pd.DataFrame(columns=["model_key", "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std", "spearman", "spearman_std"])

    metrics_df = seed_metrics_df.copy()
    metrics_df["model_key"] = metrics_df["model_name"].astype(str).str.lower()
    metrics_df["r2"] = pd.to_numeric(metrics_df["r2"], errors="coerce")
    metrics_df["rmse"] = pd.to_numeric(metrics_df["rmse"], errors="coerce")
    metrics_df["mae"] = pd.to_numeric(metrics_df["mae"], errors="coerce")
    metrics_df["spearman"] = pd.to_numeric(metrics_df["spearman"], errors="coerce")

    summary = (
        metrics_df.groupby("model_key", as_index=False)
        .agg(
            r2=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            spearman=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
        .reset_index(drop=True)
    )
    return summary


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    spearman = float("nan")
    if y_true.size >= 2 and y_pred.size >= 2:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(valid_mask) >= 2:
            spearman_value, _ = spearmanr(y_true[valid_mask], y_pred[valid_mask])
            if np.isfinite(spearman_value):
                spearman = float(spearman_value)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}


def _identify_outlier_folds(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame(columns=["model_name", "outer_seed", "outer_fold"])

    rows: list[dict] = []
    grouped = test_df.groupby(["model_name", "outer_seed", "outer_fold"], dropna=False)
    for (model_name, outer_seed, outer_fold), group in grouped:
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        fold_r2 = _calc_metrics(y_true, y_pred)["r2"]
        rows.append(
            {
                "model_name": str(model_name),
                "outer_seed": int(outer_seed),
                "outer_fold": int(outer_fold),
                "r2": fold_r2,
            }
        )

    fold_r2_df = pd.DataFrame(rows)
    if fold_r2_df.empty:
        return pd.DataFrame(columns=["model_name", "outer_seed", "outer_fold"])

    outliers = fold_r2_df[fold_r2_df["r2"] < float(OUTLIER_R2_THRESHOLD)]
    return outliers[["model_name", "outer_seed", "outer_fold"]].drop_duplicates().reset_index(drop=True)


def _filter_outlier_predictions(test_df: pd.DataFrame, outlier_keys_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty or outlier_keys_df.empty:
        return test_df.copy()

    filtered = (
        test_df.merge(outlier_keys_df, on=["model_name", "outer_seed", "outer_fold"], how="left", indicator=True)
        .loc[lambda frame: frame["_merge"] == "left_only"]
        .drop(columns=["_merge"])
        .reset_index(drop=True)
    )
    return filtered


def _build_seed_metrics(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame(columns=["model_name", "outer_seed", "mse", "rmse", "mae", "r2", "spearman"])

    rows: list[dict] = []
    grouped = test_df.groupby(["model_name", "outer_seed"], dropna=False)
    for (model_name, outer_seed), group in grouped:
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        metrics = _calc_metrics(y_true, y_pred)
        rows.append(
            {
                "model_name": str(model_name),
                "outer_seed": int(outer_seed),
                **metrics,
            }
        )

    return pd.DataFrame(rows, columns=["model_name", "outer_seed", "mse", "rmse", "mae", "r2", "spearman"])


def _build_metrics_frame(
    *,
    target_arg: str,
    model_names: list[str],
    allfeats_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    dummy_summary: pd.DataFrame,
) -> pd.DataFrame:
    allfeats_lookup = allfeats_summary.set_index("model_key") if not allfeats_summary.empty else pd.DataFrame()
    baseline_lookup = baseline_summary.set_index("model_key") if not baseline_summary.empty else pd.DataFrame()
    dummy_lookup = dummy_summary.set_index("model_key") if not dummy_summary.empty else pd.DataFrame()

    rows: list[dict] = []
    for model_name in model_names:
        model_key = str(model_name).lower()
        allfeats_row = allfeats_lookup.loc[model_key] if model_key in allfeats_lookup.index else None
        baseline_row = baseline_lookup.loc[model_key] if model_key in baseline_lookup.index else None
        dummy_row = None
        if not dummy_lookup.empty:
            if model_key in dummy_lookup.index:
                dummy_row = dummy_lookup.loc[model_key]
            elif "mean_baseline" in dummy_lookup.index:
                dummy_row = dummy_lookup.loc["mean_baseline"]

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
                "allfeats_r2": float(allfeats_row["r2"]) if allfeats_row is not None else float("nan"),
                "allfeats_r2_std": float(allfeats_row["r2_std"]) if allfeats_row is not None else float("nan"),
                "allfeats_rmse": float(allfeats_row["rmse"]) if allfeats_row is not None else float("nan"),
                "allfeats_rmse_std": float(allfeats_row["rmse_std"]) if allfeats_row is not None else float("nan"),
                "allfeats_mae": float(allfeats_row["mae"]) if allfeats_row is not None else float("nan"),
                "allfeats_mae_std": float(allfeats_row["mae_std"]) if allfeats_row is not None else float("nan"),
                "allfeats_spearman": float(allfeats_row["spearman"]) if allfeats_row is not None else float("nan"),
                "allfeats_spearman_std": float(allfeats_row["spearman_std"]) if allfeats_row is not None else float("nan"),
                "baseline_r2": float(baseline_row["r2"]) if baseline_row is not None else float("nan"),
                "baseline_r2_std": float(baseline_row["r2_std"]) if baseline_row is not None else float("nan"),
                "baseline_rmse": float(baseline_row["rmse"]) if baseline_row is not None else float("nan"),
                "baseline_rmse_std": float(baseline_row["rmse_std"]) if baseline_row is not None else float("nan"),
                "baseline_mae": float(baseline_row["mae"]) if baseline_row is not None else float("nan"),
                "baseline_mae_std": float(baseline_row["mae_std"]) if baseline_row is not None else float("nan"),
                "baseline_spearman": float(baseline_row["spearman"]) if baseline_row is not None else float("nan"),
                "baseline_spearman_std": float(baseline_row["spearman_std"]) if baseline_row is not None else float("nan"),
                "dummy_r2": float(dummy_row["r2"]) if dummy_row is not None else float("nan"),
                "dummy_r2_std": float(dummy_row["r2_std"]) if dummy_row is not None else float("nan"),
                "dummy_rmse": float(dummy_row["rmse"]) if dummy_row is not None else float("nan"),
                "dummy_rmse_std": float(dummy_row["rmse_std"]) if dummy_row is not None else float("nan"),
                "dummy_mae": float(dummy_row["mae"]) if dummy_row is not None else float("nan"),
                "dummy_mae_std": float(dummy_row["mae_std"]) if dummy_row is not None else float("nan"),
                "dummy_spearman": float(dummy_row["spearman"]) if dummy_row is not None else float("nan"),
                "dummy_spearman_std": float(dummy_row["spearman_std"]) if dummy_row is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main(
    *,
    config_path: Path,
    num_outer_seeds: int | None,
    n_jobs: int | None,
) -> Path:
    repo_root = _find_repo_root()
    resolved_config = config_path if config_path.is_absolute() else repo_root / config_path
    config = load_evaluation_config(resolved_config)

    target_arg = str(config["target"])
    target_column = TARGET_MAPPING[target_arg]

    output_dir = repo_root / "data" / "evaluation_metrics" / "baselines_new" / target_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = repo_root / config["input_file"]
    source_df = pd.read_csv(input_file)
    source_df = source_df.dropna(subset=[target_column]).reset_index(drop=True)

    full_df = compute_features_for_sequences(df=source_df)

    final_model_names = [str(model_name) for model_name in config["final_model_names"]]
    scoring_scheme = config["scoring_scheme"]
    scale_target = config["scale_target"]
    start_seed = int(config["start_seed"])
    outer_seed_count = int(num_outer_seeds or config["num_outer_seeds"])
    outer_seed_list = generate_seed_list(start_seed, outer_seed_count)
    n_jobs_value = int(n_jobs or config["n_jobs"])
    n_trials_by_model = {str(name).lower(): int(trials) for name, trials in config["final_model_trials"].items()}
    default_trials = max(n_trials_by_model.values()) if n_trials_by_model else 100

    feature_path = output_dir / "all_sequences_features.csv"
    full_df.to_csv(feature_path, index=False)
    X, y, _, _ = load_and_prepare_data(feature_path, target=target_column)

    allfeats_results = run_baseline_1_parallel(
        X=X,
        y=y,
        full_df=full_df,
        outer_seed_list=outer_seed_list,
        final_model_names=final_model_names,
        target_arg=target_arg,
        scoring_scheme=scoring_scheme,
        n_trials=default_trials,
        n_trials_by_model=n_trials_by_model,
        scale_target=scale_target,
        n_jobs=n_jobs_value,
    )
    allfeats_outliers = _identify_outlier_folds(allfeats_results["all_test_df"])
    allfeats_test_filtered = _filter_outlier_predictions(allfeats_results["all_test_df"], allfeats_outliers)
    allfeats_seed_metrics = _build_seed_metrics(allfeats_test_filtered)
    allfeats_summary = _summarize_seed_metrics(allfeats_seed_metrics)

    missing_features = [feature for feature in BASELINE_FEATURES if feature not in X.columns]
    if missing_features:
        raise ValueError(
            "Missing baseline features in input data: "
            f"{missing_features}."
        )
    X_baseline = X[BASELINE_FEATURES].copy()
    baseline_results = run_baseline_1_parallel(
        X=X_baseline,
        y=y,
        full_df=full_df,
        outer_seed_list=outer_seed_list,
        final_model_names=final_model_names,
        target_arg=target_arg,
        scoring_scheme=scoring_scheme,
        n_trials=default_trials,
        n_trials_by_model=n_trials_by_model,
        scale_target=scale_target,
        n_jobs=n_jobs_value,
    )
    baseline_outliers = _identify_outlier_folds(baseline_results["all_test_df"])
    baseline_test_filtered = _filter_outlier_predictions(baseline_results["all_test_df"], baseline_outliers)
    baseline_seed_metrics = _build_seed_metrics(baseline_test_filtered)
    baseline_summary = _summarize_seed_metrics(baseline_seed_metrics)

    dummy_results = run_baseline_4(
        X=X,
        y=y,
        full_df=full_df,
        outer_seed_list=outer_seed_list,
        target_arg=target_arg,
    )
    dummy_outliers = _identify_outlier_folds(dummy_results["all_test_df"])
    dummy_test_filtered = _filter_outlier_predictions(dummy_results["all_test_df"], dummy_outliers)
    dummy_seed_metrics = _build_seed_metrics(dummy_test_filtered)
    dummy_summary = _summarize_seed_metrics(dummy_seed_metrics)

    metrics_df = _build_metrics_frame(
        target_arg=target_arg,
        model_names=final_model_names,
        allfeats_summary=allfeats_summary,
        baseline_summary=baseline_summary,
        dummy_summary=dummy_summary,
    )

    output_path = output_dir / f"baseline_metrics_{target_arg}.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"{Fore.GREEN}Saved baseline metrics to: {output_path}{Style.RESET_ALL}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-features and baseline metrics for final evaluation plots and write a "
            "combined CSV to data/evaluation_metrics/<target>."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("workflow_evaluation/evaluation_config.yaml"),
        help="Path to evaluation YAML config.",
    )
    parser.add_argument(
        "--num-outer-seeds",
        type=int,
        default=None,
        help="Override the number of outer seeds.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override the number of worker processes.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        num_outer_seeds=args.num_outer_seeds,
        n_jobs=args.n_jobs,
    )
