from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from scripts.baseline.baseline_functions import run_baseline_1_parallel
from src.features.compute import compute_features_for_sequences
from src.features.load_features import load_and_prepare_data
from src.selection.shared import TARGET_MAPPING, generate_seed_list
from src.util.config_loader import load_evaluation_config

OUTLIER_R2_THRESHOLD: float = -1000.0

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
]

def _load_selected_features(feature_set_path: Path, target_column: str) -> list[str]:
    if not feature_set_path.exists():
        raise FileNotFoundError(f"Manual feature set CSV not found: {feature_set_path}")

    manual_df = pd.read_csv(feature_set_path)
    if target_column not in manual_df.columns:
        available = ", ".join(sorted(str(col) for col in manual_df.columns))
        raise ValueError(
            "Manual feature set CSV is missing the target column: "
            f"{target_column}. Available columns: {available}"
        )

    raw_features = manual_df[target_column].dropna().astype(str).tolist()
    raw_features = [feature.strip() for feature in raw_features if str(feature).strip()]

    selected_features: list[str] = []
    seen: set[str] = set()
    for feature in raw_features:
        if feature in seen:
            continue
        selected_features.append(feature)
        seen.add(feature)

    if not selected_features:
        raise ValueError(f"No manual features found for target column: {target_column}")

    return selected_features


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
    summary: pd.DataFrame,
) -> pd.DataFrame:
    summary_lookup = summary.set_index("model_key") if not summary.empty else pd.DataFrame()

    rows: list[dict] = []
    for model_name in model_names:
        model_key = str(model_name).lower()
        summary_row = summary_lookup.loc[model_key] if model_key in summary_lookup.index else None
        rows.append(
            {
                "target": target_arg,
                "model": model_name,
                "r2": float(summary_row["r2"]) if summary_row is not None else float("nan"),
                "r2_std": float(summary_row["r2_std"]) if summary_row is not None else float("nan"),
                "rmse": float(summary_row["rmse"]) if summary_row is not None else float("nan"),
                "rmse_std": float(summary_row["rmse_std"]) if summary_row is not None else float("nan"),
                "mae": float(summary_row["mae"]) if summary_row is not None else float("nan"),
                "mae_std": float(summary_row["mae_std"]) if summary_row is not None else float("nan"),
                "spearman": float(summary_row["spearman"]) if summary_row is not None else float("nan"),
                "spearman_std": float(summary_row["spearman_std"]) if summary_row is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def main(
    *,
    config_path: Path,
    n_jobs: int | None,
    feature_set_path: Path | None = None,
) -> Path:
    config = load_evaluation_config(config_path)

    target_arg = str(config["target"])
    target_column = TARGET_MAPPING[target_arg]

    output_dir = Path(f"data/evaluation_metrics/manual_features/{target_arg}")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = config["input_file"]
    source_df = pd.read_csv(input_file)
    source_df = source_df.dropna(subset=[target_column]).reset_index(drop=True)

    full_df = compute_features_for_sequences(df=source_df)
    full_feature_path = output_dir / "all_sequences_features.csv"
    full_df.to_csv(full_feature_path, index=False)

    X, y, _, _ = load_and_prepare_data(full_feature_path, target=target_column)

    manual_feature_path = Path(feature_set_path)
    selected_features = _load_selected_features(manual_feature_path, target_column)

    missing_features = [feature for feature in selected_features if feature not in X.columns]
    if missing_features:
        raise ValueError(
            "Missing manual features in input data: "
            f"{missing_features}."
        )

    selected_payload = {
        "selected_features": selected_features,
        "target": target_arg,
        "feature_set_path": str(manual_feature_path),
    }
    with (output_dir / "selected_features.json").open("w") as file_handle:
        json.dump(selected_payload, file_handle, indent=2)

    X_selected = X[selected_features].copy()

    final_model_names = [str(model_name) for model_name in config["final_model_names"]]
    scoring_scheme = config["scoring_scheme"]
    scale_target = config["scale_target"]
    start_seed = int(config["start_seed"])
    outer_seed_count = config["num_outer_seeds"]
    outer_seed_list = generate_seed_list(start_seed, outer_seed_count)
    n_jobs_value = int(n_jobs or config["n_jobs"])
    n_trials_by_model = {str(name).lower(): int(trials) for name, trials in config["final_model_trials"].items()}
    default_trials = max(n_trials_by_model.values()) if n_trials_by_model else 100

    manual_results = run_baseline_1_parallel(
        X=X_selected,
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
    manual_outliers = _identify_outlier_folds(manual_results["all_test_df"])
    manual_test_filtered = _filter_outlier_predictions(manual_results["all_test_df"], manual_outliers)
    manual_seed_metrics = _build_seed_metrics(manual_test_filtered)
    manual_summary = _summarize_seed_metrics(manual_seed_metrics)

    metrics_df = _build_metrics_frame(
        target_arg=target_arg,
        model_names=final_model_names,
        summary=manual_summary,
    )

    output_path = output_dir / f"manual_feature_metrics_{target_arg}.csv"
    metrics_df.to_csv(output_path, index=False)

    print(f"{Fore.GREEN}Saved manual feature metrics to: {output_path}{Style.RESET_ALL}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute nested CV metrics using a manually selected feature set and write a "
            "summary CSV to data/evaluation_metrics/manual_features/<target>."
        )
    )
    parser.add_argument(
        "--set-path",
        type=Path,
        default="data/nested_feature_sets.csv",
        help="Path to the manual feature set CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("workflow_evaluation/evaluation_config.yaml"),
        help="Path to evaluation YAML config.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Override the number of worker processes.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        feature_set_path=args.set_path,
        n_jobs=args.n_jobs,
    )

