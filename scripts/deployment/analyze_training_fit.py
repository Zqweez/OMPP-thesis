from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from colorama import Fore, Style
from scipy.stats import spearmanr

from src.visualization.plot_final_model import plot_training_fit_scatter
from src.util.deployment_outputs import sync_latest_artifacts

COMBINED_METRICS_COLUMNS = [
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
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "spearman": float("nan")}

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


def _build_combined_metrics_frame(target_arg: str, model_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name, metrics in model_metrics.items():
        rows.append(
            {
                "target": str(target_arg),
                "model": str(model_name),
                "r2": float(metrics.get("r2", np.nan)),
                "r2_std": float("nan"),
                "rmse": float(metrics.get("rmse", np.nan)),
                "rmse_std": float("nan"),
                "mae": float(metrics.get("mae", np.nan)),
                "mae_std": float("nan"),
                "spearman": float(metrics.get("spearman", np.nan)),
                "spearman_std": float("nan"),
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

    return pd.DataFrame(rows, columns=COMBINED_METRICS_COLUMNS)


def _load_settings(run_folder: Path) -> dict:
    settings_path = run_folder / "settings.json"
    if not settings_path.exists():
        return {}
    with settings_path.open("r") as file_handle:
        return json.load(file_handle)


def main(*, run_folder: str, model_name: str | None = None) -> dict:
    run_root = Path(run_folder)
    predictions_path = run_root / "02_models" / "training_predictions_long.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Training predictions not found: {predictions_path}")

    predictions_df = pd.read_csv(predictions_path)
    required_cols = {"model_name", "y_true", "y_pred"}
    missing = required_cols - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {predictions_path}: {sorted(missing)}")

    models_summary_path = run_root / "02_models" / "models_summary.csv"
    if models_summary_path.exists():
        models_summary_df = pd.read_csv(models_summary_path)
        n_trials_lookup = (
            models_summary_df.set_index("model_name")["n_trials"].to_dict()
            if "n_trials" in models_summary_df.columns
            else {}
        )
    else:
        models_summary_df = pd.DataFrame()
        n_trials_lookup = {}

    settings = _load_settings(run_root)
    config_settings = settings.get("config", {})

    target_arg = settings.get("target_arg")
    if not target_arg:
        unique_targets = predictions_df.get("target_arg", pd.Series(dtype=str)).dropna().unique().tolist()
        if len(unique_targets) == 1:
            target_arg = str(unique_targets[0])
    if not target_arg:
        target_arg = "unknown"

    if model_name:
        model_names = [model_name]
    else:
        model_names = sorted(predictions_df["model_name"].dropna().unique().tolist())

    output_dir = run_root / "03_analysis" / "training_fit"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    model_metrics: dict[str, dict[str, float]] = {}

    deployment_artifacts: dict[str, Path] = {}

    for model in model_names:
        model_df = predictions_df[predictions_df["model_name"] == model].copy()
        if model_df.empty:
            continue

        y_true = pd.to_numeric(model_df["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(model_df["y_pred"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        metrics = _calc_metrics(y_true, y_pred)
        model_metrics[str(model)] = metrics
        summary_rows.append(
            {
                "model_name": model,
                "n_samples": int(len(y_true)),
                "rmse_train": metrics["rmse"],
                "mae_train": metrics["mae"],
                "r2_train": metrics["r2"],
            }
        )

        selection_mode = settings.get("feature_option")
        if selection_mode is None:
            selection_mode = config_settings.get("feature_option", "N/A")
        plot_settings = {
            "target_arg": settings.get("target_arg", "N/A"),
            "model_name": model,
            "scale_target": config_settings.get("scale_target", "N/A"),
            "selection_mode": selection_mode,
            "n_trials": n_trials_lookup.get(model, "N/A"),
            "selected_feature_count": settings.get("selected_feature_count", "N/A"),
        }
        output_path = output_dir / f"{model}_training_fit_scatter.pdf"
        plot_training_fit_scatter(
            y_true=pd.Series(y_true),
            y_pred=np.asarray(y_pred, dtype=float),
            output_path=output_path,
            settings=plot_settings,
        )
        deployment_artifacts[f"analysis/{model}_training_fit_scatter.pdf"] = output_path

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["rmse_train", "mae_train", "r2_train"],
            ascending=[True, True, False],
        )
    else:
        summary_df = pd.DataFrame(
            columns=["model_name", "n_samples", "rmse_train", "mae_train", "r2_train"]
        )
    summary_path = output_dir / "training_fit_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    deployment_artifacts["analysis/training_fit_summary.csv"] = summary_path

    combined_metrics_df = _build_combined_metrics_frame(target_arg, model_metrics)
    combined_metrics_path = output_dir / f"evaluation_metrics_{target_arg}.csv"
    combined_metrics_df.to_csv(combined_metrics_path, index=False)
    deployment_artifacts[f"analysis/evaluation_metrics_{target_arg}.csv"] = combined_metrics_path

    shared_metrics_root = Path("data") / "deployment_metrics" / str(target_arg)
    shared_metrics_root.mkdir(parents=True, exist_ok=True)
    shared_metrics_path = shared_metrics_root / f"evaluation_metrics_{target_arg}.csv"
    combined_metrics_df.to_csv(shared_metrics_path, index=False)

    sync_latest_artifacts(target_arg=target_arg, artifacts=deployment_artifacts)

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "evaluation_metrics_path": str(combined_metrics_path),
        "model_count": int(len(summary_rows)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training-fit scatter for deployment runs")
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Optional model name to plot")
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nAnalyzing training fit at {args.run_folder}{Style.RESET_ALL}")

    result = main(
        run_folder=args.run_folder,
        model_name=args.model,
    )

    print(f"{Fore.GREEN}Analysis folder: {result['output_dir']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Summary CSV: {result['summary_path']}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Training fit analysis completed.{Style.RESET_ALL}")
