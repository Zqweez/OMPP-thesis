from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style
from scipy.stats import spearmanr

from src.selection.shared import TARGET_MAPPING

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

MODEL_ORDER = ["ridge", "elasticnet", "random_forest", "gradient_boosted"]
MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "elasticnet": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosted": "Gradient Boosted",
}
MODEL_COLORS = {
    "ridge": "#6AACDB",
    "elasticnet": "#F4A94E",
    "random_forest": "#50D650",
    "gradient_boosted": "#E370ED",
}
MODEL_ALIASES = {
    "gradient_boosting": "gradient_boosted",
    "gbt": "gradient_boosted",
    "randomforest": "random_forest",
    "random-forest": "random_forest",
    "elastic_net": "elasticnet",
}


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _slugify(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
        }

    residuals = y_true - y_pred
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residuals**2) / denom) if denom > 0 and y_true.size >= 2 else float("nan")

    spearman = float("nan")
    if y_true.size >= 2 and y_pred.size >= 2:
        spearman_value, _ = spearmanr(y_true, y_pred)
        if np.isfinite(spearman_value):
            spearman = float(spearman_value)

    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}


def _canonical_model_key(model_name: str) -> str:
    key = model_name.lower()
    return MODEL_ALIASES.get(key, key)


def _empty_metrics_row(target_arg: str, model_name: str) -> dict[str, float | str]:
    row: dict[str, float | str] = {column: float("nan") for column in COMBINED_METRICS_COLUMNS}
    row["target"] = target_arg
    row["model"] = model_name
    return row


def _apply_metrics(row: dict[str, float | str], metrics: dict[str, float], prefix: str | None = None) -> None:
    metric_keys = ["r2", "rmse", "mae", "spearman"]
    for metric in metric_keys:
        column = f"{prefix}_{metric}" if prefix else metric
        row[column] = float(metrics.get(metric, np.nan))
        row[f"{column}_std"] = float("nan")


def _format_metrics_text(metrics: dict[str, float]) -> str:
    return (
        f"R2={metrics['r2']:.3f} | RMSE={metrics['rmse']:.3f} | "
        f"rho={metrics['spearman']:.3f}"
    )


def _axis_limits(values: list[np.ndarray]) -> tuple[float, float, float, float]:
    arrays = [array for array in values if array.size]
    if not arrays:
        return 0.0, 1.0, 0.0, 1.0

    combined = np.concatenate(arrays)
    if combined.size == 0 or not np.isfinite(combined).any():
        return 0.0, 1.0, 0.0, 1.0

    min_val = float(np.nanmin(combined))
    max_val = float(np.nanmax(combined))
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return 0.0, 1.0, 0.0, 1.0

    if min_val == max_val:
        padding = 1.0 if min_val == 0.0 else abs(min_val) * 0.1
    else:
        padding = 0.05 * (max_val - min_val)

    return min_val - padding, max_val + padding, min_val, max_val


def _plot_predictions_grid(
    *,
    plot_data: dict[str, dict[str, dict[str, object]]],
    target_order: list[str],
    output_path: Path,
) -> bool:
    if not target_order:
        return False

    nrows = len(target_order)
    ncols = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.6 * nrows),
        sharey="row",
        squeeze=False,
    )
    axes = np.atleast_2d(axes)

    for row_idx, target_arg in enumerate(target_order):
        target_data = plot_data.get(target_arg, {})
        row_values = []
        for model_key in MODEL_ORDER:
            data = target_data.get(model_key)
            if not data:
                continue
            y_true = data.get("y_true")
            y_pred = data.get("y_pred")
            if isinstance(y_true, np.ndarray):
                row_values.append(y_true)
            if isinstance(y_pred, np.ndarray):
                row_values.append(y_pred)

        plot_min, plot_max, line_min, line_max = _axis_limits(row_values)

        for col_idx, model_key in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            ax.grid(True, alpha=0.25)
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect("equal", "box")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(MODEL_DISPLAY_NAMES.get(model_key, model_key), fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{target_arg}\nPredicted")
            if row_idx == nrows - 1:
                ax.set_xlabel("True values")

            data = target_data.get(model_key)
            if not data:
                ax.text(0.5, 0.5, "No predictions", ha="center", va="center", color="gray")
                continue

            y_true = data.get("y_true")
            y_pred = data.get("y_pred")
            metrics = data.get("metrics")
            if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
                ax.text(0.5, 0.5, "No predictions", ha="center", va="center", color="gray")
                continue

            ax.scatter(
                y_true,
                y_pred,
                alpha=1,
                color=MODEL_COLORS.get(model_key, "#3454D1"),
                edgecolor="black",
                linewidth=0.5,
            )
            ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black", alpha=0.6)

            if isinstance(metrics, dict):
                metrics_text = _format_metrics_text(metrics)
                ax.text(
                    0.02,
                    0.98,
                    metrics_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="dimgray",
                )

    fig.suptitle("", fontsize=14, fontweight="bold") # Predicted vs True by Target and Model
    fig.tight_layout(w_pad=0, h_pad=0, rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(wspace=-0.4, hspace=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return True


def _select_best_models(
    plot_data: dict[str, dict[str, dict[str, object]]],
    target_order: list[str],
) -> list[tuple[str, str, dict[str, object]]]:
    best_models: list[tuple[str, str, dict[str, object]]] = []
    for target_arg in target_order:
        target_data = plot_data.get(target_arg, {})
        best_key: str | None = None
        best_r2 = float("-inf")
        for model_key in MODEL_ORDER:
            data = target_data.get(model_key)
            if not data:
                continue
            metrics = data.get("metrics")
            y_true = data.get("y_true")
            y_pred = data.get("y_pred")
            if not isinstance(metrics, dict) or not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
                continue
            if y_true.size == 0 or y_pred.size == 0:
                continue
            r2_value = float(metrics.get("r2", np.nan))
            if np.isfinite(r2_value) and r2_value > best_r2:
                best_r2 = r2_value
                best_key = model_key

        if best_key is None:
            for model_key in MODEL_ORDER:
                data = target_data.get(model_key)
                if not data:
                    continue
                y_true = data.get("y_true")
                y_pred = data.get("y_pred")
                if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray) and y_true.size and y_pred.size:
                    best_key = model_key
                    break

        if best_key:
            best_models.append((target_arg, best_key, target_data[best_key]))

    return best_models


def _plot_best_models_grid(
    *,
    best_models: list[tuple[str, str, dict[str, object]]],
    output_path: Path,
) -> bool:
    if not best_models:
        return False

    ncols = len(best_models)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.2 * ncols, 3.8),
        sharey=False,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = np.array([axes])

    for idx, (target_arg, model_key, data) in enumerate(best_models):
        ax = axes[idx]
        values = []
        y_true = data.get("y_true")
        y_pred = data.get("y_pred")
        if isinstance(y_true, np.ndarray):
            values.append(y_true)
        if isinstance(y_pred, np.ndarray):
            values.append(y_pred)
        plot_min, plot_max, line_min, line_max = _axis_limits(values)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", "box")
        ax.set_title(target_arg, fontweight="bold")
        ax.set_xlabel("True values")
        if idx == 0:
            ax.set_ylabel("Predicted")

        metrics = data.get("metrics")
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", color="gray")
            continue

        ax.scatter(
            y_true,
            y_pred,
            alpha=1,
            color=MODEL_COLORS.get(model_key, "#3454D1"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black", alpha=0.6)

        if isinstance(metrics, dict):
            metrics_text = _format_metrics_text(metrics)
            model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            ax.text(
                0.02,
                0.98,
                f"{model_name}\n{metrics_text}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="dimgray",
            )

    fig.suptitle("", fontsize=14, fontweight="bold")
    fig.tight_layout(w_pad=0, h_pad=0, rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(wspace=-0.45, hspace=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return True


def main(
    *,
    predictions_csv: str,
    targets: str | None = None,
) -> dict:
    repo_root = _find_repo_root()
    predictions_path = _resolve_path(repo_root, predictions_csv)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {predictions_path}")

    prediction_df = pd.read_csv(predictions_path)

    resolved_targets = sorted([
        (target_arg, target_column)
        for target_arg, target_column in TARGET_MAPPING.items()
        if target_column in prediction_df.columns and target_arg in ["mic","npn", "pot"]
    ])
    if not resolved_targets:
        raise ValueError("No target columns found in prediction CSV.")

    output_root = Path("outputs/predictions/analysis")
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_paths: list[str] = []
    plot_paths: list[str] = []
    plot_data: dict[str, dict[str, dict[str, object]]] = {}

    for target_arg, target_column in resolved_targets:
        prefix = f"{target_column}_"
        model_columns = sorted([col for col in prediction_df.columns if col.startswith(prefix)])
        if not model_columns:
            print(
                f"{Fore.YELLOW}No model predictions found for target '{target_arg}'. Expected columns starting with '{prefix}'.{Style.RESET_ALL}"
            )
            continue

        model_metrics: dict[str, dict[str, float]] = {}
        baseline_metrics: dict[str, dict[str, float]] = {}
        allfeats_metrics: dict[str, dict[str, float]] = {}
        plot_data[target_arg] = {}

        y_true_all = pd.to_numeric(prediction_df[target_column], errors="coerce").to_numpy(dtype=float)
        valid_true_mask = np.isfinite(y_true_all)
        y_true_clean = y_true_all[valid_true_mask]
        dummy_value = float(np.mean(y_true_clean)) if y_true_clean.size else float("nan")
        dummy_pred = np.full_like(y_true_clean, dummy_value, dtype=float)
        dummy_metrics = _calc_metrics(y_true_clean, dummy_pred)

        for model_col in model_columns:
            model_name = model_col[len(prefix) :]
            if not model_name:
                continue

            y_pred = pd.to_numeric(prediction_df[model_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y_true_all) & np.isfinite(y_pred)
            y_true = y_true_all[mask]
            y_pred = y_pred[mask]

            metrics = _calc_metrics(y_true, y_pred)

            model_key = model_name.lower()
            if model_key.endswith("_baseline"):
                base_model = model_name[: -len("_baseline")]
                if base_model:
                    baseline_metrics[base_model] = metrics
            elif model_key.endswith("_allfeats"):
                base_model = model_name[: -len("_allfeats")]
                if base_model:
                    allfeats_metrics[base_model] = metrics
            else:
                model_metrics[model_name] = metrics

            canonical_key = _canonical_model_key(model_name)
            if canonical_key in MODEL_ORDER:
                plot_data[target_arg][canonical_key] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "metrics": metrics,
                }

        model_names = sorted(set(model_metrics) | set(baseline_metrics) | set(allfeats_metrics))
        rows: list[dict[str, float | str]] = []
        for model_name in model_names:
            row = _empty_metrics_row(target_arg, model_name)
            if model_name in model_metrics:
                _apply_metrics(row, model_metrics[model_name], prefix=None)
            if model_name in allfeats_metrics:
                _apply_metrics(row, allfeats_metrics[model_name], prefix="allfeats")
            if model_name in baseline_metrics:
                _apply_metrics(row, baseline_metrics[model_name], prefix="baseline")
            _apply_metrics(row, dummy_metrics, prefix="dummy")
            rows.append(row)

        metrics_df = pd.DataFrame(rows, columns=COMBINED_METRICS_COLUMNS)
        metrics_path = output_root / "metrics" / f"{_slugify(target_arg)}" / f"evaluation_metrics_{_slugify(target_arg)}.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        metrics_paths.append(str(metrics_path))

    target_order = [target_arg for target_arg, _ in resolved_targets if target_arg in plot_data]
    if target_order:
        grid_path = output_root / "predictions_grid_all_models.pdf"
        if _plot_predictions_grid(plot_data=plot_data, target_order=target_order, output_path=grid_path):
            plot_paths.append(str(grid_path))

        best_models = _select_best_models(plot_data, target_order)
        best_path = output_root / "predictions_grid_best_models.pdf"
        if _plot_best_models_grid(best_models=best_models, output_path=best_path):
            plot_paths.append(str(best_path))

    return {
        "output_dir": str(output_root),
        "plot_paths": plot_paths,
        "metrics_paths": metrics_paths,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze deployment prediction outputs")
    parser.add_argument("--predictions", type=str, required=True, help="Prediction CSV from run_deployment_predictions.py")
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated list of target args or column names to analyze",
    )
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nAnalyzing prediction metrics...{Style.RESET_ALL}")

    result = main(
        predictions_csv=args.predictions,
        targets=args.targets,
    )

    print(f"{Fore.GREEN}Outputs saved to: {result['output_dir']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Plots generated: {len(result['plot_paths'])}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Prediction analysis completed.{Style.RESET_ALL}")
