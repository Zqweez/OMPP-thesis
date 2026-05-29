from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

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

MODEL_ORDER = ["ridge", "elasticnet", "random_forest", "gradient_boosted"]
MODEL_LABELS = {
    "ridge": "Ridge",
    "elasticnet": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosted": "Gradient Boosted",
}
MODEL_COLORS = {
    "ridge": "#75AAFF",
    "elasticnet": "#96DEEF",
    "random_forest": "#a3de8a",
    "gradient_boosted": "#F1B66E",
}
"""MODEL_COLORS = {
    "ridge": "#6AACDB",
    "elasticnet": "#F4A94E",
    "random_forest": "#50D650",
    "gradient_boosted": "#E370ED",
}"""
MODEL_LABEL_TO_KEY = {label: key for key, label in MODEL_LABELS.items()}

ALL_FEATURES_LABEL = "All Features"
BASELINE_LABEL = "Lab Baseline"
MEAN_BASELINE_LABEL = "Mean Baseline"
BASELINE_RESULTS_DIRS = {
    "evaluation": Path("data/evaluation_metrics/baselines"),
    "deployment": Path("data/deployment_metrics/baselines"),
}
DEFAULT_BASELINE_SOURCE = "evaluation"

METRIC_SPECS = [
    ("R2", "r2", "r2_std"),
    ("RMSE", "rmse", "rmse_std"),
    ("MAE", "mae", "mae_std"),
    ("Spearman", "spearman", "spearman_std"),
]
METHOD_TYPE_COLORS = {
    ALL_FEATURES_LABEL: "#EBA6DF",
    BASELINE_LABEL: "#EE7474",
    MEAN_BASELINE_LABEL: "#747880",
}
"""METHOD_TYPE_COLORS = {
    ALL_FEATURES_LABEL: "#719189",
    BASELINE_LABEL: "#E24A4A",
    MEAN_BASELINE_LABEL: "#E8BC2B",
}"""

def _compute_metric_bounds(values: np.ndarray, errors: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    safe_errors = np.where(np.isfinite(errors), np.maximum(errors, 0.0), 0.0)
    lower_candidates = values - safe_errors
    upper_candidates = values + safe_errors

    valid_mask = np.isfinite(lower_candidates) & np.isfinite(upper_candidates)
    if not np.any(valid_mask):
        return (-1.0, 1.0)

    lower = float(np.min(lower_candidates[valid_mask]))
    upper = float(np.max(upper_candidates[valid_mask]))

    lower = min(lower, 0.0)
    upper = max(upper, 0.0)

    if upper <= lower:
        span = abs(upper) if upper != 0 else 1.0
        return (lower - 0.5 * span, upper + 0.5 * span)

    pad = (upper - lower) * float(max(pad_fraction, 0.0))
    return (lower - pad, upper + pad)


def _normalize_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(column).strip().lower() for column in df.columns]
    if "target" not in df.columns or "model" not in df.columns:
        raise ValueError("Metrics CSV is missing required 'target' or 'model' columns.")

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df[REQUIRED_COLUMNS].copy()


def _merge_metric_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in REQUIRED_COLUMNS if column not in {"target", "model"}]
    metrics_df = metrics_df.copy()
    metrics_df["target"] = metrics_df["target"].astype(str)
    metrics_df["model"] = metrics_df["model"].astype(str)

    rows: list[dict] = []
    for (target, model), group in metrics_df.groupby(["target", "model"], dropna=False):
        row = {"target": target, "model": model}
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                row[column] = float("nan")
                continue

            unique_values = np.unique(values.to_numpy(dtype=float))
            if unique_values.size > 1 and not np.allclose(unique_values, unique_values[0]):
                raise ValueError(
                    "Conflicting metric values found for "
                    f"target='{target}', model='{model}', column='{column}'."
                )
            row[column] = float(unique_values[0])
        rows.append(row)

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _read_results_dirs(metrics_roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metrics_root in metrics_roots:
        if not metrics_root.exists():
            raise FileNotFoundError(f"Metrics directory does not exist: {metrics_root}")
        csv_files = sorted(metrics_root.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under: {metrics_root}")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            try:
                normalized = _normalize_metric_frame(df)
            except ValueError:
                continue
            frames.append(normalized)

    if not frames:
        raise ValueError("No metric CSVs with 'target' and 'model' columns found under provided paths.")

    combined = pd.concat(frames, ignore_index=True)
    return _merge_metric_rows(combined)


def _resolve_baseline_dir(baseline_source: str) -> Path:
    source = str(baseline_source).strip().lower()
    if source not in BASELINE_RESULTS_DIRS:
        raise ValueError(
            "Baseline source must be one of: " + ", ".join(sorted(BASELINE_RESULTS_DIRS.keys()))
        )
    return BASELINE_RESULTS_DIRS[source]



def _pick_first_value(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _extract_metrics_from_row(row: pd.Series, prefix: str | None) -> dict[str, float]:
    base = f"{prefix}_" if prefix else ""
    metrics: dict[str, float] = {}
    for metric in ["r2", "rmse", "mae", "spearman"]:
        metrics[metric] = _safe_float(row.get(f"{base}{metric}"))
        metrics[f"{metric}_std"] = _safe_float(row.get(f"{base}{metric}_std"))
    return metrics


def _extract_dummy_metrics(df_target: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in ["r2", "rmse", "mae", "spearman"]:
        metrics[metric] = _pick_first_value(df_target[f"dummy_{metric}"])
        metrics[f"{metric}_std"] = _pick_first_value(df_target[f"dummy_{metric}_std"])
    return metrics


def _row_has_values(metrics: dict[str, float]) -> bool:
    return any(np.isfinite(value) for value in metrics.values())


def _build_target_method_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df_target = df[df["target"].astype(str) == str(target)].copy()
    if df_target.empty:
        return pd.DataFrame(columns=["method", "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std", "spearman", "spearman_std"])

    df_target["model_key"] = df_target["model"].astype(str).str.lower()
    model_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for model_key in MODEL_ORDER:
        matches = df_target.loc[df_target["model_key"] == model_key]
        if matches.empty:
            continue
        row = matches.iloc[0]
        model_label = MODEL_LABELS.get(model_key, model_key)

        model_metrics = _extract_metrics_from_row(row, None)
        if _row_has_values(model_metrics):
            model_rows.append({"method": f"{model_label}", **model_metrics})

        if model_key == "ridge":
            allfeats_metrics = _extract_metrics_from_row(row, "allfeats")
            if _row_has_values(allfeats_metrics):
                baseline_rows.append({"method": f"{ALL_FEATURES_LABEL}", **allfeats_metrics})

            baseline_metrics = _extract_metrics_from_row(row, "baseline")
            if _row_has_values(baseline_metrics):
                baseline_rows.append({"method": f"{BASELINE_LABEL}", **baseline_metrics})

    dummy_metrics = _extract_dummy_metrics(df_target)
    if _row_has_values(dummy_metrics):
        baseline_rows.append({"method": MEAN_BASELINE_LABEL, **dummy_metrics})

    rows = model_rows + baseline_rows
    return pd.DataFrame(rows, columns=["method", "r2", "r2_std", "rmse", "rmse_std", "mae", "mae_std", "spearman", "spearman_std"])


def _add_dashed_errorbars(
    *,
    axis: plt.Axes,
    bar_container,
    errors: np.ndarray,
    color: str,
) -> None:
    for patch, error in zip(bar_container.patches, errors):
        if not np.isfinite(error) or float(error) < 0:
            continue
        x_center = patch.get_x() + patch.get_width() / 2.0
        y_value = patch.get_height()
        axis.errorbar(
            x_center,
            y_value,
            yerr=float(error),
            fmt="none",
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            linestyle="--",
            alpha=0.95,
            zorder=6,
        )


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_facecolor("#FFFFFF")
    axis.set_title(title, fontsize=16, pad=10)
    axis.grid(axis="y", alpha=0.18)
    axis.grid(axis="x", visible=False)


def _resolve_method_color(method_label: str) -> str:
    if method_label == MEAN_BASELINE_LABEL:
        return METHOD_TYPE_COLORS[MEAN_BASELINE_LABEL]
    if method_label in MODEL_LABEL_TO_KEY:
        model_key = MODEL_LABEL_TO_KEY.get(method_label)
        if model_key in MODEL_COLORS:
            return MODEL_COLORS[model_key]
    if method_label.endswith(f"{ALL_FEATURES_LABEL}"):
        return METHOD_TYPE_COLORS[ALL_FEATURES_LABEL]
    if method_label.endswith(f"{BASELINE_LABEL}"):
        return METHOD_TYPE_COLORS[BASELINE_LABEL]
    return "#6B7280"


def _plot_metric_subplot(
    axis: plt.Axes,
    table: pd.DataFrame,
    metric: str,
    metric_std: str,
    title: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    methods = table["method"].astype(str).tolist()
    values = table[metric].to_numpy(dtype=float)
    errors = table[metric_std].to_numpy(dtype=float)
    x = np.arange(len(methods), dtype=float)

    colors = [_resolve_method_color(method) for method in methods]
    bars = axis.bar(
        x,
        values,
        color=colors,
        edgecolor="#1F2937",
        linewidth=1.0,
        alpha=0.92,
        zorder=3,
    )

    _add_dashed_errorbars(axis=axis, bar_container=bars, errors=errors, color="#1F2937")

    if y_limits is not None:
        axis.set_ylim(y_limits[0], y_limits[1])
    elif metric in {"r2", "spearman"}:
        axis.set_ylim(-1, 1)
    else:
        ymin, ymax = _compute_metric_bounds(values, errors)
        axis.set_ylim(ymin, ymax)
    axis.axhline(0.0, color="#4B5563", linewidth=1.0, alpha=0.6)

    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=30, ha="right", fontsize=12)
    _style_axis(axis, title=title)

def plot_all_targets_metrics(
    *,
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    targets = sorted(results_df["target"].astype(str).dropna().unique().tolist())
    targets = targets[3:6]
    if not targets:
        return None

    present_models = set(results_df["model"].astype(str).str.lower().tolist())
    method_order = [
        MODEL_LABELS.get(model_key, model_key)
        for model_key in MODEL_ORDER
        if model_key in present_models
    ]
    method_order.extend([ALL_FEATURES_LABEL, BASELINE_LABEL, MEAN_BASELINE_LABEL])

    target_tables: list[tuple[str, pd.DataFrame]] = []
    for target in targets:
        table = _build_target_method_table(results_df, target)
        if table.empty:
            continue
        table = table.set_index("method").reindex(method_order).reset_index()
        target_tables.append((target, table))

    if not target_tables:
        return None

    sns.set_theme(style="whitegrid", context="talk")

    n_targets = len(target_tables)
    fig_height = max(6.0, 2.8 * n_targets)
    fig, axes = plt.subplots(n_targets, 4, figsize=(14, fig_height), squeeze=False, sharex="col")
    fig.patch.set_facecolor("#F8F9FA")

    y_tick_label_size = 9
    y_major_ticks = 6

    for row_index, (target, table) in enumerate(target_tables):
        for col_index, (label, metric, metric_std) in enumerate(METRIC_SPECS):
            axis = axes[row_index, col_index]
            _plot_metric_subplot(axis, table, metric=metric, metric_std=metric_std, title=label)

            axis.tick_params(axis="y", labelsize=y_tick_label_size)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=y_major_ticks, min_n_ticks=4))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
            axis.grid(axis="y", which="minor", alpha=0.12)
            metric_label = metric.upper()
            if metric_label == "R2":
                metric_label = r"R$^2$"

            axis.set_ylabel(metric_label, fontsize=12, color="#111827")


            if row_index != n_targets - 1:
                axis.tick_params(axis="x", bottom=False, labelbottom=False)

        axes[row_index, 0].text(
            -0.15,
            1.16,
            f"Target: {target}",
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            color="#111827",
        )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.26)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "all_targets_metrics.pdf"

    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return output_path


def plot_r2_grid(
    *,
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    targets = sorted(results_df["target"].astype(str).dropna().unique().tolist())
    if not targets:
        return None

    selected_targets = targets[:6]
    target_tables: list[tuple[str, pd.DataFrame]] = []
    for target in selected_targets:
        table = _build_target_method_table(results_df, target)
        if table.empty:
            continue
        target_tables.append((target, table))

    if not target_tables:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    fig.patch.set_facecolor("#F8F9FA")

    r2_limits = (-1, 1)
    last_row_with_data = max(index // 3 for index in range(len(target_tables)))

    for index, axis in enumerate(axes.flatten()):
        if index >= len(target_tables):
            axis.set_visible(False)
            continue

        target, table = target_tables[index]
        _plot_metric_subplot(
            axis,
            table,
            metric="r2",
            metric_std="r2_std",
            title=target,
            y_limits=r2_limits,
        )
        # Change title font size slightly
        axis.set_title(target, fontsize=20, pad=8)
        axis.set_xticklabels(axis.get_xticklabels(), fontsize=14)

        row = index // 3
        col = index % 3
        if col != 0:
            axis.tick_params(axis="y", left=False, labelleft=False)
        if row != last_row_with_data:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)

        if col == 0:
            metric_label = r"R$^2$"
            axis.set_ylabel(metric_label, fontsize=16, color="#111827")

    # fig.suptitle("R2 Comparison Across Targets", fontsize=20, fontweight="bold", y=1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r2_targets_grid.pdf"

    fig.tight_layout(w_pad=0.15, h_pad=0.15, rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return output_path

def plot_r2_spearman_grid(
    *,
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    targets = sorted(results_df["target"].astype(str).dropna().unique().tolist())
    if not targets:
        return None
    selected_targets = targets[3:6] # targets[3:6]
    target_tables: list[tuple[str, pd.DataFrame]] = []
    for target in selected_targets:
        table = _build_target_method_table(results_df, target)
        if table.empty:
            continue
        target_tables.append((target, table))

    if not target_tables:
        return None

    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col", sharey="row", squeeze=False)
    fig.patch.set_facecolor("#F8F9FA")

    r2_limits = (-1, 1)
    last_row_with_data = max(index // 3 for index in range(len(target_tables)))

    for index, axis in enumerate(axes.flatten()):
        row = index // 3
        col = index % 3

        target, table = target_tables[col]
        if row == 0:
            _plot_metric_subplot(
                axis,
                table,
                metric="r2",
                metric_std="r2_std",
                title=target,
                y_limits=r2_limits,
            )
            if col == 0:
                metric_label = r"R$^2$"
                axis.set_ylabel(metric_label, fontsize=16, color="#111827")
        else:
            _plot_metric_subplot(
                axis,
                table,
                metric="spearman",
                metric_std="spearman_std",
                title=target,
                y_limits=r2_limits,
            )
            if col == 0:
                metric_label = "Spearman"
                axis.set_ylabel(metric_label, fontsize=16, color="#111827")
        # Change title font size slightly
        axis.set_title(target, fontsize=20, pad=8)
        axis.set_xticklabels(axis.get_xticklabels(), fontsize=14)

    # fig.suptitle("R2 Comparison Across Targets", fontsize=20, fontweight="bold", y=1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "r2_spearman_targets_grid.pdf"

    fig.tight_layout(w_pad=0.15, h_pad=0.15, rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return output_path


def main(results_dir: Path, output_dir: Path, baseline_source: str = DEFAULT_BASELINE_SOURCE) -> None:
    baseline_dir = _resolve_baseline_dir(baseline_source)
    results_df = _read_results_dirs([results_dir, baseline_dir]) # Delete baseline_dir for prediction metrics
    targets = sorted(results_df["target"].astype(str).dropna().unique().tolist())
    if not targets:
        raise ValueError("No targets found in metrics data.")
    
    if output_dir is None:
        output_dir = Path("outputs/model-results") / results_dir.name

    combined_path = plot_all_targets_metrics(results_df=results_df, output_dir=output_dir)
    if combined_path is not None:
        print(f"Saved combined target plot: {combined_path}")

    grid_path = plot_r2_grid(results_df=results_df, output_dir=output_dir)
    if grid_path is not None:
        print(f"Saved R2 grid plot: {grid_path}")

    if True:
        grid_path = plot_r2_spearman_grid(results_df=results_df, output_dir=output_dir)
        if grid_path is not None:
            print(f"Saved R2/Spearman grid plot: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/model-results"),
        help="Path to a directory containing per-target metric CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saving target plots.",
    )
    parser.add_argument(
        "--baseline-source",
        type=str,
        choices=sorted(BASELINE_RESULTS_DIRS.keys()),
        default=DEFAULT_BASELINE_SOURCE,
        help="Which baseline metrics to use: evaluation or deployment.",
    )
    args = parser.parse_args()
    main(results_dir=args.results_dir, output_dir=args.output_dir, baseline_source=args.baseline_source)
