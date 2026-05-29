from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from colorama import Fore, Style
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def load_run_settings(run_folder: Path) -> dict[str, Any]:
    config_path = run_folder / "run_config.yaml"
    settings: dict[str, Any] = {}

    if config_path.exists():
        with config_path.open("r") as file_handle:
            settings = yaml.safe_load(file_handle) or {}

    return settings


def _format_final_model_value(raw_value: Any) -> str:
    if raw_value is None:
        return "N/A"

    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, list):
        parts: list[str] = []
        for item in raw_value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                model_name = str(item[0]).strip()
                model_trials = str(item[1]).strip()
                if model_name and model_trials:
                    parts.append(f"{model_name}:{model_trials}")
                continue

        return ",".join(parts) if parts else "N/A"

    return "N/A"


def _format_optuna_text(settings: dict[str, Any]) -> str:
    if settings.get("use_standard_hyperparams", False):
        return "default parameters"

    trial_value = settings.get("model_n_trials", "N/A")
    return f"optuna_trials={trial_value}"


def add_common_figure_footer(fig, settings: dict[str, Any], y_offset: float = 0.01) -> None:
    final_model_value = settings.get("final_model_name", settings.get("final_model_names", "N/A"))
    final_model_text = _format_final_model_value(final_model_value)
    optuna_text = _format_optuna_text(settings)
    footer = (
        f"target={settings.get('target', 'N/A')}, "
        f"final_model={final_model_text}, "
        f"pyramid_model={settings.get('pyramid_model_name', 'N/A')}, "
        f"scale_target={settings.get('scale_target', 'N/A')}, "
        f"feature_option={settings.get('feature_option', 'N/A')}, "
        f"top_percentage={settings.get('top_percentage', 'N/A')}, "
        f"set_size={settings.get('set_size', settings.get('target_set_size', 'N/A'))}, "
        f"outer_seeds={settings.get('num_outer_seeds', settings.get('number_of_outer_seeds', 'N/A'))}, "
        f"inner_seeds={settings.get('num_pyramid_seeds', settings.get('number_of_inner_seeds', 'N/A'))}, "
        f"start_seed={settings.get('start_outer_seed', settings.get('start_seed', 'N/A'))}, "
        f"{optuna_text}"
    )
    fig.text(0.5, y_offset, footer, ha="center", va="bottom", fontsize=9, color="gray")


def plot_seed_metric_violins(
    seed_metrics_df: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    combined_pdf: PdfPages | None = None,
) -> Path | None:
    if seed_metrics_df.empty:
        print(Fore.YELLOW + "No seed-level metrics found; skipping violin metrics plot." + Style.RESET_ALL)
        return None

    metrics = [
        ("rmse", "RMSE", "#122F6B"),
        ("r2", "R2", "#0F680F"),
        ("mae", "MAE", "#D32E2E"),
        ("spearman", "Spearman", "#1B998B"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
    axes = axes.ravel()
    rng = np.random.default_rng(42)

    for ax, (metric_col, title, color) in zip(axes, metrics):
        metric_values = pd.to_numeric(seed_metrics_df.get(metric_col), errors="coerce").dropna()
        if metric_values.empty:
            continue

        violin_parts = ax.violinplot(
            [metric_values.values],
            positions=[1],
            widths=0.7,
            showmeans=False,
            showmedians=True,
        )
        for body in violin_parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("#2b2b2b")
            body.set_linewidth(1.4)
            body.set_alpha(0.35)

        if "cbars" in violin_parts:
            violin_parts["cbars"].set_color("#3a3a3a")
            violin_parts["cbars"].set_linewidth(1.2)
        if "cmins" in violin_parts:
            violin_parts["cmins"].set_color("#3a3a3a")
            violin_parts["cmins"].set_linewidth(1.2)
        if "cmaxes" in violin_parts:
            violin_parts["cmaxes"].set_color("#3a3a3a")
            violin_parts["cmaxes"].set_linewidth(1.2)
        if "cmedians" in violin_parts:
            violin_parts["cmedians"].set_color("#2b2b2b")
            violin_parts["cmedians"].set_linewidth(2.2)

        x_jitter = 1 + rng.uniform(-0.05, 0.05, size=len(metric_values))
        ax.scatter(
            x_jitter,
            metric_values.values,
            s=10,
            color="#7a7a7a",
            alpha=0.85,
            zorder=3,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([1])
        ax.set_xticklabels(["All outer seeds"])
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 1 · Distribution of Outer-Seed Aggregated Metrics", fontsize=14, fontweight="bold", y=1.08)
    fig.text(0.5, 1.04, "One dot per seed, violin shows seed-level metric distribution", ha="center", va="top", fontsize=10, color="gray")
    add_common_figure_footer(fig, settings, y_offset=-0.04)

    output_path = output_dir / "01_seed_metric_violin_plots.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    if combined_pdf is not None:
        combined_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_seed_spearman_violin(
    seed_metrics_df: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    combined_pdf: PdfPages | None = None,
) -> Path | None:
    if seed_metrics_df.empty:
        print(Fore.YELLOW + "No seed-level metrics found; skipping spearman violin plot." + Style.RESET_ALL)
        return None

    metric_values = pd.to_numeric(seed_metrics_df.get("spearman"), errors="coerce").dropna()
    if metric_values.empty:
        print(Fore.YELLOW + "No spearman metrics found; skipping spearman violin plot." + Style.RESET_ALL)
        return None

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.2), constrained_layout=True)
    rng = np.random.default_rng(42)

    violin_parts = ax.violinplot(
        [metric_values.values],
        positions=[1],
        widths=0.7,
        showmeans=False,
        showmedians=True,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#1B998B")
        body.set_edgecolor("#2b2b2b")
        body.set_linewidth(1.4)
        body.set_alpha(0.35)

    if "cbars" in violin_parts:
        violin_parts["cbars"].set_color("#3a3a3a")
        violin_parts["cbars"].set_linewidth(1.2)
    if "cmins" in violin_parts:
        violin_parts["cmins"].set_color("#3a3a3a")
        violin_parts["cmins"].set_linewidth(1.2)
    if "cmaxes" in violin_parts:
        violin_parts["cmaxes"].set_color("#3a3a3a")
        violin_parts["cmaxes"].set_linewidth(1.2)
    if "cmedians" in violin_parts:
        violin_parts["cmedians"].set_color("#2b2b2b")
        violin_parts["cmedians"].set_linewidth(2.2)

    x_jitter = 1 + rng.uniform(-0.05, 0.05, size=len(metric_values))
    ax.scatter(
        x_jitter,
        metric_values.values,
        s=10,
        color="#7a7a7a",
        alpha=0.85,
        zorder=3,
    )

    ax.set_title("Spearman", fontsize=12, fontweight="bold")
    ax.set_xticks([1])
    ax.set_xticklabels(["All outer seeds"])
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 1B · Spearman Correlation Across Outer Seeds", fontsize=14, fontweight="bold", y=1.06)
    fig.text(0.5, 1.02, "One dot per seed, violin shows seed-level distribution", ha="center", va="top", fontsize=10, color="gray")
    add_common_figure_footer(fig, settings, y_offset=-0.04)

    output_path = output_dir / "01b_seed_spearman_violin_plot.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    if combined_pdf is not None:
        combined_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_selection_percentage_barh(
    data: pd.DataFrame,
    output_dir: Path,
    plot_mode: str,
    settings: dict[str, Any],
    subtitle_meta: dict[str, Any] | None = None,
    combined_pdf: PdfPages | None = None,
) -> Path | None:
    if data.empty:
        return None

    subtitle_meta = subtitle_meta or {}
    if plot_mode == "pyramid":
        output_path = output_dir / "02_pyramid_feature_selection_frequency_barplot.pdf"
        title = "Figure 2 · Pyramid Feature Selection Frequency"
        subtitle = (
            f"Percentage over pyramid selection events "
            f"(tables={subtitle_meta.get('table_count', 'N/A')}, "
            f"max occurrences={subtitle_meta.get('total_events', 'N/A')})"
        )
    elif plot_mode == "final":
        output_path = output_dir / "03_final_model_feature_selection_frequency_barplot.pdf"
        title = "Figure 3 · Final Model Feature Selection Frequency"
        subtitle = (
            f"Percentage of final models containing each feature "
            f"(total outer folds={subtitle_meta.get('total_folds', 'N/A')})"
        )
    else:
        raise ValueError(f"Unknown plot_mode '{plot_mode}'. Expected 'pyramid' or 'final'.")

    plot_df = data.sort_values("selection_pct", ascending=True).reset_index(drop=True)
    fig_h = max(7, 0.24 * len(plot_df) + 2.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(plot_df["feature"], plot_df["selection_pct"], color="#2e7d32", alpha=0.85)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    ax.set_xlim(0, 100)
    ax.margins(y=0)
    ax.set_xlabel("Selection percentage (%)")
    ax.set_ylabel("Feature")

    ax.set_yticks(np.arange(len(plot_df.index) - 0.5, -1, -5), minor=True)
    ax.grid(which="minor", color="#232323", linestyle="-", linewidth=2, alpha=0.85)
    ax.grid(axis="x", alpha=0.3)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(ax.get_xticks())
    top_ax.set_xticklabels(ax.get_xticklabels())

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.91)
    fig.text(0.5, 0.90, subtitle, ha="center", va="top", fontsize=10)

    add_common_figure_footer(fig, settings, y_offset=0.07)
    fig.savefig(output_path, bbox_inches="tight")
    if combined_pdf is not None:
        combined_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return output_path


def collect_pyramid_selection_counts(summary_df: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, int, int]:
    rows: list[pd.DataFrame] = []
    table_count = 0

    for _, row in summary_df.iterrows():
        summary_path = row.get("selection_summary_path")
        summary_path = settings["run_folder"] / summary_path if settings["run_folder"] is not None else summary_path
        if not summary_path:
            continue
        path = Path(str(summary_path))
        if not path.exists():
            continue
        try:
            table = pd.read_csv(path)
        except Exception:
            continue

        if "feature" not in table.columns or "selection_count" not in table.columns:
            continue

        rows.append(table[["feature", "selection_count"]].copy())
        table_count += 1

    if not rows:
        return pd.DataFrame(columns=["feature", "selection_count", "selection_pct"]), 0, 0

    full_df = pd.concat(rows, ignore_index=True)
    summed = full_df.groupby("feature", as_index=False)["selection_count"].sum()

    inner_seeds = settings.get("num_pyramid_seeds", settings.get("number_of_inner_seeds", None))
    try:
        inner_seeds = int(inner_seeds)
    except Exception:
        inner_seeds = None
    if inner_seeds is None or inner_seeds <= 0:
        inner_seeds = int(full_df["selection_count"].max()) if not full_df.empty else 1

    total_events = max(1, table_count * inner_seeds)
    summed["selection_pct"] = 100.0 * summed["selection_count"] / total_events
    return summed.sort_values("selection_pct", ascending=False), table_count, total_events


def collect_final_selection_counts(summary_df: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    selected_counter: dict[str, int] = {}
    fold_count = int(len(summary_df))

    for _, row in summary_df.iterrows():
        selected_path = row.get("selected_features_path")
        selected_path = settings["run_folder"] / selected_path if settings["run_folder"] is not None else selected_path

        if not selected_path:
            continue

        payload_path = Path(str(selected_path))
        if not payload_path.exists():
            continue

        with payload_path.open("r") as file_handle:
            payload = json.load(file_handle)

        selected_features = [str(feature) for feature in payload.get("selected_features", [])]
        for feature in selected_features:
            selected_counter[feature] = selected_counter.get(feature, 0) + 1

    if not selected_counter:
        return pd.DataFrame(columns=["feature", "selection_count", "selection_pct"]), fold_count

    rows = []
    for feature, count in selected_counter.items():
        rows.append({"feature": feature, "selection_count": int(count)})

    final_df = pd.DataFrame(rows).sort_values(["selection_count", "feature"], ascending=[False, True]).reset_index(drop=True)
    final_df["selection_pct"] = 100.0 * final_df["selection_count"] / max(1, fold_count)
    return final_df, fold_count


def _build_fold_feature_binary_matrix(summary_df: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    fold_labels: list[str] = []
    feature_sets: list[set[str]] = []

    sorted_df = summary_df.sort_values(["outer_seed", "outer_fold"]).reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        outer_seed = int(row["outer_seed"])
        outer_fold = int(row["outer_fold"])
        fold_labels.append(f"seed{outer_seed}_fold{outer_fold}")

        selected_path = row.get("selected_features_path")
        selected_path = settings["run_folder"] / selected_path if settings["run_folder"] is not None else selected_path

        if not selected_path:
            feature_sets.append(set())
            continue

        payload_path = Path(str(selected_path))
        if not payload_path.exists():
            feature_sets.append(set())
            continue

        with payload_path.open("r") as file_handle:
            payload = json.load(file_handle)
        feature_sets.append(set(str(feature) for feature in payload.get("selected_features", [])))

    all_features = sorted(set().union(*feature_sets)) if feature_sets else []
    if not all_features:
        return pd.DataFrame()

    binary_matrix = pd.DataFrame(0, index=all_features, columns=fold_labels, dtype=int)
    for col, selected in zip(fold_labels, feature_sets):
        if selected:
            binary_matrix.loc[list(selected), col] = 1

    ordered_features = binary_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
    return binary_matrix.loc[ordered_features]


def _build_co_selection_matrix(binary_matrix: pd.DataFrame, top_n_features: int = 40) -> pd.DataFrame:
    if binary_matrix.empty:
        return pd.DataFrame()

    top_n_features = max(1, int(top_n_features))
    binary_matrix = binary_matrix.head(top_n_features).copy()

    values = binary_matrix.values.astype(float)
    feature_names = binary_matrix.index.tolist()
    raw_counts = values @ values.T
    return pd.DataFrame(raw_counts, index=feature_names, columns=feature_names)


def _compute_linkage_from_similarity(similarity_matrix: pd.DataFrame):
    if similarity_matrix.empty or len(similarity_matrix.index) < 3:
        return None

    sim = similarity_matrix.values.astype(float)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    sim = (sim + sim.T) / 2

    max_value = float(np.nanmax(sim)) if sim.size else 0.0
    if max_value > 1.0:
        sim = sim / max_value

    sim = np.clip(sim, 0.0, 1.0)
    distance = 1.0 - sim
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    condensed_distance = squareform(distance, checks=False)
    return linkage(condensed_distance, method="average")


def plot_co_selection_clustermap(
    matrix: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    combined_pdf: PdfPages | None = None,
) -> Path | None:
    if matrix.empty:
        print(Fore.YELLOW + "No co-selection data found; skipping co-selection heatmap." + Style.RESET_ALL)
        return None

    output_path = output_dir / "04_feature_co_selection_heatmap.pdf"

    n_features = len(matrix.index)
    fig_size = max(10, min(42, 0.28 * n_features + 6))
    linkage_matrix = _compute_linkage_from_similarity(matrix)

    clustermap_kwargs = {
        "data": matrix,
        "cmap": "RdBu_r",
        "linewidths": 0,
        "figsize": (fig_size, fig_size),
        "xticklabels": True,
        "yticklabels": True,
        "cbar_kws": {"label": "Co-selection count"},
    }

    if linkage_matrix is None:
        clustermap_kwargs.update({"row_cluster": False, "col_cluster": False})
    else:
        clustermap_kwargs.update({"row_linkage": linkage_matrix, "col_linkage": linkage_matrix})

    cg = sns.clustermap(**clustermap_kwargs)

    tick_fontsize = 5 if n_features > 90 else 6 if n_features > 60 else 7
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=tick_fontsize)
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=tick_fontsize)

    cg.fig.suptitle("Figure 4 · Feature Co-selection Frequency Heatmap", fontsize=14, fontweight="bold", y=1.02)
    cg.fig.text(
        0.5,
        0.99,
        "Top selected features across outer folds; values are pairwise co-selection counts",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )
    add_common_figure_footer(cg.fig, settings, y_offset=0.01)

    cg.fig.savefig(output_path, bbox_inches="tight")
    if combined_pdf is not None:
        combined_pdf.savefig(cg.fig, bbox_inches="tight")
    plt.close(cg.fig)
    return output_path


def plot_three_diagnostics(axes_row, y_true_values: np.ndarray, y_pred_values: np.ndarray, row_title: str) -> None:
    residuals = y_true_values - y_pred_values

    ax1 = axes_row[0]
    ax1.scatter(y_true_values, y_pred_values, alpha=0.75, color="#3454D1", edgecolor="white", linewidth=0.5)
    min_val = float(min(np.min(y_true_values), np.min(y_pred_values)))
    max_val = float(max(np.max(y_true_values), np.max(y_pred_values)))
    ax1.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", alpha=0.6)
    ax1.set_title(f"{row_title}: Predictions vs True Values", fontweight="bold")
    ax1.set_xlabel("True values")
    ax1.set_ylabel("Predicted values")
    ax1.grid(True, alpha=0.25)

    ax2 = axes_row[1]
    ax2.scatter(y_pred_values, residuals, alpha=0.75, color="#E4572E", edgecolor="white", linewidth=0.5)
    ax2.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    residual_abs_max = float(np.max(np.abs(residuals))) if residuals.size else 1.0
    if residual_abs_max <= 0:
        residual_abs_max = 1.0
    ax2.set_ylim(-1.05 * residual_abs_max, 1.05 * residual_abs_max)
    ax2.set_title(f"{row_title}: Residuals vs Predicted Values", fontweight="bold")
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residual (y_true - y_pred)")
    ax2.grid(True, alpha=0.25)

    ax3 = axes_row[2]
    ax3.scatter(y_true_values, residuals, alpha=0.75, color="#E4572E", edgecolor="white", linewidth=0.5)
    ax3.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    ax3.set_ylim(-1.05 * residual_abs_max, 1.05 * residual_abs_max)
    ax3.set_title(f"{row_title}: Residuals vs True Values", fontweight="bold")
    ax3.set_xlabel("True values")
    ax3.set_ylabel("Residual (y_true - y_pred)")
    ax3.grid(True, alpha=0.25)


def plot_train_test_diagnostics(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    settings: dict[str, Any],
    combined_pdf: PdfPages | None = None,
) -> Path | None:
    if train_df.empty or test_df.empty:
        print(Fore.YELLOW + "Missing train/test predictions; skipping combined diagnostics." + Style.RESET_ALL)
        return None

    output_path = output_dir / "05_train_test_predictions_and_residuals.pdf"

    train_true = train_df["y_true"].to_numpy(dtype=float)
    train_pred = train_df["y_pred"].to_numpy(dtype=float)
    test_true = test_df["y_true"].to_numpy(dtype=float)
    test_pred = test_df["y_pred"].to_numpy(dtype=float)

    train_metrics = _calc_metrics(train_true, train_pred)
    test_metrics = _calc_metrics(test_true, test_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    plot_three_diagnostics(axes[0], train_true, train_pred, "Training data")
    plot_three_diagnostics(axes[1], test_true, test_pred, "Test data")

    fig.suptitle("Figure 5 · Combined Train/Test Fit Diagnostics", fontsize=14, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        1.01,
        (
            f"Train RMSE={train_metrics['rmse']:.4f}, R2={train_metrics['r2']:.4f}, MAE={train_metrics['mae']:.4f} | "
            f"Test RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.4f}"
        ),
        ha="center",
        va="center",
        fontsize=11,
        color="gray",
    )
    add_common_figure_footer(fig, settings, y_offset=-0.03)
    fig.savefig(output_path, bbox_inches="tight")
    if combined_pdf is not None:
        combined_pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_evaluation_plots(
    *,
    run_folder: Path,
    output_dir: Path,
    summary_df: pd.DataFrame,
    seed_metrics_df: pd.DataFrame,
    all_train_df: pd.DataFrame,
    all_test_df: pd.DataFrame,
    model_name: str | None = None,
    model_optuna_trials: int | str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_run_settings(run_folder)
    if model_name is not None:
        settings["final_model_name"] = str(model_name)
    if model_optuna_trials is not None:
        settings["model_n_trials"] = model_optuna_trials

    combined_pdf_path = output_dir / "00_combined_plots.pdf"

    # Check if the paths in summary_df are absolute or relative, and if relative, resolve them against the run_folder
    if Path(summary_df["selected_features_path"].iloc[0]).is_absolute():
        settings["run_folder"] = None
    else:
        settings["run_folder"] = Path(run_folder).resolve()

    with PdfPages(combined_pdf_path) as combined_pdf:
        # Figure 1 - Seed-level metric distributions
        violin_plot = plot_seed_metric_violins(seed_metrics_df, output_dir, settings, combined_pdf=combined_pdf)

        # Figure 2 - Pyramid feature selection frequency
        pyramid_counts, pyramid_table_count, pyramid_total_events = collect_pyramid_selection_counts(summary_df, settings)
        if not pyramid_counts.empty:
            pyramid_csv = output_dir / "pyramid_feature_selection_frequency.csv"
            pyramid_counts.to_csv(pyramid_csv, index=False)
            pyramid_plot = _plot_selection_percentage_barh(
                data=pyramid_counts,
                output_dir=output_dir,
                plot_mode="pyramid",
                settings=settings,
                subtitle_meta={"table_count": pyramid_table_count, "total_events": pyramid_total_events},
                combined_pdf=combined_pdf,
            )
        else:
            print(Fore.YELLOW + "No pyramid feature selection data found; skipping pyramid selection frequency plot." + Style.RESET_ALL)

        # Figure 3 - Final model feature selection frequency
        final_counts, total_folds = collect_final_selection_counts(summary_df, settings)
        if not final_counts.empty:
            final_csv = output_dir / "final_model_feature_selection_frequency.csv"
            final_counts.to_csv(final_csv, index=False)
            final_plot = _plot_selection_percentage_barh(
                data=final_counts,
                output_dir=output_dir,
                plot_mode="final",
                settings=settings,
                subtitle_meta={"total_folds": total_folds},
                combined_pdf=combined_pdf,
            )
        else:
            print(Fore.YELLOW + "No final model feature selection data found; skipping final selection frequency plot." + Style.RESET_ALL)

        # Figure 4 - Feature co-selection heatmap
        binary_matrix = _build_fold_feature_binary_matrix(summary_df, settings)
        if not binary_matrix.empty:
            binary_csv = output_dir / "feature_binary_selection_matrix.csv"
            binary_matrix.to_csv(binary_csv)

            co_selection_matrix = _build_co_selection_matrix(binary_matrix, top_n_features=40)
            if not co_selection_matrix.empty:
                co_selection_csv = output_dir / "feature_co_selection_counts.csv"
                co_selection_matrix.to_csv(co_selection_csv)

                co_selection_plot = plot_co_selection_clustermap(
                    co_selection_matrix,
                    output_dir=output_dir,
                    settings=settings,
                    combined_pdf=combined_pdf,
                )
            else:
                print(Fore.YELLOW + "Co-selection matrix is empty; skipping co-selection heatmap." + Style.RESET_ALL)

        # Figure 5 - Combined train/test diagnostics
        train_test_plot = plot_train_test_diagnostics(
            train_df=all_train_df,
            test_df=all_test_df,
            output_dir=output_dir,
            settings=settings,
            combined_pdf=combined_pdf,
        )

