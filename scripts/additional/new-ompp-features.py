from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from src.features.compute import compute_features_for_sequences

MASTER_PATH = Path("data/OMPP_new_master.csv")
OUTPUT_DIR = Path("outputs/additional/ompp_descriptor_heatmap")
OUTPUT_FIG = OUTPUT_DIR / "ompp_descriptor_heatmap_allompps.pdf"

DESCRIPTORS = [
    "Length",
    "MW",
    "Charge",
    "Discrimination_Factor",
    "Mean_Hydrophobicity",
    "Hydrophobic_Moment",
    "pI",
    "BalabanJ",
    "T_Scale_3",
    "EState_VSA2"
]

RENAMNE_MAPPING = {
    "MW": "Molecular Weight",
    "Discrimination_Factor": "Discrimination Factor",
    "Mean_Hydrophobicity": "Mean Hydrophobicity",
    "Hydrophobic_Moment": "Hydrophobic Moment",
    "pI": "Isoelectric Point (pI)",
    "BalabanJ": "Balaban's J",
    "T_Scale_3": "T-Scale 3",
    "EState_VSA2": "EState VSA2",
}

ASSAY_COLUMNS = ["MIC_Mean", "NPN_Mean", "Potentiation_Mean"]
ASSAY_RENAME = {
    "MIC_Mean": "MIC",
    "NPN_Mean": "NPN",
    "Potentiation_Mean": "Potentiation",
}
ASSAY_ORDER = ["MIC", "NPN", "Potentiation"]
SAMPLE_COL = "Sample"
STAT_ROWS = ("Mean", "Std")
HIGHLIGHT_OMPP = {"53", "55"}

def _load_master_df() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {MASTER_PATH}")
    df = pd.read_csv(MASTER_PATH)
    if "Sequence" not in df.columns:
        raise ValueError("OMPP_new_master.csv must include a 'Sequence' column.")
    df["Sequence"] = df["Sequence"].astype(str).str.strip()
    return df

def _resolve_filter_column(master_df: pd.DataFrame, filter_df: pd.DataFrame) -> str:
    for candidate in ["OMPP nr", "Peptide", "Sequence"]:
        if candidate in filter_df.columns and candidate in master_df.columns:
            return candidate
    for candidate in filter_df.columns:
        if candidate in master_df.columns:
            return candidate
    if not filter_df.columns.tolist():
        raise ValueError("new_OMPPs.csv has no columns to use as identifiers.")
    return filter_df.columns[0]


def _normalize_ompp_label(label: str) -> str:
    text = str(label).strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"^ompp\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return match.group(1)
    return text


def _get_highlight_rows(index: pd.Index) -> list[int]:
    highlight_rows: list[int] = []
    highlight_set = {str(item) for item in HIGHLIGHT_OMPP}
    for row_idx, label in enumerate(index):
        normalized = _normalize_ompp_label(label)
        if normalized in highlight_set:
            highlight_rows.append(row_idx)
    return highlight_rows


def _apply_row_highlights(ax: plt.Axes, *, n_cols: int, row_indices: list[int]) -> None:
    for row_idx in row_indices:
        rect = Rectangle(
            (0, row_idx),
            n_cols,
            1,
            fill=False,
            edgecolor="#111111",
            linewidth=2,
            zorder=6,
        )
        ax.add_patch(rect)


def _highlight_row_labels(ax: plt.Axes, row_indices: list[int]) -> None:
    for row_idx, tick in enumerate(ax.get_yticklabels()):
        if row_idx in row_indices:
            tick.set_color("#B91C1C")
            tick.set_fontweight("bold")

def _format_assay_value(column: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if column == "NPN":
        return f"{value:.1f}"
    if column == "Potentiation":
        return f"{value:.2f}"
    return f"{value:g}"


def _format_descriptor_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2g}"


def _normalize_assays(assay_df: pd.DataFrame) -> pd.DataFrame:
    normalized = assay_df.copy()
    for col in assay_df.columns:
        values = assay_df[col].dropna()
        if values.empty:
            continue
        min_val = float(values.min())
        max_val = float(values.max())
        if np.isclose(min_val, max_val):
            normalized.loc[values.index, col] = 0.5
        else:
            normalized.loc[values.index, col] = (values - min_val) / (max_val - min_val)
    return normalized


def _build_final_table() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    master_df = _load_master_df()
    features_df = compute_features_for_sequences(master_df, include_raw_charge=True)

    descriptor_cols = [name for name in DESCRIPTORS if name in features_df.columns]
    missing = [name for name in DESCRIPTORS if name not in descriptor_cols]
    if missing:
        print(f"Warning: missing descriptor columns: {', '.join(missing)}")

    # Rename descriptors using RENAMNE_MAPPING for better readability in the heatmap
    features_df = features_df.rename(columns=RENAMNE_MAPPING)
    descriptor_cols = [RENAMNE_MAPPING.get(name, name) for name in descriptor_cols]

    if not descriptor_cols:
        raise ValueError("No requested descriptors were found in the computed features.")

    filter_df = pd.read_csv("data/OMPP_new_master.csv")
    filter_col = _resolve_filter_column(features_df, filter_df)
    filter_values = filter_df[filter_col].astype(str).str.strip().dropna()
    ordered_values = filter_values.unique()

    features_df[filter_col] = features_df[filter_col].astype(str).str.strip()
    subset_df = features_df[features_df[filter_col].isin(ordered_values)].copy()
    if subset_df.empty:
        raise ValueError("Filtering by new_OMPPs.csv removed all rows. Check the identifier column.")

    subset_df["_order"] = pd.Categorical(subset_df[filter_col], categories=ordered_values, ordered=True)
    subset_df = subset_df.sort_values("_order").drop(columns="_order")

    assays_df = subset_df[ASSAY_COLUMNS].rename(columns=ASSAY_RENAME)
    assays_df = assays_df.reindex(columns=ASSAY_ORDER)

    table_df = pd.concat(
        [subset_df[[filter_col]].rename(columns={filter_col: SAMPLE_COL}), assays_df, subset_df[descriptor_cols]],
        axis=1,
    )

    descriptor_means = features_df[descriptor_cols].mean(numeric_only=True)
    descriptor_stds = features_df[descriptor_cols].std(numeric_only=True)

    mean_row = {SAMPLE_COL: "Mean"}
    std_row = {SAMPLE_COL: "Std"}
    for col in ASSAY_ORDER:
        mean_row[col] = np.nan
        std_row[col] = np.nan
    for col in descriptor_cols:
        mean_row[col] = descriptor_means.get(col, np.nan)
        std_row[col] = descriptor_stds.get(col, np.nan)

    stats_df = pd.DataFrame([mean_row, std_row])
    final_df = pd.concat([table_df, stats_df], ignore_index=True)
    final_df = final_df.set_index(SAMPLE_COL)

    return final_df, descriptor_cols, descriptor_means, descriptor_stds


def _scale_descriptor_colors(
    descriptor_df: pd.DataFrame,
    descriptor_means: pd.Series,
    descriptor_stds: pd.Series,
) -> pd.DataFrame:
    descriptor_means = descriptor_means.reindex(descriptor_df.columns)
    descriptor_stds = descriptor_stds.reindex(descriptor_df.columns)

    data_mask = ~descriptor_df.index.isin(STAT_ROWS)
    centered = descriptor_df.subtract(descriptor_means, axis="columns")
    scale = centered.loc[data_mask].abs().max(axis=0)
    scale = scale.where(scale > 0, 1.0)

    scaled = centered.divide(scale, axis="columns")

    if "Std" in descriptor_df.index:
        scaled.loc["Std"] = 0

    return scaled


def _make_heatmap(
    table_df: pd.DataFrame,
    descriptor_cols: list[str],
    descriptor_means: pd.Series,
    descriptor_stds: pd.Series,
) -> None:
    assay_df = table_df[ASSAY_ORDER]
    descriptor_df = table_df[descriptor_cols]
    
    assay_color_df = _normalize_assays(assay_df)
    descriptor_color_df = _scale_descriptor_colors(
        descriptor_df,
        descriptor_means,
        descriptor_stds,
    )

    assay_annot = pd.DataFrame(index=assay_df.index, columns=assay_df.columns, dtype=object)
    for col in assay_df.columns:
        assay_annot[col] = assay_df[col].apply(lambda value: _format_assay_value(col, value))

    descriptor_annot = pd.DataFrame(index=descriptor_df.index, columns=descriptor_df.columns, dtype=object)
    for col in descriptor_df.columns:
        descriptor_annot[col] = descriptor_df[col].apply(_format_descriptor_value)

    n_rows = len(table_df.index)
    n_assays = len(ASSAY_ORDER)
    n_desc = len(descriptor_cols)
    fig_width = max(9.0, 0.6 * (n_assays + n_desc) + 2.0)
    fig_height = max(5.0, 0.4 * n_rows + 1.5)

    sns.set_theme(style="white", context="paper")
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [max(n_assays, 1), max(n_desc, 1)], "wspace": 0.2},
    )

    assay_cmap = sns.color_palette("Blues", as_cmap=True)
    descriptor_cmap = sns.color_palette("RdBu_r", as_cmap=True)

    sns.heatmap(
        assay_color_df,
        mask=assay_color_df.isna(),
        cmap=assay_cmap,
        vmin=0.0,
        vmax=1.0,
        annot=assay_annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Assay (normalized)"},
        ax=ax_left,
    )

    sns.heatmap(
        descriptor_color_df,
        mask=descriptor_color_df.isna(),
        cmap=descriptor_cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=descriptor_annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Relative to mean (per descriptor)"},
        ax=ax_right,
    )

    descriptor_cbar = ax_right.collections[0].colorbar
    descriptor_cbar.set_ticks([-1.0, 0.0, 1.0])
    descriptor_cbar.set_ticklabels(["Below mean", "Mean", "Above mean"])

    ax_left.set_xlabel("")
    ax_left.set_ylabel("")
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")

    ax_left.set_xticklabels(ax_left.get_xticklabels(), rotation=30, ha="right")
    ax_right.set_xticklabels(ax_right.get_xticklabels(), rotation=30, ha="right")

    ax_right.tick_params(axis="y", left=False, labelleft=False)

    highlight_rows = _get_highlight_rows(table_df.index)
    if highlight_rows:
        _apply_row_highlights(ax_left, n_cols=n_assays, row_indices=highlight_rows)
        _apply_row_highlights(ax_right, n_cols=n_desc, row_indices=highlight_rows)
        _highlight_row_labels(ax_left, highlight_rows)

    for spine in ax_left.spines.values():
        spine.set_visible(False)
    #ax_left.spines["right"].set_visible(True)
    #ax_left.spines["right"].set_linewidth(3.0)
    #ax_left.spines["right"].set_color("black")

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    table_df, descriptor_cols, descriptor_means, descriptor_stds = _build_final_table()
    _make_heatmap(table_df, descriptor_cols, descriptor_means, descriptor_stds)
    print(f"Saved heatmap to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
