from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Edit this list to include/exclude targets without using the CLI.
INCLUDE_TARGETS = ["npn", "mic", "pot", "disc", "cyto", "hemo"]

DEFAULT_TOP_N = 10
DEFAULT_MAX_RANK = 500


def _resolve_frequency_path(deployment_root: Path, target: str) -> Path | None:
    candidates = [
        deployment_root / target / "selection" / "feature_frequency.csv",
        deployment_root / target / "feature_frequency.csv",
        deployment_root / target / "01_feature_selection" / "feature_frequency.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_feature_column(df: pd.DataFrame) -> str:
    for candidate in ["feature", "feature_name"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No feature column found. Expected 'feature' or 'feature_name'.")


def _resolve_frequency_column(df: pd.DataFrame) -> str:
    for candidate in ["selection_count", "selection_fraction", "frequency", "count", "selection_freq"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No frequency column found. Expected one of: selection_count, selection_fraction, frequency, count, selection_freq."
    )


def _rank_target_features(df: pd.DataFrame, max_rank: int | None = None) -> pd.DataFrame:
    feature_col = _resolve_feature_column(df)
    freq_col = _resolve_frequency_column(df)

    df = df.copy()
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")

    sort_cols = [freq_col]
    sort_ascending = [False]
    if "mean_abs_coefficient" in df.columns:
        df["mean_abs_coefficient"] = pd.to_numeric(df["mean_abs_coefficient"], errors="coerce")
        sort_cols.append("mean_abs_coefficient")
        sort_ascending.append(False)
    sort_cols.append(feature_col)
    sort_ascending.append(True)

    ranked = df.sort_values(by=sort_cols, ascending=sort_ascending).reset_index(drop=True)
    if max_rank is not None:
        keep_n = min(max_rank, len(ranked))
        ranked = ranked.head(keep_n).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked = ranked.rename(columns={feature_col: "feature", freq_col: "frequency"})
    return ranked[["feature", "rank", "frequency"]]


def _build_rank_matrices(
    *,
    target_files: dict[str, Path],
    targets: Iterable[str],
    top_n: int,
    max_rank: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_tables: dict[str, pd.DataFrame] = {}
    all_features: set[str] = set()

    for target in targets:
        path = target_files.get(target)
        if path is None:
            continue
        df = pd.read_csv(path)
        ranked = _rank_target_features(df, max_rank=max_rank)
        rank_tables[target] = ranked
        top_features = ranked.loc[ranked["rank"] <= top_n, "feature"].astype(str).tolist()
        all_features.update(top_features)

    if not rank_tables:
        raise ValueError("No feature tables were loaded. Check target list and deployment paths.")

    ordered_targets = [target for target in targets if target in rank_tables]
    top_rank_df = pd.DataFrame(index=sorted(all_features), columns=ordered_targets, dtype=float)
    extra_rank_df = pd.DataFrame(index=sorted(all_features), columns=ordered_targets, dtype=float)
    valid_index = top_rank_df.index

    for target, ranked in rank_tables.items():
        rank_series = pd.Series(ranked["rank"].values, index=ranked["feature"].astype(str))
        if rank_series.index.has_duplicates:
            rank_series = rank_series[~rank_series.index.duplicated(keep="first")]
        top_series = rank_series[rank_series <= top_n]
        extra_series = rank_series[rank_series > top_n]
        top_series = top_series.loc[top_series.index.intersection(valid_index)]
        extra_series = extra_series.loc[extra_series.index.intersection(valid_index)]
        top_rank_df.loc[top_series.index, target] = top_series
        extra_rank_df.loc[extra_series.index, target] = extra_series

    top_rank_df = top_rank_df.dropna(how="all")
    extra_rank_df = extra_rank_df.reindex(index=top_rank_df.index, columns=top_rank_df.columns)
    return top_rank_df, extra_rank_df


def _apply_sorting(rank_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = rank_df.copy()
    sorted_df["mean_rank"] = sorted_df.mean(axis=1, skipna=True)
    sorted_df = (
        sorted_df.reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(by=["mean_rank", "feature"], ascending=[True, True])
        .set_index("feature")
    )
    return sorted_df


def _plot_heatmap(
    *,
    rank_df: pd.DataFrame,
    extra_rank_df: pd.DataFrame,
    output_path: Path,
    top_n: int,
    title: str,
) -> None:
    plot_df = rank_df.copy()
    if "mean_rank" in plot_df.columns:
        plot_df = plot_df.drop(columns=["mean_rank"])

    n_features = plot_df.shape[0]
    n_targets = plot_df.shape[1]

    fig_width = max(7.0, n_targets * 1.1)
    fig_height = max(6.0, n_features * 0.35)

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    mask = plot_df.isna()
    annot = plot_df.copy()
    annot = annot.applymap(lambda value: "" if pd.isna(value) else f"{int(value)}")

    cmap = sns.color_palette("Reds_r", as_cmap=True)
    cmap.set_bad(color="white")
    norm = mcolors.Normalize(vmin=1, vmax=max(top_n, 1))

    annot_size = 10
    if n_features > 40:
        annot_size = 7
    elif n_features > 25:
        annot_size = 8

    sns.heatmap(
        plot_df,
        mask=mask,
        cmap=cmap,
        norm=norm,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="#E6E6E6",
        cbar_kws={"label": "Rank (1 = highest)"},
        annot_kws={"fontsize": annot_size},
        ax=ax,
    )

    extra_fontsize = max(6, annot_size - 3)
    extra_df = extra_rank_df.reindex(index=plot_df.index, columns=plot_df.columns)
    for row_idx, _ in enumerate(plot_df.index):
        for col_idx, _ in enumerate(plot_df.columns):
            extra_value = extra_df.iat[row_idx, col_idx]
            if pd.isna(extra_value):
                continue
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{int(extra_value)}",
                ha="center",
                va="center",
                color="black",
                fontsize=extra_fontsize,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a heatmap of top-N deployment feature ranks across targets."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top features to keep per target.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/additional/feature_importance_heatmap",
        help="Folder to save the ranked tables and heatmap.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Optional comma-separated list of targets to include (overrides INCLUDE_TARGETS).",
    )
    args = parser.parse_args()

    deployment_root = Path("outputs/deployment")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.targets:
        targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    else:
        targets = list(INCLUDE_TARGETS)

    if not targets:
        raise ValueError("No targets specified. Provide --targets or edit INCLUDE_TARGETS.")

    target_files: dict[str, Path] = {}
    missing_targets: list[str] = []

    for target in targets:
        resolved = _resolve_frequency_path(deployment_root, target)
        if resolved is None:
            missing_targets.append(target)
            continue
        target_files[target] = resolved

    if missing_targets:
        print(f"Skipping targets with missing feature_frequency.csv: {', '.join(missing_targets)}")

    if not target_files:
        raise FileNotFoundError("No feature_frequency.csv files found. Check deployment_root or targets.")

    max_rank = max(args.top_n, DEFAULT_MAX_RANK)
    rank_df, extra_rank_df = _build_rank_matrices(
        target_files=target_files,
        targets=targets,
        top_n=args.top_n,
        max_rank=max_rank,
    )
    rank_df = _apply_sorting(rank_df)
    target_columns = [col for col in rank_df.columns if col != "mean_rank"]
    extra_rank_df = extra_rank_df.reindex(index=rank_df.index, columns=target_columns)

    ranked_table_path = output_dir / f"feature_rank_table_top{args.top_n}.csv"
    rank_df.to_csv(ranked_table_path)

    heatmap_data = rank_df.drop(columns=["mean_rank"])
    heatmap_table_path = output_dir / f"feature_rank_matrix_top{args.top_n}.csv"
    heatmap_data.to_csv(heatmap_table_path)

    extra_table_path = output_dir / f"feature_rank_matrix_extra_top{args.top_n}_max{max_rank}.csv"
    extra_rank_df.to_csv(extra_table_path)

    plot_path = output_dir / f"feature_rank_heatmap_top{args.top_n}.pdf"
    title = ""#f"Top {args.top_n} Feature Ranks by Target"
    _plot_heatmap(
        rank_df=rank_df,
        extra_rank_df=extra_rank_df,
        output_path=plot_path,
        top_n=args.top_n,
        title=title,
    )

    print("Saved ranked table:", ranked_table_path)
    print("Saved heatmap matrix:", heatmap_table_path)
    print("Saved extra rank matrix:", extra_table_path)
    print("Saved heatmap:", plot_path)


if __name__ == "__main__":
    main()
