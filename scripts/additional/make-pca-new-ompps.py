from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.features.compute import compute_features_for_sequences
from src.selection.shared import TARGET_MAPPING
EXCLUDE_COLS = {
    "Sequence",
    "OMPP nr",
    "Peptide",
    "Potentiation_Mean",
    "MIC_Mean",
    "NPN_Mean",
    "DISC_Mean",
    "Cytotoxicity_Mean",
    "Hemolysis_Mean",
}


def load_feature_sets(feature_sets_path: Path) -> dict[str, list[str]]:
    if not feature_sets_path.is_file():
        raise FileNotFoundError(f"Feature sets file not found: {feature_sets_path}")

    feature_sets_df = pd.read_csv(feature_sets_path)
    feature_sets: dict[str, list[str]] = {}
    for target_col in TARGET_MAPPING.values():
        if target_col not in feature_sets_df.columns:
            continue
        features = (
            feature_sets_df[target_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        features = [feature for feature in features if feature]
        feature_sets[target_col] = features

    return feature_sets


def resolve_feature_columns(
    full_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label: str,
) -> list[str]:
    missing_cols = [col for col in feature_cols if col not in full_df.columns]
    if missing_cols:
        print(
            "Warning: dropping missing features for",
            label,
            f"({len(missing_cols)} missing).",
        )
        feature_cols = [col for col in feature_cols if col in full_df.columns]

    if not feature_cols:
        raise ValueError(f"No usable features found for {label}.")

    return feature_cols


def run_pca(feature_df: pd.DataFrame, *, n_components: int = 3) -> tuple[pd.DataFrame, list[float]]:
    filled = feature_df.fillna(feature_df.mean(numeric_only=True))
    scaled = StandardScaler().fit_transform(filled)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled)
    pc_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(n_components)])
    explained_variance = pca.explained_variance_ratio_.tolist()
    return pc_df, explained_variance


def make_pair_plot(
    *,
    pc_df: pd.DataFrame,
    explained_variance: list[float],
    ompp_values: pd.Series,
    output_path: Path,
    plot_label: str | None = None,
) -> None:
    if ompp_values.isna().any():
        print("Warning: missing OMPP nr values detected; labeling as 'Unknown'.")

    ompp_values_str = ompp_values.astype(str).fillna("Unknown")
    ompp_labels = ompp_values_str
    ompp_unique = sorted(ompp_labels.unique(), key=str)
    edge_palette = sns.color_palette("tab10", n_colors=len(ompp_unique))
    ompp_colors = dict(zip(ompp_unique, edge_palette))

    for key, val in ompp_colors.items():
        # set color of OMPP nr below 48 to light gray for better visibility
        nr = key.lstrip("OMPP-").strip()
        if nr.isdigit() and int(nr) < 48:
            ompp_colors[key] = "#A0A0A0"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_specs = [(1, 2), (1, 3), (2, 3)]
    for ax, (pc_x, pc_y) in zip(axes, plot_specs):
        for label in ompp_unique:
            label_mask = ompp_values_str == label
            ax.scatter(
                pc_df.loc[label_mask, f"PC{pc_x}"],
                pc_df.loc[label_mask, f"PC{pc_y}"],
                s=60,
                alpha=1,
                c=[ompp_colors[label]],
                marker="o",
                edgecolors="white",
                linewidths=0.9,
            )

        ax.set_title(f"PCA: PC{pc_x} vs PC{pc_y}", fontsize=16)
        ax.set_xlabel(
            f"PC{pc_x} ({explained_variance[pc_x - 1] * 100:.1f}% variance)",
            fontsize=14,
        )
        ax.set_ylabel(
            f"PC{pc_y} ({explained_variance[pc_y - 1] * 100:.1f}% variance)",
            fontsize=14,
        )
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    title = "PCA Pairwise Scatter Plots | colored by OMPP nr"
    if plot_label:
        title = f"{title} | features: {plot_label}"
    fig.suptitle(
        title,
        fontsize=16,
        y=1,
    )
    fig.text(
        0.5,
        0.945,
        "Explained variance: "
        f"PC1={explained_variance[0]*100:.1f}%, "
        f"PC2={explained_variance[1]*100:.1f}%, "
        f"PC3={explained_variance[2]*100:.1f}%",
        ha="center",
        va="center",
        fontsize=11,
        color="gray",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.4,
            markersize=8,
            label=str(label),
        )
        for label, color in ompp_colors.items()
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            title="OMPP nr",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=min(8, len(legend_handles)),
            frameon=False,
            fontsize=14,
            title_fontsize=16,
        )

    fig.subplots_adjust(bottom=0.25, top=0.86, wspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(
    *,
    input_path: str,
    output_dir: str,
    mode: str,
    feature_sets_path: str,
) -> None:
    input_path = Path(input_path)
    output_root = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    source_df = pd.read_csv(input_path)
    if "Sequence" not in source_df.columns:
        raise ValueError("Input file must include a 'Sequence' column.")
    if "OMPP nr" not in source_df.columns:
        raise ValueError("Input file must include an 'OMPP nr' column for outline colors.")

    full_df = compute_features_for_sequences(df=source_df)

    if mode == "all":
        feature_cols = [col for col in full_df.columns if col not in EXCLUDE_COLS]
        feature_cols = resolve_feature_columns(full_df, feature_cols, label="all features")
        feature_df = full_df[feature_cols].select_dtypes(include="number")
        if feature_df.shape[1] < 3:
            raise ValueError("Fewer than 3 numeric features available for PCA.")

        pc_df, explained_variance = run_pca(feature_df, n_components=3)

        output_path = output_root / "pca_new_ompps.pdf"
        make_pair_plot(
            pc_df=pc_df,
            explained_variance=explained_variance,
            ompp_values=full_df["OMPP nr"],
            output_path=output_path,
        )
        print(f"Saved: {output_path}")
        return

    feature_sets = load_feature_sets(Path(feature_sets_path))
    for target_key, target_col in TARGET_MAPPING.items():
        if target_col not in feature_sets:
            print(f"Skipping {target_key}: missing feature set for '{target_col}'.")
            continue

        try:
            feature_cols = resolve_feature_columns(
                full_df,
                feature_sets[target_col],
                label=target_col,
            )
        except ValueError as exc:
            print(f"Skipping {target_key}: {exc}")
            continue

        feature_df = full_df[feature_cols].select_dtypes(include="number")
        if feature_df.shape[1] < 3:
            print(f"Skipping {target_key}: fewer than 3 numeric features available.")
            continue

        pc_df, explained_variance = run_pca(feature_df, n_components=3)
        output_path = output_root / f"pca_new_ompps_{target_key}_feature_sets.pdf"
        make_pair_plot(
            pc_df=pc_df,
            explained_variance=explained_variance,
            ompp_values=full_df["OMPP nr"],
            output_path=output_path,
            plot_label=target_col,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PCA for new OMPP sequences and save a pairwise PC plot.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/new_OMPPs.csv",
        help="Input CSV with new sequences and targets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pca_new_ompps",
        help="Directory to save PCA plots.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "feature_sets"],
        default="all",
        help="Use all descriptors or target-specific feature sets.",
    )
    parser.add_argument(
        "--feature_sets",
        type=str,
        default="data/deployment_feature_sets.csv",
        help="CSV with target-specific feature sets.",
    )
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        feature_sets_path=args.feature_sets,
    )
