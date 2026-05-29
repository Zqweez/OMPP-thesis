from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
TARGET_MAPPING = {
	"npn": "NPN_Mean",
	"mic": "MIC_Mean",
	"pot": "Potentiation_Mean",
	"disc": "DISC_Mean",
	"cyto": "Cytotoxicity_Mean",
	"hemo": "Hemolysis_Mean",
}


def make_pair_plot(
	pc_df: pd.DataFrame,
	explained_variance: list[float],
	target_values: pd.Series | None,
	target_key: str | None,
	output_path: Path,
) -> None:
    is_mic_target = target_key == "mic" and target_values is not None
    is_continuous_target = target_key is not None and target_key != "mic" and target_values is not None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6 if is_mic_target else 8))

    if is_mic_target:
        mic_values = pd.to_numeric(target_values, errors="coerce")
        target_is_high = mic_values >= 256
    else:
        mic_values = None
        target_is_high = None

    if is_continuous_target:
        cont_values = pd.to_numeric(target_values, errors="coerce")
        valid_mask = cont_values.notna()
        cont_values_valid = cont_values[valid_mask]
    else:
        cont_values = None
        valid_mask = None
        cont_values_valid = None

    scatter_for_colorbar = None

    plot_specs = [(1, 2), (1, 3), (2, 3)]
    for ax, (pc_x, pc_y) in zip(axes, plot_specs):
        if is_mic_target:
            low_mask = ~target_is_high.fillna(False)
            high_mask = target_is_high.fillna(False)

            ax.scatter(
                pc_df.loc[low_mask, f"PC{pc_x}"],
                pc_df.loc[low_mask, f"PC{pc_y}"],
                s=30,
                alpha=0.8,
                c="#2E86AB",
                edgecolor="white",
                linewidth=0.5,
                label="MIC Low (<256)",
            )
            ax.scatter(
                pc_df.loc[high_mask, f"PC{pc_x}"],
                pc_df.loc[high_mask, f"PC{pc_y}"],
                s=30,
                alpha=0.8,
                c="#D1495B",
                edgecolor="white",
                linewidth=0.5,
                label="MIC High (≥256)",
            )
            ax.legend(frameon=False, loc="best")
        elif is_continuous_target:
            scatter_for_colorbar = ax.scatter(
                pc_df.loc[valid_mask, f"PC{pc_x}"],
                pc_df.loc[valid_mask, f"PC{pc_y}"],
                s=40,
                alpha=1,
                c=cont_values_valid,
                cmap="RdYlBu",
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_title(f"PCA: PC{pc_x} vs PC{pc_y}", fontsize=14, weight="bold")
        ax.set_xlabel(f"PC{pc_x} ({explained_variance[pc_x - 1] * 100:.1f}% variance)", fontsize=11)
        ax.set_ylabel(f"PC{pc_y} ({explained_variance[pc_y - 1] * 100:.1f}% variance)", fontsize=11)
        ax.grid(alpha=0.25)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    target_title = f" | colored by {TARGET_MAPPING[target_key]}" if target_key is not None else ""
    fig.suptitle(f"PCA Pairwise Scatter Plots{target_title}", fontsize=16, weight="bold", y=1.04)
    fig.text(0.5, 0.97, f"Explained variance: PC1={explained_variance[0]*100:.1f}%, PC2={explained_variance[1]*100:.1f}%, PC3={explained_variance[2]*100:.1f}%", ha="center", va="center", fontsize=12, color="gray")

    if is_continuous_target and scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=axes, orientation="horizontal", pad=0.15, shrink=0.5, aspect=20)
        cbar.set_label(TARGET_MAPPING[target_key])

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(input_path: str, input_feats: str, output_dir: str, target: str | None) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_col = TARGET_MAPPING[target]

    df = pd.read_csv(input_path)
    if input_feats is not None:
        input_feats = Path(input_feats)
        if input_feats.is_file():
            # Feature frequency from a pyramid run
            manual_features = pd.read_csv(input_feats)
            # Take top 50 features by frequency
            if target_col in manual_features.columns:
                feature_cols = [col for col in df.columns if col in manual_features[target_col].tolist()]
                feature_df = df[feature_cols].select_dtypes(include="number")
            else:
                raise ValueError(f"Provided input_feats file does not contain column for target '{target_col}': {input_feats}")
        else:
            raise ValueError(f"Provided input_feats path is not a file: {input_feats}")
    else:
        feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
        feature_df = df[feature_cols].select_dtypes(include="number")

    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True))

    scaled = StandardScaler().fit_transform(feature_df)
    pd.DataFrame(scaled, columns=feature_cols).to_csv(output_dir / "scaled_features.csv", index=False)
    pca = PCA()
    pcs = pca.fit_transform(scaled)
    explained_variance = pca.explained_variance_ratio_.tolist()
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(explained_variance))], "Explained_Variance": explained_variance}).to_csv(output_dir / "pca_components.csv", index=False)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scaled)

    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"])
    explained_variance = pca.explained_variance_ratio_.tolist()

    
    target_values = df[target_col]
    output_path = output_dir / f"pca_plot_{target}_{'manual' if input_feats else 'all'}.pdf"
    make_pair_plot(pc_df, explained_variance, target_values, target, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA and save pairwise PC scatter plots.")
    parser.add_argument("--input", type=str, default="data/OMPP_master_long_features_reduced.csv", help="Input CSV file.",)
    parser.add_argument("--input_feats", type=str, default=None, help="Optional path to manual feature csv")
    parser.add_argument("--target", type=str, choices=list(TARGET_MAPPING.keys()), default="npn")
    parser.add_argument("--output_dir", type=str, default="outputs/pca", help="Directory to save PCA plots.",)
    args = parser.parse_args()
    main(args.input, args.input_feats, args.output_dir, args.target)
