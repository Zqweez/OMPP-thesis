import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def plot_clustergram(corr_abs, distance, save_path: str, threshold: float = 0.85, method: str = "average"):
    """Plot a clustered heatmap (clustergram) of the absolute correlation matrix."""
    cdict = [
        (0.0, "#FFFFFF"),
        (threshold, "#5DB7C3"),
        (threshold, "#F44F4F"),
        (1.0, "#8F1313"),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list("sharp_thresh", cdict)

    D = distance.to_numpy(dtype=float)
    # Handle NaNs/infs
    D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0)
    # Force exact symmetry
    D = (D + D.T) / 2
    # Guard against tiny floating-point negatives from corr > 1 by epsilon
    D = np.clip(D, 0.0, 1.0)
    # Force zero diagonal
    np.fill_diagonal(D, 0.0)

    linkage_matrix = linkage(squareform(D), method=method)
    cg = sns.clustermap(
        corr_abs,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap=cmap,
        linewidths=0.5,
        figsize=(55, 50),
    )

    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=7)
    plt.tight_layout()
    plt.suptitle(f"Clustered feature correlation heatmap \n Using threshold = {threshold}", y=1.02)

    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_dendrogram(
    linkage_matrix,
    feature_order: list[str],
    threshold: float,
    save_path: str,
):
    """Plot and save feature dendrogram from an already computed linkage matrix."""
    plt.figure(figsize=(25, 8))
    dendrogram(linkage_matrix, labels=feature_order, leaf_rotation=180, leaf_font_size=2)
    plt.axhline(threshold, linestyle="--")
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.suptitle(f"Feature Dendrogram\nthreshold={round(threshold, 3)}", y=1.05)

    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


