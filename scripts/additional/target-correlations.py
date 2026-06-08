import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.compute import compute_features_for_sequences
from src.selection.shared import TARGET_MAPPING
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def load_data(file_path: Path):
    """Load the dataset and calculate features."""
    df = pd.read_csv(file_path)
    full_df = compute_features_for_sequences(df=df)
    return full_df

def plot_clustermap(corr, save_path: str, method: str = "average"):
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    D = (1-corr.abs()).to_numpy(dtype=float)

    linkage_matrix = linkage(squareform(D), method=method)

    cg = sns.clustermap(
        corr,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap=cmap,
        linewidths=0.5,
        figsize=(5, 5),
    )
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0, ha="left", fontsize=7)

    plt.tight_layout()
    plt.suptitle(f"Clustered target correlation heatmap", y=1.02)
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def target_correlations(df: pd.DataFrame, out_dir: Path):
    """Calculate correlations between the targets"""
    targets_df = df[[col for col in df.columns if col in TARGET_MAPPING.values()]]
    correlations_pearson = targets_df.corr(method="pearson")
    correlations_spearman = targets_df.corr(method="spearman")

    # Plot correlation heatmaps
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(correlations_pearson, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson Correlation")

    plt.subplot(1, 2, 2)
    sns.heatmap(correlations_spearman, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation")
    plt.tight_layout()
    plt.savefig(out_dir / "target_correlations.pdf", bbox_inches='tight')
    plt.close()

    # Make clustermaps
    distance_pearson = 1 - correlations_pearson.abs()
    distance_spearman = 1 - correlations_spearman.abs()
    
    plot_clustermap(corr=correlations_spearman, save_path=out_dir / "target_correlation_clustermap_spearman.pdf")
    plot_clustermap(corr=correlations_pearson, save_path=out_dir / "target_correlation_clustermap_pearson.pdf")

def target_feature_correlations(df: pd.DataFrame, out_dir: Path):
    """Calculate correlations between the targets and features."""
    # Get only numeric columns for correlation calculation
    head_n = 4
    df = df.select_dtypes(include=[np.number])
    target_columns = ["NPN_Mean","MIC_Mean","Potentiation_Mean"] # or TARGET_MAPPING.values()
    target_cols = [col for col in df.columns if col in target_columns]

    available_features_df = pd.read_csv("data/deployment_feature_sets.csv")
    available_features = set()
    for col in target_cols:
        available_features.update(available_features_df[col].dropna().tolist())

    # available_features = set(["T_Scale_1", "T_Scale_2", "T_Scale_3", "T_Scale_4", "T_Scale_5", "Z_Scale_1", "Z_Scale_2", "Z_Scale_3", "Z_Scale_4", "Z_Scale_5"])
    available_features = set(["Discrimination_Factor", "Lenght", "Hydrophobic_Moment", "ChargeDensity", "MW"])
    df = df[target_cols + list(available_features)]
    # df = df.drop(columns=[col for col in df.columns if col in TARGET_MAPPING.values() and col not in target_cols])
    corr_spearman = df.corr(method="spearman")
    corr_pearson = df.corr(method="pearson")
    # Only keep the 3 features that has highest abs correlation with each target
    top_features_spearman = {}
    top_features_pearson = {}
    for target in target_cols:
        top_features_spearman[target] = corr_spearman[target].abs().sort_values(ascending=False).drop(target_columns).head(head_n).index.tolist()
        top_features_pearson[target] = corr_pearson[target].abs().sort_values(ascending=False).drop(target_columns).head(head_n).index.tolist()

    # Save the top features for each target as csv
    top_features_df = pd.DataFrame.from_dict(top_features_spearman, orient="index", columns=[f"Top Feature {i+1}" for i in range(head_n)])
    top_features_df.to_csv(out_dir / "top_target_features_spearman.csv")
    top_features_df = pd.DataFrame.from_dict(top_features_pearson, orient="index", columns=[f"Top Feature {i+1}" for i in range(head_n)])
    top_features_df.to_csv(out_dir / "top_target_features_pearson.csv")

    # Make heatmaps of the correlations between targets and top features
    columns_to_keep_s = []
    for k, v in top_features_spearman.items():
        columns_to_keep_s.append(k)
        columns_to_keep_s.extend(v)
    columns_to_keep_s = list(set(columns_to_keep_s))
    filtered_corr_spearman = corr_spearman.loc[columns_to_keep_s,columns_to_keep_s]

    columns_to_keep_p = []
    for k, v in top_features_pearson.items():
        columns_to_keep_p.append(k)
        columns_to_keep_p.extend(v)
    columns_to_keep_p = list(set(columns_to_keep_p))
    filtered_corr_pearson = corr_pearson.loc[columns_to_keep_p, columns_to_keep_p]

    # Plot heatmaps of the correlations between targets and top features
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(filtered_corr_spearman, annot=True, annot_kws={"size": 5}, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation between targets and top features")
    plt.subplot(1, 2, 2)
    sns.heatmap(filtered_corr_pearson, annot=True, annot_kws={"size": 5}, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Pearson Correlation between targets and top features")
    plt.tight_layout()
    plt.savefig(out_dir / "target_feature_correlations.pdf", bbox_inches='tight')
    plt.close()

    # Plot dendrograms of the correlations between targets and top features
    distance_spearman = 1 - filtered_corr_spearman.abs()
    distance_pearson = 1 - filtered_corr_pearson.abs()

    linkage_matrix_spearman = linkage(squareform(distance_spearman), method="average")
    linkage_matrix_pearson = linkage(squareform(distance_pearson), method="average")
    
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix_spearman, labels=filtered_corr_spearman.index, leaf_rotation=180, leaf_font_size=2)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.suptitle(f"Dendrogram - Spearman", y=1.05)
    plt.savefig(out_dir / "dendrogram_spearman.pdf", bbox_inches="tight", transparent=True)
    plt.close()

    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix_pearson, labels=filtered_corr_pearson.index, leaf_rotation=180, leaf_font_size=2)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.suptitle(f"Dendrogram - Pearson", y=1.05)
    plt.savefig(out_dir / "dendrogram_pearson.pdf", bbox_inches="tight", transparent=True)
    plt.close()


def main():
    out_dir = Path("outputs/additional/target_correlations")
    out_dir.mkdir(exist_ok=True, parents=True)
    data_path = Path("data/OMPP_master_long.csv")
    # data_path = Path("data/OMPP_new_master.csv")
    full_df = load_data(data_path)

    full_feature_path = out_dir / "all_sequences_features.csv"
    full_df.to_csv(full_feature_path, index=False)

    target_correlations(full_df, out_dir)

    target_feature_correlations(full_df, out_dir)

if __name__ == "__main__":
    main()