import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path

from src.features.compute import (
    compute_features_from_file,
    get_modlamp_descriptors,
    get_heliquest_descriptors,
    get_peptide_factors,
    get_rdkit_descriptors,
    getMD_analysis_descriptors,
)
from src.visualization.plot_feature_clustering import plot_clustergram, plot_feature_dendrogram

def run_cluster_pipeline_with_amp(
    *,
    ompp_feature_path: str | Path,
    ompp_features_df: pd.DataFrame,
    amp_sequence_path: str | Path,
    amp_feature_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.95,
    method: str = "complete",
    max_cluster_size: int = 1,
    variance_threshold: float = 0.0,
    perfect_corr_threshold: float = 1.0,
    make_plots: bool = False,
    save_clustering_csv: bool = True,
) -> dict:
    """Run one clustering pipeline using a fixed AMP sequence input.

    This helper intentionally assumes AMP sequences are located at
    `data/amp/apm_helix.csv` and raises an error if that file is missing.
    It computes AMP features, runs clustering against the provided OMPP
    feature table, builds keep/remove summary, and saves reduced clusters.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    amp_sequence_path = Path(amp_sequence_path)
    if not amp_sequence_path.exists():
        raise FileNotFoundError(
            "Expected AMP sequence file at data/APD6/amp_helices.csv. "
            "Create it or update AMP_SEQUENCE_INPUT_PATH in src/features/clustering.py."
        )

    ompp_path = Path(ompp_feature_path)
    if not ompp_path.exists():
        raise FileNotFoundError(f"OMPP feature file not found: {ompp_path}")

    amp_feature_path = Path(amp_feature_path)
    if not amp_feature_path.exists():
        amp_feature_path.parent.mkdir(parents=True, exist_ok=True)
        compute_features_from_file(
            file_name=str(amp_sequence_path),
            output_path=str(amp_feature_path),
        )

    clustered_path = output_path / "clustered_features.csv"
    cluster_df, corr_abs, distance, linkage_matrix = get_clustered_groups(
        data_path_1=str(amp_feature_path),
        data_path_2=str(ompp_path),
        threshold=(1 - threshold),
        method=method,
        clustered_path=str(clustered_path) if save_clustering_csv else None,
    )

    summary_df = summarize_correlations(
        ompp_features=ompp_features_df,
        cluster_df=cluster_df,
        corr_abs=corr_abs,
        threshold=threshold,
        summary_save_path=str(output_path / "correlation_summary.csv") if save_clustering_csv else None,
    )
    keep_summary_df, features_to_remove = get_features_to_keep(
        summary_df,
        corr_abs,
        max_cluster_size=max_cluster_size,
        variance_threshold=variance_threshold,
        perfect_corr_threshold=perfect_corr_threshold,
        keep_save_path=str(output_path / "correlation_summary_with_keep.csv") if save_clustering_csv else None,
    )
    

    cluster_df_reduced = cluster_df.loc[~cluster_df["feature_name"].isin(features_to_remove)].copy()
    cluster_df_reduced = cluster_df_reduced.drop(columns=["cluster_size"], errors="ignore")
    cluster_df_reduced["cluster_size"] = cluster_df_reduced["cluster_id"].value_counts()
    cluster_df_reduced = cluster_df_reduced.sort_values(["cluster_id", "feature_name"]).reset_index(drop=True)
    
    if save_clustering_csv:
        keep_summary_path = output_path / "correlation_summary_with_keep.csv"
        keep_summary_df.to_csv(keep_summary_path, index=False)

        clustered_reduced_path = output_path / "clustered_features_reduced.csv"
        cluster_df_reduced.to_csv(clustered_reduced_path, index=False)

    if make_plots:
        plot_clustergram(
            corr_abs,
            distance,
            save_path=str(output_path / "clustergram.pdf"),
            threshold=threshold,
            method=method,
        )
        plot_feature_dendrogram(
            linkage_matrix=linkage_matrix,
            feature_order=list(corr_abs.columns),
            threshold=(1 - threshold),
            save_path=str(output_path / "dendrogram.pdf"),
        )        

    return {
        "cluster_df": cluster_df,
        "cluster_df_reduced": cluster_df_reduced,
        "corr_abs": corr_abs,
        "distance": distance,
    }


def _load_feature_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_to_drop = [
        "Potentiation_Mean",
        "MIC_Mean",
        "NPN_Mean",
        "DISC_Mean",
        "Cytotoxicity_Mean",
        "Hemolysis_Mean",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.select_dtypes(include=[np.number])
    return df


def get_clustered_groups(
    data_path_1: str,
    data_path_2: str,
    threshold: float = 0.15,
    method: str = "complete",
    clustered_path: str | None = None,
):
    """Get clustered feature groups from two feature tables.

    `data_path_1` is treated as base, and `data_path_2`
    may contain additional feature columns.
    """
    df1 = _load_feature_df(data_path_1)
    df2 = _load_feature_df(data_path_2)

    feats1 = set(df1.columns)
    feats2 = set(df2.columns)

    shared = sorted(list(feats1 & feats2))
    only2 = sorted(list(feats2 - feats1))
    if clustered_path is not None:
        print(f"\nShared features: {len(shared)}")
        print(f"Only in df2 (extra-only): {len(only2)}")

    # Correlations
    corr1 = df1[shared].corr(method="pearson").abs() if shared else pd.DataFrame()
    corr2 = df2[shared + only2].corr(method="pearson").abs() if (shared or only2) else pd.DataFrame()

    # Build full corr matrix
    feature_order = shared + only2
    corr_abs = pd.DataFrame(0.0, index=feature_order, columns=feature_order)

    if shared:
        corr_abs.loc[shared, shared] = corr1

    if only2:
        corr_abs.loc[only2, only2] = corr2.loc[only2, only2]
        if shared:
            corr_abs.loc[shared, only2] = corr2.loc[shared, only2]
            corr_abs.loc[only2, shared] = corr2.loc[only2, shared]

    # Ensure diagonal is 1
    for feature in corr_abs.columns:
        corr_abs.loc[feature, feature] = 1.0

    distance = 1 - corr_abs

    distance = distance.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # Guard against tiny floating-point negatives from corr > 1
    distance = distance.clip(0, 1)
    
    for feature in distance.columns:
        distance.loc[feature, feature] = 0.0

    linkage_matrix = linkage(squareform(distance.values), method=method)
    cluster_labels = fcluster(linkage_matrix, threshold, criterion="distance")
    cluster_df = pd.DataFrame({"feature_name": feature_order, "cluster_id": cluster_labels})
    cluster_sizes = cluster_df["cluster_id"].value_counts()
    cluster_df["cluster_size"] = cluster_df["cluster_id"].map(cluster_sizes)
    cluster_df = cluster_df.sort_values(["cluster_id", "feature_name"]).reset_index(drop=True)

    if clustered_path is not None:
        Path(clustered_path).parent.mkdir(parents=True, exist_ok=True)
        cluster_df.to_csv(clustered_path.replace(".csv", "_unfiltered.csv"), index=False)

    return cluster_df, corr_abs, distance, linkage_matrix


def get_descriptor_origin(ompp_features: pd.DataFrame):
    """Track which library produced each descriptor name."""
    pH = 7.4
    df_seqs = ompp_features.copy()
    seq = df_seqs["Sequence"].iloc[0]

    feature_origins: list[dict[str, str]] = []

    modlamp_features = get_modlamp_descriptors(seq, pH=pH)
    for feat in modlamp_features.keys():
        feature_origins.append({"feature_name": feat, "library": "modlAMP"})

    hydro_features = get_heliquest_descriptors(seq, modlamp_features["Charge"])
    for feat in hydro_features.keys():
        feature_origins.append({"feature_name": feat, "library": "calc_hydrophob"})

    peptide_features = get_peptide_factors(seq)
    for feat in peptide_features.keys():
        feature_origins.append({"feature_name": feat, "library": "peptides.py"})

    rdkit_desc = get_rdkit_descriptors(seq)
    for feat in rdkit_desc.keys():
        feature_origins.append({"feature_name": feat, "library": "rdkit"})
    
    # Commented out mordred for now since its not currently used and raises a numpy warning 
    #mordred_desc = get_mordred_descriptors(seq)
    #for feat in mordred_desc.keys():
        #feature_origins.append({"feature_name": feat, "library": "mordred"})

    if "OMPP nr" in df_seqs.columns:
        ompp_id = str(df_seqs["OMPP nr"].iloc[0])
        try:
            md_output = getMD_analysis_descriptors(ompp_id)
            for feat in md_output.keys():
                feature_origins.append({"feature_name": feat, "library": "MDAnalysis"})
        except FileNotFoundError:
            pass

    origin_df = pd.DataFrame(feature_origins).drop_duplicates(subset=["feature_name"], keep="first")

    return origin_df


def summarize_correlations(
    ompp_features: pd.DataFrame,
    cluster_df: pd.DataFrame,
    corr_abs: pd.DataFrame,
    threshold: float,
    summary_save_path: str | None = None,
):
    origin_df = get_descriptor_origin(ompp_features)

    summary_data = []
    for _, row in cluster_df.iterrows():
        feature_name = row["feature_name"]
        cluster_id = row["cluster_id"]
        cluster_size = row["cluster_size"]

        if feature_name in origin_df["feature_name"].values:
            library = origin_df.loc[origin_df["feature_name"] == feature_name, "library"].values[0]
        else:
            library = "Unknown"

        cluster_features = cluster_df[cluster_df["cluster_id"] == cluster_id]["feature_name"].tolist()
        cluster_features_others = [f for f in cluster_features if f != feature_name]
        other_cluster_features = cluster_df[cluster_df["cluster_id"] != cluster_id]["feature_name"].tolist()

        if feature_name in ompp_features.columns:
            variance = ompp_features[feature_name].var()
            mean_val = ompp_features[feature_name].mean()

        if feature_name in corr_abs.index:
            # Mean correlation with features in the same cluster (excluding self)
            if cluster_features_others:
                mean_corr_within_cluster = corr_abs.loc[feature_name, cluster_features_others].mean()
                max_corr_within_cluster = corr_abs.loc[feature_name, cluster_features_others].max()
            else:
                mean_corr_within_cluster = 0.0
                max_corr_within_cluster = 0.0
            
            # Mean correlation with features NOT in the same cluster
            if other_cluster_features:
                mean_corr_outside_cluster = corr_abs.loc[feature_name, other_cluster_features].mean()
            else:
                mean_corr_outside_cluster = 0.0
            
            # Overall mean correlation (excluding self-correlation)
            all_others = [f for f in corr_abs.columns if f != feature_name]
            mean_corr_overall = corr_abs.loc[feature_name, all_others].mean() if all_others else 0.0
        else:
            mean_corr_within_cluster = np.nan
            max_corr_within_cluster = np.nan
            mean_corr_outside_cluster = np.nan
            mean_corr_overall = np.nan

        summary_data.append(
            {
                "feature_name": feature_name,
                "library": library,
                "variance": variance,
                "mean_value": mean_val,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "mean_corr_within_cluster": mean_corr_within_cluster,
                "max_corr_within_cluster": max_corr_within_cluster,
                "mean_corr_outside_cluster": mean_corr_outside_cluster,
                "mean_corr_overall": mean_corr_overall,
            }
        )
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(["cluster_id", "mean_corr_overall"]).reset_index(drop=True)

    if summary_save_path is not None:
        Path(summary_save_path).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_save_path, index=False)

        print(f"\nFeature correlation summary saved to: {summary_save_path}")
        print(f"Summary contains {len(summary_df)} features across {summary_df['cluster_id'].nunique()} clusters")
        
        print(f"\nLibrary distribution:")
        print(summary_df['library'].value_counts())
    
    return summary_df


def get_features_to_keep(
    summary_df: pd.DataFrame,
    corr_abs: pd.DataFrame | None = None,
    max_cluster_size: int = 2,
    variance_threshold: float = 0.0,
    perfect_corr_threshold: float = 1.0,
    keep_save_path: str | None = None,
):
    """ 
    This function will read the summary df to determine which features to keep based on the correlation analysis.
    
    Rules:
     - Remove features with variance <= variance_threshold (constant features)
     - Within each cluster, identify pairs/groups of features that correlate perfectly (1.0) and keep only one from each group
     - If cluster_size > max_cluster_size after perfect correlation removal, keep only the max_cluster_size features with lowest mean_corr_overall
     - Otherwise, keep all remaining features in the cluster
    """
    if corr_abs is None:
        print("Warning: No correlation matrix provided.")

    summary_df = summary_df.copy()
    summary_df["keep"] = True
    summary_df["note"] = ""

    summary_df.loc[summary_df["variance"] <= variance_threshold, "keep"] = False
    summary_df.loc[summary_df["variance"] <= variance_threshold, "note"] = f"Removed due to low variance (<= {variance_threshold})"

    def _find_perfect_corr_groups(features: list[str], cluster_corr: pd.DataFrame) -> list[list[str]]:
        perfect_mask = cluster_corr == perfect_corr_threshold
        for f in features:
            perfect_mask.loc[f, f] = False

        adjacency = {f: set(cluster_corr.columns[perfect_mask.loc[f]].tolist()) for f in features}

        groups = []
        visited = set()

        for f in features:
            if f in visited:
                continue

            # Run DFS starting from f
            stack = [f]
            component = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                # Add to visited and component
                visited.add(cur)
                component.add(cur)
                # Add neighbors to stack
                stack.extend(adjacency[cur] - visited)

            if len(component) > 1:
                groups.append(sorted(component))

        return groups

    for cluster_id, group in summary_df.groupby("cluster_id"):
        group_to_process = group[group["keep"]].copy()
        if group_to_process.empty:
            continue
    
        # Identify perfect correlations within the cluster
        if corr_abs is not None and len(group_to_process) > 1:
            features_in_cluster = group_to_process["feature_name"].tolist()
            cluster_corr = corr_abs.loc[features_in_cluster, features_in_cluster]
            
            # Find groups of perfectly correlated features
            perfect_groups = _find_perfect_corr_groups(features_in_cluster, cluster_corr)
            
            # For each group of perfectly correlated features, keep only the best one
            for group_idx, perfect_group in enumerate(perfect_groups):
                group_data = group_to_process[group_to_process["feature_name"].isin(perfect_group)]
                best_feature = group_data.sort_values("mean_corr_overall", ascending=True).iloc[0]["feature_name"]

                group_tag = f"cluster_{cluster_id}_group_{group_idx}"
                for feature in perfect_group:
                    if feature != best_feature:
                        summary_df.loc[summary_df["feature_name"] == feature, "keep"] = False
                        summary_df.loc[summary_df["feature_name"] == feature, "note"] = f"Perfectly correlated with {best_feature} in cluster {cluster_id}"
                    else:
                        summary_df.loc[summary_df["feature_name"] == feature, "note"] = f"Best of perfectly correlated features in cluster {cluster_id}"

        cluster_feature_names = group["feature_name"].tolist()
        remaining_features = summary_df[
            (summary_df["feature_name"].isin(cluster_feature_names)) & (summary_df["keep"])
        ]

        if len(remaining_features) > max_cluster_size:
            sorted_remaining = remaining_features.sort_values("mean_corr_overall", ascending=True)
            features_to_keep = sorted_remaining.iloc[:max_cluster_size]["feature_name"].tolist()

            for feature in remaining_features["feature_name"].tolist():
                if feature not in features_to_keep:
                    summary_df.loc[summary_df["feature_name"] == feature, "keep"] = False
                    summary_df.loc[summary_df["feature_name"] == feature, "note"] = f"Removed because cluster {cluster_id} exceeded max size {max_cluster_size}"
    if keep_save_path is not None:
        Path(keep_save_path).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(keep_save_path, index=False)

        print("\nFeature selection summary:")
        print(f"Total features: {len(summary_df)}")
        print(f"Features to keep: {summary_df['keep'].sum()}")
        print(f"Features to remove: {(~summary_df['keep']).sum()}")

    # Make a csv with only the features to keep for inspection
    features_to_keep_df = summary_df[summary_df["keep"]].copy()
    # Then also make a csv with counts of where the features come from,
    # Both before and after clustering
    before_counts = summary_df['library'].value_counts()
    after_counts = features_to_keep_df['library'].value_counts()
    counts_df = pd.DataFrame({
        "library": before_counts.index,
        "count_before": before_counts.values,
        "count_after": after_counts.reindex(before_counts.index, fill_value=0).values
    })
    if keep_save_path is not None:
        counts_save_path = Path("outputs/feature_investigation/feature_library_counts.csv")
        Path(counts_save_path).parent.mkdir(parents=True, exist_ok=True)
        counts_df.to_csv(counts_save_path, index=False)

    features_to_remove = summary_df[~summary_df["keep"]]["feature_name"].tolist()
    return summary_df, features_to_remove
