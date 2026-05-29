from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
from colorama import Fore, Style

from src.features.clustering import run_cluster_pipeline_with_amp
from src.features.compute import compute_features_for_sequences
from src.selection.shared import TARGET_MAPPING, save_json
from src.util.config_loader import load_deployment_config
from src.util.deployment_outputs import sync_latest_artifacts, write_latest_run_marker
from src.util.get_run_folder import build_deployment_run_folder_from_config


def main(*, config_path: str, run_folder: str | None = None) -> dict:
    config = load_deployment_config(config_path)
    repo_root = Path(__file__).resolve().parents[2]

    run_root = Path(run_folder) if run_folder else build_deployment_run_folder_from_config(
        config_path=config_path,
        project_root=repo_root,
    )
    run_root.mkdir(parents=True, exist_ok=True)

    config_path_resolved = Path(config_path).resolve()
    shutil.copy2(config_path_resolved, run_root / "run_config.yaml")

    input_path = Path(config["input_file"])
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    target_arg = config["target"]
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")

    source_df = pd.read_csv(input_path)
    target_column = TARGET_MAPPING[target_arg]
    if target_column not in source_df.columns:
        raise ValueError(f"Target column '{target_column}' is missing in source data.")

    source_df = source_df.dropna(subset=[target_column]).reset_index(drop=True)

    feature_root = run_root / "00_features"
    feature_root.mkdir(parents=True, exist_ok=True)

    full_feature_path = feature_root / "all_sequences_features.csv"
    full_df = compute_features_for_sequences(df=source_df)
    full_df.to_csv(full_feature_path, index=False)

    if "OMPP nr" not in full_df.columns:
        raise ValueError("Missing required unique identifier column 'OMPP nr' in feature table.")
    if full_df["OMPP nr"].isna().any():
        raise ValueError("Column 'OMPP nr' contains missing values.")
    if not full_df["OMPP nr"].is_unique:
        raise ValueError("Column 'OMPP nr' must be unique.")

    amp_sequence_path = Path(config["amp_sequence_path"])
    amp_feature_path = Path(config["amp_feature_path"])
    if not amp_sequence_path.is_absolute():
        amp_sequence_path = repo_root / amp_sequence_path
    if not amp_feature_path.is_absolute():
        amp_feature_path = repo_root / amp_feature_path

    clustering_dir = feature_root / "global_clustering"
    run_cluster_pipeline_with_amp(
        ompp_feature_path=full_feature_path,
        ompp_features_df=full_df,
        amp_sequence_path=amp_sequence_path,
        amp_feature_path=amp_feature_path,
        output_dir=clustering_dir,
        threshold=config["cluster_threshold"],
        method=config["cluster_method"],
        max_cluster_size=config["max_cluster_size"],
        variance_threshold=config["variance_threshold"],
        perfect_corr_threshold=config["perfect_corr_threshold"],
        make_plots=config["make_global_cluster_plots"],
        save_clustering_csv=True,
    )

    clustered_features_path = clustering_dir / "clustered_features_reduced.csv"

    settings_payload = {
        "run_folder": str(run_root.resolve()),
        "run_folder_relative": str(run_root),
        "input_file": str(input_path),
        "target_arg": target_arg,
        "target_column": target_column,
        "full_feature_path": str(full_feature_path),
        "clustered_features_path": str(clustered_features_path),
    }
    save_json(run_root / "settings.json", settings_payload)

    write_latest_run_marker(target_arg=target_arg, run_root=run_root)
    sync_latest_artifacts(
        target_arg=target_arg,
        artifacts={
            "config.yaml": run_root / "run_config.yaml",
            "settings.json": run_root / "settings.json",
            "features/clustered_features_reduced.csv": clustered_features_path,
        },
    )

    return {
        "run_folder": str(run_root),
        "full_feature_path": str(full_feature_path),
        "clustered_features_path": str(clustered_features_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare deployment features and clustering outputs")
    parser.add_argument("--config", type=str, default="workflow_deployment/deployment_config.yaml")
    parser.add_argument("--run_folder", type=str, default=None)
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nPreparing deployment features at {datetime.now()}{Style.RESET_ALL}")

    result = main(
        config_path=args.config,
        run_folder=args.run_folder,
    )

    print(f"{Fore.GREEN}Run folder: {result['run_folder']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Feature table: {result['full_feature_path']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Clustering table: {result['clustered_features_path']}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Feature preparation completed at {datetime.now()}{Style.RESET_ALL}")
