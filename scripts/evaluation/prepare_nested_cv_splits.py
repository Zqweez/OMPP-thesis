from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import KFold

from src.features.clustering import run_cluster_pipeline_with_amp
from src.features.compute import compute_features_for_sequences
from src.selection.shared import TARGET_MAPPING, compute_cv_folds, generate_seed_list, save_json
from src.util.config_loader import load_evaluation_config
from src.util.get_run_folder import build_evaluation_run_folder


def _save_split_data(path: Path, split_df: pd.DataFrame) -> None:
    """Save one train/test split as full data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)

def main(
    *,
    config_path: str,
    repo_root: str = "",
    run_folder: str | None = None,
) -> dict:
    """Create full-feature table, global clustering artifacts, and split manifest."""
    config = load_evaluation_config(config_path)
    input_file = Path(repo_root) / config["input_file"]
    amp_feature_path = Path(repo_root) / config["amp_feature_path"]
    amp_sequence_path = Path(repo_root) / config["amp_sequence_path"]
    target_arg = config["target"]
    scale_target = config["scale_target"]
    start_seed = config["start_seed"]
    num_outer_seeds = config["num_outer_seeds"]
    cluster_threshold = config["cluster_threshold"]
    cluster_method = config["cluster_method"]
    max_cluster_size = config["max_cluster_size"]
    variance_threshold = config["variance_threshold"]
    perfect_corr_threshold = config["perfect_corr_threshold"]
    make_global_cluster_plots = config["make_global_cluster_plots"]

    # Step 1: Validate target argument and source data path.
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    source_df = pd.read_csv(input_file)

    target = TARGET_MAPPING[target_arg]
    if target not in source_df.columns:
        raise ValueError(f"Target column '{target}' is missing in source data.")
    # Only keep rows where the target value is not nan such that if the input file contains rows without target values, they are automatically dropped
    print(f"{Fore.GREEN}Loaded source data with {len(source_df)} rows.{Style.RESET_ALL}")
    source_df = source_df.dropna(subset=[target]).reset_index(drop=True)
    print(f"{Fore.BLUE}Loaded source data with {len(source_df)} rows after dropping missing target values.{Style.RESET_ALL}")

    # Step 2: Resolve run folder so all generated artifacts live under one root.
    run_folder_path = Path(run_folder) if run_folder else build_evaluation_run_folder(
        target_arg=target_arg,
        scaled=scale_target,
        num_outer_seeds=num_outer_seeds,
        n_samples=len(source_df),
    )

    feature_root = run_folder_path / "00_features"
    feature_root.mkdir(parents=True, exist_ok=True)

    # Step 3: Resolve or compute full feature table for all sequences once.
    full_feature_path = feature_root / "all_sequences_features.csv"
    full_df = compute_features_for_sequences(df=source_df)
    full_df.to_csv(full_feature_path, index=False)

    if "OMPP nr" not in full_df.columns:
        raise ValueError("Missing required unique identifier column 'OMPP nr' in feature table.")
    if full_df["OMPP nr"].isna().any():
        raise ValueError("Column 'OMPP nr' contains missing values.")
    if not full_df["OMPP nr"].is_unique:
        raise ValueError("Column 'OMPP nr' must be unique.")

    # Step 4: Compute global clustering/plots on full feature table for diagnostics.
    global_cluster_result = run_cluster_pipeline_with_amp(
        ompp_feature_path=full_feature_path,
        output_dir=feature_root / "global_clustering",
        amp_feature_path=amp_feature_path,
        amp_sequence_path=amp_sequence_path,
        ompp_features_df=full_df,
        threshold=cluster_threshold,
        method=cluster_method,
        max_cluster_size=max_cluster_size,
        variance_threshold=variance_threshold,
        perfect_corr_threshold=perfect_corr_threshold,
        make_plots=make_global_cluster_plots,
    )

    # Step 5: Resolve outer-CV folds and generate train/test split datasets.
    outer_folds = compute_cv_folds(len(full_df))[0]

    split_root = run_folder_path / "00_splits"
    split_root.mkdir(parents=True, exist_ok=True)

    # Build one manifest row per (outer_seed, outer_fold) with explicit train/test CSV paths.
    outer_seed_list = generate_seed_list(start_seed, num_outer_seeds)
    manifest_rows: list[dict] = []

    for outer_seed in outer_seed_list:
        kf = KFold(n_splits=outer_folds, shuffle=True, random_state=outer_seed)
        for outer_fold, (train_idx, test_idx) in enumerate(kf.split(full_df), start=1):
            split_dir = split_root / f"outer_seed_{outer_seed}" / f"fold_{outer_fold}"
            train_path = split_dir / "train_data.csv"
            test_path = split_dir / "test_data.csv"

            train_indices = train_idx.astype(int).tolist()
            test_indices = test_idx.astype(int).tolist()
            _save_split_data(train_path, full_df.iloc[train_indices].reset_index(drop=True))
            _save_split_data(test_path, full_df.iloc[test_indices].reset_index(drop=True))

            manifest_rows.append(
                {
                    "outer_seed": int(outer_seed),
                    "outer_fold": int(outer_fold),
                    "train_data_csv": str(train_path),
                    "test_data_csv": str(test_path),
                    "split_dir": str(split_dir),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )

    # Step 6: Persist split manifest and run settings used by downstream fold jobs.
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["outer_seed", "outer_fold"]).reset_index(drop=True)
    manifest_path = split_root / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    copied_config_path = run_folder_path / "run_config.yaml"
    shutil.copy2(Path(config_path), copied_config_path)

    return {
        "run_folder": str(run_folder_path),
        "split_manifest": str(manifest_path),
        "copied_config": str(copied_config_path),
        "num_manifest_rows": int(len(manifest_df)),
        "outer_seed_list": [int(seed) for seed in outer_seed_list],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare nested-CV outer split manifest for evaluation runs")
    parser.add_argument("--config", type=str, default="workflow_evaluation/evaluation_config.yaml")
    parser.add_argument("--repo_root", type=str, default="", help="Root path of the repository for correct data import in nextflow.")
    parser.add_argument("--run_folder", type=str, default=None, help="Optional run folder override. If omitted, default timestamped folder name is used.",)
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nPreparing nested-CV splits at {datetime.now()}{Style.RESET_ALL}")

    result = main(
        config_path=args.config,
        repo_root=args.repo_root,
        run_folder=args.run_folder,
    )

    print(f"{Fore.GREEN}Run folder: {result['run_folder']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Split manifest: {result['split_manifest']}{Style.RESET_ALL}")

    print(f"{Fore.LIGHTGREEN_EX}Finished nested CV split generation at {datetime.now()}{Style.RESET_ALL}")
