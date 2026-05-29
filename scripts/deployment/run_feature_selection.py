from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from colorama import Fore, Style

from src.features.load_features import load_and_prepare_data
from src.selection.double_pyramid import (
    PyramidConfig,
    run_double_pyramid_selection,
    run_single_pyramid_selection,
)
from src.selection.shared import TARGET_MAPPING, generate_seed_list, save_json
from src.util.config_loader import load_deployment_config
from src.util.deployment_outputs import sync_latest_artifacts, write_latest_run_marker


def _resolve_config_path(run_root: Path, config_path: str | None) -> Path:
    if config_path:
        return Path(config_path)
    return run_root / "run_config.yaml"


def _load_settings(run_root: Path) -> dict:
    settings_path = run_root / "settings.json"
    if not settings_path.exists():
        return {}
    with settings_path.open("r") as file_handle:
        return json.load(file_handle)


def main(*, run_folder: str, config_path: str | None = None) -> dict:
    run_root = Path(run_folder)
    if not run_root.exists():
        raise FileNotFoundError(f"Run folder not found: {run_root}")

    resolved_config_path = _resolve_config_path(run_root, config_path)
    config = load_deployment_config(resolved_config_path)

    target_arg = config["target"]
    if target_arg not in TARGET_MAPPING:
        raise ValueError(f"Unknown target '{target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")

    target_column = TARGET_MAPPING[target_arg]

    full_feature_path = run_root / "00_features" / "all_sequences_features.csv"
    if not full_feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {full_feature_path}")

    cluster_path = run_root / "00_features" / "global_clustering" / "clustered_features_reduced.csv"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Clustered feature table not found: {cluster_path}")

    X, y, _, _ = load_and_prepare_data(str(full_feature_path), target_column)
    cluster_df = pd.read_csv(cluster_path)

    available_features = [
        feature
        for feature in cluster_df["feature_name"].astype(str).tolist()
        if feature in X.columns
    ]
    if not available_features:
        raise ValueError("No clustering features available in deployment feature matrix.")

    selection_root = run_root / "01_feature_selection"
    selection_root.mkdir(parents=True, exist_ok=True)

    seed_list = generate_seed_list(config["start_pyramid_seed"], config["num_pyramid_seeds"])
    base_config = PyramidConfig(
        model_name=config["pyramid_model_name"],
        scoring_scheme=config["scoring_scheme"],
        top_percentage=config["top_percentage"],
        set_size=config["set_size"],
        scale_target=config["scale_target"],
        random_state=config["random_state"],
        num_pyramid_seeds=config["num_pyramid_seeds"],
        start_seed=config["start_pyramid_seed"],
    )

    stage1_summary_df = None
    if config["feature_option"] == "single":
        selected_features, ranking_df, selection_meta = run_single_pyramid_selection(
            X=X,
            y=y,
            available_features=available_features,
            config=base_config,
            seed_list=seed_list,
            save_root=selection_root,
        )
    elif config["feature_option"] == "double":
        selected_features, ranking_df, stage1_summary_df, selection_meta = run_double_pyramid_selection(
            X=X,
            y=y,
            available_features=available_features,
            config=base_config,
            seed_list=seed_list,
            double_top_percent=config["double_top_percent"],
            save_root=selection_root,
        )
    else:
        raise ValueError("feature_option must be 'single' or 'double'.")

    stage1_path = None
    if stage1_summary_df is not None:
        stage1_path = selection_root / "stage1_feature_frequency.csv"
        stage1_summary_df.to_csv(stage1_path, index=False)

    ranking_df.to_csv(selection_root / "feature_frequency.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(selection_root / "selected_features.csv", index=False)
    save_json(
        selection_root / "selected_features.json",
        {
            "selection_meta": selection_meta,
            "selected_features": [str(feature) for feature in selected_features],
        },
    )

    settings_payload = _load_settings(run_root)
    settings_payload.update(
        {
            "target_arg": target_arg,
            "target_column": target_column,
            "selected_feature_count": int(len(selected_features)),
            "selected_features_path": str(selection_root / "selected_features.json"),
            "feature_frequency_path": str(selection_root / "feature_frequency.csv"),
            "selection_meta": selection_meta,
        }
    )
    save_json(run_root / "settings.json", settings_payload)

    write_latest_run_marker(target_arg=target_arg, run_root=run_root)
    sync_latest_artifacts(
        target_arg=target_arg,
        artifacts={
            "selection/feature_frequency.csv": selection_root / "feature_frequency.csv",
            "selection/selected_features.csv": selection_root / "selected_features.csv",
            "selection/selected_features.json": selection_root / "selected_features.json",
            "selection/stage1_feature_frequency.csv": stage1_path,
        },
    )

    return {
        "run_folder": str(run_root),
        "selected_feature_count": int(len(selected_features)),
        "feature_frequency_path": str(selection_root / "feature_frequency.csv"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deployment feature selection")
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Override config path (default: run_config.yaml)")
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nRunning deployment feature selection at {datetime.now()}{Style.RESET_ALL}")

    result = main(
        run_folder=args.run_folder,
        config_path=args.config,
    )

    print(f"{Fore.GREEN}Run folder: {result['run_folder']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Feature frequency: {result['feature_frequency_path']}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Feature selection completed at {datetime.now()}{Style.RESET_ALL}")
