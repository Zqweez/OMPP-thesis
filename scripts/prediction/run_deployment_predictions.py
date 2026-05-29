from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from colorama import Fore, Style

from src.features.compute import compute_features_for_sequences
from src.selection.shared import TARGET_MAPPING


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path

def _parse_targets(targets_arg: str | None, deployment_root: Path) -> list[str]:
    if targets_arg:
        items = [item for item in re.split(r"[,\s]+", targets_arg.strip()) if item]
        return sorted({item.lower() for item in items})

    if not deployment_root.exists():
        raise FileNotFoundError(f"Deployment root not found: {deployment_root}")

    return sorted({path.name.lower() for path in deployment_root.iterdir() if path.is_dir()})


def _resolve_target_dir(deployment_root: Path, target_arg: str) -> Path:
    if not deployment_root.exists():
        return deployment_root / target_arg

    name_lookup = {path.name.lower(): path.name for path in deployment_root.iterdir() if path.is_dir()}
    return deployment_root / name_lookup.get(target_arg.lower(), target_arg)


def _load_top_features(features_path: Path) -> list[str]:
    if features_path.suffix.lower() == ".json":
        with features_path.open("r") as file_handle:
            payload = json.load(file_handle)
        if isinstance(payload, list):
            features = [str(feature) for feature in payload]
        elif isinstance(payload, dict):
            features = [str(feature) for feature in payload.get("selected_features", [])]
        else:
            features = []
    return features

def main(
    *,
    input_csv: str,
    targets: str | None = None,
) -> dict:
    repo_root = _find_repo_root()
    input_path = _resolve_path(repo_root, input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    input_df = pd.read_csv(input_path)

    deployment_root_path = Path("outputs/deployment")
    target_args = _parse_targets(targets, deployment_root_path)
    if not target_args:
        raise ValueError("No deployment targets found.")

    if "Sequence" not in input_df.columns:
        raise ValueError("Input CSV must contain a 'Sequence' column.")
    if input_df["Sequence"].isna().any():
        raise ValueError("Sequence column contains missing values.")

    feature_df = compute_features_for_sequences(df=input_df.copy())
    feature_df.to_csv(Path("outputs/predictions") / f"features_{input_path.stem}.csv", index=False)

    prediction_df = feature_df[["Sequence"]].copy()

    total_models = 0
    for target_arg in target_args:
        if target_arg not in TARGET_MAPPING:
            raise ValueError(
                f"Unknown target '{target_arg}'. Choose one of: {sorted(TARGET_MAPPING.keys())}"
            )

        target_root = _resolve_target_dir(deployment_root_path, target_arg)
        models_dir = target_root / "models"
        if not models_dir.exists():
            raise FileNotFoundError(f"Model folder not found: {models_dir}")

        joblib_paths = sorted(models_dir.glob("*.joblib"))
        if not joblib_paths:
            raise FileNotFoundError(f"No .joblib models found in {models_dir}")

        target_column = TARGET_MAPPING[target_arg]

        for model_path in joblib_paths:
            model_name = model_path.stem
            column_name = f"{target_column}_{model_name}"
            if column_name in prediction_df.columns:
                raise ValueError(f"Prediction column already exists: {column_name}")

            if "baseline" in str(model_path):
                selected_features = ["Discrimination_Factor", "Hydrophobic_Moment"]
                missing_features = [feature for feature in selected_features if feature not in feature_df.columns]
                if missing_features:
                    raise ValueError(
                        "Missing selected features in computed table for target "
                        f"'{target_arg}': {missing_features}"
                    )
                X_selected = feature_df[selected_features].copy()
            elif "allfeats" not in str(model_path):
                selected_features_path = models_dir / f"top_features.json"
                selected_features = _load_top_features(selected_features_path)
                missing_features = [feature for feature in selected_features if feature not in feature_df.columns]
                if missing_features:
                    raise ValueError(
                        "Missing selected features in computed table for target "
                        f"'{target_arg}': {missing_features}"
                    )
                X_selected = feature_df[selected_features].copy()
            else:
                # Remove TARGET_MAPPING values
                X_selected = feature_df[[col for col in feature_df.columns if col not in TARGET_MAPPING.values() and col not in ["Sequence", "Peptide", "OMPP nr"]]].copy()

            model = joblib.load(model_path)
            y_pred = model.predict(X_selected)
            prediction_df[column_name] = np.asarray(y_pred, dtype=float)
            total_models += 1

    conflicts = (set(prediction_df.columns) - {"Sequence"}) & set(input_df.columns)
    if conflicts:
        raise ValueError(
            "Prediction columns already exist in input data: "
            + ", ".join(sorted(conflicts))
        )

    output_df = input_df.merge(prediction_df, on="Sequence", how="left", sort=False)

    output_path = Path("outputs/predictions") / f"predictions_{input_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    return {
        "output_path": str(output_path),
        "row_count": int(len(output_df)),
        "model_count": int(total_models),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deployment models to predict unseen sequences")
    parser.add_argument("--input", type=str, required=True, help="CSV file containing sequences to predict")
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated list of target args (default: all targets in outputs/deployment)",
    )
    args = parser.parse_args()

    print(f"{Fore.LIGHTGREEN_EX}\nRunning deployment predictions...{Style.RESET_ALL}")

    result = main(
        input_csv=args.input,
        targets=args.targets,
    )

    print(f"{Fore.GREEN}Saved predictions: {result['output_path']}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Rows predicted: {result['row_count']}{Style.RESET_ALL}")
    print(f"{Fore.LIGHTGREEN_EX}Completed predictions for {result['model_count']} models.{Style.RESET_ALL}")
