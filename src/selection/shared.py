from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

TARGET_MAPPING = {
    "npn": "NPN_Mean",
    "mic": "MIC_Mean",
    "pot": "Potentiation_Mean",
    "disc": "DISC_Mean",
    "cyto": "Cytotoxicity_Mean",
    "hemo": "Hemolysis_Mean",
}

def compute_cv_folds(n_samples: int) -> Tuple[int, int]:
    if n_samples == 14:
        return 7, 6
    elif n_samples == 12:
        return 6, 5
    return 5, 5


def generate_seed_list(start_seed: int, num_seeds: int) -> List[int]:
    return [start_seed + index for index in range(num_seeds)]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(path: Path, payload: Any) -> None:
    with path.open("w") as file_handle:
        json.dump(_to_jsonable(payload), file_handle, indent=2)
        

def generate_even_feature_sets(
    num_sets: int,
    available_features: Sequence[str],
    seed_value: int = 42,
) -> List[List[str]]:
    rng = np.random.default_rng(seed_value)
    shuffled_features = list(available_features)
    rng.shuffle(shuffled_features)

    base_size = len(shuffled_features) // num_sets
    remainder = len(shuffled_features) % num_sets
    target_sizes = [base_size + (1 if index < remainder else 0) for index in range(num_sets)]

    feature_sets = []
    start = 0
    for size in target_sizes:
        end = start + size
        feature_sets.append(shuffled_features[start:end])
        start = end

    return feature_sets