from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "input_file",
    "target",
    "start_seed",
    "num_outer_seeds",
    "n_jobs",
    "feature_option",
    "pyramid_model_name",
    "final_model_names",
    "scoring_scheme",
    "top_percentage",
    "double_top_percent",
    "set_size",
    "num_pyramid_seeds",
    "start_pyramid_seed",
    "scale_target",
    "cluster_threshold",
    "cluster_method",
    "max_cluster_size",
    "variance_threshold",
    "perfect_corr_threshold",
    "make_global_cluster_plots",
}

DEPLOYMENT_REQUIRED_KEYS = {
    "input_file",
    "amp_feature_path",
    "amp_sequence_path",
    "target",
    "feature_option",
    "pyramid_model_name",
    "final_model_names",
    "scoring_scheme",
    "top_percentage",
    "double_top_percent",
    "set_size",
    "num_pyramid_seeds",
    "start_pyramid_seed",
    "scale_target",
    "cluster_threshold",
    "cluster_method",
    "max_cluster_size",
    "variance_threshold",
    "perfect_corr_threshold",
    "make_global_cluster_plots",
    "random_state",
    "n_jobs",
}


@dataclass
class FinalModelSpec:
    model_name: str
    n_trials: int

def _parse_one_model_entry(entry: Any) -> tuple[str, int]:
    """Parse one strict model specification into (model_name, n_trials)."""
    if not isinstance(entry, (list, tuple)):
        raise ValueError(
            "'final_model_names' entries must be [model_name, n_trials]. "
            "Example: ['ridge', 100]"
        )

    if len(entry) != 2:
        raise ValueError(
            "Each entry in 'final_model_names' must contain exactly two values: [model_name, n_trials]."
        )

    model_name = str(entry[0]).strip()
    if not model_name:
        raise ValueError("'final_model_names' contains an empty model name.")

    model_trials =  entry[1]
    return model_name, model_trials


def _parse_final_model_config(raw_value: Any) -> tuple[list[str], dict[str, int]]:
    """Normalize final model config into names and per-model trial mapping."""
    if not isinstance(raw_value, list):
        raise ValueError(
            "'final_model_names' must be a non-empty list of [model_name, n_trials] entries."
        )
    entries: list[Any] = raw_value

    if not entries:
        raise ValueError("'final_model_names' must contain at least one model.")

    model_names: list[str] = []
    model_trials: dict[str, int] = {}

    for entry in entries:
        model_name, trials = _parse_one_model_entry(entry)
        normalized = model_name.lower()
        if normalized in model_trials:
            raise ValueError("'final_model_names' contains duplicate model names.")
        model_names.append(model_name)
        model_trials[normalized] = int(trials)

    return model_names, model_trials


def _parse_n_jobs(value: Any) -> int:
    """Parse n_jobs config value, allowing 'auto' for CPU-based default."""
    if value is None:
        raise ValueError("'n_jobs' must be provided in the config.")

    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "auto":
            cpu_count = os.cpu_count() or 1
            return max(1, cpu_count - 2)
        return int(cleaned)

    return int(value)


def load_evaluation_config(config_path: str | Path) -> dict:
    """Load and validate evaluation YAML config used by pipeline scripts."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as handle:
        config = yaml.safe_load(handle) or {}

    missing = sorted(REQUIRED_KEYS - set(config.keys()))
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    final_model_names_clean, final_model_trials = _parse_final_model_config(config.get("final_model_names"))

    final_model_settings = [
        {
            "model_name": model_name,
            "n_trials": int(final_model_trials[model_name.lower()]),
        }
        for model_name in final_model_names_clean
    ]

    return {
        "input_file": str(config["input_file"]),
        "amp_feature_path": str(config["amp_feature_path"]),
        "amp_sequence_path": str(config["amp_sequence_path"]),
        "target": str(config["target"]),
        "start_seed": int(config["start_seed"]),
        "num_outer_seeds": int(config["num_outer_seeds"]),
        "n_jobs": _parse_n_jobs(config["n_jobs"]),
        "feature_option": str(config["feature_option"]),
        "pyramid_model_name": str(config["pyramid_model_name"]),
        "final_model_names": final_model_names_clean,
        "final_model_trials": final_model_trials,
        "final_model_settings": final_model_settings,
        "scoring_scheme": str(config["scoring_scheme"]),
        "top_percentage": float(config["top_percentage"]),
        "double_top_percent": float(config["double_top_percent"]),
        "set_size": int(config["set_size"]),
        "num_pyramid_seeds": int(config["num_pyramid_seeds"]),
        "start_pyramid_seed": int(config["start_pyramid_seed"]),
        "scale_target": bool(config["scale_target"]),
        "cluster_threshold": float(config["cluster_threshold"]),
        "cluster_method": str(config["cluster_method"]),
        "max_cluster_size": int(config["max_cluster_size"]),
        "variance_threshold": float(config["variance_threshold"]),
        "perfect_corr_threshold": float(config["perfect_corr_threshold"]),
        "make_global_cluster_plots": bool(config["make_global_cluster_plots"]),
    }


def load_deployment_config(config_path: str | Path) -> dict:
    """Load and validate deployment YAML config used by deployment pipeline scripts."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as handle:
        config = yaml.safe_load(handle) or {}

    missing = sorted(DEPLOYMENT_REQUIRED_KEYS - set(config.keys()))
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    final_model_names_clean, final_model_trials = _parse_final_model_config(config.get("final_model_names"))

    final_model_settings = [
        FinalModelSpec(model_name=model_name, n_trials=int(final_model_trials[model_name.lower()]))
        for model_name in final_model_names_clean
    ]

    return {
        "input_file": str(config["input_file"]),
        "amp_feature_path": str(config["amp_feature_path"]),
        "amp_sequence_path": str(config["amp_sequence_path"]),
        "target": str(config["target"]),
        "feature_option": str(config["feature_option"]),
        "pyramid_model_name": str(config["pyramid_model_name"]),
        "final_model_names": final_model_names_clean,
        "final_model_trials": final_model_trials,
        "final_model_settings": final_model_settings,
        "scoring_scheme": str(config["scoring_scheme"]),
        "top_percentage": float(config["top_percentage"]),
        "double_top_percent": float(config["double_top_percent"]),
        "set_size": int(config["set_size"]),
        "num_pyramid_seeds": int(config["num_pyramid_seeds"]),
        "start_pyramid_seed": int(config["start_pyramid_seed"]),
        "scale_target": bool(config["scale_target"]),
        "cluster_threshold": float(config["cluster_threshold"]),
        "cluster_method": str(config["cluster_method"]),
        "max_cluster_size": int(config["max_cluster_size"]),
        "variance_threshold": float(config["variance_threshold"]),
        "perfect_corr_threshold": float(config["perfect_corr_threshold"]),
        "make_global_cluster_plots": bool(config["make_global_cluster_plots"]),
        "random_state": int(config["random_state"]),
        "n_jobs": _parse_n_jobs(config["n_jobs"]),
        "use_default_params": bool(config.get("use_default_params", False)),
        "top_n_features": int(config.get("top_n_features", 20)),
    }
