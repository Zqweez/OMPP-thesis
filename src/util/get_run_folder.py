from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.selection.shared import TARGET_MAPPING


def build_evaluation_run_folder(
    *,
    target_arg: str,
    scaled: bool,
    num_outer_seeds: int,
    num_pyrseeds: int,
    n_samples: int,
    set_size: int,
    timestamp: datetime | None = None,
) -> Path:
    """Build deterministic timestamped evaluation run-folder path."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    run_name = f"{ts}_{target_arg}_{n_samples}samples_{num_outer_seeds}oseeds_{num_pyrseeds}pyrseeds{'_scaled' if scaled else ''}{f'_ss{set_size}' if set_size != 20 else ''}"
    return Path("outputs/evaluation-runs") / run_name


def build_deployment_run_folder(
    *,
    target_arg: str,
    feature_option: str,
    num_pyrseeds: int,
    n_samples: int,
    timestamp: datetime | None = None,
) -> Path:
    """Build deterministic timestamped deployment run-folder path."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    run_name = (
        f"{ts}_{target_arg}_{n_samples}samples_{feature_option}_"
        f"{num_pyrseeds}pyrseeds"
    )
    return Path("outputs/deployment-runs") / run_name


def _count_csv_samples(csv_path: Path, target_arg: str) -> int:
    """Return number of data rows in a CSV (excluding header)."""
    df = pd.read_csv(csv_path)
    target = TARGET_MAPPING.get(target_arg)
    if not target:
        raise ValueError(f"Unknown target '{target_arg}'.")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    return len(df)


def build_evaluation_run_folder_from_config(
    *,
    config_path: str | Path,
    project_root: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Build evaluation run-folder path from evaluation config values and input CSV size."""
    from src.util.config_loader import load_evaluation_config

    config = load_evaluation_config(config_path)

    input_path = Path(str(config["input_file"]))
    if not input_path.is_absolute() and project_root is not None:
        input_path = Path(project_root) / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    return build_evaluation_run_folder(
        target_arg=str(config["target"]),
        scaled=bool(config["scale_target"]),
        num_outer_seeds=int(config["num_outer_seeds"]),
        num_pyrseeds=int(config["num_pyramid_seeds"]),
        n_samples=int(_count_csv_samples(input_path, config["target"])),
        set_size=int(config["set_size"]),
        timestamp=timestamp,
    )


def build_deployment_run_folder_from_config(
    *,
    config_path: str | Path,
    project_root: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Build deployment run-folder path from deployment config values and input CSV size."""
    from src.util.config_loader import load_deployment_config

    config = load_deployment_config(config_path)

    input_path = Path(str(config["input_file"]))
    if not input_path.is_absolute() and project_root is not None:
        input_path = Path(project_root) / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    return build_deployment_run_folder(
        target_arg=str(config["target"]),
        feature_option=str(config["feature_option"]),
        num_pyrseeds=int(config["num_pyramid_seeds"]),
        n_samples=int(_count_csv_samples(input_path, config["target"])),
        timestamp=timestamp,
    )