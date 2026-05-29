from __future__ import annotations

import shutil
from pathlib import Path


def resolve_latest_deployment_root(target_arg: str) -> Path:
    """Return stable deployment output root for a target."""
    return Path("outputs") / "deployment" / str(target_arg)


def write_latest_run_marker(*, target_arg: str, run_root: Path) -> Path:
    """Persist latest run path for a target."""
    root = resolve_latest_deployment_root(target_arg)
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / "latest_run.txt"
    marker_path.write_text(str(run_root.resolve()))
    return marker_path


def sync_latest_artifacts(*, target_arg: str, artifacts: dict[str, Path]) -> list[Path]:
    """Copy artifacts into stable deployment output paths."""
    root = resolve_latest_deployment_root(target_arg)
    root.mkdir(parents=True, exist_ok=True)

    synced: list[Path] = []
    for relative_path, source_path in artifacts.items():
        if source_path is None:
            continue
        src = Path(source_path)
        if not src.exists():
            continue

        dst = root / relative_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(dst)

    return synced
