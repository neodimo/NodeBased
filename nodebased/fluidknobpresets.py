"""Named knob snapshots for fluid nodes, stored separately from simulation cache data."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QStandardPaths

from .core import SPECS
from .nodecatalog import node_category

FLUID_NODE_TYPES = tuple(kind for kind in SPECS if node_category(kind) == "Fluids")


def user_directory() -> Path:
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation)) / "fluid_knob_presets"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "preset"


def save(directory: Optional[Path], node_type: str, name: str, params: dict) -> Path:
    if node_type not in FLUID_NODE_TYPES:
        raise ValueError(f"knob presets are only supported for fluid nodes: {node_type}")
    name = name.strip()
    if not name:
        raise ValueError("preset name must not be empty")
    expected = set(SPECS[node_type]["params"])
    if set(params) != expected:
        raise ValueError(f"{node_type} preset must contain exactly its current knob set")
    # Validate JSON round-tripping before creating/updating a user file.
    encoded = json.dumps({"node_type": node_type, "name": name, "params": params}, indent=2)
    root = Path(directory) if directory is not None else user_directory()
    folder = root / node_type
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_slug(name)}.json"
    path.write_text(encoded + "\n", encoding="utf-8")
    return path


def list_presets(directory: Optional[Path], node_type: str) -> list[tuple[str, Path]]:
    if node_type not in FLUID_NODE_TYPES:
        return []
    root = Path(directory) if directory is not None else user_directory()
    folder = root / node_type
    result = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("node_type") == node_type and isinstance(data.get("name"), str):
                    result.append((data["name"], path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return result


def load(path: Path, node_type: str) -> dict:
    if node_type not in FLUID_NODE_TYPES:
        raise ValueError(f"knob presets are only supported for fluid nodes: {node_type}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("node_type") != node_type or not isinstance(data.get("params"), dict):
        raise ValueError(f"this preset does not belong to {node_type}")
    params = data["params"]
    if set(params) != set(SPECS[node_type]["params"]):
        raise ValueError(f"{node_type} preset has an outdated knob set")
    return params
