"""Generic preset system (particle artist tools, DiMo 9/27 -- "Are there artist tools for
particles?"). A preset is one JSON file: a `name`, a `category`, an optional `description` and
`thumbnail` (a small PNG next to the manifest), and an `ops` list -- the same op vocabulary and
placeholders `radialcommands` already gives a user command ($selected, $selected[N],
$downstream_of(ID), $new:NAME), so a preset can wire its sub-graph onto whatever is selected the
same way a radial command does. See docs/PRESETS.md for the file format, written so lane 6
(fluids) can add its own presets to the same browser.

Shipped presets live under `nodebased/data/presets/<category>/*.json`; user presets live in the
user's app-data directory (`user_presets_directory()`), exactly like `radialcommands`' user
command folder. Loading never raises: a bad file becomes a `PresetLoadError` and is left off the
list rather than stopping every other preset from loading.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QStandardPaths

from .radialcommands import _fill_missing_positions, ordered_chain, resolve_placeholders

_MANIFEST_SUFFIX = ".json"
_SHIPPED_ROOT = Path(__file__).resolve().parent / "data" / "presets"


def user_presets_directory() -> Path:
    """Where user preset files live. A "Save selection as preset" dialogue writes here."""
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation)) / "presets"


# ---- the preset model ---------------------------------------------------------------------------

@dataclass
class Preset:
    id: str                 # the manifest's filename stem: also the file's identity
    path: Path
    name: str
    category: str
    description: str
    thumbnail: Optional[Path]
    ops: list
    user: bool = False       # True for a user preset, False for one shipped with the package


@dataclass
class PresetLoadError:
    path: Path
    error: str


def _parse_manifest(path: Path, user: bool) -> Preset:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("a preset file must hold a single JSON object")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError('"name" must be a non-empty string')
    category = data.get("category", "Other")
    if not isinstance(category, str) or not category:
        raise ValueError('"category" must be a non-empty string')
    description = data.get("description", "")
    if not isinstance(description, str):
        raise ValueError('"description" must be a string')
    ops = data.get("ops")
    if not isinstance(ops, list) or not ops:
        raise ValueError('"ops" must be a non-empty list')
    thumbnail_name = data.get("thumbnail")
    if thumbnail_name is not None and not isinstance(thumbnail_name, str):
        raise ValueError('"thumbnail" must be a filename')
    thumbnail = (path.parent / thumbnail_name) if thumbnail_name else None
    return Preset(id=path.stem, path=path, name=name, category=category, description=description,
                  thumbnail=thumbnail, ops=ops, user=user)


def load_all(user_directory: Optional[Path] = None, shipped_directory: Optional[Path] = None
             ) -> Tuple[List[Preset], List[PresetLoadError]]:
    """Every shipped preset plus every user preset, split into the ones that parsed and the ones
    that did not. A user preset with the same filename stem as a shipped one is a distinct entry
    (its `id` collides, but callers key browsing off the `(user, id)` pair, never `id` alone)."""
    shipped_directory = shipped_directory if shipped_directory is not None else _SHIPPED_ROOT
    user_directory = user_directory if user_directory is not None else user_presets_directory()
    presets, errors = [], []
    for directory, user in ((shipped_directory, False), (user_directory, True)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob(f"*{_MANIFEST_SUFFIX}")):
            try:
                presets.append(_parse_manifest(path, user))
            except (ValueError, OSError, json.JSONDecodeError) as error:
                errors.append(PresetLoadError(path=path, error=str(error)))
    return presets, errors


def categories(presets: List[Preset]) -> List[str]:
    """Every category name in `presets`, in first-seen order (what a category filter combo lists)."""
    seen: List[str] = []
    for preset in presets:
        if preset.category not in seen:
            seen.append(preset.category)
    return seen


def search(presets: List[Preset], text: str = "", category: Optional[str] = None) -> List[Preset]:
    """`presets` narrowed to `category` (when given) and to those whose name or description
    contains `text` (case-insensitive); an empty `text` matches everything."""
    result = presets
    if category:
        result = [preset for preset in result if preset.category == category]
    text = text.strip().lower()
    if text:
        result = [preset for preset in result
                  if text in preset.name.lower() or text in preset.description.lower()]
    return result


def build_ops(preset: Preset, selected_ids: List[str], nodes: Dict[str, dict], pos) -> list:
    """The concrete op list for placing `preset` in the graph at `pos`: its placeholders resolved
    against the current selection and graph, exactly like a radial command's `ops` body."""
    new_ids: Dict[str, str] = {}
    ops = resolve_placeholders(copy.deepcopy(preset.ops), selected_ids, nodes, new_ids)
    return _fill_missing_positions(ops, pos)


# ---- saving a selection back out as a preset ------------------------------------------------------

def capture_selection_ops(nodes: Dict[str, dict], selected_ids: List[str]) -> list:
    """An `ops` body recreating `selected_ids`: each selected node becomes a `create` with a fresh
    `$new:` id, every wire between two selected nodes is preserved, and -- when the selection forms
    one unbranched chain (`radialcommands.ordered_chain`) -- an input the chain's root node reads
    from outside the selection becomes `$selected[0]`, so placing the saved preset back onto a new
    selection rewires its root onto whatever is selected then. A selection that does not form one
    chain (branches, merges, or several disconnected nodes) is still captured node-for-node and
    wire-for-wire; it is simply not re-anchored onto a future selection, since there is no single
    "the" entry point to rewire.
    """
    idset = set(selected_ids)
    order = ordered_chain(nodes, selected_ids)
    root = order[0] if order else None
    order = order or list(selected_ids)
    name_of = {node_id: f"$new:n{index}" for index, node_id in enumerate(order)}
    ops: list = []
    for node_id in order:
        node = nodes[node_id]
        ops.append({"op": "create", "id": name_of[node_id], "type": node["type"],
                    "params": copy.deepcopy(node["params"])})
    for node_id in order:
        node = nodes[node_id]
        for slot, source in (node.get("inputs") or {}).items():
            if source is None:
                continue
            if source in idset:
                ops.append({"op": "connect", "id": name_of[node_id], "input": slot,
                           "source": name_of[source]})
            elif node_id == root:
                ops.append({"op": "connect", "id": name_of[node_id], "input": slot,
                           "source": "$selected[0]"})
    return ops


def save_selection_as_preset(directory: Path, preset_id: str, name: str, category: str,
                             nodes: Dict[str, dict], selected_ids: List[str],
                             description: str = "", thumbnail: Optional[str] = None) -> Path:
    """Writes (or overwrites) a preset file capturing `selected_ids`, validating it the same way
    loading it later would, before the caller believes the save succeeded."""
    directory.mkdir(parents=True, exist_ok=True)
    ops = capture_selection_ops(nodes, selected_ids)
    manifest = {"name": name, "category": category, "description": description, "ops": ops}
    if thumbnail:
        manifest["thumbnail"] = thumbnail
    path = directory / f"{preset_id}{_MANIFEST_SUFFIX}"
    path.write_text(json.dumps(manifest, indent=2))
    _parse_manifest(path, user=True)
    return path
