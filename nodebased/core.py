"""Serializable graph and the shared human/agent command boundary (no Qt imports)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

# Parameter schemas are also consumed by the inspector and agent discovery.
# Filter nodes accept an optional "mask" image (alpha gates where the filter applies) and a
# per-node "mix" (blend between original input and filtered output). The mask slot is listed in
# "optional_inputs" rather than "inputs" so a node validates without it wired — the evaluator
# treats None there as full opacity (M.a = 1).
IMAGE_FILTER_KINDS = ("Grade", "ColorCorrect", "Blur", "Transform", "Crop")

# The version `upgrade_document` migrates to and `validate` accepts. Tests and callers should refer
# to this rather than hard-coding a number, so a schema bump does not spray stale literals.
SCHEMA_VERSION = 7
DEFAULT_SETTINGS = {
    "color": {
        "config": "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5",
        "working_space": "ACEScg",
        "display": "sRGB - Display",
        "view": "ACES 2.0",
    },
    "viewer": {"background": "black"},
}
SPECS = {
    # Read owns its own timeline-frame -> source-frame mapping (see docs/TIME_MODEL.md). "path" may
    # be a padded sequence pattern (plate.%04d.exr / plate.####.exr) or a still; "frame_offset"
    # shifts the source relative to the timeline, and "missing" decides what an absent frame does.
    "Read": {"inputs": [], "params": {"path": "", "colorspace": "Auto", "alpha_mode": "Auto", "layer": "", "subimage": 0,
                                      "frame_offset": 0, "missing": "error"}},
    "Constant": {"inputs": [], "params": {"width": 960, "height": 540, "red": 0.12, "green": 0.3, "blue": 0.6, "alpha": 1.0}},
    "Checker": {"inputs": [], "params": {"width": 960, "height": 540, "size": 64}},
    "Grade": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
    "ColorCorrect": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0, "mix": 1.0}},
    "Blur": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"radius": 8.0, "mix": 1.0}},
    "Transform": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                                                 "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest", "mix": 1.0}},
    "Crop": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"x": 0, "y": 0, "width": 960, "height": 540, "mix": 1.0}},
    "Shuffle": {"inputs": ["image"], "params": {"red_from": "R", "green_from": "G", "blue_from": "B", "alpha_from": "A"}},
    "Merge": {"inputs": ["A", "B"], "params": {"operation": "over", "mix": 1.0}},
    "Premult": {"inputs": ["image"], "params": {}},
    "Unpremult": {"inputs": ["image"], "params": {}},
    "Dot": {"inputs": ["input"], "params": {}},
    "Switch": {"inputs": ["0", "1"], "params": {"which": 0}},
    "Viewer": {"inputs": ["image"], "params": {}},
}
LIMITS = {"width": (1, 8192), "height": (1, 8192), "size": (1, 4096),
          "exposure": (-20, 20), "multiply": (-100, 100), "offset": (-100, 100),
          "red": (-100, 100), "green": (-100, 100), "blue": (-100, 100),
          "alpha": (0, 1), "mix": (0, 1),
          "x": (-8192, 8192), "y": (-8192, 8192), "subimage": (0, 1023),
          "translate_x": (-8192.0, 8192.0), "translate_y": (-8192.0, 8192.0),
          "rotate": (-100000.0, 100000.0), "scale": (0.001, 1000.0),
          "center_x": (-8192.0, 8192.0), "center_y": (-8192.0, 8192.0),
          "lift": (-10, 10), "gamma": (0.01, 100), "gain": (0, 100), "saturation": (0, 10),
          "radius": (0, 500), "which": (0, 1),
          "frame_offset": (-1000000, 1000000)}

# Document time range limits (schema v5). FPS is stored from v5 onward so the field exists for a
# future clip timeline; nothing consumes it yet — see docs/TIME_MODEL.md.
TIME_LIMITS = {"first": (-1000000, 1000000), "last": (-1000000, 1000000),
               "current": (-1000000, 1000000), "fps": (0.01, 1000.0)}
DEFAULT_TIME = {"first": 1, "last": 1, "current": 1, "fps": 24.0}
MISSING_FRAME_POLICIES = ("error", "hold", "black")


# Merge keeps A as foreground and B as background, mirroring Nuke's wiring convention.
MERGE_OPERATIONS = ("over", "under", "plus", "minus", "multiply", "screen", "max", "min",
                    "difference", "divide", "mask", "stencil", "in", "out", "atop", "xor")
TRANSFORM_FILTERS = ("nearest", "bilinear", "cubic")
CHOICES = {"colorspace": ["Auto", "sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1", "Raw"],
           "alpha_mode": ["Auto", "Straight", "Premultiplied"],
           "operation": list(MERGE_OPERATIONS),
           "filter": list(TRANSFORM_FILTERS),
           "red_from": ["R", "G", "B", "A", "0", "1"], "green_from": ["R", "G", "B", "A", "0", "1"],
           "blue_from": ["R", "G", "B", "A", "0", "1"], "alpha_from": ["R", "G", "B", "A", "0", "1"],
           "missing": list(MISSING_FRAME_POLICIES)}


def upgrade_document(document):
    doc = copy.deepcopy(document)
    if isinstance(doc, dict) and doc.get("version") == 1:
        for node in doc.get("nodes", {}).values():
            if node.get("type") == "Read":
                node["params"] = {**SPECS["Read"]["params"], **node["params"]}
        doc["version"] = 2
    if isinstance(doc, dict) and doc.get("version") == 2:
        # v2 -> v3: rename Transform x/y to translate_x/translate_y (int -> float) and add the
        # rotate/scale/center/filter params with identity defaults; add Merge.operation = "over"
        # so v0.3.0 comps render identically and existing projects opt into the new ops explicitly.
        for node in doc.get("nodes", {}).values():
            kind = node.get("type")
            if kind == "Transform":
                old_params = node.get("params", {})
                upgraded = {
                    "translate_x": float(old_params.get("x", 0)),
                    "translate_y": float(old_params.get("y", 0)),
                    "rotate": 0.0,
                    "scale": 1.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "filter": "nearest",
                }
                upgraded.update({k: v for k, v in old_params.items() if k not in ("x", "y")})
                node["params"] = upgraded
            elif kind == "Merge":
                params = node.get("params", {})
                if "operation" not in params:
                    params = {**params, "operation": "over"}
                    node["params"] = params
        doc["version"] = 3
    if isinstance(doc, dict) and doc.get("version") == 3:
        # v3 -> v4: every image-filter node (Grade, ColorCorrect, Blur, Transform, Crop) gains an
        # optional "mask" input slot and a per-node "mix" param (defaults to 1.0 so existing
        # projects render byte-identically). Dot and Switch are new in v4; no upgrade path
        # synthesizes them — they appear only when the user adds them.
        for node in doc.get("nodes", {}).values():
            kind = node.get("type")
            if kind in IMAGE_FILTER_KINDS:
                inputs = node.setdefault("inputs", {})
                if "mask" not in inputs:
                    inputs["mask"] = None
                params = node.setdefault("params", {})
                if "mix" not in params:
                    params["mix"] = 1.0
        doc["version"] = 4
    if isinstance(doc, dict) and doc.get("version") == 4:
        # v4 -> v5: the document gains a time range, and Read gains its source-time mapping.
        # A v4 comp had no time axis at all, so it upgrades to a single-frame range — every
        # still-image Read renders byte-identically at frame 1, which is the only frame there is.
        # frame_offset=0 and missing="error" preserve v4 behaviour for literal (non-pattern) paths.
        doc["time"] = dict(DEFAULT_TIME)
        for node in doc.get("nodes", {}).values():
            if node.get("type") == "Read":
                params = node.setdefault("params", {})
                params.setdefault("frame_offset", 0)
                params.setdefault("missing", "error")
        doc["version"] = 5
    if isinstance(doc, dict) and doc.get("version") == 5:
        # v5 -> v6: the document gains a top-level "animation" section. v5 had no curves, so the
        # upgrade is just an empty {"curves": {}} shape. Animation never changes a stored
        # node["params"]; it only adds an evaluation-time override layer, so v5 graphs render
        # byte-identically after upgrade.
        doc["animation"] = {"curves": {}}
        doc["version"] = 6
    if isinstance(doc, dict) and doc.get("version") == 6:
        # v6 -> v7: make the colour contract explicit in the project. The processing space is
        # ACEScg and the default view is the ACES 2.0 SDR Rec.709 transform; the viewer background
        # also becomes a saved project preference. Existing comps previously used an implicit
        # sRGB view, so upgrading them keeps that view to preserve their appearance.
        doc["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        doc["settings"]["color"]["view"] = "sRGB"
        # The literal 7, never SCHEMA_VERSION: a step must write the version it actually emits.
        # If this said SCHEMA_VERSION, the day a v8 lands this block would stamp a document "8"
        # while having done only v7's work, and the v7 -> v8 step below it would never fire — a
        # document tagged with the new version but missing the new section. Every step above
        # writes its own literal for the same reason.
        doc["version"] = 7
    return doc


def empty_document():
    return {"version": SCHEMA_VERSION, "nodes": {}, "view": None, "time": dict(DEFAULT_TIME),
            "animation": {"curves": {}}, "settings": copy.deepcopy(DEFAULT_SETTINGS)}


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"color", "viewer"}:
        raise ValueError("settings must define color and viewer")
    color = settings["color"]
    if not isinstance(color, dict) or set(color) != {"config", "working_space", "display", "view"}:
        raise ValueError("settings.color is malformed")
    # This release ships one self-contained ACES pipeline. Persisting these fields now gives the
    # project format a clean extension point for user OCIO configs without pretending arbitrary
    # configs are already supported.
    if color["config"] != DEFAULT_SETTINGS["color"]["config"]:
        raise ValueError("Unsupported OCIO config")
    if color["working_space"] != "ACEScg":
        raise ValueError("Working space must be ACEScg")
    if color["display"] != "sRGB - Display":
        raise ValueError("Unsupported display")
    if color["view"] not in ("sRGB", "ACES 2.0", "Linear"):
        raise ValueError("Unsupported display view")
    viewer = settings["viewer"]
    if not isinstance(viewer, dict) or set(viewer) != {"background"}:
        raise ValueError("settings.viewer is malformed")
    if viewer["background"] not in ("black", "checker"):
        raise ValueError("Viewer background must be black or checker")


def validate_time(time):
    """The document's frame range. Kept separate so the UI can validate an edit before applying it."""
    if not isinstance(time, dict) or set(time) != set(DEFAULT_TIME):
        raise ValueError("Document time must define first, last, current and fps")
    for name, default in DEFAULT_TIME.items():
        value = time[name]
        if type(value) not in (float, int) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"time.{name} must be a finite number")
        if type(default) is int and type(value) is not int:
            raise ValueError(f"time.{name} must be an integer")
        lo, hi = TIME_LIMITS[name]
        if not lo <= value <= hi:
            raise ValueError(f"time.{name} must be between {lo} and {hi}")
    if time["last"] < time["first"]:
        raise ValueError("time.last must not precede time.first")
    if not time["first"] <= time["current"] <= time["last"]:
        raise ValueError("time.current must fall inside the frame range")


def validate(doc):
    if not isinstance(doc, dict) or set(doc) != {"version", "nodes", "view", "time", "animation", "settings"} or doc["version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed NodeBased document")
    validate_time(doc["time"])
    validate_settings(doc["settings"])
    nodes = doc["nodes"]
    if not isinstance(nodes, dict) or len(nodes) > 1000:
        raise ValueError("Document must contain at most 1000 nodes")
    if doc["view"] is not None and doc["view"] not in nodes:
        raise ValueError("Viewer target does not exist")
    for key, node in nodes.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("Invalid node ID")
        if not isinstance(node, dict) or set(node) != {"type", "name", "params", "inputs", "pos", "disabled"}:
            raise ValueError("Malformed node")
        kind = node["type"]
        if kind not in SPECS:
            raise ValueError(f"Unknown node type: {kind}")
        spec = SPECS[kind]
        if not isinstance(node["name"], str) or not 1 <= len(node["name"]) <= 128:
            raise ValueError("Node name must contain 1–128 characters")
        if type(node["disabled"]) is not bool:
            raise ValueError("disabled must be boolean")
        if not isinstance(node["pos"], list) or len(node["pos"]) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e6 for v in node["pos"]):
            raise ValueError("Invalid node position")
        if not isinstance(node["params"], dict) or set(node["params"]) != set(spec["params"]):
            raise ValueError(f"Invalid parameters for {kind}")
        for name, default in spec["params"].items():
            value = node["params"][name]
            if name in CHOICES and value not in CHOICES[name]:
                raise ValueError(f"Invalid {name}: {value}")
            if isinstance(default, str):
                if not isinstance(value, str) or len(value) > 32768:
                    raise ValueError(f"{name} must be a path string")
            else:
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if type(default) is int and type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
                lo, hi = LIMITS[name]
                if not lo <= value <= hi:
                    raise ValueError(f"{name} must be between {lo} and {hi}")
        # Inputs cover both required slots (in SPECS[kind]["inputs"]) and optional slots (in
        # SPECS[kind].get("optional_inputs")). Required must be wired before evaluation; optional
        # may be None and acts as identity (full opacity mask, no input selection).
        expected_inputs = set(spec["inputs"]) | set(spec.get("optional_inputs", []))
        if not isinstance(node["inputs"], dict) or set(node["inputs"]) != expected_inputs:
            raise ValueError(f"Invalid inputs for {kind}")
        for slot, source in node["inputs"].items():
            if source is not None and (not isinstance(source, str) or source not in nodes):
                raise ValueError(f"Input {slot!r} references a missing node")
    # Iterative topological walk avoids recursion-limit crashes on long graphs.
    pending = {key: sum(v is not None for v in n["inputs"].values()) for key, n in nodes.items()}
    children = {key: [] for key in nodes}
    for key, node in nodes.items():
        for source in node["inputs"].values():
            if source is not None:
                children[source].append(key)
    ready = [key for key, count in pending.items() if count == 0]
    count = 0
    while ready:
        key = ready.pop()
        count += 1
        for child in children[key]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if count != len(nodes):
        raise ValueError("Image graphs cannot contain cycles; AI loops will use structured tasks")
    # Animation section is always present post-v5: {"curves": {node_id: {param: curve}}}. Validate
    # structural shape plus that each curve points at a real node and a numeric parameter on it,
    # and that the curve itself satisfies validate_curve().
    animation = doc["animation"]
    if not isinstance(animation, dict) or set(animation) != {"curves"}:
        raise ValueError("animation must define exactly 'curves'")
    curves_root = animation["curves"]
    if not isinstance(curves_root, dict):
        raise ValueError("animation.curves must be an object")
    # Local import to avoid a circular dependency at module import time.
    from .animation import validate_curve as _validate_curve
    for node_id, params_curves in curves_root.items():
        if node_id not in nodes:
            raise ValueError(f"animation.curves references missing node {node_id!r}")
        if not isinstance(params_curves, dict):
            raise ValueError(f"animation.curves[{node_id!r}] must be an object")
        node = nodes[node_id]
        spec_params = SPECS[node["type"]]["params"]
        for param_name, curve in params_curves.items():
            if not isinstance(param_name, str) or not param_name:
                raise ValueError("animation curve key must be a non-empty string")
            if param_name not in spec_params:
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: unknown parameter for "
                    f"{node['type']!r}")
            spec_default = spec_params[param_name]
            if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: only numeric parameters can "
                    f"be animated (this is a {type(spec_default).__name__})")
            if not isinstance(curve, dict):
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: curve must be an object")
            try:
                _validate_curve(curve)
            except Exception as error:
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: {error}") from error


def atomic_save(path, doc):
    path = Path(path).expanduser().resolve()
    validate(doc)
    # Store read paths relative to the project for portable folder trees.
    portable = copy.deepcopy(doc)
    for node in portable["nodes"].values():
        if node["type"] == "Read" and node["params"]["path"]:
            try:
                node["params"]["path"] = os.path.relpath(node["params"]["path"], path.parent)
            except ValueError:  # Windows different drive.
                pass
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(portable, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_document(path):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Project exceeds 4 MiB document limit")
    doc = upgrade_document(json.loads(path.read_text(encoding="utf-8")))
    validate(doc)
    for node in doc["nodes"].values():
        if node["type"] == "Read" and node["params"]["path"]:
            node["params"]["path"] = str((path.parent / node["params"]["path"]).resolve())
    return doc


class Dispatcher:
    """Each edit/batch is validated atomically and takes one undo slot."""
    def __init__(self, document=None):
        self.document = upgrade_document(document or empty_document())
        validate(self.document)
        self.undo_stack = []
        self.redo_stack = []
        self.revision = 0

    def execute(self, request):
        if not isinstance(request, dict):
            raise ValueError("Command must be an object")
        op = request.get("op")
        if op == "describe":
            # Local import keeps the top-level Dispatcher import cycle-free.
            from .animation import CURVE_INTERPOLATIONS as _CURVE_INTERPOLATIONS, FRAME_LIMITS as _FRAME_LIMITS
            return {"protocol": 1, "nodes": copy.deepcopy(SPECS), "limits": LIMITS, "choices": CHOICES,
                    "time": copy.deepcopy(self.document["time"]), "time_limits": TIME_LIMITS,
                    "sequence_patterns": ["printf (plate.%04d.exr)", "hash (plate.####.exr)", "still (plate.exr)"],
                    "animation": {
                        "interpolations": list(_CURVE_INTERPOLATIONS),
                        "frame_limits": list(_FRAME_LIMITS),
                        "shape": {"interpolation": "constant | linear",
                                  "keys": [{"frame": "int", "value": "number"}]},
                        "operations": ["set_key", "delete_key", "clear_curve"],
                        "set_key": {"id": "string (node id)", "param": "string (numeric parameter name)",
                                     "frame": "int (timeline frame)", "value": "number (finite)",
                                     "interpolation": "constant | linear (optional, default 'linear')"},
                        "delete_key": {"id": "string", "param": "string", "frame": "int"},
                        "clear_curve": {"id": "string", "param": "string"}},
                    "settings": copy.deepcopy(self.document["settings"]),
                    "operations": ["describe", "inspect", "create", "set", "connect", "move", "rename", "disable", "delete", "view", "time", "settings", "set_key", "delete_key", "clear_curve", "batch", "undo", "redo", "save", "load"]}
        if op == "inspect":
            return {"revision": self.revision, "document": copy.deepcopy(self.document)}
        if op == "save":
            atomic_save(request["path"], self.document)
            return {"path": str(Path(request["path"]).resolve())}
        if op in ("undo", "redo"):
            source, target = (self.undo_stack, self.redo_stack) if op == "undo" else (self.redo_stack, self.undo_stack)
            if source:
                target.append(self.document)
                self.document = source.pop()
                self.revision += 1
            return {"revision": self.revision}
        draft = copy.deepcopy(self.document)
        if op == "load":
            draft = load_document(request["path"])
            result = {}
        elif op == "batch":
            commands = request.get("commands")
            if not isinstance(commands, list) or len(commands) > 1000:
                raise ValueError("batch requires at most 1000 commands")
            result = [self._edit(draft, command) for command in commands]
        else:
            result = self._edit(draft, request)
        validate(draft)
        if draft != self.document:
            # Transport playback updates the persisted playhead through the same validated command
            # boundary, but must not fill the artist's undo history once per frame.
            transient_time = op == "time" and request.get("transient") is True
            if not transient_time:
                self.undo_stack.append(self.document)
                self.undo_stack = self.undo_stack[-100:]
                self.redo_stack.clear()
            self.document = draft
            self.revision += 1
        return {"revision": self.revision, "result": result}

    def _edit(self, doc, cmd):
        op = cmd["op"]
        nodes = doc["nodes"]
        if op == "create":
            kind = cmd["type"]
            if kind not in SPECS:
                raise ValueError(f"Unknown node type: {kind}")
            key = cmd.get("id", uuid.uuid4().hex[:12])
            if key in nodes:
                raise ValueError("Node ID already exists")
            params = {**SPECS[kind]["params"], **cmd.get("params", {})}
            if kind == "Read" and params["path"]:
                params["path"] = str(Path(params["path"]).expanduser().resolve())
            nodes[key] = {"type": kind, "name": cmd.get("name", kind), "params": params,
                          "inputs": {slot: None for slot in
                                     list(SPECS[kind]["inputs"]) + list(SPECS[kind].get("optional_inputs", []))},
                          "pos": cmd.get("pos", [0, 0]), "disabled": False}
            return {"id": key}
        if op == "view":
            doc["view"] = cmd.get("id")
            return {}
        if op in ("set_key", "delete_key", "clear_curve"):
            return self._animation_edit(doc, cmd)
        if op == "time":
            # Scrubbing the playhead and re-ranging the comp are both document edits, so they are
            # undoable and reach an attached agent through the same validated boundary as any other
            # change. Clamp current into the range so setting first/last cannot strand the playhead.
            time = dict(doc["time"])
            for name in DEFAULT_TIME:
                if name in cmd:
                    time[name] = cmd[name]
            if "current" not in cmd:
                time["current"] = min(max(time["current"], time["first"]), time["last"])
            validate_time(time)
            doc["time"] = time
            return dict(time)
        if op == "settings":
            changes = cmd.get("settings")
            if not isinstance(changes, dict):
                raise ValueError("settings operation requires a settings object")
            unknown = set(changes) - {"color", "viewer"}
            if unknown:
                raise ValueError(f"Unknown settings groups: {sorted(unknown)}")
            for group, values in changes.items():
                if not isinstance(values, dict):
                    raise ValueError(f"settings.{group} must be an object")
                unknown_fields = set(values) - set(doc["settings"][group])
                if unknown_fields:
                    raise ValueError(f"Unknown settings.{group} fields: {sorted(unknown_fields)}")
                doc["settings"][group].update(values)
            return copy.deepcopy(doc["settings"])
        key = cmd["id"]
        node = nodes[key]
        if op == "set":
            name = cmd["param"]
            value = cmd["value"]
            if node["type"] == "Read" and name == "path" and isinstance(value, str) and value:
                value = str(Path(value).expanduser().resolve())
            node["params"][name] = value
        elif op == "connect":
            node["inputs"][cmd["input"]] = cmd.get("source")
        elif op == "move":
            node["pos"] = cmd["pos"]
        elif op == "rename":
            node["name"] = cmd["name"]
        elif op == "disable":
            if not SPECS[node["type"]]["inputs"]:
                raise ValueError("Source nodes cannot be bypassed")
            node["disabled"] = cmd["value"]
        elif op == "delete":
            del nodes[key]
            for other in nodes.values():
                other["inputs"] = {slot: None if value == key else value for slot, value in other["inputs"].items()}
            if doc["view"] == key:
                doc["view"] = None
            # Atomically drop any curves targeting the deleted node. Without this, validate()
            # would reject the post-delete document for referencing a missing node. The undo
            # stack already holds a deep copy of the pre-delete document (including the node's
            # animation entry), so undo restores both the node and its curves automatically.
            doc["animation"]["curves"].pop(key, None)
        else:
            raise ValueError(f"Unknown edit operation: {op}")
        return {}

    def _animation_edit(self, doc, cmd):
        """Atomic, undoable edits to doc["animation"]. The node's stored params are never
        touched: a curve is an evaluation-time override only. See ``nodebased/animation.py``."""
        from .animation import (CURVE_INTERPOLATIONS as _CURVE_INTERPOLATIONS,
                                  coerce_value_for_param as _coerce,
                                  drop_key as _drop_key, merge_key as _merge_key,
                                  validate_curve as _validate_curve)
        op = cmd["op"]
        node_id = cmd.get("id")
        param = cmd.get("param")
        if not isinstance(node_id, str) or node_id not in doc["nodes"]:
            raise ValueError(f"animation edit {op!r}: unknown node id {node_id!r}")
        if not isinstance(param, str) or not param:
            raise ValueError(f"animation edit {op!r}: 'param' must be a non-empty string")
        node = doc["nodes"][node_id]
        spec_params = SPECS[node["type"]]["params"]
        if param not in spec_params:
            raise ValueError(f"animation edit {op!r}: {node['type']!r} has no parameter {param!r}")
        spec_default = spec_params[param]
        if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
            raise ValueError(
                f"animation edit {op!r}: parameter {param!r} is not numeric "
                f"({type(spec_default).__name__})")
        curves_root = doc["animation"]["curves"]
        node_curves = curves_root.setdefault(node_id, {})
        curve = node_curves.get(param)
        if op == "set_key":
            interpolation = cmd.get("interpolation", curve["interpolation"] if curve else "linear")
            if interpolation not in _CURVE_INTERPOLATIONS:
                raise ValueError(
                    f"animation set_key: interpolation must be one of {list(_CURVE_INTERPOLATIONS)}")
            value = cmd["value"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("animation set_key: 'value' must be a number")
            import math as _math
            if not _math.isfinite(value):
                raise ValueError("animation set_key: 'value' must be a finite number")
            # Range check using the spec's limit when present. The set_key layer rejects out-of-
            # range values explicitly so the caller gets a clear error rather than silent clamping.
            if param in LIMITS:
                lo, hi = LIMITS[param]
                if value < lo or value > hi:
                    raise ValueError(
                        f"animation set_key: value {value} outside [{lo}, {hi}] for {param!r}")
            frame = cmd["frame"]
            if type(frame) is not int or isinstance(frame, bool):
                raise ValueError("animation set_key: 'frame' must be an integer")
            base = {"interpolation": interpolation, "keys": []} if curve is None else curve
            new_curve = _merge_key(base, frame, value)
            _validate_curve(new_curve)
            node_curves[param] = new_curve
            return {"id": node_id, "param": param, "frame": frame, "value": float(value),
                    "interpolation": interpolation}
        if op == "delete_key":
            frame = cmd["frame"]
            if type(frame) is not int or isinstance(frame, bool):
                raise ValueError("animation delete_key: 'frame' must be an integer")
            if curve is None:
                raise ValueError(f"animation delete_key: no curve on {node_id!r}.{param!r}")
            new_curve = _drop_key(curve, frame)
            if new_curve is None:
                del node_curves[param]
                if not node_curves:
                    del curves_root[node_id]
                return {"id": node_id, "param": param, "frame": frame, "removed": True}
            node_curves[param] = new_curve
            return {"id": node_id, "param": param, "frame": frame, "remaining_keys": len(new_curve["keys"])}
        if op == "clear_curve":
            if param not in node_curves:
                raise ValueError(f"animation clear_curve: no curve on {node_id!r}.{param!r}")
            del node_curves[param]
            if not node_curves:
                del curves_root[node_id]
            return {"id": node_id, "param": param, "cleared": True}
        raise ValueError(f"Unknown animation operation: {op}")


def demo_document():
    d = Dispatcher()
    d.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Checker", "name": "Checker · procedural plate", "pos": [-210, -150]},
        {"op": "create", "id": "grade", "type": "Grade", "pos": [-210, -30], "params": {"exposure": 0.35}},
        {"op": "connect", "id": "grade", "input": "image", "source": "plate"},
        {"op": "create", "id": "wash", "type": "Constant", "name": "Constant · blue wash", "pos": [100, -100], "params": {"alpha": 0.22}},
        {"op": "create", "id": "merge", "type": "Merge", "pos": [-110, 100]},
        {"op": "connect", "id": "merge", "input": "A", "source": "wash"},
        {"op": "connect", "id": "merge", "input": "B", "source": "grade"},
        {"op": "create", "id": "viewer", "type": "Viewer", "pos": [-110, 230]},
        {"op": "connect", "id": "viewer", "input": "image", "source": "merge"},
        {"op": "view", "id": "viewer"}]})
    return d.document
