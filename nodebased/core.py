"""Serializable graph and the shared human/agent command boundary (no Qt imports)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from . import shapes

# Parameter schemas are also consumed by the inspector and agent discovery.
# Filter nodes accept an optional "mask" image (alpha gates where the filter applies) and a
# per-node "mix" (blend between original input and filtered output). The mask slot is listed in
# "optional_inputs" rather than "inputs" so a node validates without it wired — the evaluator
# treats None there as full opacity (M.a = 1).
IMAGE_FILTER_KINDS = ("Grade", "ColorCorrect", "Blur", "Transform", "Crop")

# Kinds that honour the optional-mask + mix contract. IMAGE_FILTER_KINDS is frozen history — the
# v3 -> v4 upgrade is written against it — so a kind that adopts the contract later joins this
# list instead, which is what the inspector and the docs read.
MASK_MIX_KINDS = IMAGE_FILTER_KINDS + ("Tracker",)

# The version `upgrade_document` migrates to and `validate` accepts. Tests and callers should refer
# to this rather than hard-coding a number, so a schema bump does not spray stale literals.
SCHEMA_VERSION = 7
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
    # Two-input explicit channel routing. Separate from Shuffle (one input, kept unchanged) because
    # CHOICES is keyed by parameter name globally — sharing "red_from" would force one option list
    # on both nodes. Naming a B.* source with B unwired is an error, never a silent black channel.
    "ChannelShuffle": {"inputs": ["A"], "optional_inputs": ["B"],
                       "params": {"out_red": "A.r", "out_green": "A.g", "out_blue": "A.b", "out_alpha": "A.a"}},
    # Roto is a generator: it states its own format rather than inheriting one from an image input
    # and then quietly disagreeing with it. Shapes live in document["node_data"], not in params —
    # see docs/ROTO_TRACKING.md.
    "Roto": {"inputs": [], "params": {"width": 960, "height": 540, "invert": 0}},
    # Tracker is a Transform whose transform is solved from tracks in node_data instead of typed
    # in. It honours the same optional-mask + mix contract as the image filters.
    "Tracker": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"reference_frame": 1, "mode": "match_move", "apply_translate": 1,
                           "apply_rotate": 1, "apply_scale": 1, "filter": "bilinear", "mix": 1.0}},
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
          "invert": (0, 1), "reference_frame": (-1000000, 1000000),
          "apply_translate": (0, 1), "apply_rotate": (0, 1), "apply_scale": (0, 1),
          "frame_offset": (-1000000, 1000000)}

# Declared artifact type per node kind, for docs/EVALUATION_TIERS.md clause C6. The cache does not
# yet *store* the type — typed cache entries are v0.9.0 work — so this is the declaration the
# scheduler will read, not a claim that typed storage exists.
ARTIFACT_TYPES = {"Roto": "matte"}
DEFAULT_ARTIFACT_TYPE = "image"


def artifact_type(kind):
    if kind not in SPECS:
        raise ValueError(f"Unknown node type: {kind}")
    return ARTIFACT_TYPES.get(kind, DEFAULT_ARTIFACT_TYPE)

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
TRACKER_MODES = ("match_move", "stabilise")
# Each ChannelShuffle output names its source explicitly. "0"/"1" are constants; there is no
# "leave it alone" option, because that is the one that hides a mistake.
CHANNEL_SOURCES = ("A.r", "A.g", "A.b", "A.a", "B.r", "B.g", "B.b", "B.a", "0", "1")
CHOICES = {"colorspace": ["Auto", "sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1", "Raw"],
           "alpha_mode": ["Auto", "Straight", "Premultiplied"],
           "operation": list(MERGE_OPERATIONS),
           "filter": list(TRANSFORM_FILTERS),
           "red_from": ["R", "G", "B", "A", "0", "1"], "green_from": ["R", "G", "B", "A", "0", "1"],
           "blue_from": ["R", "G", "B", "A", "0", "1"], "alpha_from": ["R", "G", "B", "A", "0", "1"],
           "missing": list(MISSING_FRAME_POLICIES),
           "mode": list(TRACKER_MODES),
           "out_red": list(CHANNEL_SOURCES), "out_green": list(CHANNEL_SOURCES),
           "out_blue": list(CHANNEL_SOURCES), "out_alpha": list(CHANNEL_SOURCES)}


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
        # ---------------------------------------------------------------------------------------
        # v5 -> v6 BRIDGE STUB. Schema v6 (parameter animation curves) is owned by branch
        # `m3/animation-curves` and is NOT merged. This branch needs a v7 section that layers on
        # top of v6, so it carries the smallest v6 step that can exist: a v5 document has no
        # curves, so creating the empty section is the whole migration.
        #
        # DELETE THIS BLOCK at merge time and take M3's real v5 -> v6 step, together with
        # `nodebased.animation`'s curve validation, which is stricter than the structural check in
        # `validate_animation` below. Nothing on this branch evaluates a curve on a node parameter.
        # See docs/ROTO_TRACKING.md.
        # ---------------------------------------------------------------------------------------
        doc["animation"] = {"curves": {}}
        doc["version"] = 6
    if isinstance(doc, dict) and doc.get("version") == 6:
        # v6 -> v7: the document gains `node_data`, a generic per-node structured payload keyed by
        # node id the way animation.curves is. No v6 node type carries a payload, so every existing
        # comp upgrades to an empty section and renders byte-identically.
        doc["node_data"] = {}
        doc["version"] = SCHEMA_VERSION
    return doc


def empty_document():
    return {"version": SCHEMA_VERSION, "nodes": {}, "view": None, "time": dict(DEFAULT_TIME),
            "animation": {"curves": {}}, "node_data": {}}


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


def validate_animation(animation, nodes):
    """Structural check on the v6 animation section.

    Intentionally shallow: branch `m3/animation-curves` owns the real per-curve rules and its
    `nodebased.animation.validate_curve` is strictly stronger than this. Rejecting an orphan node
    id is the one rule this branch needs, because the v7 `delete` path has to clear both sections
    and a test should notice if it stops doing so.
    """
    if not isinstance(animation, dict) or set(animation) != {"curves"}:
        raise ValueError("animation must define exactly 'curves'")
    curves = animation["curves"]
    if not isinstance(curves, dict):
        raise ValueError("animation.curves must be an object keyed by node id")
    for key, slots in curves.items():
        if key not in nodes:
            raise ValueError(f"animation.curves[{key!r}] does not name a node in this document")
        if not isinstance(slots, dict):
            raise ValueError(f"animation.curves[{key!r}] must be an object keyed by parameter name")


def validate(doc):
    if (not isinstance(doc, dict)
            or set(doc) != {"version", "nodes", "view", "time", "animation", "node_data"}
            or doc["version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported or malformed NodeBased document")
    validate_time(doc["time"])
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
    validate_animation(doc["animation"], nodes)
    shapes.validate_node_data(doc["node_data"], nodes)
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
            return {"protocol": 1, "nodes": copy.deepcopy(SPECS), "limits": LIMITS, "choices": CHOICES,
                    "time": copy.deepcopy(self.document["time"]), "time_limits": TIME_LIMITS,
                    "sequence_patterns": ["printf (plate.%04d.exr)", "hash (plate.####.exr)", "still (plate.exr)"],
                    "artifact_types": {kind: artifact_type(kind) for kind in SPECS},
                    # Structured per-node payloads (schema v7). An agent discovers which node types
                    # carry one, and the envelope every time-varying number inside it uses.
                    "node_data": {"payloads": dict(shapes.NODE_DATA_SCHEMA),
                                  "shape_modes": list(shapes.SHAPE_MODES),
                                  "curve_interpolations": list(shapes.CURVE_INTERPOLATIONS),
                                  "scalar_limits": {k: list(v) for k, v in shapes.SHAPE_LIMITS.items()},
                                  "point_fields": list(shapes.POINT_FIELDS)},
                    "operations": ["describe", "inspect", "create", "set", "connect", "move", "rename", "disable", "delete", "view", "time", "set_shapes", "set_tracks", "batch", "undo", "redo", "save", "load"]}
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
        elif op in ("set_shapes", "set_tracks"):
            # Whole-payload replacement, validated by `validate` like any other edit and taking one
            # undo slot. There is no per-key op yet: keying a single point waits for v6 so it can
            # reuse `nodebased.animation.merge_key` instead of growing a second key-insert path.
            slot = "shapes" if op == "set_shapes" else "tracks"
            if shapes.payload_slot(node["type"]) != slot:
                raise ValueError(f"{op}: {node['type']} nodes do not carry {slot}")
            items = cmd[slot]
            if not isinstance(items, list):
                raise ValueError(f"{op}: {slot!r} must be a list")
            # An empty payload is stored as an absent entry so "no shapes" has one representation
            # in the document and two comps that look identical also serialize identically.
            if items:
                doc["node_data"][key] = {slot: copy.deepcopy(items)}
            else:
                doc["node_data"].pop(key, None)
            return {slot: len(items)}
        elif op == "disable":
            if not SPECS[node["type"]]["inputs"]:
                raise ValueError("Source nodes cannot be bypassed")
            node["disabled"] = cmd["value"]
        elif op == "delete":
            del nodes[key]
            # Side-data sections are keyed by node id, so deleting a node must clear them in the
            # same atomic edit or the document fails validation with an orphan entry.
            doc["node_data"].pop(key, None)
            doc["animation"]["curves"].pop(key, None)
            for other in nodes.values():
                other["inputs"] = {slot: None if value == key else value for slot, value in other["inputs"].items()}
            if doc["view"] == key:
                doc["view"] = None
        else:
            raise ValueError(f"Unknown edit operation: {op}")
        return {}


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
