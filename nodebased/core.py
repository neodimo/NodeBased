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
SPECS = {
    "Read": {"inputs": [], "params": {"path": "", "colorspace": "Auto", "alpha_mode": "Auto", "layer": "", "subimage": 0}},
    "Constant": {"inputs": [], "params": {"width": 960, "height": 540, "red": 0.12, "green": 0.3, "blue": 0.6, "alpha": 1.0}},
    "Checker": {"inputs": [], "params": {"width": 960, "height": 540, "size": 64}},
    "Grade": {"inputs": ["image"], "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0}},
    "ColorCorrect": {"inputs": ["image"], "params": {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0}},
    "Blur": {"inputs": ["image"], "params": {"radius": 8.0}},
    "Transform": {"inputs": ["image"], "params": {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                                                 "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest"}},
    "Crop": {"inputs": ["image"], "params": {"x": 0, "y": 0, "width": 960, "height": 540}},
    "Shuffle": {"inputs": ["image"], "params": {"red_from": "R", "green_from": "G", "blue_from": "B", "alpha_from": "A"}},
    "Merge": {"inputs": ["A", "B"], "params": {"operation": "over", "mix": 1.0}},
    "Premult": {"inputs": ["image"], "params": {}},
    "Unpremult": {"inputs": ["image"], "params": {}},
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
          "lift": (-10, 10), "gamma": (0.01, 100), "gain": (0, 100), "saturation": (0, 10), "radius": (0, 500)}


# Merge keeps A as foreground and B as background, mirroring Nuke's wiring convention.
MERGE_OPERATIONS = ("over", "under", "plus", "minus", "multiply", "screen", "max", "min",
                    "difference", "divide", "mask", "stencil", "in", "out", "atop", "xor")
TRANSFORM_FILTERS = ("nearest", "bilinear", "cubic")
CHOICES = {"colorspace": ["Auto", "sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1", "Raw"],
           "alpha_mode": ["Auto", "Straight", "Premultiplied"],
           "operation": list(MERGE_OPERATIONS),
           "filter": list(TRANSFORM_FILTERS),
           "red_from": ["R", "G", "B", "A", "0", "1"], "green_from": ["R", "G", "B", "A", "0", "1"],
           "blue_from": ["R", "G", "B", "A", "0", "1"], "alpha_from": ["R", "G", "B", "A", "0", "1"]}


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
    return doc


def empty_document():
    return {"version": 3, "nodes": {}, "view": None}


def validate(doc):
    if not isinstance(doc, dict) or set(doc) != {"version", "nodes", "view"} or doc["version"] != 3:
        raise ValueError("Unsupported or malformed NodeBased document")
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
        if not isinstance(node["inputs"], dict) or set(node["inputs"]) != set(spec["inputs"]):
            raise ValueError(f"Invalid inputs for {kind}")
        for source in node["inputs"].values():
            if source is not None and (not isinstance(source, str) or source not in nodes):
                raise ValueError("Input references a missing node")
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
                    "operations": ["describe", "inspect", "create", "set", "connect", "move", "rename", "disable", "delete", "view", "batch", "undo", "redo", "save", "load"]}
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
                          "inputs": {slot: None for slot in SPECS[kind]["inputs"]},
                          "pos": cmd.get("pos", [0, 0]), "disabled": False}
            return {"id": key}
        if op == "view":
            doc["view"] = cmd.get("id")
            return {}
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
