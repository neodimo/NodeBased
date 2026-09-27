"""Group, Input and Output: a node graph nested inside a node (Nuke's Group / Input / Output).

A `Group` node carries its own graph under the node key ``"graph"``: ``{"nodes", "animation",
"node_data"}``, laid out exactly like the same sections of a document so every existing rule
(validation, the edit operations, curves, shapes) applies inside it unchanged. Inside the graph,
`Input` nodes (numbered, ``input_number``) stand for the group's input slots ``in<N>`` and exactly
one `Output` node names what the group produces.

Evaluation never sees a Group. `flatten_groups` rewrites a document into the equivalent plain
graph (the Input nodes bound to whatever is wired into the group, the Output node replaced by its
source, a bypassed group replaced by its first input) before the evaluator and the tile executor
read it. Grouped and ungrouped graphs therefore run the same kernels on the same inputs in the
same order, produce identical pixels, and share cache keys: a cache key is a hash of the kernel,
its parameters and its input keys, so editing a knob inside a group changes the keys of exactly
the nodes that depend on it.
"""
from __future__ import annotations

import copy

MAX_GROUP_DEPTH = 8
GROUP_KINDS = ("Group", "Input", "Output")
GRAPH_KEYS = {"nodes", "animation", "node_data"}
SEPARATOR = "/"


def empty_graph():
    return {"nodes": {}, "animation": {"curves": {}}, "node_data": {}}


def input_number(node):
    return int(node["params"]["input_number"])


def group_slots(node):
    """The input slots of a Group node, one per Input node in its graph, lowest number first."""
    graph = node.get("graph")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), dict):
        return []
    numbers = sorted({input_number(inner) for inner in graph["nodes"].values()
                      if isinstance(inner, dict) and inner.get("type") == "Input"
                      and type(inner.get("params", {}).get("input_number")) is int})
    return [f"in{number}" for number in numbers]


def new_group_graph(input_id="input1", output_id="output"):
    """The graph of a fresh Group: Input 1 wired straight through to the Output."""
    from .core import SPECS
    graph = empty_graph()
    graph["nodes"][input_id] = {"type": "Input", "name": "Input1", "params": dict(SPECS["Input"]["params"]),
                                "inputs": {}, "pos": [0, 0], "disabled": False}
    graph["nodes"][output_id] = {"type": "Output", "name": "Output", "params": {},
                                 "inputs": {"image": input_id}, "pos": [0, 160], "disabled": False}
    return graph


def has_groups(doc):
    return any(node["type"] == "Group" for node in doc["nodes"].values())


# --- validation --------------------------------------------------------------------------------

def validate_graph(node, doc, depth):
    """Raise ValueError unless the graph inside Group `node` is a sound graph of its own."""
    from .core import validate
    label = node.get("name", "Group")
    graph = node.get("graph")
    if not isinstance(graph, dict) or set(graph) != GRAPH_KEYS or not isinstance(graph["nodes"], dict):
        raise ValueError(f"{label}: a Group needs a graph of nodes, animation and node_data")
    if depth >= MAX_GROUP_DEPTH:
        raise ValueError(f"Groups nest at most {MAX_GROUP_DEPTH} deep")
    validate(scope_document(doc, graph), depth + 1)
    inner = graph["nodes"]
    outputs = [key for key, value in inner.items() if value["type"] == "Output"]
    if len(outputs) != 1:
        raise ValueError(f"{label}: a Group needs exactly one Output node, found {len(outputs)}")
    numbers = [value["params"]["input_number"] for value in inner.values() if value["type"] == "Input"]
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"{label}: two Input nodes share a number")


def scope_document(doc, graph):
    """A document-shaped view of `graph` (sharing its dicts) for validation and for edits inside it.

    Only the graph sections belong to the group; time and settings come from the document. The
    settings are a private copy, so an edit run against the view cannot reach the real ones.
    """
    from .core import VIEWER_EXTRA_KEYS, VIEWER_STATE_KEYS
    settings = copy.deepcopy(doc["settings"])
    for name in VIEWER_STATE_KEYS + VIEWER_EXTRA_KEYS:
        settings["viewer"].pop(name, None)
    return {"version": doc["version"], "nodes": graph["nodes"], "view": None, "time": doc["time"],
            "animation": graph["animation"], "settings": settings,
            "node_data": graph["node_data"], "expressions": {}, "references": []}


def resolve_path(doc, path):
    """The Group nodes named by `path` (outermost first) and the innermost graph."""
    if not isinstance(path, list) or not path or any(type(step) is not str for step in path):
        raise ValueError("path must be a non-empty list of group ids")
    nodes, chain = doc["nodes"], []
    for step in path:
        node = nodes.get(step)
        if node is None or node["type"] != "Group":
            raise ValueError(f"path: {step!r} is not a Group here")
        chain.append((nodes, node))
        nodes = node["graph"]["nodes"]
    return chain


def sync_group_slots(node):
    """Bring a Group node's input dict in line with the Input nodes of its graph, keeping the
    connections of every slot that still exists."""
    node["inputs"] = {slot: node["inputs"].get(slot) for slot in group_slots(node)}


# --- flattening --------------------------------------------------------------------------------

class _Scope:
    def __init__(self, graph, prefix, bindings, inner):
        self.nodes, self.prefix, self.bindings, self.inner = graph["nodes"], prefix, bindings, inner
        self.curves = graph["animation"]["curves"]
        self.data = graph["node_data"]
        self.children = {}


def flatten_groups(doc, target=None):
    """`(document, target)` with every Group expanded in place.

    A document without groups comes back untouched (the same object), so ungrouped graphs keep
    their exact behaviour. `target` may name a Group: it maps to the node feeding its Output.
    Raises ValueError when the target is a group whose output is not connected.
    """
    if not has_groups(doc):
        return doc, target
    from .core import bypass_slot
    top = _Scope({"nodes": doc["nodes"], "animation": doc.get("animation") or {"curves": {}},
                  "node_data": doc.get("node_data") or {}}, "", {}, False)

    def enter(scope, gid):
        child = scope.children.get(gid)
        if child is None:
            node = scope.nodes[gid]
            bindings = {input_number(inner): output_of(scope, node["inputs"].get(f"in{input_number(inner)}"))
                        for inner in node["graph"]["nodes"].values() if inner["type"] == "Input"}
            child = scope.children[gid] = _Scope(node["graph"], scope.prefix + gid + SEPARATOR, bindings, True)
        return child

    def output_of(scope, key):
        """The flat id whose value node `key` of `scope` stands for (None when it stands for nothing)."""
        while key is not None:
            node = scope.nodes[key]
            kind = node["type"]
            if kind == "Group":
                if node["disabled"]:
                    slot = bypass_slot(node)
                    key = None if slot is None else node["inputs"].get(slot)
                    continue
                scope = enter(scope, key)
                key = next(inner["inputs"]["image"] for inner in scope.nodes.values() if inner["type"] == "Output")
                continue
            if scope.inner and kind == "Output":
                key = node["inputs"]["image"]
                continue
            if scope.inner and kind == "Input":
                return scope.bindings.get(input_number(node))
            return scope.prefix + key
        return None

    flat_nodes, curves, data = {}, {}, {}

    def emit(scope):
        for key, node in scope.nodes.items():
            kind = node["type"]
            if kind == "Group":
                if not node["disabled"]:
                    emit(enter(scope, key))
                continue
            if scope.inner and kind in ("Input", "Output"):
                continue
            flat = scope.prefix + key
            flat_nodes[flat] = {**node, "inputs": {slot: output_of(scope, source)
                                                    for slot, source in node["inputs"].items()}}
            if key in scope.curves:
                curves[flat] = scope.curves[key]
            if key in scope.data:
                data[flat] = scope.data[key]

    emit(top)
    mapped = output_of(top, target) if target is not None else None
    if target is not None and mapped is None:
        raise ValueError(f"{doc['nodes'][target]['name']}: the group output is not connected")
    flat_doc = {**doc, "nodes": flat_nodes, "animation": {**(doc.get("animation") or {}), "curves": curves},
                "node_data": data, "view": None if doc.get("view") is None else output_of(top, doc["view"])}
    return flat_doc, mapped


# --- grouping and ungrouping -------------------------------------------------------------------

def _unique(taken, wanted):
    key, count = wanted, 1
    while key in taken:
        count += 1
        key = f"{wanted}{count}"
    return key


def make_group(doc, ids, group_id, name="Group"):
    """Move the nodes `ids` into a new Group node of `doc` (a document or a scope document).

    Every connection from outside the selection into it becomes a numbered Input node (one per
    distinct source) and a slot of the group; the one selected node the outside reads (or, when
    nothing outside reads any, the one selected node nothing else in the selection reads) feeds
    the group's Output. The outside is rewired to the group. Returns the new group's id.
    """
    from .core import viewer_state
    nodes = doc["nodes"]
    if not isinstance(ids, list) or not ids or any(type(key) is not str for key in ids):
        raise ValueError("group: ids must be a non-empty list of node ids")
    selected = list(dict.fromkeys(ids))
    missing = [key for key in selected if key not in nodes]
    if missing:
        raise ValueError(f"group: unknown node id {missing[0]!r}")
    if not isinstance(group_id, str) or not group_id or len(group_id) > 128 or group_id in nodes:
        raise ValueError("group: the group id must be a new, non-empty string")
    inside = set(selected)
    for key in doc["expressions"]:
        if key in inside:
            raise ValueError("group: nodes with expressions cannot be grouped yet")
    for params in doc["expressions"].values():
        for text in params.values():
            from . import expressions as expr
            if any(target in inside for target, _ in expr.parse(text).references):
                raise ValueError("group: nodes read by an expression cannot be grouped yet")
    if inside & set(doc["references"]):
        raise ValueError("group: a reference node cannot be grouped")

    # Which selected nodes does the outside read, and which does nothing inside the selection read?
    read_outside = [key for key in selected
                    if any(other not in inside and key in node["inputs"].values() for other, node in nodes.items())]
    viewed = doc["view"] if doc["view"] in inside else None
    viewer_inputs = [value for value in viewer_state(doc)["inputs"] if value in inside]
    if viewed is not None:
        read_outside.append(viewed)
    read_outside += viewer_inputs
    read_outside = list(dict.fromkeys(read_outside))
    if len(read_outside) > 1:
        names = ", ".join(repr(nodes[key]["name"]) for key in read_outside)
        raise ValueError(f"group: a group has one output, but the outside reads {names}")
    if read_outside:
        output_source = read_outside[0]
    else:
        consumed = {source for key in selected for source in nodes[key]["inputs"].values() if source in inside}
        sinks = [key for key in selected if key not in consumed and _produces_image(nodes[key])]
        if len(sinks) > 1:
            raise ValueError("group: the selection ends in more than one node; select a single chain to group")
        output_source = sinks[0] if sinks else None

    # Input nodes: one per distinct outside source, numbered in the order they are met.
    graph = empty_graph()
    inner = graph["nodes"]
    for key in selected:
        inner[key] = nodes[key]
    outer_sources = []
    for key in selected:
        for source in nodes[key]["inputs"].values():
            if source is not None and source not in inside and source not in outer_sources:
                outer_sources.append(source)
    xs = [nodes[key]["pos"][0] for key in selected]
    ys = [nodes[key]["pos"][1] for key in selected]
    input_ids = {}
    for number, source in enumerate(outer_sources, 1):
        input_id = _unique(inner, f"input{number}")
        input_ids[source] = input_id
        inner[input_id] = {"type": "Input", "name": f"Input{number}", "params": {"input_number": number},
                           "inputs": {}, "pos": [min(xs) + 200 * (number - 1), min(ys) - 140], "disabled": False}
    for key in selected:
        node = inner[key]
        node["inputs"] = {slot: input_ids.get(source, source) for slot, source in node["inputs"].items()}
    output_id = _unique(inner, "output")
    inner[output_id] = {"type": "Output", "name": "Output", "params": {}, "inputs": {"image": output_source},
                        "pos": [sum(xs) / len(xs), max(ys) + 140], "disabled": False}
    for section, target in (("curves", graph["animation"]["curves"]), ("data", graph["node_data"])):
        source = doc["animation"]["curves"] if section == "curves" else doc["node_data"]
        for key in selected:
            if key in source:
                target[key] = source.pop(key)

    group = {"type": "Group", "name": name, "params": {}, "graph": graph,
             "inputs": {f"in{number}": source for number, source in enumerate(outer_sources, 1)},
             "pos": [sum(xs) / len(xs), sum(ys) / len(ys)], "disabled": False}
    for key in selected:
        del nodes[key]
    for node in nodes.values():
        node["inputs"] = {slot: group_id if source == output_source and output_source is not None else source
                          for slot, source in node["inputs"].items()}
    nodes[group_id] = group
    _remap_view(doc, {output_source: group_id} if output_source is not None else {}, inside)
    return group_id


def _produces_image(node):
    from .core import OUTPUT_TYPES
    return OUTPUT_TYPES.get(node["type"], "image") == "image"


def _remap_view(doc, mapping, gone):
    """Point the viewer at the replacement of a moved node; forget nodes that left the graph."""
    from .core import _store_viewer_state, viewer_state
    if doc["view"] in gone:
        doc["view"] = mapping.get(doc["view"])
    state = viewer_state(doc)
    state["inputs"] = [mapping.get(value) if value in gone else value for value in state["inputs"]]
    _store_viewer_state(doc, state)


def ungroup(doc, group_id):
    """The inverse of make_group: the group's nodes come back into `doc` and the group goes.

    Inner connections from an Input node return to whatever was wired into that group slot, and
    everything that read the group reads the node that fed its Output.
    """
    nodes = doc["nodes"]
    group = nodes.get(group_id)
    if group is None or group["type"] != "Group":
        raise ValueError(f"ungroup: {group_id!r} is not a Group")
    if group["disabled"]:
        raise ValueError("ungroup: enable the group first, so its bypass does not change the result")
    graph = group["graph"]
    inner = graph["nodes"]
    clash = sorted(key for key, node in inner.items() if node["type"] not in ("Input", "Output") and key in nodes)
    if clash:
        raise ValueError(f"ungroup: node id {clash[0]!r} already exists outside the group")
    bound = {key: group["inputs"].get(f"in{input_number(node)}") for key, node in inner.items()
             if node["type"] == "Input"}
    output = next(node for node in inner.values() if node["type"] == "Output")

    def outer(source):
        seen = set()
        while source in inner and source not in seen:
            seen.add(source)
            node = inner[source]
            if node["type"] == "Input":
                return bound[source]
            if node["type"] == "Output":
                source = node["inputs"]["image"]
                continue
            return source
        return source

    replacement = outer(output["inputs"]["image"])
    del nodes[group_id]
    for key, node in inner.items():
        if node["type"] in ("Input", "Output"):
            continue
        node["inputs"] = {slot: outer(source) for slot, source in node["inputs"].items()}
        nodes[key] = node
    for node in nodes.values():
        node["inputs"] = {slot: replacement if source == group_id else source
                          for slot, source in node["inputs"].items()}
    doc["animation"]["curves"].update(graph["animation"]["curves"])
    doc["node_data"].update(graph["node_data"])
    _remap_view(doc, {group_id: replacement}, {group_id})
