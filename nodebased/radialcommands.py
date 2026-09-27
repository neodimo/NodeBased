"""User-defined radial menu commands (plan "Radial menu", DiMo 9/27, deliverable R2).

A command is one JSON file in the user's command folder (`user_commands_directory`, created
next to the app's other per-machine state): `label`, an optional `when` (`count`, `types`,
`categories`, `output_types` -- all optional, all ANDed, and each checked against every node in
the selection), an optional preferred `slot` (0-7), `enabled`, and either an inline `ops` list
(the agent op vocabulary -- see `core.Dispatcher`) or a `script` naming a co-located Python file
exposing `run(ctx)` (and, optionally, a stricter `when(ctx)` that can inspect the graph, not just
the selection's own types). Either body runs as one undo unit: `ops` are resolved and sent as a
single `batch` command; a script's `ctx.dispatch` calls are collected and sent the same way once
`run` returns.

Files reload whenever the folder's contents change (checked by mtime, no watcher thread needed:
`load_all` is cheap and already called once per ring-open). A file that fails to parse or
validate is reported as an error and simply does not appear on the ring -- it never stops the
other commands from loading.

Placeholders inside an `ops` list (each a plain JSON string, resolved before the ops run):
  $selected            the whole selection, as a list of node ids
  $selected[N]          the Nth selected node id (negative indices count from the end)
  $downstream_of(ID)     every node id downstream of ID (ID may itself be a placeholder)
  $new:NAME              a fresh node id, the same value every time $new:NAME appears in one
                         run -- lets a `create` op be wired to by a later op in the same list,
                         since `core.Dispatcher` only assigns an id when the caller omits one.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QStandardPaths

from .core import SPECS, OUTPUT_TYPES
from .nodecatalog import node_category

SLOT_COUNT = 8
_MANIFEST_SUFFIX = ".json"


def user_commands_directory() -> Path:
    """Where user command files live. A dialogue box or a text editor can be pointed here."""
    return Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation)) / "radial_commands"


# ---- the command model -------------------------------------------------------------------------

@dataclass
class UserCommand:
    id: str                         # the manifest's filename stem: also the file's identity
    path: Path
    label: str
    when: dict
    slot: Optional[int]
    enabled: bool
    ops: Optional[list] = None       # set when the body is an inline op list
    script_path: Optional[Path] = None   # set when the body is a Python file
    script_run: Optional[Callable] = None
    script_when: Optional[Callable] = None
    requires_types: Tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def kind(self) -> str:
        return "script" if self.script_path is not None else "ops"


@dataclass
class CommandLoadError:
    path: Path
    error: str


def _check_requires(names) -> Optional[str]:
    missing = [name for name in names if name not in SPECS]
    if not missing:
        return None
    return f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not available in this build yet."


def _load_script(directory: Path, script_name: str):
    script_path = directory / script_name
    if not script_path.is_file():
        raise ValueError(f"script {script_name!r} not found next to the manifest")
    spec = importlib.util.spec_from_file_location(f"nodebased_radial_command_{script_path.stem}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise ValueError(f"{script_name} does not define run(ctx)")
    return script_path, run, getattr(module, "when", None)


def _parse_manifest(path: Path) -> UserCommand:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("a command file must hold a single JSON object")
    label = data.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError('"label" must be a non-empty string')
    when = data.get("when", {})
    if not isinstance(when, dict):
        raise ValueError('"when" must be an object')
    slot = data.get("slot")
    if slot is not None and (not isinstance(slot, int) or not (0 <= slot < SLOT_COUNT)):
        raise ValueError(f'"slot" must be an integer from 0 to {SLOT_COUNT - 1}, or absent')
    enabled = bool(data.get("enabled", True))
    requires_types = tuple(data.get("requires_types", ()))
    has_ops, has_script = "ops" in data, "script" in data
    if has_ops == has_script:
        raise ValueError('a command needs exactly one of "ops" or "script"')
    command = UserCommand(id=path.stem, path=path, label=label, when=when, slot=slot,
                          enabled=enabled, requires_types=requires_types)
    if has_ops:
        if not isinstance(data["ops"], list):
            raise ValueError('"ops" must be a list')
        command.ops = data["ops"]
    else:
        if not isinstance(data["script"], str):
            raise ValueError('"script" must be a filename')
        script_path, run, when_fn = _load_script(path.parent, data["script"])
        command.script_path, command.script_run, command.script_when = script_path, run, when_fn
    command.unavailable_reason = _check_requires(command.requires_types)
    return command


def load_all(directory: Optional[Path] = None) -> Tuple[List[UserCommand], List[CommandLoadError]]:
    """Every command file in `directory` (default `user_commands_directory()`), split into the
    ones that parsed and the ones that did not. Never raises: a bad file becomes an error entry."""
    directory = directory if directory is not None else user_commands_directory()
    commands, errors = [], []
    if not directory.is_dir():
        return commands, errors
    for path in sorted(directory.glob(f"*{_MANIFEST_SUFFIX}")):
        try:
            commands.append(_parse_manifest(path))
        except (ValueError, OSError, json.JSONDecodeError, SyntaxError, ImportError) as error:
            errors.append(CommandLoadError(path=path, error=str(error)))
    return commands, errors


# ---- selection/graph helpers shared by placeholders, `when`, and scripts -----------------------

def downstream_of(nodes: Dict[str, dict], key: str) -> List[str]:
    """Every node id downstream of `key`, transitively, in discovery order."""
    result, seen, frontier = [], {key}, [key]
    while frontier:
        current = frontier.pop()
        for node_id, node in nodes.items():
            if node_id in seen:
                continue
            if current in (node.get("inputs") or {}).values():
                seen.add(node_id)
                result.append(node_id)
                frontier.append(node_id)
    return result


def consumers_of(nodes: Dict[str, dict], key: str) -> List[str]:
    """Every node, anywhere in the graph, that wires `key` into one of its own inputs."""
    return [node_id for node_id, node in nodes.items()
            if key in (node.get("inputs") or {}).values()]


def ordered_chain(nodes: Dict[str, dict], ids) -> Optional[List[str]]:
    """`ids` reordered root-first if they form one simple, unbranched chain inside the selection
    (each link fed only by the previous one, feeding only into the next); `None` if the selection
    branches, merges, or is disconnected, since duplicating an ambiguous shape has no one right
    answer."""
    idset = set(ids)
    if not idset:
        return None

    def selected_inputs(node_id):
        return [v for v in (nodes[node_id].get("inputs") or {}).values() if v in idset]

    def selected_consumers(node_id):
        return [other for other in idset if node_id in (nodes[other].get("inputs") or {}).values()]

    heads = [node_id for node_id in idset if not selected_inputs(node_id)]
    if len(heads) != 1:
        return None
    order, seen, current = [heads[0]], {heads[0]}, heads[0]
    while len(order) < len(idset):
        forward = selected_consumers(current)
        if len(forward) != 1 or forward[0] in seen or selected_inputs(forward[0]) != [current]:
            return None
        current = forward[0]
        order.append(current)
        seen.add(current)
    return order


def _when_matches(when: dict, selection: List[dict]) -> bool:
    count = when.get("count")
    if count is not None:
        low, high = (count, count) if isinstance(count, int) else tuple(count)
        if not (low <= len(selection) <= high):
            return False
    types = when.get("types")
    if types is not None and any(node["type"] not in types for node in selection):
        return False
    categories = when.get("categories")
    if categories is not None and any(node_category(node["type"]) not in categories for node in selection):
        return False
    output_types = when.get("output_types")
    if output_types is not None and any(OUTPUT_TYPES.get(node["type"], "image") not in output_types
                                        for node in selection):
        return False
    return True


# ---- placeholder resolution ---------------------------------------------------------------------

_SELECTED_INDEX_RE = re.compile(r"^\$selected\[(-?\d+)\]$")
_DOWNSTREAM_RE = re.compile(r"^\$downstream_of\((.+)\)$")
_NEW_ID_RE = re.compile(r"^\$new:(\w+)$")


def resolve_placeholders(value, selected_ids: List[str], nodes: Dict[str, dict], new_ids: Dict[str, str]):
    """Walks `value` (an op, or a whole ops list) replacing placeholder strings; `new_ids` is a
    per-run cache so every `$new:NAME` occurrence in one command resolves to the same fresh id."""
    if isinstance(value, str):
        if value == "$selected":
            return list(selected_ids)
        match = _SELECTED_INDEX_RE.match(value)
        if match:
            return selected_ids[int(match.group(1))]
        match = _DOWNSTREAM_RE.match(value)
        if match:
            target = resolve_placeholders(match.group(1).strip(), selected_ids, nodes, new_ids)
            return downstream_of(nodes, target)
        match = _NEW_ID_RE.match(value)
        if match:
            name = match.group(1)
            if name not in new_ids:
                new_ids[name] = uuid.uuid4().hex[:12]
            return new_ids[name]
        return value
    if isinstance(value, list):
        return [resolve_placeholders(item, selected_ids, nodes, new_ids) for item in value]
    if isinstance(value, dict):
        return {key: resolve_placeholders(item, selected_ids, nodes, new_ids) for key, item in value.items()}
    return value


def _fill_missing_positions(ops: List[dict], pos) -> List[dict]:
    """A `create` op with no `pos` of its own lands near where the ring opened, offset so several
    creates in one command do not stack exactly on top of each other."""
    filled, offset = [], 0
    for op in ops:
        if op.get("op") == "create" and "pos" not in op:
            op = {**op, "pos": [pos.x() + offset * 40, pos.y() + offset * 140]}
            offset += 1
        filled.append(op)
    return filled


# ---- execution context for a script body --------------------------------------------------------

class CommandContext:
    """What a command script's `run(ctx)` (and optional `when(ctx)`) sees: the selection and the
    graph on screen, read-only, plus `dispatch` to queue a document op. Every queued op runs, in
    order, as a single `batch` command once `run` returns -- one undo step, whatever `run` did."""

    def __init__(self, selected_ids: List[str], nodes: Dict[str, dict]):
        self.selected_ids = list(selected_ids)
        self.nodes = nodes
        self._ops: List[dict] = []

    def selected(self) -> List[str]:
        return list(self.selected_ids)

    def node(self, key: str) -> Optional[dict]:
        return self.nodes.get(key)

    def downstream_of(self, key: str) -> List[str]:
        return downstream_of(self.nodes, key)

    def consumers_of(self, key: str) -> List[str]:
        return consumers_of(self.nodes, key)

    def ordered_chain(self, ids) -> Optional[List[str]]:
        return ordered_chain(self.nodes, ids)

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def dispatch(self, op: dict):
        self._ops.append(op)

    def collected_ops(self) -> List[dict]:
        return list(self._ops)


# ---- fitting a command into the ring -------------------------------------------------------------

def _selected_node_dicts(nodes: Dict[str, dict], selected_ids: List[str]) -> List[dict]:
    return [nodes[key] for key in selected_ids if key in nodes]


def _command_applies(command: UserCommand, selection: List[dict], nodes: Dict[str, dict], selected_ids) -> bool:
    if not command.enabled or not command.available:
        return False
    if not _when_matches(command.when, selection):
        return False
    if command.script_when is not None:
        try:
            return bool(command.script_when(CommandContext(selected_ids, nodes)))
        except Exception:
            return False
    return True


def _run_command(command: UserCommand, window, selected_ids: List[str], pos):
    nodes = window.graph_nodes()
    if command.script_run is not None:
        ctx = CommandContext(selected_ids, nodes)
        command.script_run(ctx)
        ops = ctx.collected_ops()
    else:
        new_ids: Dict[str, str] = {}
        ops = resolve_placeholders(copy.deepcopy(command.ops), selected_ids, nodes, new_ids)
        ops = _fill_missing_positions(ops, pos)
    if ops:
        window.command({"op": "batch", "commands": ops})


def wrap_command(command: UserCommand):
    """The `RadialCommand` a user command file appears as on the ring, `id`-prefixed so it can
    never collide with a built-in id (`radialrules.commands_for` and its learned-slot overlay
    both look user commands up again by this same `user:`-prefixed id)."""
    from .radialrules import RadialCommand   # deferred: radialrules imports this module
    return RadialCommand(
        id=f"user:{command.id}", label=command.label,
        when=lambda _selection: True,
        run=lambda window, ids, pos, _command=command: _run_command(_command, window, ids, pos))


def overlay_user_commands(table: list, nodes: Dict[str, dict], selected_ids: List[str],
                          directory: Optional[Path] = None,
                          commands: Optional[List[UserCommand]] = None) -> list:
    """`table`: the built-in slots already resolved for this selection (a `RadialCommand` or
    `None` per slot -- see `radialrules.commands_for`). Every applicable, enabled user command
    fills its own preferred slot if that slot is empty, or the first empty slot otherwise; a
    command with no empty slot left is simply not offered this time (its file and its other
    contexts are unaffected). `commands` lets a caller that already loaded (and possibly
    filtered, e.g. for local-usage learning) the command list hand it in directly instead of
    this loading its own copy."""
    if commands is None:
        commands, _errors = load_all(directory)
    if not commands:
        return table
    selection = _selected_node_dicts(nodes, selected_ids)
    table = list(table)
    for command in commands:
        if not _command_applies(command, selection, nodes, selected_ids):
            continue
        entry = wrap_command(command)
        target = command.slot if command.slot is not None and table[command.slot] is None else None
        if target is None:
            target = next((index for index, existing in enumerate(table) if existing is None), None)
        if target is not None:
            table[target] = entry
    return table


# ---- saving/editing --------------------------------------------------------------------------

def save_ops_command(directory: Path, command_id: str, label: str, ops: list, when: Optional[dict] = None,
                     slot: Optional[int] = None, enabled: bool = True) -> Path:
    """Write (or overwrite) an inline-`ops` command file, validating it first the same way
    loading it later would."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"label": label, "when": when or {}, "slot": slot, "enabled": enabled, "ops": ops}
    path = directory / f"{command_id}{_MANIFEST_SUFFIX}"
    path.write_text(json.dumps(manifest, indent=2))
    _parse_manifest(path)   # raises here, before the caller believes the save succeeded
    return path


DEFAULT_COMMANDS = {
    "copy_nodes_to_each_branch.json": """{
  "label": "Copy nodes to each branch",
  "when": {"count": [1, 1000]},
  "slot": null,
  "enabled": true,
  "script": "copy_nodes_to_each_branch.py"
}
""",
    "copy_nodes_to_each_branch.py": '''"""Built-in radial command, doubling as the worked example for a script-bodied command file.

When the selected nodes form one unbranched chain that feeds a single node consumed by several
other nodes (several "branches" reading the same upstream work), duplicate the chain and that
shared node into every branch but the first, so each branch gets its own independent copy instead
of all of them reading the one original. The first branch is left exactly as it was: with N
branches this adds N-1 copies, which is the smallest edit that gives every branch its own.
"""
import copy


def when(ctx):
    chain = ctx.ordered_chain(ctx.selected())
    if not chain:
        return False
    consumers = ctx.consumers_of(chain[-1])
    if len(consumers) != 1:
        return False
    return len(ctx.consumers_of(consumers[0])) >= 2


def run(ctx):
    chain = ctx.ordered_chain(ctx.selected())
    if not chain:
        return
    tail_consumers = ctx.consumers_of(chain[-1])
    if len(tail_consumers) != 1:
        return
    branch_node = tail_consumers[0]
    branches = ctx.consumers_of(branch_node)
    if len(branches) < 2:
        return
    unit = chain + [branch_node]
    for branch_consumer in branches[1:]:
        mapping = {}
        for node_id in unit:
            node = ctx.node(node_id)
            new_id = ctx.new_id()
            mapping[node_id] = new_id
            ctx.dispatch({"op": "create", "id": new_id, "type": node["type"],
                         "params": copy.deepcopy(node["params"]),
                         "pos": [node["pos"][0] + 220, node["pos"][1]]})
        for node_id in unit:
            node = ctx.node(node_id)
            for slot, source in (node.get("inputs") or {}).items():
                if source is None:
                    continue
                # Inside the unit: rewire to this branch's own copy. Outside it (the chain's
                # external upstream): keep reading the one shared original -- only the unit
                # itself needs its own copy, not whatever feeds it.
                ctx.dispatch({"op": "connect", "id": mapping[node_id], "input": slot,
                             "source": mapping.get(source, source)})
        consumer_node = ctx.node(branch_consumer)
        for slot, source in (consumer_node.get("inputs") or {}).items():
            if source == branch_node:
                ctx.dispatch({"op": "connect", "id": branch_consumer, "input": slot,
                             "source": mapping[branch_node]})
''',
    "shrinkwrap_uv_geometry_around_mesh.json": """{
  "label": "Shrinkwrap UV'd geometry around mesh",
  "when": {"count": 1, "output_types": ["geometry"]},
  "slot": null,
  "enabled": true,
  "requires_types": ["Shrinkwrap3D"],
  "ops": [
    {"op": "create", "id": "$new:wrap", "type": "Shrinkwrap3D"},
    {"op": "connect", "id": "$new:wrap", "input": "target", "source": "$selected[0]"}
  ]
}
""",
}


def install_default_commands(directory: Optional[Path] = None) -> bool:
    """Drops the two built-in examples into `directory` the first time it holds no command file
    yet, so they are there out of the box but are ordinary, editable, deletable files from then
    on (an empty directory still counts as "not set up yet": nothing to protect by skipping it).
    Returns whether anything was installed."""
    directory = directory if directory is not None else user_commands_directory()
    if any(directory.glob(f"*{_MANIFEST_SUFFIX}")):
        return False
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in DEFAULT_COMMANDS.items():
        (directory / name).write_text(content)
    return True
