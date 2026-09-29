"""The radial menu's command model and its per-context slot tables (plan "Radial menu", DiMo
9/27): `RadialCommand(id, label, when, run)` is the one shape every ring slice takes, whether it
adds a node or runs an action, and `commands_for` picks the eight for the current selection.

Each context in `CONTEXTS` is a fixed, eight-long tuple (`None` where a context simply has fewer
than eight things to offer): a command's slot in its context never moves, even when that
command's own `when` happens to be false for the current selection -- the slot is just empty that
time, per `radialmenu.RadialMenu`'s deliverable. `context_for_selection` picks the context from
DiMo's six named cases: nothing selected; one image node; a keyer; two nodes; 3D geometry, Scene3D
or Render3D; several nodes.
"""
from dataclasses import dataclass
from typing import Callable, List

from .core import bypass_slot, OUTPUT_TYPES
from . import fluidshelf
from .nodecatalog import node_category

SLOT_COUNT = 8

# Backdrop/Input/Output carry no image of their own, so viewing one is meaningless.
_NON_VIEWABLE_TYPES = frozenset({"Backdrop", "Input", "Output"})
_SCENE_CATEGORIES = ("3D", "Particles", "Fluids")


@dataclass(frozen=True)
class RadialCommand:
    """One ring slice. `when(selection)` takes the list of selected node dicts (document-shaped,
    each carrying at least "type") and says whether the slice is offered. `run(window, ids, pos)`
    executes through the window's own command path -- so it is one undo unit, the same as any
    other graph edit -- given the selected node ids and the scene point the menu opened at."""
    id: str
    label: str
    when: Callable[[List[dict]], bool]
    run: Callable[["object", List[str], "object"], None]


def _always(selection):
    return True


def _one_bypassable(selection):
    return len(selection) == 1 and bypass_slot(selection[0]) is not None


def _any_bypassable(selection):
    return any(bypass_slot(node) is not None for node in selection)


def _viewable_alone(selection):
    return len(selection) == 1 and selection[0]["type"] not in _NON_VIEWABLE_TYPES


def _image_output_alone(selection):
    return len(selection) == 1 and OUTPUT_TYPES.get(selection[0]["type"], "image") == "image"


def _has_group(selection):
    return any(node["type"] == "Group" for node in selection)


def _add(kind):
    def run(window, ids, pos):
        window.add_node(kind, position=pos)
    return run


def _view(slot):
    def run(window, ids, pos):
        target = ids[0] if slot == 1 else ids[-1]
        window.command({"op": "viewer_input", "slot": slot, "id": target})
    return run


def _toggle_bypass(window, ids, pos):
    nodes = window.graph_nodes()
    commands = [{"op": "disable", "id": key, "value": not nodes[key]["disabled"]}
                for key in ids if key in nodes and bypass_slot(nodes[key]) is not None]
    if commands:
        window.command({"op": "batch", "commands": commands})


def _group(window, ids, pos):
    window.graph.group_selection()


def _ungroup(window, ids, pos):
    window.graph.ungroup_selection()


def _backdrop(window, ids, pos):
    window.add_node("Backdrop", position=pos)


def _align(window, ids, pos):
    window.graph.align_selection()


def _geometry_selection(selection):
    return bool(selection) and all(OUTPUT_TYPES.get(node.get("type")) in ("geometry", "scene")
                                   for node in selection)


def _make_fluid_tool(tool):
    def run(window, ids, pos):
        document = window.graph_document()
        ops = fluidshelf.build_ops(tool, window.graph_nodes(), ids,
                                   document.get("animation") or {}, (pos.x(), pos.y()))
        if ops:
            window.command({"op": "batch", "commands": ops})
    return run


_EMPTY = (
    RadialCommand("add_read", "Add Read", _always, _add("Read")),
    RadialCommand("add_checker", "Add Checker", _always, _add("Checker")),
    RadialCommand("add_constant", "Add Constant", _always, _add("Constant")),
    RadialCommand("add_text", "Add Text", _always, _add("Text")),
    RadialCommand("add_backdrop", "Add Backdrop", _always, _add("Backdrop")),
    RadialCommand("add_dot", "Add Dot", _always, _add("Dot")),
    RadialCommand("add_scene3d", "Add Scene3D", _always, _add("Scene3D")),
    RadialCommand("add_camera3d", "Add Camera3D", _always, _add("Camera3D")),
)

_ONE_IMAGE = (
    RadialCommand("add_grade", "Add Grade", _always, _add("Grade")),
    RadialCommand("add_merge", "Add Merge", _always, _add("Merge")),
    RadialCommand("add_blur", "Add Blur", _always, _add("Blur")),
    RadialCommand("add_transform", "Add Transform", _always, _add("Transform")),
    RadialCommand("view", "View", _viewable_alone, _view(1)),
    RadialCommand("bypass", "Bypass", _one_bypassable, _toggle_bypass),
    RadialCommand("backdrop", "Backdrop around", _always, _backdrop),
    RadialCommand("group", "Group", _always, _group),
)

_KEYER = (
    RadialCommand("add_premult", "Add Premult", _always, _add("Premult")),
    RadialCommand("add_merge", "Add Merge", _always, _add("Merge")),
    RadialCommand("add_blur", "Add Blur", _always, _add("Blur")),
    RadialCommand("add_erode", "Add Erode", _always, _add("Erode")),
    RadialCommand("view", "View", _viewable_alone, _view(1)),
    RadialCommand("bypass", "Bypass", _one_bypassable, _toggle_bypass),
    RadialCommand("backdrop", "Backdrop around", _always, _backdrop),
    RadialCommand("group", "Group", _always, _group),
)

_TWO_NODES = (
    RadialCommand("add_merge", "Add Merge", _always, _add("Merge")),
    RadialCommand("add_dissolve", "Add Dissolve", _always, _add("Dissolve")),
    RadialCommand("group", "Group", _always, _group),
    RadialCommand("backdrop", "Backdrop around", _always, _backdrop),
    RadialCommand("align", "Align", _always, _align),
    RadialCommand("view_a", "View as A", _always, _view(1)),
    RadialCommand("view_b", "View as B", _always, _view(2)),
    RadialCommand("bypass", "Bypass", _any_bypassable, _toggle_bypass),
)

_THREE_D = (
    RadialCommand("add_scene3d", "Add Scene3D", _always, _add("Scene3D")),
    RadialCommand("add_axis3d", "Add Axis3D", _always, _add("Axis3D")),
    RadialCommand("add_render3d", "Add Render3D", _always, _add("Render3D")),
    RadialCommand("add_transformgeo", "Add TransformGeo3D", _always, _add("TransformGeo3D")),
    RadialCommand("view", "View", _image_output_alone, _view(1)),
    RadialCommand("bypass", "Bypass", _one_bypassable, _toggle_bypass),
    RadialCommand("backdrop", "Backdrop around", _always, _backdrop),
    RadialCommand("group", "Group", _always, _group),
)

# Geometry selections get direct, one-gesture fluid shelf actions in the ring. Other 3D
# selections keep the general scene/camera/render commands above.
_FLUID_GEOMETRY = (
    RadialCommand("fluid_make_smoke", "Make smoke from selected geometry", _geometry_selection,
                  _make_fluid_tool("make_smoke")),
    RadialCommand("fluid_make_liquid", "Make liquid from selected geometry", _geometry_selection,
                  _make_fluid_tool("make_liquid")),
    RadialCommand("fluid_make_collider", "Make collider", _geometry_selection,
                  _make_fluid_tool("make_collider")),
    RadialCommand("fluid_make_fire", "Make fire from selected", _geometry_selection,
                  _make_fluid_tool("make_fire")),
    _THREE_D[4], _THREE_D[5], _THREE_D[6], _THREE_D[7],
)

_SEVERAL = (
    RadialCommand("group", "Group", _always, _group),
    RadialCommand("ungroup", "Ungroup", _has_group, _ungroup),
    RadialCommand("backdrop", "Backdrop around", _always, _backdrop),
    RadialCommand("align", "Align", _always, _align),
    RadialCommand("bypass", "Bypass all", _any_bypassable, _toggle_bypass),
    None,
    None,
    None,
)

CONTEXTS = {
    "empty": _EMPTY,
    "one_image": _ONE_IMAGE,
    "keyer": _KEYER,
    "two_nodes": _TWO_NODES,
    "3d": _THREE_D,
    "several": _SEVERAL,
}


def context_for_selection(nodes, selected_ids):
    """Which of `CONTEXTS`' rows fits the current selection."""
    selected = [nodes[key] for key in selected_ids if key in nodes]
    if not selected:
        return "empty"
    categories = [node_category(node["type"]) for node in selected]
    if any(category in _SCENE_CATEGORIES for category in categories):
        return "3d"
    if len(selected) == 1:
        return "keyer" if categories[0] == "Keyer" else "one_image"
    if len(selected) == 2:
        return "two_nodes"
    return "several"


def builtin_ids_for_context(context_kind):
    """Every command id the rule table alone offers for `context_kind`, regardless of whether its
    `when` currently hides it. Local-usage learning (`app.Preferences.record_radial_usage`) uses
    this to avoid "promoting" a command that already has a slot of its own whenever it applies --
    only a command that has to compete for a shared, unreserved slot needs one won for it."""
    tables = [CONTEXTS.get(context_kind, ())]
    if context_kind == "3d":
        tables.append(_FLUID_GEOMETRY)
    return {command.id for table in tables for command in table if command is not None}


def _lookup_by_id(command_id, context_kind, user_commands):
    """The `RadialCommand` (built-in or wrapped user command) `command_id` names, or `None` if it
    no longer exists -- a deleted user command file, or an id from a different context's table."""
    for command in CONTEXTS.get(context_kind, ()):
        if command is not None and command.id == command_id:
            return command
    if context_kind == "3d":
        for command in _FLUID_GEOMETRY:
            if command is not None and command.id == command_id:
                return command
    from .radialcommands import wrap_command
    for command in user_commands:
        if f"user:{command.id}" == command_id:
            return wrap_command(command)
    return None


def _apply_slot_overrides(table, mapping, context_kind, selection, user_commands):
    """Force whichever command each `{slot: command_id}` entry in `mapping` names into that slot,
    leaving every other slot exactly as `table` already had it. A command that no longer exists,
    or whose own `when` says no for this selection, is skipped rather than leaving a hole or
    raising -- the slot simply keeps whatever `table` already resolved there."""
    if not mapping:
        return table
    table = list(table)
    for slot_key, command_id in mapping.items():
        try:
            slot = int(slot_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= slot < len(table)):
            continue
        candidate = _lookup_by_id(command_id, context_kind, user_commands)
        if candidate is not None and candidate.when(selection):
            table[slot] = candidate
    return table


def commands_for(nodes, selected_ids, preferences=None):
    """The eight ring slices for the current selection: `commands_for(...)[i]` is the
    `RadialCommand` filed at slot i, or `None` when that context has nothing there or the
    command's own `when` says not for this selection. User-defined commands (`radialcommands.py`)
    then fill whatever slots are still empty.

    `preferences` (an `app.Preferences`, or any object with its `radial_bucket`/`radial_learned`/
    `radial_pins` methods) layers local-usage learning and right-click pins on top: a learned or
    pinned slot is forced to hold its remembered command before the ordinary free-slot fill runs,
    so that command is never also placed a second time by the ordinary fill. Pins are applied
    after learning, so a pin always wins a slot a promotion would otherwise also want."""
    selected = [nodes[key] for key in selected_ids if key in nodes]
    context = context_for_selection(nodes, selected_ids)
    table = _FLUID_GEOMETRY if context == "3d" and _geometry_selection(selected) else CONTEXTS[context]
    resolved = [command if command is not None and command.when(selected) else None for command in table]

    from .radialcommands import overlay_user_commands, load_all   # deferred: breaks the import cycle
    user_commands, _errors = load_all()

    if preferences is not None:
        node_type = selected[0]["type"] if len(selected) == 1 else None
        bucket = preferences.radial_bucket(context, node_type)
        resolved = _apply_slot_overrides(resolved, preferences.radial_learned().get(bucket, {}),
                                         context, selected, user_commands)
        resolved = _apply_slot_overrides(resolved, preferences.radial_pins().get(bucket, {}),
                                         context, selected, user_commands)
        placed_ids = {command.id for command in resolved if command is not None}
        user_commands = [command for command in user_commands
                        if f"user:{command.id}" not in placed_ids]

    return overlay_user_commands(resolved, nodes, selected_ids, commands=user_commands)
