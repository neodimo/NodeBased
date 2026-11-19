"""The Project Settings panel: Nuke's Root panel over the document's own settings.

In Nuke, S in the Node Graph (and Edit > Project Settings) opens the Root node's panel in the Properties bin
(17.0 Reference Guide, Other > Root). NodeBased keeps those settings in the document (`time`, `settings.project`,
`settings.viewer.proxy`, `settings.color`), and `docs/parity/nodebased_to_nuke.json` pairs each Root knob with one
document field. This module is the layout of that panel and nothing else: the guide's tabs, order, labels and widget kinds,
the field each control edits, and the one operation each edit goes through, so one edit is one undo step. The window code
(`nodebased/app.py`, `build_root_panel`) only draws what `TABS` lists, and the fidelity audit reads the same table for
Root's layout score (`knob_groups`).

A row's `fields` are dotted document paths; `names` are the guide's scripting names, in the order the guide lists them.
A knob of the guide that NodeBased has no field for (views, live groups, the LUT list, font paths) has no control here.
"""
from dataclasses import dataclass

from . import projectsettings as _project

ROOT_KEY = "@root"            # the bin's key for this panel, beside the node ids of the node panels
TITLE = "Project Settings"
COLOR_VIEWS = ("sRGB", "ACES 2.0", "Linear")
PROXY_SCALE_FIELD = "settings.viewer.proxy_scale"     # the viewer tier read as a fraction of full size (a view of `settings.viewer.proxy`)
PROXY_SCALES = (0.5, 0.25, 0.125)     # tiers 2, 4 and 8 as a fraction of full size (Nuke's proxy scale)


@dataclass(frozen=True)
class Row:
    tab: str
    label: str
    kind: str                      # string, bool, enum, int, float, or row (several controls side by side, `members` says which)
    fields: tuple                  # document paths this row edits
    names: tuple                   # the guide's scripting names of the knobs in it
    members: tuple = ()            # a row's widget kind per field
    choices: tuple = ()            # an enum's fixed choices (empty: see `choices_for`)
    section: str = ""              # a heading shown above the first row of a run that shares it
    help: str = ""


PROJECT_PREFIX = _P = "settings.project."
ROWS = (
    Row("Root", "name", "string", (_P + "name",), ("name",), help="The project's name, shown in the window title."),
    Row("Root", "project directory", "string", (_P + "project_directory",), ("project_directory",),
        help="A Read's relative path is resolved against this folder."),
    Row("Root", "Script Directory", "string", (_P + "script_directory",), ("Script_directory",),
        help="The folder the document lives in; used when the project directory is empty."),
    Row("Root", "comment", "string", (_P + "label",), ("label",), help="A note on the project."),
    Row("Root", "lock all connections", "bool", (_P + "lock_connections",), ("lock_connections",),
        help="On, no node's input can be rewired."),
    Row("Root", "frame range", "row", ("time.first", "time.last"), ("first_frame", "last_frame"), members=("int", "int"),
        help="The project's first and last frame."),
    Row("Root", "lock range", "bool", (_P + "lock_range",), ("lock_range",),
        help="On, loading a sequence leaves the frame range where it is."),
    Row("Root", "fps", "float", ("time.fps",), ("fps",), help="The project's frame rate."),
    Row("Root", "full size format", "enum", (_P + "format",), ("format",), help="The project format."),
    Row("Root", "proxy mode", "row", ("settings.viewer.proxy", _P + "proxy_type"), ("proxy", "proxy_type"),
        members=("bool", "enum"), choices=("scale", "format"),
        help="Proxy on, and whether it is a scale or a format."),
    Row("Root", "proxy scale", "float", (PROXY_SCALE_FIELD,), ("proxy_scale",),
        help="The fraction of full size the proxy renders at: 1/2, 1/4 or 1/8."),
    Row("Root", "proxy format", "enum", (_P + "proxy_format",), ("proxy_format",), help="The proxy format."),
    Row("Root", "read proxy files", "enum", (_P + "read_proxy_files",), ("proxySetting",),
        choices=_project.CHOICE_FIELDS["read_proxy_files"], help="When a Read with a proxy file reads it."),
    Row("Color", "OCIO config", "string", ("settings.color.config",), ("OCIO_config",),
        help="The OCIO config the colour nodes use when their own is empty."),
    Row("Color", "monitor", "enum", ("settings.color.view",), ("monitorLUT",), choices=COLOR_VIEWS,
        section="default LUT settings", help="The viewer's display transform."),
    Row("Color", "8-bit files", "enum", (_P + "int8_lut",), ("int8LUT",), choices=_project.READ_SPACES,
        help="The colour space a Read takes an 8-bit file in when its own colorspace is Auto."),
    Row("Color", "16-bit files", "enum", (_P + "int16_lut",), ("int16LUT",), choices=_project.READ_SPACES,
        help="The colour space a Read takes a 16-bit file in when its own colorspace is Auto."),
    Row("Color", "log files", "enum", (_P + "log_lut",), ("logLUT",), choices=_project.READ_SPACES,
        help="The colour space a Read takes a Cineon or DPX file in when its own colorspace is Auto."),
    Row("Color", "float files", "enum", (_P + "float_lut",), ("floatLUT",), choices=_project.READ_SPACES,
        help="The colour space a Read takes an EXR in when its own colorspace is Auto."),
)
TABS = tuple(dict.fromkeys(row.tab for row in ROWS))


def get(document, path):
    """The value of one document field, with the defaults filled in for a field the document does not hold."""
    if path == "settings.viewer.proxy":
        from .core import viewer_proxy
        return viewer_proxy(document)
    if path == PROXY_SCALE_FIELD:
        from .core import viewer_proxy
        return proxy_scale(viewer_proxy(document))
    if path.startswith(_P):
        return _project.project(document)[path[len(_P):]]
    value = document
    for part in path.split("."):
        value = value[part]
    return value


def choices_for(document, row, index=0):
    """The choices of an enum control: the document's formats for a format knob, otherwise the row's fixed ones."""
    if row.fields[index] in (_P + "format", _P + "proxy_format"):
        from .core import document_formats
        return tuple(document_formats(document))
    return row.choices


def proxy_scale(tier):
    """The proxy scale Nuke shows for one of our proxy tiers (1 is full size, which has no scale to show)."""
    return 1.0 / tier if tier > 1 else PROXY_SCALES[0]


def tier_for_scale(scale):
    """The tier nearest a typed proxy scale, in log scale."""
    import math
    from .core import VIEWER_PROXY_TIERS
    scale = max(float(scale), 1e-6)
    return min((t for t in VIEWER_PROXY_TIERS if t > 1), key=lambda t: abs(math.log2(1.0 / t) - math.log2(scale)))


def edit(document, path, value):
    """The one operation that sets document field `path` to `value`, or None when it already holds it.

    Every control goes through this, so an edit is one undo step and the agent's `settings`, `time` and `viewer_proxy`
    operations stay the only writers. (`proxy` and `proxy scale` share `settings.viewer.proxy`: the tier.)"""
    if path in ("settings.viewer.proxy", PROXY_SCALE_FIELD):
        raise ValueError("a proxy edit names its control: use `edit_proxy`")
    if path in ("time.first", "time.last", "time.fps"):
        name = path.split(".")[1]
        if get(document, path) == value:
            return None
        command = {"op": "time", name: value}
        if name == "first" and value > document["time"]["last"]:
            command["last"] = value
        if name == "last" and value < document["time"]["first"]:
            command["first"] = value
        return command
    if path.startswith(_P):
        if get(document, path) == value:
            return None
        return {"op": "settings", "settings": {"project": {path[len(_P):]: value}}}
    if path.startswith("settings.color."):
        if get(document, path) == value:
            return None
        return {"op": "settings", "settings": {"color": {path.split(".")[2]: value}}}
    if path in _EXTRA_FIELDS:
        return _EXTRA_FIELDS[path](document, value)
    raise KeyError(path)


_EXTRA_FIELDS = {}


def edit_proxy(document, *, on=None, scale=None):
    """The `viewer_proxy` operation for the proxy mode check box (on) or the proxy scale (a fraction), or None for no change."""
    from .core import viewer_proxy
    now = viewer_proxy(document)
    if scale is not None:
        tier = tier_for_scale(scale)
    elif on:
        tier = now if now > 1 else 2
    else:
        tier = 1
    return None if tier == now else {"op": "viewer_proxy", "tier": tier}


def knob_groups():
    """The panel as `KnobGroup`s over the document paths, in order: what the fidelity audit reads as Root's layout."""
    from .knobs import KnobGroup
    groups = []
    for row in ROWS:
        if row.kind == "row":
            groups.append(KnobGroup("row", row.fields, label=row.label, members=row.members, section=row.section or None))
        else:
            groups.append(KnobGroup(row.kind, row.fields, label=row.label, section=row.section or None))
    return tuple(groups)
