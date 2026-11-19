"""The Project Settings panel: Nuke's Root panel over the document's own settings.

In Nuke, S in the Node Graph (and Edit > Project Settings) opens the Root node's panel in the Properties bin
(17.0 Reference Guide, Other > Root). NodeBased keeps those settings in the document (`time`, `settings.project`,
`settings.viewer.proxy`, `settings.color`), and `docs/parity/nodebased_to_nuke.json` pairs each Root knob with one
document field. This module is the layout of that panel and nothing else: the guide's tabs, order, labels and widget kinds,
the field each control edits, and the one operation each edit goes through, so one edit is one undo step. The window code
(`nodebased/app.py`, `build_root_panel`) only draws what `TABS` lists, and the fidelity audit reads the same table for
Root's layout score (`knob_groups`).

A row's `fields` are dotted document paths; `names` are the guide's scripting names, in the order the guide lists them.
A knob of the guide that NodeBased has no field for (live groups, the LUT list, font paths, callbacks) has no control here.
The Views tab edits `settings.views` (`nodebased/projectviews.py`): recorded with the document, rendering stays mono.
"""
from dataclasses import dataclass

from . import ocioconfigs as _ocio
from . import projectsettings as _project
from . import projectviews as _views

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
    shown_when: tuple = ()         # (field, value): the row shows only while that field holds the value (Nuke's custom config path)


COLOR_CONFIG = "settings.color.config"
COLOR_CUSTOM = "settings.color.custom_config"
VIEWS_PREFIX = _V = "settings.views."
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
    Row("Color", "OCIO config", "enum", (COLOR_CONFIG,), ("OCIO_config",), choices=_ocio.entry_ids(),
        help="The OCIO config the colour nodes use when their own is empty. Nuke's shipped configs are listed as Nuke lists them; "
             "NodeBased cannot load those, so choosing one shows a warning and the colour-space menus use the built-in list."),
    Row("Color", "custom OCIO config", "string", (COLOR_CUSTOM,), ("customOCIOConfigPath",),
        shown_when=(COLOR_CONFIG, _ocio.CUSTOM), help="The path of the .ocio config file, used when OCIO config is custom."),
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
    Row("Views", "views", "views", (_V + "names",), ("views#5",), help=_views.TOOLTIP),
    Row("Views", "hero", "enum", (_V + "hero",), ("hero_view",),
        help="The view loaded by default when the project is loaded. " + _views.TOOLTIP),
    Row("Views", "Set up views for stereo", "button", (), ("setlr",),
        help="Replace the views with left and right. " + _views.TOOLTIP),
    Row("Views", "View selection uses buttons?", "bool", (_V + "buttons",), ("views_button",),
        help="On, the views show as buttons above the viewer; off, in a menu. " + _views.TOOLTIP),
    Row("Views", "Use colors in UI?", "bool", (_V + "use_colours",), ("views_colours",),
        help="On, each view's colour shows in the views list. " + _views.TOOLTIP),
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
    if path.startswith(_V):
        return _views.views(document)[path[len(_V):]]
    color = document.get("settings", {}).get("color", {})
    if path == COLOR_CONFIG:
        return _ocio.selection(color)[0]
    if path == COLOR_CUSTOM:
        return _ocio.selection(color)[1]
    value = document
    for part in path.split("."):
        value = value[part]
    return value


def choices_for(document, row, index=0):
    """The choices of an enum control: the document's formats for a format knob, otherwise the row's fixed ones."""
    if row.fields[index] == _V + "hero":
        return tuple(_views.views(document)["names"])
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
    if path.startswith(_V):
        if path == _V + "names":
            raise ValueError("the views list is edited by name: use `edit_views`")
        if get(document, path) == value:
            return None
        return {"op": "settings", "settings": {"views": {path[len(_V):]: value}}}
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


def edit_views(document, action, name=None, step=0):
    """The `settings` operation for one of the Views tab's buttons (`add`, `remove`, `move` by `step`, `stereo`), or None for no change."""
    current = _views.views(document)
    changes = {"add": lambda: _views.add(current, name), "remove": lambda: _views.remove(current, name),
               "move": lambda: _views.move(current, name, step), "stereo": lambda: _views.stereo(current)}[action]()
    return None if changes is None else {"op": "settings", "settings": {"views": changes}}


def config_labels():
    """(entry id, label) for the OCIO config menu, in the menu's order."""
    return _ocio.entries()


def tab_tooltip(row):
    """The tooltip of a row's control: the row's help, which on the Views tab says views are recorded and rendering is mono."""
    return row.help


def visible(document, row):
    """Whether a row shows: a row with `shown_when` shows only while its field holds the value (the custom config path)."""
    return not row.shown_when or get(document, row.shown_when[0]) == row.shown_when[1]


def knob_groups():
    """The panel as `KnobGroup`s over the document paths, in order: what the fidelity audit reads as Root's layout."""
    from .knobs import KnobGroup
    groups = []
    for row in ROWS:
        if not row.fields:
            continue            # a button with no field of its own (Set up views for stereo) edits the views list
        if row.kind == "row":
            groups.append(KnobGroup("row", row.fields, label=row.label, members=row.members, section=row.section or None))
        else:
            kind = "string" if row.kind == "views" else row.kind       # the guide types the views list as a string
            groups.append(KnobGroup(kind, row.fields, label=row.label, section=row.section or None))
    return tuple(groups)
