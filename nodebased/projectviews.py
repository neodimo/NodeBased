"""The project's views: Nuke's Root Views tab (17.0 Reference Guide, Other > Root, Views Tab) over `settings.views`.

A project starts with one view, `main`. The Views tab lists the views and the colour of each, adds, removes and moves them,
names the hero view (the one loaded when the script opens), sets up `left` and `right` for stereo, chooses whether views show as
buttons above the viewer, and whether the views' colours show (`Use colors in UI?`).

NodeBased records the views and acts on none of them: rendering is mono, no node reads a view, and the viewer shows one picture.
They save and load with the document so a project that names its views keeps them. The group is optional: a document that never
changed them has no `views` key, and one whose fields are all at their defaults is stored without it, so untouched documents
keep their exact serialized form (`settings.project` follows the same rule).

Fields: `names` (distinct, at least one), `colours` (a `#rrggbb` for each name, same order), `hero` (one of the names),
`buttons` (Nuke's `views_button`, on by default), `use_colours` (Nuke's `views_colours`, off by default). The default colour of
`main` and the left and right colours are ours: the guide gives none.
"""
import re

DEFAULTS = {"names": ["main"], "colours": ["#ffffff"], "hero": "main", "buttons": True, "use_colours": False}
FLAG_FIELDS = ("buttons", "use_colours")
NAME_LIMIT = 64
COUNT_LIMIT = 16
STEREO = (("left", "#ff0000"), ("right", "#00ff00"))
PALETTE = ("#00a0ff", "#ffa000", "#c040ff", "#00d0a0", "#ff4090", "#a0c000")      # colours handed to views added after the first two
TOOLTIP = ("Views are recorded in the project and saved with it. NodeBased renders one picture (mono): no node and no viewer "
           "reads a view yet.")


def views(document):
    """The views group with the defaults filled in for any field the document does not hold."""
    stored = (document.get("settings", {}) or {}).get("views") or {}
    return {**{k: list(v) if isinstance(v, list) else v for k, v in DEFAULTS.items()}, **stored}


def validate(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError(f"settings.views fields must be among {sorted(DEFAULTS)}")
    names, colours = values.get("names", DEFAULTS["names"]), values.get("colours", DEFAULTS["colours"])
    if (not isinstance(names, list) or not names or len(names) > COUNT_LIMIT
            or any(not isinstance(n, str) or not n.strip() or len(n) > NAME_LIMIT or n != n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"settings.views.names must be 1 to {COUNT_LIMIT} distinct names of 1 to {NAME_LIMIT} characters")
    if (not isinstance(colours, list) or len(colours) != len(names)
            or any(not isinstance(c, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colours)):
        raise ValueError("settings.views.colours must hold one #rrggbb colour for each view")
    if values.get("hero", "main") not in names:
        raise ValueError("settings.views.hero must be one of the views")
    for name in FLAG_FIELDS:
        if name in values and type(values[name]) is not bool:
            raise ValueError(f"settings.views.{name} must be true or false")


def merge(settings, changes):
    """Apply `changes` to settings["views"]; the group is dropped again when every field is back at its default."""
    values = {**{k: list(v) if isinstance(v, list) else v for k, v in DEFAULTS.items()}, **settings.get("views", {}), **changes}
    validate(values)
    if values == DEFAULTS:
        settings.pop("views", None)
    else:
        settings["views"] = values
    return values


# The edits behind the tab's buttons: each returns the fields to merge, or None when there is nothing to do.
def add(current, name):
    name = str(name).strip()
    if not name or name in current["names"] or len(current["names"]) >= COUNT_LIMIT:
        return None
    colour = PALETTE[len(current["names"]) % len(PALETTE)]
    return {"names": current["names"] + [name], "colours": current["colours"] + [colour]}


def remove(current, name):
    """Remove a view; the only view cannot be removed, and removing the hero hands the role to the first view left."""
    if name not in current["names"] or len(current["names"]) == 1:
        return None
    keep = [i for i, n in enumerate(current["names"]) if n != name]
    names = [current["names"][i] for i in keep]
    changes = {"names": names, "colours": [current["colours"][i] for i in keep]}
    if current["hero"] == name:
        changes["hero"] = names[0]
    return changes


def move(current, name, step):
    """Move a view up (-1) or down (+1) the list; the view at the top is the first one above the viewer."""
    names = list(current["names"])
    if name not in names:
        return None
    index, target = names.index(name), names.index(name) + step
    if not 0 <= target < len(names):
        return None
    order = list(range(len(names)))
    order[index], order[target] = order[target], order[index]
    return {"names": [names[i] for i in order], "colours": [current["colours"][i] for i in order]}


def stereo(current):
    """`Set up views for stereo`: the views become left and right, and the hero is left."""
    names = [n for n, _ in STEREO]
    changes = {"names": names, "colours": [c for _, c in STEREO], "hero": names[0]}
    return None if all(current[k] == v for k, v in changes.items()) else changes
