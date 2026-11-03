"""The project settings Nuke keeps on the Root node that a NodeBased document can hold: `settings.project`.

The group is optional. A document that never set a field has no `project` key, and one whose fields are all at
their defaults is stored without it, so old and untouched documents keep their exact serialized form (the
`formats` registry follows the same rule). Each field acts:

- `name`, `label`: the project's name and comment, shown in the window title and the Project Settings dialog.
- `project_directory`: the folder a Read's relative path is resolved against when the path is set (Nuke's
  [value root.project_directory]); an absolute path is left alone, and with no directory a relative path
  resolves against the working directory as before.
- `lock_connections`: no node's input can be rewired while it is on (Nuke's lock all connections).
- `lock_range`: loading a sequence leaves the project frame range where it is (Nuke's lock range).
"""
from pathlib import Path

DEFAULTS = {"name": "", "label": "", "project_directory": "", "lock_connections": False, "lock_range": False}
TEXT_FIELDS = ("name", "label", "project_directory")
FLAG_FIELDS = ("lock_connections", "lock_range")


def project(document):
    """The project fields with the defaults filled in for any the document does not hold."""
    return {**DEFAULTS, **(document.get("settings", {}).get("project") or {})}


def validate(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError(f"settings.project fields must be among {sorted(DEFAULTS)}")
    for name, value in values.items():
        if name in TEXT_FIELDS and not isinstance(value, str):
            raise ValueError(f"settings.project.{name} must be a string")
        if name in FLAG_FIELDS and type(value) is not bool:
            raise ValueError(f"settings.project.{name} must be true or false")


def merge(settings, changes):
    """Apply `changes` to settings["project"]; the group is dropped again when every field is back at its default."""
    values = {**DEFAULTS, **settings.get("project", {}), **changes}
    validate(values)
    if values == DEFAULTS:
        settings.pop("project", None)
    else:
        settings["project"] = values
    return values


def resolve_media_path(document, value):
    """A Read's path as stored: absolute, a relative one taken from the project directory when one is set."""
    path = Path(value).expanduser()
    directory = project(document)["project_directory"]
    if directory and not path.is_absolute():
        path = Path(directory).expanduser() / path
    return str(path.resolve())
