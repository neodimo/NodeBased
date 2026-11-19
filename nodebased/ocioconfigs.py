"""The OCIO config menu of Project Settings: Nuke's Root `OCIO config` knob over `settings.color`.

Nuke 17.0's Root panel (Reference Guide, Other > Root, Color tab) has an `OCIO config` menu (scripting name `OCIO_config`,
default `nuke-default`) and, below it, a `custom OCIO config` path field (`customOCIOConfigPath`) that applies when the menu
says `custom`. The Reference Guide does not list the menu's entries and the User Guide (Configuring Nuke > OCIO Color
Management) names only `nuke-default`, the ACES 1.3 configs and `custom`. The full list of what 17.0 ships comes from a
measurement of Nuke 17.0 published by Andrea Geremia, "ACES in Nuke: the FAQ" (andreageremia.it/tutorial_aces_faq.html,
"Which config do I choose in Nuke 17?"). That is a third-party convention, not Foundry text. The order here is that page's.

NodeBased does not ship Foundry's config files, so those entries are listed (the menu reads like Nuke's) but cannot be loaded:
choosing one shows a warning and the colour-space menus use the built-in list, the ACES configs OpenColorIO itself carries. Those
built-in configs are listed after Nuke's, because they are the ones that load; the project default stays the ACES 2.0 CG config
(its working space is ACEScg), where Nuke's default `nuke-default` is a linear sRGB set-up without ACES.

Storage: `settings.color.config` holds the menu entry (a Nuke name, an `ocio://` built-in, or `custom`) and the optional
`settings.color.custom_config` the path (schema 37; an older document that held a path or an unknown URI keeps it as a custom path).
"""
from functools import lru_cache
from pathlib import Path

CUSTOM = "custom"
DEFAULT = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"      # the built-in the project starts on (core.DEFAULT_SETTINGS)
NUKE_DEFAULT = "nuke-default"                               # the Reference Guide's default for the knob
# What Nuke 17.0 ships, in the order of the source above (see the module text). NodeBased cannot load any of these.
NUKE_SHIPPED = (
    "fn-nuke_studio-config-v3.0.0_aces-v2.0_ocio-v2.4",
    "fn-nuke_cg-config-v3.0.0_aces-v2.0_ocio-v2.4",
    "fn-nuke_studio-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "fn-nuke_cg-config-v2.2.0_aces-v1.3_ocio-v2.4",
    "aces_1.2",
    NUKE_DEFAULT,
)
SOURCE = ("Nuke 17.0 Reference Guide, Other > Root (OCIO_config, customOCIOConfigPath); Nuke 17.0 User Guide, OCIO Color Management; "
          "list of shipped configs: Andrea Geremia, ACES in Nuke: the FAQ (a measurement of Nuke 17.0)")


@lru_cache(maxsize=1)
def builtin_names():
    """The built-in configs OpenColorIO carries, as `ocio://<name>` ids: the project default first, then the registry's rest, newest first."""
    try:
        import PyOpenColorIO as ocio
        names = [row[0] for row in ocio.BuiltinConfigRegistry().getBuiltinConfigs()]
    except Exception:
        names = [DEFAULT[len("ocio://"):]]
    uris = ["ocio://" + name for name in reversed(names)]
    return tuple(sorted(uris, key=lambda uri: uri != DEFAULT))


def entries():
    """Every menu entry as (id, label), in the menu's order: Nuke's shipped configs, the built-ins that load, then custom."""
    rows = [(name, name) for name in NUKE_SHIPPED]
    rows += [(uri, uri[len("ocio://"):] + " (built in)") for uri in builtin_names()]
    rows.append((CUSTOM, CUSTOM))
    return tuple(rows)


def entry_ids():
    return tuple(row[0] for row in entries())


def selection(color):
    """(menu entry, custom path) the panel shows for the `settings.color` group: a stored value that is no entry (a path or an
    unknown URI from before the menu) reads as `custom` with that value as its path."""
    config = str(color.get("config", "") or "")
    custom = str(color.get("custom_config", "") or "")
    if config in entry_ids():
        return config, custom
    return CUSTOM, custom or config


def effective(color):
    """The config string the colour nodes load: a built-in URI, or a file path for `custom`; a Nuke name stays a name (it will not load)."""
    entry, path = selection(color)
    return path if entry == CUSTOM else entry


@lru_cache(maxsize=32)
def _file_problem(path, stamp):
    import PyOpenColorIO as ocio
    try:
        ocio.Config.CreateFromFile(path)
    except Exception as error:
        return f"{Path(path).name}: {str(error).splitlines()[0][:120]}"
    return None


def problem(color):
    """None when NodeBased can load the selected config, otherwise the sentence the panel's warning shows."""
    entry, path = selection(color)
    if entry in NUKE_SHIPPED:
        return (f"NodeBased cannot load the Nuke config '{entry}': it is Foundry's file and is not shipped here. "
                "Colour-space menus use the built-in list.")
    if entry == CUSTOM:
        if not path.strip():
            return "Enter the path of a custom OCIO config. Until then colour-space menus use the built-in list."
        file = Path(path).expanduser()
        if path.startswith("ocio://"):
            return None if path in builtin_names() else f"'{path}' is not a built-in OCIO config. Colour-space menus use the built-in list."
        if not file.is_file():
            return f"The custom OCIO config '{path}' does not exist. Colour-space menus use the built-in list."
        detail = _file_problem(str(file), file.stat().st_mtime_ns)
        return None if detail is None else f"Could not load the custom OCIO config ({detail}). Colour-space menus use the built-in list."
    return None


def choices_config(color):
    """The config the colour-space menus read: the selected one when it loads, otherwise the built-in default."""
    return DEFAULT if problem(color) else effective(color)


def migrate(color):
    """The schema 37 form of a `settings.color` group: a stored path or unknown URI becomes `custom` plus `custom_config`."""
    config = str(color.get("config", "") or "")
    if config in entry_ids():
        return
    color["config"] = CUSTOM
    color["custom_config"] = config
