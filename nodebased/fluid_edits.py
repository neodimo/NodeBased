"""Artist-facing invalidation classes for fluid edits.

The display-only stages consume cached simulation output. Changes to a time-varying
simulation input are reported with their earliest affected frame; static simulation
changes invalidate from the beginning of the solve.
"""
from __future__ import annotations

from .core import SPECS


FLUID_KINDS = frozenset(kind for kind in SPECS if "Fluid" in kind or kind in {"ReadVDB3D", "WriteVDB3D"})
LOOK_KINDS = frozenset({"FluidSurface3D", "FluidFoam3D", "FluidCache3D", "FluidUpres3D",
                        "FluidVolumeRender3D", "ReadVDB3D", "WriteVDB3D"})
TIME_VARYING_KINDS = frozenset({"FluidSource3D", "FluidForce3D", "FluidCollide3D"})


def knob_class(kind: str, knob: str) -> str | None:
    """Return ``look``, ``from-frame`` or ``from-start`` for a fluid-node knob."""
    if kind not in FLUID_KINDS or knob not in SPECS[kind]["params"]:
        return None
    if kind in LOOK_KINDS:
        return "look"
    if kind in TIME_VARYING_KINDS:
        return "from-frame"
    return "from-start"


def edit_scope(document: dict, command: dict) -> tuple[str, int | None]:
    """Classify an edit and return its restart boundary where one is available."""
    if command.get("op") == "batch":
        scopes = [edit_scope(document, item) for item in command.get("commands", ())]
        scopes = [scope for scope in scopes if scope[0] != "look"]
        if not scopes:
            return "look", None
        starts = [frame for category, frame in scopes if category == "from-frame" and frame is not None]
        return ("from-frame", min(starts)) if starts else ("from-start", None)
    node = document.get("nodes", {}).get(command.get("id"), {})
    kind = node.get("type")
    knob = command.get("param")
    category = knob_class(kind, knob) if knob else None
    if category is None:
        return "from-start", None
    if category == "look":
        return "look", None
    if category == "from-frame":
        frame = command.get("frame") if command.get("op") in {"set_key", "delete_key"} else None
        if frame is None and command.get("op") == "set_curve":
            curve = command.get("curve") or ()
            frames = [key.get("frame") for key in curve if isinstance(key, dict)]
            frame = min(frames) if frames else None
        return ("from-frame", int(frame)) if frame is not None else ("from-start", None)
    return "from-start", None


def status_text(document: dict, command: dict) -> str:
    node = document.get("nodes", {}).get(command.get("id"), {})
    if node.get("type") not in FLUID_KINDS:
        return ""
    category, frame = edit_scope(document, command)
    if category == "look":
        return "LOOK UPDATE · simulation cache reused"
    if category == "from-frame" and frame is not None:
        return f"SIM UPDATE · keyed at frame {frame} · checkpoint before {frame}"
    return "SIM UPDATE · re-bake from start"
