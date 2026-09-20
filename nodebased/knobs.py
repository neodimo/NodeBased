"""Presentation metadata for node properties (kept separate from the document schema)."""
from __future__ import annotations

from dataclasses import dataclass

from .core import CHOICES, LIMITS, SPECS


KNOB_KINDS = ("float_slider", "int", "bool", "enum", "xy", "xyz", "float", "color", "file_read",
              "file_write", "string", "multiline")


def _default_label(params):
    known_stems = {
        ("translate_x", "translate_y"): "Translate",
        ("center_x", "center_y"): "Center",
        ("red", "green", "blue", "alpha"): "Color",
    }
    if params in known_stems:
        return known_stems[params]
    return params[0].replace("_", " ").title()


@dataclass(frozen=True)
class KnobGroup:
    kind: str
    params: tuple[str, ...]
    label: str | None = None
    soft_range: tuple[float, float] | None = None

    def __post_init__(self):
        if self.label is None:
            object.__setattr__(self, "label", _default_label(self.params))


def _groups(*groups):
    return list(groups)


KNOB_LAYOUT = {
    "ReadSplat3D": _groups(
        KnobGroup("string", ("splat_path",), label="Splat file"),
        KnobGroup("enum", ("splat_orientation",)), KnobGroup("enum", ("splat_colorspace",)),
        KnobGroup("int", ("splat_sh_degree",)),
        KnobGroup("float_slider", ("splat_relight",), label="Relight", soft_range=(0, 1)),
        KnobGroup("float_slider", ("splat_opacity",), soft_range=(0, 2)),
        KnobGroup("float_slider", ("splat_scale",), soft_range=(0.01, 4)),
        KnobGroup("xyz", ("tx", "ty", "tz"), label="Translate"),
        KnobGroup("xyz", ("rx", "ry", "rz"), label="Rotate"),
        KnobGroup("xyz", ("sx", "sy", "sz"), label="Scale"),
        KnobGroup("float", ("uscale",), label="Uniform scale"),
        KnobGroup("enum", ("rot_order",), label="Rotation order"),
        KnobGroup("xyz", ("pivot_x", "pivot_y", "pivot_z"), label="Pivot")),
    "Read": _groups(
        KnobGroup("file_read", ("path",)), KnobGroup("enum", ("colorspace",)),
        KnobGroup("enum", ("alpha_mode",)), KnobGroup("string", ("layer",)),
        KnobGroup("int", ("subimage",)), KnobGroup("int", ("frame_offset",)),
        KnobGroup("enum", ("missing",))),
    "Constant": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("color", ("red", "green", "blue", "alpha"))),
    "Checker": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("int", ("size",))),
    "Grade": _groups(
        KnobGroup("float_slider", ("exposure",), soft_range=(-10, 10)),
        KnobGroup("float_slider", ("multiply",), soft_range=(-4, 4)),
        KnobGroup("float_slider", ("offset",), soft_range=(-10, 10)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "ColorCorrect": _groups(
        KnobGroup("float_slider", ("lift",), soft_range=(-2, 2)),
        KnobGroup("float_slider", ("gamma",), soft_range=(0.1, 4)),
        KnobGroup("float_slider", ("gain",), soft_range=LIMITS["gain"]),
        KnobGroup("float_slider", ("saturation",), soft_range=LIMITS["saturation"]),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Blur": _groups(
        KnobGroup("float_slider", ("radius",), soft_range=(0, 100)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Transform": _groups(
        KnobGroup("xy", ("translate_x", "translate_y")),
        KnobGroup("float_slider", ("rotate",), soft_range=(-180, 180)),
        KnobGroup("float_slider", ("scale",), soft_range=(0.1, 4.0)),
        KnobGroup("xy", ("center_x", "center_y")), KnobGroup("enum", ("filter",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Crop": _groups(
        KnobGroup("int", ("x",)), KnobGroup("int", ("y",)),
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Shuffle": _groups(
        KnobGroup("enum", ("red_from",)), KnobGroup("enum", ("green_from",)),
        KnobGroup("enum", ("blue_from",)), KnobGroup("enum", ("alpha_from",))),
    "ChannelShuffle": _groups(
        KnobGroup("enum", ("out_red",)), KnobGroup("enum", ("out_green",)),
        KnobGroup("enum", ("out_blue",)), KnobGroup("enum", ("out_alpha",))),
    "Roto": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("bool", ("invert",))),
    "Tracker": _groups(
        KnobGroup("int", ("reference_frame",)), KnobGroup("enum", ("mode",)),
        KnobGroup("bool", ("apply_translate",)), KnobGroup("bool", ("apply_rotate",)),
        KnobGroup("bool", ("apply_scale",)), KnobGroup("enum", ("filter",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Merge": _groups(
        KnobGroup("enum", ("operation",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Premult": [], "Unpremult": [], "Dot": [],
    "Switch": _groups(KnobGroup("int", ("which",))),
    "Viewer": [],
    "Write": _groups(
        KnobGroup("file_write", ("path",)), KnobGroup("enum", ("file_type",)),
        KnobGroup("enum", ("bit_depth",))),
}

_XFORM_KNOBS = tuple(KnobGroup("float_slider", (name,), label=label, soft_range=soft) for name, label, soft in (
    ("tx", "Translate X", (-10, 10)), ("ty", "Translate Y", (-10, 10)), ("tz", "Translate Z", (-10, 10)),
    ("rx", "Rotate X", (-180, 180)), ("ry", "Rotate Y", (-180, 180)), ("rz", "Rotate Z", (-180, 180)),
    ("sx", "Scale X", (0.01, 5)), ("sy", "Scale Y", (0.01, 5)), ("sz", "Scale Z", (0.01, 5))))
_SURFACE_KNOB = KnobGroup("color", ("red", "green", "blue", "alpha"))
_MATERIAL_KNOBS = tuple(KnobGroup("float_slider", (key,), label=label, soft_range=LIMITS[key])
                        for key, label in (("spec_amount", "Specular"),
                                           ("spec_shininess", "Shininess"), ("emission", "Emission")))
_TARGET_KNOBS = tuple(KnobGroup("float_slider", (f"target_{a}",), soft_range=(-10, 10)) for a in "xyz")
KNOB_LAYOUT.update({
    "Card3D": _groups(KnobGroup("float_slider", ("card_width",), label="Width", soft_range=(0.01, 10)),
                      KnobGroup("float_slider", ("card_height",), label="Height", soft_range=(0.01, 10)),
                      *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Cube3D": _groups(KnobGroup("float_slider", ("cube_size",), label="Size", soft_range=(0.01, 10)),
                      *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Sphere3D": _groups(KnobGroup("float_slider", ("sphere_radius",), label="Radius", soft_range=(0.01, 10)),
                        KnobGroup("int", ("segments",)), *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "ReadAlembic3D": _groups(KnobGroup("string", ("abc_path",), label="Alembic file"),
                             KnobGroup("string", ("abc_root",), label="Root object")),
    "ReadAlembicCamera3D": _groups(KnobGroup("string", ("abc_path",), label="Alembic file"),
                                   KnobGroup("string", ("abc_camera",), label="Camera object")),
    "ReadUSD3D": _groups(KnobGroup("string", ("usd_path",), label="USD file"),
                         KnobGroup("string", ("usd_root",), label="Root prim")),
    "ReadUSDCamera3D": _groups(KnobGroup("string", ("usd_path",), label="USD file"),
                               KnobGroup("string", ("usd_camera",), label="Camera prim")),
    "ReadGeo3D": _groups(KnobGroup("string", ("geo_path",), label="OBJ file"), *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Light3D": _groups(KnobGroup("enum", ("light_type",), label="Type"),
                       KnobGroup("enum", ("shadows",), label="Shadows"),
                       *_XFORM_KNOBS[:3], *_TARGET_KNOBS,
                       KnobGroup("float_slider", ("red",), soft_range=(0, 1)),
                       KnobGroup("float_slider", ("green",), soft_range=(0, 1)),
                       KnobGroup("float_slider", ("blue",), soft_range=(0, 1)),
                       KnobGroup("float_slider", ("intensity",), soft_range=(0, 5))),
    "Camera3D": _groups(*_XFORM_KNOBS[:3], *_TARGET_KNOBS,
                        KnobGroup("float_slider", ("roll",), soft_range=(-180, 180)),
                        KnobGroup("float_slider", ("fov",), label="Vertical FOV", soft_range=(5, 120)),
                        KnobGroup("float_slider", ("near",), soft_range=(0.01, 10)),
                        KnobGroup("float_slider", ("far",), soft_range=(10, 10000))),
    "Project3D": _groups(KnobGroup("enum", ("project_outside",), label="Outside"),
                         KnobGroup("enum", ("project_backfaces",), label="Backfaces"),
                         KnobGroup("enum", ("project_occlusion",), label="Occlusion")),
    "WriteGeo3D": _groups(KnobGroup("string", ("geo_write_path",), label="OBJ / USD file")),
    "Scene3D": _groups(*_XFORM_KNOBS),
    "Render3D": _groups(KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
                        KnobGroup("color", ("red", "green", "blue", "alpha"), label="Background"),
                        KnobGroup("float_slider", ("ambient",), soft_range=(0, 1)),
                        KnobGroup("int", ("samples",), label="Antialiasing samples"),
                        KnobGroup("enum", ("render_output",), label="Output"),
                        KnobGroup("enum", ("render_backend",), label="Backend"),
                        KnobGroup("enum", ("render_mode",), label="Mode")),
})


def _inverted_binary_name(param):
    return param in {"invert", "apply_translate", "apply_rotate", "apply_scale"}


def _scalar_kind(node_type, param, value=None):
    if param in CHOICES:
        return "enum"
    if param == "path" and node_type == "Read":
        return "file_read"
    if param == "path" and node_type == "Write":
        return "file_write"
    if isinstance(value, str):
        return "string"
    if _inverted_binary_name(param):
        return "bool"
    if param in {"which", "subimage"}:
        return "int"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "bool" if LIMITS.get(param) == (0, 1) else "int"
    return "float_slider"


def _fallback_layout(node_type):
    return [KnobGroup(_scalar_kind(node_type, param, default), (param,))
            for param, default in SPECS[node_type]["params"].items()]


def knob_layout(node_type):
    """Return presentation groups, deriving simple groups for newly added node types."""
    return KNOB_LAYOUT.get(node_type, _fallback_layout(node_type))


def resolve_kind(node_type, param, value):
    """Resolve one parameter to the kind used by its node's presentation layout."""
    for group in knob_layout(node_type):
        if param in group.params:
            return group.kind
    return _scalar_kind(node_type, param, value)
