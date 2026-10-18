"""The viewer's display transform as a wgpu fragment shader (the direct viewer's linear path).

`to_qimage` turns a premultiplied scene-linear float frame into display bytes on the processor:
gain, channel select, unpremultiply, gamma, the view transform, re-associate, the checker
background, the clipping warning and quantise. At 1080p that costs about 41 ms a frame
(docs/PERFORMANCE.md). This module builds the same chain as one GLSL 4.50 fragment shader that
wgpu compiles through naga, so the direct viewer can upload the float frame once and do every one
of those steps on the card.

The view transform is OCIO's own GPU shader for the view (`GpuShaderDesc`, the text gpudisplay
already hands to OpenGL), so the card runs the processor the CPU path runs, not an approximation of
it. OCIO's small lookup tables (ACES 2.0 ships two 1D textures, the reach and gamut-cusp tables,
nearest-sampled at texel centres, and one constant hue array) go into one read-only storage buffer
per view, bound as group 1. A view that needs a 2D or 3D lookup texture, or a filtered 1D one, raises
`Unsupported` and the direct viewer falls back to uploading `to_qimage`'s bytes.

`settings_from` reads the same arguments `to_qimage` takes; `uniform_values` packs them for the
shader. tests/test_carddisplay.py holds the card to `to_qimage` within 1/255 per channel.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

CHANNELS = ("RGB", "R", "G", "B", "A")
IDENTITY_VIEWS = ("Linear", "Raw")
FUNCTION = "OCIODisplay"


class Unsupported(Exception):
    """This view cannot run as a card shader; show `to_qimage`'s bytes instead."""


def settings_from(exposure=0.0, channel="RGB", background="black", view="sRGB", look=None):
    """The display arguments of one picture, in `to_qimage`'s own terms, as a hashable tuple."""
    look = look or {}
    if channel not in CHANNELS:
        raise Unsupported(f"channel {channel!r}")
    return (float(exposure), channel, background, view, float(look.get("gamma", 1.0)), bool(look.get("zebra")))


def _gpu_processor(view):
    from . import color
    if view in ("sRGB", "ACES 2.0"):
        return color.display_gpu_processor(view)
    import PyOpenColorIO as ocio
    if view not in color.viewer_displays():
        raise Unsupported(f"unknown view {view!r}")
    transform = ocio.DisplayViewTransform(src=color.WORKING, display="sRGB - Display", view=view)
    return color.config().getProcessor(transform).getDefaultGPUProcessor()


def _move_tables(text, desc):
    """Move OCIO's lookup data out of the shader text into one storage buffer of vec4 rows.

    Each nearest-sampled 1D texture becomes `NAME_fetch(u)` and each `const float NAME[N]` array an
    indexed read of `ocio_tables`. Constant arrays inlined in the shader were 40 times slower: naga
    copies a constant array into function memory for every dynamic index, and ACES 2.0's cusp lookup
    indexes its 363 hues in a loop (17 ms against 0.4 ms per 960 x 540 draw on the RTX 3080 Ti).
    Returns (glsl, rows) where rows is an (N, 4) float32 array, at least one row long."""
    if list(desc.get3DTextures()):
        raise Unsupported("view needs a 3D lookup table")
    rows, helpers = [], []
    for texture in desc.getTextures():
        if texture.height != 1 or "NEAREST" not in str(texture.interpolation):
            raise Unsupported(f"view needs a {texture.width}x{texture.height} {texture.interpolation} lookup")
        width = int(texture.width)
        values = np.asarray(texture.getValues(), np.float32)
        values = values.reshape(width, 3 if "RGB" in str(texture.channel) else 1)
        offset = len(rows)
        rows.extend(np.pad(values, ((0, 0), (0, 4 - values.shape[1]))))
        sampler = texture.samplerName
        name = re.sub(r"\W", "_", sampler)
        helpers.append(f"vec4 {name}_fetch(float u) {{ return ocio_tables.data[{offset} + "
                       f"clamp(int(floor(u * {width}.0)), 0, {width - 1})]; }}\n")
        text = re.sub(r"uniform\s+sampler1D\s+" + re.escape(sampler) + r"\s*;", "", text)
        text = re.sub(r"\btexture\(\s*" + re.escape(sampler) + r"\s*,", f"{name}_fetch(", text)
    for match in list(re.finditer(r"const\s+float\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*float\s*\[\s*\d+\s*\]\s*\(([^)]*)\)\s*;", text)):
        name, count = match.group(1), int(match.group(2))
        values = np.array([float(v) for v in match.group(3).split(",")], np.float32)
        if values.size != count:
            raise Unsupported(f"could not read the {name} table")
        offset = len(rows)
        rows.extend(np.pad(values[:, None], ((0, 0), (0, 3))))
        text = text.replace(match.group(0), "")
        text = re.sub(r"\b" + re.escape(name) + r"\[([^\]]*)\]", rf"ocio_tables.data[{offset} + (\1)].x", text)
    if re.search(r"\buniform\b|\bsampler[123]D\b|\bconst\s+\w+\s+\w+\s*\[", text):
        raise Unsupported("view shader declares uniforms, samplers or arrays the card path does not bind")
    table = np.array(rows, np.float32).reshape(-1, 4) if rows else np.zeros((1, 4), np.float32)
    return "".join(helpers) + text, table


@lru_cache(maxsize=16)
def view_program(view):
    """(GLSL defining `vec4 OCIODisplay(vec4)` for `view`, its lookup rows) (identity for Linear and Raw)."""
    if view in IDENTITY_VIEWS:
        return f"vec4 {FUNCTION}(vec4 c) {{ return c; }}\n", np.zeros((1, 4), np.float32)
    import PyOpenColorIO as ocio
    try:
        desc = ocio.GpuShaderDesc.CreateShaderDesc()
        desc.setLanguage(ocio.GPU_LANGUAGE_GLSL_4_0)
        desc.setFunctionName(FUNCTION)
        _gpu_processor(view).extractGpuShaderInfo(desc)
    except Unsupported:
        raise
    except Exception as exc:
        raise Unsupported(str(exc)) from exc
    return _move_tables(desc.getShaderText(), desc)


def view_source(view):
    return view_program(view)[0]


def view_tables(view):
    return view_program(view)[1]


FRAGMENT = """#version 450
layout(set = 0, binding = 0) uniform texture2D img;
layout(set = 0, binding = 1) uniform View {
    vec2 viewport; vec2 centre;
    vec4 image;            // scene rectangle the picture covers: x, y, w, h
    vec4 outline;          // scene rectangle of the format (overlay pass only)
    vec2 tex; float zoom; float srgb;
    vec4 surround;
    vec4 disp0;            // gain (2^exposure), channel (0 RGB, 1 R, 2 G, 3 B, 4 A), gamma, zebra
    vec4 disp1;            // checker, checker level 0, checker level 1, unused
} v;
layout(set = 0, binding = 2) uniform sampler smp;
layout(set = 1, binding = 0, std430) readonly buffer OcioTables { vec4 data[]; } ocio_tables;
layout(location = 0) out vec4 fragColor;

%(view)s

vec3 encode(vec3 c) {
    // An sRGB surface encodes on write; the shader already produced display values, so undo it once.
    if (v.srgb < 0.5) { return c; }
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(hi, lo, vec3(lessThanEqual(c, vec3(0.04045))));
}

vec3 display(ivec2 t) {
    vec4 p = texelFetch(sampler2D(img, smp), t, 0);
    float channel = v.disp0.y;
    if (channel > 3.5) { return clamp(vec3(p.a), 0.0, 1.0); }
    vec3 rgb = p.rgb * v.disp0.x;
    if (channel > 0.5) {
        float c = channel < 1.5 ? rgb.r : (channel < 2.5 ? rgb.g : rgb.b);
        rgb = vec3(c);
    }
    float w = clamp(p.a, 0.0, 1.0);
    vec3 straight = w > 1e-8 ? rgb / w : vec3(0.0);
    float gamma = v.disp0.z;
    if (gamma != 1.0) {
        vec3 a = abs(straight);
        vec3 bent = mix(pow(max(a, vec3(1e-30)), vec3(1.0 / gamma)), vec3(0.0), vec3(equal(a, vec3(0.0))));
        straight = sign(straight) * bent;
    }
    rgb = %(function)s(vec4(straight, 1.0)).rgb * w;
    if (v.disp1.x > 0.5) {
        float level = ((t.x / 16 + t.y / 16) %% 2) == 0 ? v.disp1.y : v.disp1.z;
        rgb = rgb + vec3(level) * (1.0 - w);
    }
    rgb = clamp(rgb, 0.0, 1.0);
    if (v.disp0.w > 0.5 && ((t.x + t.y) / 6) %% 2 == 0) {
        if (any(lessThan(straight, vec3(0.0)))) { rgb = vec3(0.0, 90.0 / 255.0, 1.0); }
        else if (any(greaterThan(straight, vec3(1.0)))) { rgb = vec3(1.0, 0.0, 0.0); }
    }
    return rgb;
}

void main() {
    vec2 s = v.centre + (gl_FragCoord.xy - v.viewport * 0.5) / v.zoom;
    vec2 rel = (s - v.image.xy) / v.image.zw;
    if (rel.x < 0.0 || rel.y < 0.0 || rel.x >= 1.0 || rel.y >= 1.0) {
        fragColor = vec4(encode(v.surround.rgb), 1.0);
        return;
    }
    ivec2 t = ivec2(min(rel * v.tex, v.tex - vec2(1.0)));
    fragColor = vec4(encode(display(t)), 1.0);
}
"""


@lru_cache(maxsize=16)
def fragment_source(view):
    return FRAGMENT % {"view": view_source(view), "function": FUNCTION}


@lru_cache(maxsize=16)
def checker_levels(view):
    """The two checker shades `to_qimage` composites, through `view` (computed once on the processor)."""
    from .color import display_rgb
    levels = display_rgb(np.array([[[0.055] * 3, [0.095] * 3]], np.float32), view)[0, :, 0]
    return float(levels[0]), float(levels[1])


def uniform_values(settings):
    """The 8 floats after the view block: disp0 and disp1."""
    exposure, channel, background, view, gamma, zebra = settings
    checker = background == "checker" and channel != "A"
    levels = checker_levels(view) if checker else (0.0, 0.0)
    return [float(np.float32(2.0) ** np.float32(exposure)), float(CHANNELS.index(channel)), gamma,
            1.0 if zebra and channel != "A" else 0.0, 1.0 if checker else 0.0, levels[0], levels[1], 0.0]


@lru_cache(maxsize=16)
def supported(view):
    """True when `view` builds as a card shader (the GPU itself may still refuse; the presenter says so)."""
    try:
        view_source(view)
    except Unsupported:
        return False
    return True


class LazyPicture:
    """A frame the direct viewer shows on the card; the 8-bit QImage is built only if something asks.

    Playback with the direct viewer on hands one of these to the window instead of `to_qimage`'s
    result. `image()` runs `to_qimage` once, on first request (sampling, export, a fallback when the
    card refuses the view), and keeps the result."""

    def __init__(self, frame, exposure=0.0, channel="RGB", background="black", view="sRGB", look=None):
        self.frame = frame
        self.args = (exposure, channel, background, view, look)
        self.settings = settings_from(exposure, channel, background, view, look)
        self._image = None

    def width(self):
        return int(self.frame.shape[1])

    def height(self):
        return int(self.frame.shape[0])

    def built(self):
        return self._image is not None

    def image(self):
        if self._image is None:
            from .imaging import to_qimage
            exposure, channel, background, view, look = self.args
            self._image = to_qimage(self.frame, exposure, channel, background=background, view=view, look=look)
        return self._image
