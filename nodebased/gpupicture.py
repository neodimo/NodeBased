"""The 8-bit display picture of a played 3D frame, made on the render card in the same submission as the frame.

A played 3D frame used to leave the card as 33 MB of float colour at 1080p, go up again to whatever graphics context
the display conversion owned (software OpenGL on a private Xvfb display, the integrated GPU on a hybrid laptop) and come back as
a 6 MB picture, all on the conversion thread that is supposed to run beside the next render. The conversion alone took 14 to
27 ms of the 41.7 ms a 24 fps frame has (docs/PERFORMANCE.md, "3D playback on the render card").

Here the render pass is followed, in the same command encoder, by one more pass that reads the frame texture it just drew and
writes the picture: the view transform, `to_qimage`'s own order of operations (unpremultiply, the view's OCIO shader, associate
again) and the 8-bit quantise. The picture is read back beside the float frame (8 MB against 33 MB of float colour) and handed to
the conversion stage through a small registry: `to_qimage` asks it first, by the identity of the float frame, and only builds
the picture itself when nothing matches. The float frame is untouched (the display cache, pixel readouts and exports use it).

Arm it for the thread that evaluates a played frame with `arm(view)`; `gpu3d.render` then adds the pass when the frame is a
plain opaque-or-transparent RGBA draw of one sample with a zero background, on a discrete or integrated adapter, and registers the
picture against the array it returns. Any other frame, and any card that cannot build the shader, leaves the old path in charge.
"""
import collections
import os
import threading
import weakref

import numpy as np

from . import carddisplay

VERTEX = """
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
"""

# to_qimage with exposure 0, gamma 1, the RGB channels and a black background, on a premultiplied frame:
# unpremultiply by the clamped alpha, the view, associate again. An opaque frame divides and multiplies by one.
FRAGMENT = """#version 450
layout(set = 0, binding = 0) uniform texture2D img;
layout(set = 1, binding = 0, std430) readonly buffer OcioTables { vec4 data[]; } ocio_tables;
layout(location = 0) out vec4 fragColor;

%(view)s

void main() {
    vec4 p = texelFetch(img, ivec2(gl_FragCoord.xy), 0);
    float w = clamp(p.a, 0.0, 1.0);
    vec3 straight = w > 1e-8 ? p.rgb / w : vec3(0.0);
    vec3 rgb = %(function)s(vec4(straight, 1.0)).rgb * w;
    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
"""

ENABLED = os.environ.get("NODEBASED_CARD_PICTURE", "1") != "0"      # 0: the conversion stage always makes the picture itself
MAX_PENDING = 3          # pictures waiting for their frame to be converted; a dropped frame's picture ages out
ARMED_KINDS = ("discrete", "integrated")

_local = threading.local()
_lock = threading.Lock()
_pending = collections.OrderedDict()      # id(frame) -> (weakref, view, rgba8 picture)
counters = collections.Counter()


def arm(view):
    """The frames this thread renders from now on want their picture made for `view` (until `disarm`)."""
    _local.view = view if ENABLED else None


def disarm():
    _local.view = None
    _local.warm = None


def arm_warm(view):
    """The frames this thread renders from now on are not played, but a played frame will want the picture for `view`: they build
    its display pass (`warm`) without making a picture."""
    _local.warm = view if ENABLED else None


def warm_view():
    return getattr(_local, "warm", None)


def armed_view():
    return getattr(_local, "view", None)


def register(frame, view, rgba):
    """Keep `rgba` (HxWx4 uint8, alpha 255) as the picture of `view` for exactly this float frame."""
    key = id(frame)

    def forget(_ref, key=key):
        with _lock:
            _pending.pop(key, None)
    with _lock:
        _pending[key] = (weakref.ref(frame, forget), view, rgba)
        while len(_pending) > MAX_PENDING:
            _pending.popitem(last=False)
        counters["registered"] += 1


def take(frame, exposure, channel, background, view, look):
    """The RGB888 QImage the card made of `frame` for these display arguments, or None. A picture is handed out once."""
    if not _pending:
        return None
    look = look or {}
    if (exposure or channel not in ("RGB", "RGBA") or background == "checker" or float(look.get("gamma", 1.0)) != 1.0
            or look.get("zebra")):
        return None
    with _lock:
        entry = _pending.get(id(frame))
        if entry is None or entry[0]() is not frame or entry[1] != view:
            return None
        del _pending[id(frame)]
    rgba = entry[2]
    if rgba.shape[:2] != frame.shape[:2]:
        return None
    from PySide6.QtGui import QImage
    height, width = rgba.shape[:2]
    image = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBX8888)
    counters["taken"] += 1
    return image.convertToFormat(QImage.Format.Format_RGB888)


def clear():
    with _lock:
        _pending.clear()


class Recorder:
    """The display pass for one frame: `record` adds it (and the copy of its picture) to the command encoder that drew the
    frame, `read` returns the picture once that encoder's work is done."""

    def __init__(self, state, view, width, height):
        self.state, self.view, self.width, self.height = state, view, width, height
        self.resources = _resources(state, view, width, height)

    @property
    def bytes(self):
        """Room the picture needs in the frame's readback buffer, after the float bands (offset to 256)."""
        return self.resources["stride"] * self.height

    def record(self, encoder, source_view, staging, offset):
        """The display pass, then the copy of its picture into `staging` at `offset` (a multiple of 256)."""
        device = self.state["device"]
        pipeline, tables_group, layout = self.resources["pipeline"]
        group = device.create_bind_group(layout=layout, entries=[{"binding": 0, "resource": source_view}])
        rp = encoder.begin_render_pass(color_attachments=[{"view": self.resources["target_view"], "resolve_target": None,
                                                           "clear_value": (0, 0, 0, 1), "load_op": "clear",
                                                           "store_op": "store"}])
        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, group)
        rp.set_bind_group(1, tables_group)
        rp.draw(3, 1, 0, 0)
        rp.end()
        self.offset = offset
        encoder.copy_texture_to_buffer({"texture": self.resources["target"]},
                                       {"buffer": staging, "offset": offset, "bytes_per_row": self.resources["stride"],
                                        "rows_per_image": self.height}, (self.width, self.height, 1))

    def read(self, staging):
        """The picture out of the mapped readback buffer (the submission is done) as an HxWx4 uint8 array."""
        stride = self.resources["stride"]
        view = staging.read_mapped(copy=False)
        raw = np.frombuffer(view, np.uint8, count=self.height * stride, offset=self.offset).reshape(self.height, stride)
        rgba = raw[:, :self.width * 4].copy().reshape(self.height, self.width, 4)       # a copy: the mapping is released after this
        del raw, view
        return rgba


def prepare(state, view, width, height):
    """A Recorder for this card, view and frame size, or None when the card cannot make the picture (the old path then runs).
    Call with the GPU lock held, before the frame's command encoder exists."""
    if state.get("picture_failed"):
        return None
    from . import gpu3d
    if gpu3d._adapter_kind(state) not in ARMED_KINDS:
        return None
    try:
        return Recorder(state, view, width, height)
    except carddisplay.Unsupported as error:
        state.setdefault("picture_unsupported", {})[view] = str(error)
        return None
    except Exception as error:      # a card or driver that refuses the pass: stop trying on it
        state["picture_failed"] = f"{type(error).__name__}: {error}"
        return None


def _view_pipeline(state, view):
    """The display pass for `view` on this card: (pipeline, tables bind group, source layout, tables buffer), built once."""
    wgpu, device = state["wgpu"], state["device"]
    held = state.setdefault("picture", {})
    if "layouts" not in held:
        held["vertex"] = device.create_shader_module(code=VERTEX, label="picture_vertex")
        held["source_layout"] = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"}}])
        held["tables_layout"] = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "read-only-storage"}}])
        held["layouts"] = True
        held["pipelines"] = {}
    pipelines = held["pipelines"]
    if view not in pipelines:
        source = FRAGMENT % {"view": carddisplay.view_source(view), "function": carddisplay.FUNCTION}
        tables = carddisplay.view_tables(view)
        module = device.create_shader_module(code=source, label=f"picture_{view}_frag")
        pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[held["source_layout"], held["tables_layout"]]),
            vertex={"module": held["vertex"], "entry_point": "vs"},
            fragment={"module": module, "entry_point": "main", "targets": [{"format": "rgba8unorm"}]})
        buffer = device.create_buffer_with_data(data=tables.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        group = device.create_bind_group(layout=held["tables_layout"], entries=[
            {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": tables.nbytes}}])
        pipelines[view] = (pipeline, group, held["source_layout"], buffer)
    return pipelines[view]


def warm(state, view):
    """Build the display pass for `view` before the first played frame needs it (the frames before playback start call this, so
    the first played frame creates no shader module or pipeline). Quietly does nothing on a card that cannot make the picture."""
    if state.get("picture_failed"):
        return False
    from . import gpu3d
    if gpu3d._adapter_kind(state) not in ARMED_KINDS:
        return False
    try:
        _view_pipeline(state, view)
    except carddisplay.Unsupported as error:
        state.setdefault("picture_unsupported", {})[view] = str(error)
        return False
    except Exception as error:
        state["picture_failed"] = f"{type(error).__name__}: {error}"
        return False
    return True


def _resources(state, view, width, height):
    wgpu, device = state["wgpu"], state["device"]
    held = state.setdefault("picture", {})
    pipeline, group, layout, _buffer = _view_pipeline(state, view)
    key = (width, height)
    if held.get("size") != key:
        old = held.pop("target", None)
        if old is not None:
            old.destroy()
        held["target"] = device.create_texture(size=(width, height, 1), format="rgba8unorm",
                                               usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
        held["target_view"] = held["target"].create_view()
        held["stride"] = ((width * 4 + 255) // 256) * 256
        held["size"] = key
    return {"pipeline": (pipeline, group, layout), "target": held["target"], "target_view": held["target_view"],
            "stride": held["stride"]}
