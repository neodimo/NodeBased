"""VP2a upload timing: how long a picture takes from hand-over to being on the card, per adapter.

No window: the presenter draws into an offscreen texture of the dock's size. For each picture it records
the Qt-thread time of the hand-over (staging slot map included), the filler thread's copy, the Qt-thread
time of the draw that records the GPU copy, and the time until the card has finished (a queue idle wait
after the draw, which a real dock never does). For comparison it also times VP1's path, one
`queue.write_texture` of the same frame on the Qt thread plus the same idle wait.

    python tools/viewer_direct_upload.py OUT.json --adapter discrete|integrated|cpu [--size 3840x2160] [--count 24]
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--adapter", default="discrete")
p.add_argument("--size", default="3840x2160")
p.add_argument("--count", type=int, default=24)
p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
args = p.parse_args()

import wgpu  # noqa: E402

from nodebased import carddisplay, gpu3d  # noqa: E402
from nodebased import directviewer as dv  # noqa: E402

W, H = (int(v) for v in args.size.lower().split("x"))
DOCK = (960, 540)


class Target:
    def __init__(self, size):
        self.size, self.texture = size, None

    def get_preferred_format(self, adapter):
        return "rgba8unorm"

    def configure(self, device, format, alpha_mode):
        self.texture = device.create_texture(size=(*self.size, 1), format=format,
                                             usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)

    def get_current_texture(self):
        return self.texture


def dist(values):
    v = sorted(values)
    pick = lambda q: v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))]
    return {"n": len(v), "p50": round(pick(50), 2), "p95": round(pick(95), 2), "max": round(v[-1], 2),
            "mean": round(statistics.fmean(v), 2)}


kind = {"discrete": "discretegpu", "integrated": "integratedgpu", "cpu": "cpu"}[args.adapter]
adapters = [a for a in gpu3d.rank_adapters(list(wgpu.gpu.enumerate_adapters_sync())) if gpu3d._kind_of(a.info) == kind]
adapter = adapters[0]
presenter = dv.RetainedPresenter(Target(DOCK), adapter=adapter, fill_thread=True)
settings = carddisplay.settings_from(0.0, "RGB", "black", "sRGB", None)
rng = np.random.default_rng(3)
frames = [rng.random((H, W, 4), np.float32).astype(args.dtype) for _ in range(3)]
view = dv.fit_view(DOCK, (0, 0, W, H))

hand, fill, draw, total = [], [], [], []
for i in range(args.count + 2):
    fills_before = presenter.counters["fill_ms_total"]
    start = time.perf_counter()
    presenter.offer_linear(frames[i % 3], settings, (0, 0, W, H), (0, 0, W, H), i)
    handed = time.perf_counter()
    presenter.ring.wait()
    filled = time.perf_counter()
    presenter.draw(view, DOCK)
    drawn = time.perf_counter()
    presenter.wait_idle()
    done = time.perf_counter()
    if i >= 2:                       # the first two make the buffers and the texture
        hand.append((handed - start) * 1000)
        fill.append(presenter.counters["fill_ms_total"] - fills_before)
        draw.append((drawn - filled) * 1000)
        total.append((done - start) * 1000)

# VP1's path for comparison: one write_texture of the whole frame on the calling thread.
device = presenter.device
fmt = "rgba16float" if args.dtype == "float16" else "rgba32float"
texel = 8 if args.dtype == "float16" else 16
texture = device.create_texture(size=(W, H, 1), format=fmt, usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
old_call, old_total = [], []
for i in range(args.count):
    start = time.perf_counter()
    device.queue.write_texture({"texture": texture}, frames[i % 3], {"bytes_per_row": W * texel, "rows_per_image": H}, (W, H, 1))
    called = time.perf_counter()
    presenter.wait_idle()
    old_call.append((called - start) * 1000)
    old_total.append((time.perf_counter() - start) * 1000)
texture.destroy()

result = {
    "adapter": adapter.info.get("device", ""), "backend": adapter.info.get("backend_type", ""), "machine": platform.node(),
    "picture": f"{W}x{H} {args.dtype}", "bytes_per_picture": W * H * texel, "pictures": args.count,
    "hand_over_qt_thread_ms": dist(hand), "filler_thread_copy_ms": dist(fill), "draw_qt_thread_ms": dist(draw),
    "hand_over_to_on_card_ms": dist(total),
    "vp1_write_texture_qt_thread_ms": dist(old_call), "vp1_write_texture_to_on_card_ms": dist(old_total),
    "counters": {k: round(v, 1) for k, v in presenter.counters.items()},
}
presenter.lose_device()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(result, fh, indent=1)
print(json.dumps({k: result[k] for k in ("adapter", "hand_over_qt_thread_ms", "filler_thread_copy_ms", "draw_qt_thread_ms",
                                          "hand_over_to_on_card_ms", "vp1_write_texture_qt_thread_ms",
                                          "vp1_write_texture_to_on_card_ms")}))
