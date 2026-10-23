"""What the render card's display picture pass costs on its own (nodebased/gpupicture.py).

Uploads a 1920 x 1080 HDR float frame once, then times 40 times: the pass for a view (no transform, sRGB, ACES 2.0), the copy of its
8-bit picture into a mappable buffer, the wait for it and the copy out. `submit_to_mapped_ms` is the card's share (the shader plus the
picture's trip over the link); the difference between "ACES 2.0" and "Linear" is the shader. One JSON line per adapter.

    python tools/benchmark_card_picture.py [default|discrete|integrated|cpu]      # run under /tmp/nb-gpu.lock
"""
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodebased import gpu3d, gpupicture  # noqa: E402

kind = sys.argv[1] if len(sys.argv) > 1 else "default"
orig = gpu3d._state
gpu3d._state = lambda choice=None: orig(kind if (choice or 'default') == 'default' else choice)
state = gpu3d._state(); wgpu = state['wgpu']; device = state['device']
W, H = 1920, 1080
gpupicture.ARMED_KINDS = ("discrete", "integrated", "cpu", "other")
rng = np.random.default_rng(3)
fmt = state['format']
data = (rng.random((H, W, 4), dtype=np.float32) * 2.0).astype(np.float32); data[..., 3] = 1
tex = device.create_texture(size=(W, H, 1), format=fmt, usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
if fmt == 'rgba32float':
    device.queue.write_texture({'texture': tex}, data, {'bytes_per_row': W*16, 'rows_per_image': H}, (W, H, 1))
else:
    device.queue.write_texture({'texture': tex}, data.astype('f2'), {'bytes_per_row': W*8, 'rows_per_image': H}, (W, H, 1))
view = tex.create_view()
out = {"adapter": state['info'].get('device'), "format": fmt}
for name in ("Linear", "sRGB", "ACES 2.0"):
    rec = gpupicture.prepare(state, name, W, H)
    staging = device.create_buffer(size=rec.bytes, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    def once(read):
        enc = device.create_command_encoder()
        rec.record(enc, view, staging, 0)
        t = time.perf_counter()
        device.queue.submit([enc.finish()]); staging.map_sync(wgpu.MapMode.READ)
        gpu = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if read: rec.read(staging)
        cpu = (time.perf_counter() - t) * 1000
        staging.unmap()
        return gpu, cpu
    for _ in range(5): once(True)
    runs = [once(True) for _ in range(40)]
    out[name] = {"submit_to_mapped_ms": round(statistics.median(r[0] for r in runs), 2), "copy_out_ms": round(statistics.median(r[1] for r in runs), 2)}
print(json.dumps(out)); sys.stdout.flush(); os._exit(0)
