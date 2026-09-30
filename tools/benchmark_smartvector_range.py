import time
import numpy as np
from nodebased import gpu3d
from nodebased.flow_nodes import accumulated_vectors
from nodebased.opticalflow import _sample

h, w, count = 1080, 1920, 20
rng = np.random.default_rng(91)
base = rng.random((h, w), dtype=np.float32)
for _ in range(4):
    base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0)
            + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5

y, x = np.mgrid[:h, :w].astype(np.float32)
frames = [_sample(base, x - 0.5 * i, y) for i in range(count)]
print("adapter:", gpu3d.describe(), flush=True)
for backend in ("cpu", "gpu"):
    started = time.perf_counter()
    forward, backward = accumulated_vectors(frames, count // 2, vector_detail=4,
        smoothness=1.0, reanchor_interval=5, backend=backend)
    print(f"{backend}_seconds={time.perf_counter() - started:.3f}",
          f"frames={count}", f"forward_layers={len(forward)}",
          f"backward_layers={len(backward)}", flush=True)
