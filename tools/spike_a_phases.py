"""Spike A: where the time of one warm edit goes (a probe for the 12 GB offloaded variant, run by the provider interpreter).

    python tools/spike_a_phases.py SESSION_DIR [--offload never]

SESSION_DIR is the `session` folder of a `spike_a_interactive.py` run (the control images of the shot as .npy). Loads the
engine like `serve` does, makes three single-frame edits and prints, per edit, the wall time of each model call (text
encoders, ControlNet, UNet step by step, VAE) measured with CUDA synchronisation, so the first UNet call of an edit (which
carries the weights' transfer from host memory) can be told from the later ones. Run under the GPU lock.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

session = Path(sys.argv[1])
if "--offload" in sys.argv:
    os.environ["NODEBASED_SDXL_OFFLOAD"] = sys.argv[sys.argv.index("--offload") + 1]
spec = importlib.util.spec_from_file_location("sdxl", Path(__file__).resolve().parents[1] / "nodebased" / "providers" / "sdxl_controlnet.py")
sdxl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdxl)
torch = sdxl._setup_torch()
settings = sdxl.resolve_settings({"fast_preview": True})
engine = sdxl._Engine(torch, settings, preview_dir="probe")
calls = []


def wrap(name, module):
    def pre(_module, _args, _kwargs=None):
        torch.cuda.synchronize()
        calls.append([name, time.perf_counter(), None])

    def post(_module, _args, _kwargs=None, _output=None):
        torch.cuda.synchronize()
        calls[-1][2] = time.perf_counter()
    module.register_forward_pre_hook(pre)
    module.register_forward_hook(lambda m, a, o: post(m, a))


pipe = engine.pipe
for name in ("text_encoder", "text_encoder_2", "unet", "controlnet"):
    wrap(name, getattr(pipe, name))
engine.preview_vae.decode  # the previews' own VAE is not timed here: see the interaction bench
rows = sorted(session.glob("frame_*.depth.npy"))
frame = 12
base = session / f"frame_{frame:04d}"
row = {"frame": frame, "depth": f"{base}.depth.npy", "normals": f"{base}.normals.npy", "motion_backward": f"{base}.motion.npy",
       "z": f"{base}.z.npy", "ids": f"{base}.ids.npy", "result": f"{session}/probe.result.npy"}
job = {"frames": [row], "width": 480, "height": 270, "prompt": "replace the backdrop with a rainy street", "seed": 7, "settings": settings}
report = []
for edit in range(3):
    calls.clear()
    began = time.perf_counter()
    torch.cuda.synchronize()
    engine.run_job(job, sink=None, cancelled=None, on_frame=lambda *a: None)
    torch.cuda.synchronize()
    total = time.perf_counter() - began
    report.append({"edit": edit + 1, "total_s": round(total, 3),
                   "calls": [(n, round(e - s, 3), round(s - began, 3)) for n, s, e in calls]})
    print(json.dumps(report[-1]), flush=True)
print("peak allocated MiB", round(torch.cuda.max_memory_allocated() / 2**20), "offloaded", engine.offloaded)
