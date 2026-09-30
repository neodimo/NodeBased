"""4K/8K memory ceiling and throughput benchmark (docs/VISION.md's M2 gate, the second half:
docs/BENCHMARKS-v0.33-m2.md's "4K/8K memory ceilings and throughput" section).

Builds the ten-node graph the step brief names -- Read, Grade, Transform, Merge, Blur, Roto,
Tracker, ColorCorrect, Reformat, Write -- reading a short real EXR sequence at 4K and at 8K, and
plays it back through the full-frame reference evaluator (`Evaluator.evaluate_raster`) and
through the tile-path executor (`TileExecutor.compose_region`, a full-canvas region so both paths
render the same pixels). Transform, Roto, Tracker and Reformat are not tile-native (the
precedented exclusion `docs/M1_GATE.md` and `tileexec.SUPPORTED_TILED_KINDS` already state for
Transform/Crop/Mirror); on the tile path those four fall back to the full-frame evaluator per
node, which is the documented, expected behaviour, not a bug in this benchmark.

Each (resolution, path) combination is measured twice: "cold" is a fresh in-memory cache with an
empty on-disk result tier (`cachetier.DiskCache`), so every frame is a genuine miss; "warm" reuses
the same on-disk tier (now populated by the cold pass) but a fresh in-memory cache, so a hit comes
from disk, not from the process's own memory -- the on-disk tier is what "warm" names in this
project (see `cachetier.py`'s module docstring), not the OS page cache.

    python tools/benchmark_memory_throughput.py [--json out.json] [--frames 6] [--label "..."]
"""
import argparse
import json
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from nodebased import cachetier
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tiles import TileRegion
from nodebased.tileexec import TileExecutor

RESOLUTIONS = (("4K", 3840, 2160), ("8K", 7680, 4320))
DEFAULT_FRAMES = 6


def peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def write_sequence(directory, width, height, frames):
    """A short real EXR sequence at `width`x`height`; a different seed per frame so each frame is
    genuinely distinct pixel data, not the same still repeated under a new frame number."""
    pattern = str(Path(directory) / "plate.%04d.exr")
    for frame in range(1, frames + 1):
        rng = np.random.default_rng(frame)
        data = rng.random((height, width, 4)).astype(np.float32)
        data[..., 3] = 1.0
        write_exr(pattern % frame, data)
    return pattern


def build_document(sequence_path, width, height):
    """Read, Grade, Transform, Merge, Blur, Roto, Tracker, ColorCorrect, Reformat, Write."""
    d = Dispatcher()
    d.execute({"op": "create", "type": "Read", "id": "read", "params": {"path": sequence_path}})
    d.execute({"op": "create", "type": "Grade", "id": "grade", "params": {"exposure": 0.1}})
    d.execute({"op": "connect", "id": "grade", "input": "image", "source": "read"})
    d.execute({"op": "create", "type": "Transform", "id": "transform",
               "params": {"translate_x": 2.0, "translate_y": 1.0}})
    d.execute({"op": "connect", "id": "transform", "input": "image", "source": "grade"})
    d.execute({"op": "create", "type": "Checker", "id": "checker_b",
               "params": {"width": width, "height": height, "size": 48}})
    d.execute({"op": "create", "type": "Merge", "id": "merge", "params": {"operation": "over"}})
    d.execute({"op": "connect", "id": "merge", "input": "A", "source": "transform"})
    d.execute({"op": "connect", "id": "merge", "input": "B", "source": "checker_b"})
    d.execute({"op": "create", "type": "Blur", "id": "blur", "params": {"radius": 3.0}})
    d.execute({"op": "connect", "id": "blur", "input": "image", "source": "merge"})
    d.execute({"op": "create", "type": "Tracker", "id": "tracker", "params": {}})
    d.execute({"op": "connect", "id": "tracker", "input": "image", "source": "blur"})
    # Roto has no image input of its own (a shape-source matte generator, like Nuke's); it
    # feeds ColorCorrect's optional mask input instead of sitting in the main chain.
    d.execute({"op": "create", "type": "Roto", "id": "roto",
               "params": {"width": width, "height": height}})
    d.execute({"op": "create", "type": "ColorCorrect", "id": "colorcorrect", "params": {}})
    d.execute({"op": "connect", "id": "colorcorrect", "input": "image", "source": "tracker"})
    d.execute({"op": "connect", "id": "colorcorrect", "input": "mask", "source": "roto"})
    d.execute({"op": "create", "type": "Reformat", "id": "reformat",
               "params": {"reformat_type": "scale", "scale": 1.0}})
    d.execute({"op": "connect", "id": "reformat", "input": "image", "source": "colorcorrect"})
    d.execute({"op": "create", "type": "Write", "id": "write", "params": {"path": ""}})
    d.execute({"op": "connect", "id": "write", "input": "image", "source": "reformat"})
    return d, "write"


def _full_canvas_region(exe, document, target):
    bounds = exe.canvas_region(document, target, frame=1, tier=1)
    return TileRegion(x=bounds.full_x, y=bounds.full_y, width=bounds.full_width, height=bounds.full_height,
                      full_width=bounds.full_width, full_height=bounds.full_height,
                      full_x=bounds.full_x, full_y=bounds.full_y)


def _playback(evaluate_frame, frames):
    start = time.perf_counter()
    for frame in range(1, frames + 1):
        evaluate_frame(frame)
    elapsed = time.perf_counter() - start
    return frames / elapsed if elapsed > 0 else float("inf"), elapsed


def measure(width, height, frames, path_kind, disk_root):
    """One (resolution, tile|full-frame) pair, cold then warm. Returns a dict of both passes."""
    with tempfile.TemporaryDirectory() as seq_dir:
        sequence_path = write_sequence(seq_dir, width, height, frames)
        results = {}
        for state in ("cold", "warm"):
            disk = cachetier.DiskCache(root=disk_root, enabled=True)
            d, target = build_document(sequence_path, width, height)
            if path_kind == "full-frame":
                ev = Evaluator(disk=disk)

                def evaluate_frame(frame, ev=ev, d=d, target=target):
                    ev.evaluate_raster(d.document, target, frame=frame, tier=1, typed=True)
            else:
                exe = TileExecutor(evaluator=Evaluator(disk=disk))
                region = _full_canvas_region(exe, d.document, target)

                def evaluate_frame(frame, exe=exe, d=d, target=target, region=region):
                    exe.compose_region(d.document, target, region, frame=frame, tier=1)

            mem_before = peak_memory_mb()
            fps, elapsed = _playback(evaluate_frame, frames)
            mem_after = peak_memory_mb()
            results[state] = {"fps": fps, "elapsed_s": elapsed, "peak_mb": mem_after,
                              "peak_delta_mb": mem_after - mem_before}
        return results


def run(frames=DEFAULT_FRAMES, label=None):
    table = {}
    for res_name, width, height in RESOLUTIONS:
        table[res_name] = {}
        for path_kind in ("full-frame", "tile"):
            with tempfile.TemporaryDirectory() as disk_dir:
                table[res_name][path_kind] = measure(width, height, frames, path_kind, disk_dir)
    return {"label": label, "platform": platform.platform(), "python": platform.python_version(),
            "frames": frames, "results": table}


def _print_table(report):
    for res_name, per_path in report["results"].items():
        for path_kind, states in per_path.items():
            for state, m in states.items():
                print(f"{res_name:>4} {path_kind:>10} {state:>4}: "
                      f"{m['fps']:6.2f} fps, peak {m['peak_mb']:8.1f} MB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()
    report = run(frames=args.frames, label=args.label)
    _print_table(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
