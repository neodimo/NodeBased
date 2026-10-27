"""Bake the fuel source and render three fire-shot frames plus a mesh-light comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from nodebased import gpu3d, simcache
from nodebased.core import load_document
from nodebased.imaging import Evaluator, write_png

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


@contextmanager
def gpu_lock():
    with open('/tmp/nb-gpu.lock', 'a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True, help='disposable simulation cache outside the repository')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    images = REPO / 'docs' / 'images' / 'fire_shot'
    images.mkdir(parents=True, exist_ok=True)
    document = load_document(ROOT / 'fire_shot.nbcomp')
    params = document['nodes']['render']['params']
    params.update(width=args.width, height=args.height, pt_samples=args.samples, sampling='fixed')
    evaluator = Evaluator(sim=simcache.SimCache(root=args.workdir, memory_budget=1 << 30,
                                                disk_budget=8 << 30))
    with gpu_lock():
        device = gpu3d._state()['info'].get('device', 'unknown')
    started = time.perf_counter()
    with gpu_lock():
        evaluator.evaluate_raster(document, 'cache', frame=24, typed=True)
    bake_s = time.perf_counter() - started
    frames = {}
    for frame in (8, 16, 24):
        started = time.perf_counter()
        with gpu_lock():
            pixels = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
        seconds = time.perf_counter() - started
        write_png(images / f'fire-{frame:03d}.png', pixels)
        frames[str(frame)] = {'seconds': round(seconds, 3), 'path': f'docs/images/fire_shot/fire-{frame:03d}.png',
                              'mean_linear_rgb': float(pixels[..., :3].mean())}
        if frame == 16:
            on = pixels
    off_document = load_document(ROOT / 'fire_shot.nbcomp')
    off_document['nodes']['render']['params'].update(width=args.width, height=args.height,
                                                     pt_samples=args.samples, sampling='fixed',
                                                     volume_fire_light_meshes=0.0)
    started = time.perf_counter()
    with gpu_lock():
        off = evaluator.evaluate_raster(off_document, 'render', frame=16).pixels.copy()
    off_s = time.perf_counter() - started
    write_png(images / 'fire-016-mesh-light-off.png', off)
    result = {'date': '2026-10-27', 'device': device, 'resolution': [args.width, args.height],
              'samples_per_pixel': args.samples, 'bake_seconds': round(bake_s, 3), 'frames': frames,
              'comparison_off_seconds': round(off_s, 3),
              'comparison_mean_linear_rgb_delta': float((on - off)[..., :3].mean()),
              'comparison_off_path': 'docs/images/fire_shot/fire-016-mesh-light-off.png'}
    benchmark = REPO / 'benchmarks' / 'fire_shot' / 'fire-glow-2026-10-27.json'
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
