"""Bake the fuel source and render three fire-shot frames plus a mesh-light comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from nodebased import gpu3d, simcache
from nodebased.core import load_document
from nodebased.imaging import Evaluator, write_png, _volume_settings

from make_shot import FUEL_TOP, SHAPED_RENDER, SHAPED_SOLVER, build_document
import flame_metrics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


try:
    import fcntl
except ImportError:  # Windows has no flock; the GPU lock is a Linux workstation convention
    fcntl = None


@contextmanager
def gpu_lock():
    if fcntl is None:
        yield
        return
    with open('/tmp/nb-gpu.lock', 'a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


FRAMES = (8, 16, 24)


def flame_occupancy(volume, threshold_fraction=.05):
    """Measure how continuous the cached flame field is, with no rendering involved."""
    flame = np.asarray(volume.flame, dtype=np.float32)
    peak = float(flame.max())
    if peak <= 0:
        return {'peak': 0.0, 'occupied_cells': 0, 'occupied_fraction': 0.0, 'empty_slices': 0,
                'height_slices': 0, 'lateral_sway_cells': 0.0}
    live = flame > peak * threshold_fraction
    # Volumes are indexed [ix, iy, iz]: height is the second axis.
    per_slice = live.reshape(live.shape[0], live.shape[1], -1).sum(axis=(0, 2))
    rows = np.flatnonzero(per_slice)
    span = rows[-1] - rows[0] + 1 if rows.size else 0
    empty = int(span - rows.size)
    sway = 0.0
    if rows.size > 1:
        centroids = []
        for y in rows:
            plane = flame[:, y, :]
            total = float(plane.sum())
            if total > 0:
                xi, zi = np.indices(plane.shape)
                centroids.append(((plane * xi).sum() / total, (plane * zi).sum() / total))
        if len(centroids) > 1:
            sway = float(np.asarray(centroids).std(axis=0).max())
    return {'peak': peak, 'occupied_cells': int(live.sum()), 'occupied_fraction': float(live.mean()),
            'empty_slices': empty, 'height_slices': int(span), 'lateral_sway_cells': round(sway, 3)}


def downsample(pixels, factor=2):
    h, w = pixels.shape[0] // factor * factor, pixels.shape[1] // factor * factor
    return pixels[:h, :w].reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def render_variant(cells, samples, workdir, width, height, frames=FRAMES, seeds=(1,), tag=None, **shot):
    """Bake one cell count and render the same frames at one sample count, once per seed."""
    document = build_document(width, height, division_size=2.0 / cells, pt_samples=samples, **shot)
    document['nodes']['render']['params']['pt_seed'] = seeds[0]
    evaluator = Evaluator(sim=simcache.SimCache(root=workdir / (tag or f'cells{cells}'), memory_budget=1 << 30,
                                                disk_budget=8 << 30))
    started = time.perf_counter()
    with gpu_lock():
        evaluator.evaluate_raster(document, 'cache', frame=max(frames), typed=True)
    bake_s = time.perf_counter() - started
    result = {'cells': cells, 'samples': samples, 'shot_options': shot, 'bake_seconds': round(bake_s, 3), 'frames': {}}
    images = {}
    for frame in frames:
        with gpu_lock():
            volume = evaluator.evaluate_raster(document, 'cache', frame=frame, typed=True)
        occupancy = flame_occupancy(volume)
        started = time.perf_counter()
        with gpu_lock():
            pixels = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
        seconds = time.perf_counter() - started
        entry = {'seconds': round(seconds, 3), 'mean_linear_rgb': float(pixels[..., :3].mean()),
                 'flame': occupancy}
        images[frame] = pixels
        if frame == 16 and len(seeds) > 1:
            other = []
            for seed in seeds[1:]:
                document['nodes']['render']['params']['pt_seed'] = seed
                with gpu_lock():
                    other.append(evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy())
            document['nodes']['render']['params']['pt_seed'] = seeds[0]
            # Independent seeds differ only by Monte Carlo noise: rms difference / sqrt(2) is the grain.
            entry['grain_rms'] = float(np.sqrt(((pixels[..., :3] - other[0][..., :3]) ** 2).mean() / 2))
        result['frames'][str(frame)] = entry
    return result, images


def compare(args):
    workdir = args.workdir
    images_dir = REPO / 'docs' / 'images' / 'fire_shot'
    images_dir.mkdir(parents=True, exist_ok=True)
    runs, pictures = {}, {}
    for cells in (32, 64):
        for samples in (32, 128):
            key = f'{cells}c-{samples}spp'
            runs[key], pictures[key] = render_variant(cells, samples, workdir, args.width, args.height,
                                                      seeds=(1, 2), tag=f'cells{cells}-{samples}', curl=0)
            print(key, json.dumps({f: (v['seconds'], round(v['mean_linear_rgb'], 4), v.get('grain_rms'))
                                   for f, v in runs[key]['frames'].items()}), flush=True)
    differences = {}
    for samples in (32, 128):
        a, b = pictures[f'32c-{samples}spp'], pictures[f'64c-{samples}spp']
        differences[f'32c_vs_64c_at_{samples}spp'] = {
            str(f): float(np.abs(a[f][..., :3] - b[f][..., :3]).mean()) for f in FRAMES}
    for cells in (32, 64):
        a, b = pictures[f'{cells}c-32spp'], pictures[f'{cells}c-128spp']
        differences[f'32spp_vs_128spp_at_{cells}c'] = {
            str(f): float(np.abs(a[f][..., :3] - b[f][..., :3]).mean()) for f in FRAMES}
    # Contact sheet: rows are the four variants, columns are frames 8, 16 and 24, at half size.
    rows = [np.concatenate([downsample(pictures[key][f]) for f in FRAMES], axis=1) for key in runs]
    sheet = np.concatenate(rows, axis=0)
    sheet_name = f'fire-compare-{args.tag}.png'
    write_png(images_dir / sheet_name, sheet)
    result = {'date': args.date, 'device': args.device, 'resolution': [args.width, args.height],
              'seeds': [1, 2], 'variants': runs, 'image_differences_mean_abs_linear': differences,
              'contact_sheet': f'docs/images/fire_shot/{sheet_name}',
              'contact_sheet_rows': list(runs), 'contact_sheet_columns': list(FRAMES)}
    benchmark = REPO / 'benchmarks' / 'fire_shot' / f'fire-look-{args.tag}.json'
    benchmark.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(differences, indent=2), flush=True)


def curl_sweep(args):
    """Render 64 cells at 128 samples with several seeded sideways-curl strengths."""
    images_dir = REPO / 'docs' / 'images' / 'fire_shot'
    runs, pictures = {}, {}
    for name, shot in (('straight', {'curl': 0}), ('curl-0.10', {'curl': .10}), ('curl-0.20', {'curl': .20}),
                       ('curl-0.20-noise', {'curl': .20, 'source_noise': .35})):
        runs[name], pictures[name] = render_variant(64, 128, args.workdir, args.width, args.height,
                                                    seeds=(1, 2), tag=f'sweep-{name}', **shot)
        print(name, json.dumps({f: (v['seconds'], v['flame']['lateral_sway_cells'], v['flame']['occupied_cells'])
                                for f, v in runs[name]['frames'].items()}), flush=True)
    rows = [np.concatenate([downsample(pictures[key][f]) for f in FRAMES], axis=1) for key in runs]
    sheet_name = f'fire-curl-{args.tag}.png'
    write_png(images_dir / sheet_name, np.concatenate(rows, axis=0))
    result = {'date': args.date, 'resolution': [args.width, args.height], 'cells': 64, 'samples': 128,
              'variants': runs, 'contact_sheet': f'docs/images/fire_shot/{sheet_name}',
              'contact_sheet_rows': list(runs), 'contact_sheet_columns': list(FRAMES)}
    benchmark = REPO / 'benchmarks' / 'fire_shot' / f'fire-curl-{args.tag}.json'
    benchmark.write_text(json.dumps(result, indent=2) + '\n')


GRAIN_COMPONENTS = ('rgba', 'emission', 'diffuse', 'diffuse_indirect')
GRAIN_TIMING_REPEATS = 5


def grain_estimator(args):
    """One process of the grain comparison: render the shot with one fire-emission estimator and save arrays and numbers.

    `--estimator old` sums the fire's glow at delta tracking's tentative collisions (what the shot used before the grain step) on
    both tracers; `new` integrates it along each flight. Run as a separate process per estimator so no prepared scene is shared."""
    from nodebased import gpupathtrace, ptvolume
    estimator = args.grain_estimator
    old = estimator == 'old'
    gpupathtrace.ENABLE_EXPECTED_EMISSION = not old
    ptvolume.ENABLE_EXPECTED_EMISSION = not old
    out = args.workdir / f'grain-{estimator}'
    out.mkdir(parents=True, exist_ok=True)
    document = build_document(args.width, args.height)
    params = document['nodes']['render']['params']
    evaluator = Evaluator(sim=simcache.SimCache(root=args.workdir / 'cache', memory_budget=1 << 30, disk_budget=8 << 30))
    with gpu_lock():
        device = gpu3d._state()['info'].get('device', 'unknown')
        started = time.perf_counter()
        evaluator.evaluate_raster(document, 'cache', frame=max(FRAMES), typed=True)
    result = {'estimator': estimator, 'device': device, 'resolution': [args.width, args.height], 'frames': {}}
    result['bake_seconds'] = round(time.perf_counter() - started, 3)

    def render(frame, seed, samples, output='rgba'):
        params.update(pt_samples=samples, pt_seed=seed, render_output=output)
        with gpu_lock():
            started = time.perf_counter()
            pixels = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
            return pixels, time.perf_counter() - started

    for frame in FRAMES:
        render(frame, 1, 8)                                                        # a warm-up: shader, scene and grid are built
        first, _ = render(frame, 1, args.samples)
        second, _ = render(frame, 2, args.samples)
        # a new seed each time: an identical request is answered from the evaluator's image cache in a millisecond
        seconds = [render(frame, 100 + i, args.samples)[1] for i in range(GRAIN_TIMING_REPEATS)]
        np.save(out / f'f{frame}-s1.npy', first)
        np.save(out / f'f{frame}-s2.npy', second)
        entry = {'grain_rms': float(np.sqrt(((first[..., :3] - second[..., :3]) ** 2).mean() / 2)),
                 'mean_linear_rgb': float((first[..., :3].mean() + second[..., :3].mean()) / 2),
                 'seconds_min': round(min(seconds), 3), 'seconds_median': round(float(np.median(seconds)), 3),
                 'seconds_all': [round(x, 3) for x in seconds]}
        if frame == 16:
            entry['components'] = {}
            for name in GRAIN_COMPONENTS[1:]:
                a, _ = render(frame, 1, args.samples, name)
                b, _ = render(frame, 2, args.samples, name)
                entry['components'][name] = {'grain_rms': float(np.sqrt(((a[..., :3] - b[..., :3]) ** 2).mean() / 2)),
                                             'mean_linear_rgb': float((a[..., :3].mean() + b[..., :3].mean()) / 2)}
        result['frames'][str(frame)] = entry
        print(estimator, frame, json.dumps({k: v for k, v in entry.items() if k != 'seconds_all'}), flush=True)
    result['reference'] = {'samples': args.reference_samples, 'frames': {}}
    for frame in FRAMES:
        seconds = []
        for seed in (1, 2):
            pixels, took = render(frame, seed, args.reference_samples)
            np.save(out / f'f{frame}-ref-s{seed}.npy', pixels)
            seconds.append(round(took, 2))
        result['reference']['frames'][str(frame)] = {'seconds': seconds}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def grain(args):
    """Old versus new fire-emission estimator on the same shot: grain, time, mean agreement with a 2048-sample reference."""
    images_dir = REPO / 'docs' / 'images' / 'fire_shot'
    images_dir.mkdir(parents=True, exist_ok=True)
    for estimator in ('old', 'new'):
        command = [sys.executable, str(Path(__file__).resolve()), '--workdir', str(args.workdir), '--width', str(args.width),
                   '--height', str(args.height), '--grain-estimator', estimator, '--samples', str(args.samples),
                   '--reference-samples', str(args.reference_samples)]
        subprocess.run(command, check=True)
    runs, arrays = {}, {}
    for estimator in ('old', 'new'):
        folder = args.workdir / f'grain-{estimator}'
        runs[estimator] = json.loads((folder / 'result.json').read_text())
        arrays[estimator] = {f: [np.load(folder / f'f{f}-s{s}.npy') for s in (1, 2)] for f in FRAMES}
        arrays[estimator].update({f'ref{f}': [np.load(folder / f'f{f}-ref-s{s}.npy') for s in (1, 2)] for f in FRAMES})
    agreement = {}
    for frame in FRAMES:
        reference = sum(a[..., :3] for a in arrays['old'][f'ref{frame}']) / 2          # the pre-change renderer at 2048 samples
        new_reference = sum(a[..., :3] for a in arrays['new'][f'ref{frame}']) / 2
        luminance = reference.mean(-1)
        core = luminance > np.quantile(luminance, .98)
        floor = float(np.abs(arrays['old'][f'ref{frame}'][0][..., :3] - arrays['old'][f'ref{frame}'][1][..., :3]).mean())
        agreement[str(frame)] = {
            'relative_mean_difference': float((new_reference.mean() - reference.mean()) / reference.mean()),
            'relative_mean_difference_flame_core': float((new_reference[core].mean() - reference[core].mean()) / reference[core].mean()),
            'mean_abs_difference': float(np.abs(new_reference - reference).mean()),
            'old_reference_two_seed_mean_abs_difference': floor}
    ratios = {str(f): runs['new']['frames'][str(f)]['seconds_median'] / runs['old']['frames'][str(f)]['seconds_median']
              for f in FRAMES}
    gate = {'grain_at_frame_16_old': runs['old']['frames']['16']['grain_rms'],
            'grain_at_frame_16_new': runs['new']['frames']['16']['grain_rms'],
            'grain_ratio': runs['new']['frames']['16']['grain_rms'] / runs['old']['frames']['16']['grain_rms'],
            'render_time_ratio_median_by_frame': {k: round(v, 3) for k, v in ratios.items()},
            'worst_relative_mean_difference_at_2048': max(abs(v['relative_mean_difference']) for v in agreement.values())}
    rows = [np.concatenate([downsample(arrays[e][f][0]) for f in FRAMES], axis=1) for e in ('old', 'new')]
    sheet_name = f'fire-grain-before-after-{args.tag}.png'
    write_png(images_dir / sheet_name, np.concatenate(rows, axis=0))
    # The upper plume at frame 16, 3 x enlarged without smoothing: before above, after below.
    crops = []
    for e in ('old', 'new'):
        crop = arrays[e][16][0][40:200, 330:650]
        crops.append(np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1))
    zoom_name = f'fire-grain-before-after-zoom-{args.tag}.png'
    write_png(images_dir / zoom_name, np.concatenate(crops, axis=0))
    result = {'date': args.date, 'device': runs['new']['device'], 'resolution': [args.width, args.height],
              'samples': args.samples, 'reference_samples': args.reference_samples, 'seeds': [1, 2],
              'gate': gate, 'agreement_with_old_estimator_at_reference_samples': agreement,
              'old': runs['old'], 'new': runs['new'],
              'contact_sheet': f'docs/images/fire_shot/{sheet_name}', 'contact_sheet_rows': ['before', 'after'],
              'contact_sheet_columns': list(FRAMES), 'zoom': f'docs/images/fire_shot/{zoom_name}'}
    benchmark = REPO / 'benchmarks' / 'fire_shot' / f'fire-grain-{args.tag}.json'
    benchmark.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(gate, indent=2), flush=True)


def smoke_grain_estimator(args):
    """One process of the smoke-grain comparison: render the shot with the fire sampled as a light at smoke vertices or without
    (`--smoke-grain-estimator old|new`, both tracers), saving arrays and numbers. The new process also renders frame 16 with the
    fire's light on smoke switched off (`volume_fire_light` 0), which no fire sampling can reach: the floor the gate is read against."""
    from nodebased import gpupathtrace, ptvolume
    estimator = args.smoke_grain_estimator
    old = estimator == 'old'
    gpupathtrace.ENABLE_FIRE_NEE = not old
    ptvolume.ENABLE_FIRE_NEE = not old
    out = args.workdir / f'smoke-{estimator}'
    out.mkdir(parents=True, exist_ok=True)
    document = build_document(args.width, args.height)
    params = document['nodes']['render']['params']
    evaluator = Evaluator(sim=simcache.SimCache(root=args.workdir / 'cache', memory_budget=1 << 30, disk_budget=8 << 30))
    with gpu_lock():
        device = gpu3d._state()['info'].get('device', 'unknown')
        evaluator.evaluate_raster(document, 'cache', frame=max(FRAMES), typed=True)
    result = {'estimator': estimator, 'device': device, 'resolution': [args.width, args.height], 'frames': {}}

    def render(frame, seed, samples, output='rgba', **extra):
        params.update(pt_samples=samples, pt_seed=seed, render_output=output, **extra)
        with gpu_lock():
            started = time.perf_counter()
            pixels = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
            return pixels, time.perf_counter() - started

    def grain_of(frame, output='rgba', **extra):
        a, _ = render(frame, 1, args.samples, output, **extra)
        b, _ = render(frame, 2, args.samples, output, **extra)
        return a, b, {'grain_rms': float(np.sqrt(((a[..., :3] - b[..., :3]) ** 2).mean() / 2)),
                      'mean_linear_rgb': float((a[..., :3].mean() + b[..., :3].mean()) / 2)}

    for frame in FRAMES:
        render(frame, 1, 8)                                                        # a warm-up: shader, scene and grids are built
        first, second, entry = grain_of(frame)
        seconds = [render(frame, 100 + i, args.samples)[1] for i in range(GRAIN_TIMING_REPEATS)]
        np.save(out / f'f{frame}-s1.npy', first)
        np.save(out / f'f{frame}-s2.npy', second)
        entry.update({'seconds_min': round(min(seconds), 3), 'seconds_median': round(float(np.median(seconds)), 3),
                      'seconds_all': [round(x, 3) for x in seconds]})
        if frame == 16:
            entry['components'] = {name: grain_of(frame, name)[2] for name in GRAIN_COMPONENTS[1:]}
            if not old:
                entry['fire_light_off'] = {'rgba': grain_of(frame, 'rgba', volume_fire_light=0.0)[2],
                                           'diffuse': grain_of(frame, 'diffuse', volume_fire_light=0.0)[2]}
                params.update(volume_fire_light=1.0)
        result['frames'][str(frame)] = entry
        print(estimator, frame, json.dumps({k: v for k, v in entry.items() if k != 'seconds_all'}), flush=True)
    result['reference'] = {'samples': args.reference_samples, 'frames': {}}
    for frame in FRAMES:
        seconds = []
        for seed in (1, 2):
            pixels, took = render(frame, seed, args.reference_samples)
            np.save(out / f'f{frame}-ref-s{seed}.npy', pixels)
            seconds.append(round(took, 2))
        result['reference']['frames'][str(frame)] = {'seconds': seconds}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def smoke_grain(args):
    """Fire sampled as a light at smoke vertices against the previous renderer: grain, time, agreement with a 2048-sample reference."""
    images_dir = REPO / 'docs' / 'images' / 'fire_shot'
    images_dir.mkdir(parents=True, exist_ok=True)
    for estimator in ('old', 'new'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--workdir', str(args.workdir), '--width', str(args.width),
                        '--height', str(args.height), '--smoke-grain-estimator', estimator, '--samples', str(args.samples),
                        '--reference-samples', str(args.reference_samples)], check=True)
    runs, arrays = {}, {}
    for estimator in ('old', 'new'):
        folder = args.workdir / f'smoke-{estimator}'
        runs[estimator] = json.loads((folder / 'result.json').read_text())
        arrays[estimator] = {f: [np.load(folder / f'f{f}-s{s}.npy') for s in (1, 2)] for f in FRAMES}
        arrays[estimator].update({f'ref{f}': [np.load(folder / f'f{f}-ref-s{s}.npy') for s in (1, 2)] for f in FRAMES})
    agreement = {}
    for frame in FRAMES:
        reference = sum(a[..., :3] for a in arrays['old'][f'ref{frame}']) / 2          # the previous renderer at 2048 samples
        new_reference = sum(a[..., :3] for a in arrays['new'][f'ref{frame}']) / 2
        core = reference.mean(-1) > np.quantile(reference.mean(-1), .98)
        agreement[str(frame)] = {
            'relative_mean_difference': float((new_reference.mean() - reference.mean()) / reference.mean()),
            'relative_mean_difference_flame_core': float((new_reference[core].mean() - reference[core].mean()) / reference[core].mean()),
            'mean_abs_difference': float(np.abs(new_reference - reference).mean()),
            'old_reference_two_seed_mean_abs_difference': float(np.abs(arrays['old'][f'ref{frame}'][0][..., :3]
                                                                      - arrays['old'][f'ref{frame}'][1][..., :3]).mean())}
    ratios = {str(f): runs['new']['frames'][str(f)]['seconds_median'] / runs['old']['frames'][str(f)]['seconds_median'] for f in FRAMES}
    old16, new16 = runs['old']['frames']['16'], runs['new']['frames']['16']
    gate = {'grain_at_frame_16_old': old16['grain_rms'], 'grain_at_frame_16_new': new16['grain_rms'],
            'grain_ratio': new16['grain_rms'] / old16['grain_rms'], 'grain_gate_at_most': old16['grain_rms'] / 2,
            'grain_with_the_fire_light_on_smoke_switched_off': new16['fire_light_off']['rgba']['grain_rms'],
            'render_time_ratio_median_by_frame': {k: round(v, 3) for k, v in ratios.items()}, 'render_time_gate_at_most': 1.3,
            'worst_relative_mean_difference_at_2048': max(abs(v['relative_mean_difference']) for v in agreement.values())}
    rows = [np.concatenate([downsample(arrays[e][f][0]) for f in FRAMES], axis=1) for e in ('old', 'new')]
    sheet_name = f'fire-smoke-grain-before-after-{args.tag}.png'
    write_png(images_dir / sheet_name, np.concatenate(rows, axis=0))
    crops = []
    for e in ('old', 'new'):
        crop = arrays[e][16][0][40:200, 330:650]
        crops.append(np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1))
    zoom_name = f'fire-smoke-grain-before-after-zoom-{args.tag}.png'
    write_png(images_dir / zoom_name, np.concatenate(crops, axis=0))
    result = {'date': args.date, 'device': runs['new']['device'], 'resolution': [args.width, args.height],
              'samples': args.samples, 'reference_samples': args.reference_samples, 'seeds': [1, 2],
              'gate': gate, 'agreement_with_old_estimator_at_reference_samples': agreement, 'old': runs['old'], 'new': runs['new'],
              'contact_sheet': f'docs/images/fire_shot/{sheet_name}', 'contact_sheet_rows': ['before', 'after'],
              'contact_sheet_columns': list(FRAMES), 'zoom': f'docs/images/fire_shot/{zoom_name}'}
    (REPO / 'benchmarks' / 'fire_shot' / f'fire-smoke-grain-{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(gate, indent=2), flush=True)


METRIC_FRAMES = tuple(range(8, 25))
FLAME_SWEEP = {
    # the solver controls, each read with the flame glow following the gas temperature (`SHAPED_RENDER`), except the first row
    'soft shot of the grain step (old flame colour)': dict(colour='old'),
    'old solver settings, glow follows temperature': dict(),
    'turbulence 3': dict(turbulence=3),
    'turbulence 3, swirl 6': dict(turbulence=3, swirl_size=6),
    'turbulence 8': dict(turbulence=8),
    'shredding 2': dict(shredding=2),
    'disturbance 1': dict(disturbance=1),
    'cooling 0.3': dict(cooling_rate=.3),
    'cooling 0.5': dict(cooling_rate=.5),
    'cooling 1.0': dict(cooling_rate=1.0),
    'cooling 2.0': dict(cooling_rate=2.0),
    'cooling 0.5, vorticity 1.5': dict(cooling_rate=.5, vorticity=1.5),
    'cooling 0.5, vorticity 2, expansion 2': dict(cooling_rate=.5, vorticity=2.0, burn_expansion=2.0),
    'cooling 0.5, vorticity 3, expansion 2': dict(cooling_rate=.5, vorticity=3.0, burn_expansion=2.0),
    'cooling 0.5, vorticity 1.5, expansion 2.5': dict(cooling_rate=.5, vorticity=1.5, burn_expansion=2.5),
    'cooling 1.0, vorticity 1.5, expansion 2, turbulence 1, swirl 3 (blows up after frame 30)': dict(
        cooling_rate=1.0, vorticity=1.5, burn_expansion=2.0, turbulence=1.0, swirl_size=3.0),
    'cooling 1.0, vorticity 1, expansion 2, turbulence 1, swirl 3 (frame 0.49 to 0.56 s)': dict(
        cooling_rate=1.0, vorticity=1.0, burn_expansion=2.0, turbulence=1.0, swirl_size=3.0),
    'cooling 1.0, vorticity 1, expansion 1.5, turbulence 1, swirl 3': dict(
        cooling_rate=1.0, vorticity=1.0, burn_expansion=1.5, turbulence=1.0, swirl_size=3.0),
    'cooling 1.0, vorticity 1, expansion 1.5': dict(cooling_rate=1.0, vorticity=1.0, burn_expansion=1.5),
    'burn rate 6, cooling 0.5, vorticity 1.5, expansion 2.5': dict(burn_rate=6, cooling_rate=.5, vorticity=1.5, burn_expansion=2.5),
    'shaped shot (this step): cooling 1.0, vorticity 1, expansion 1.25': dict(SHAPED_SOLVER),
}


def bake_and_measure(document, workdir, tag):
    """Bake a fire-shot document to frame 24 in a fresh cache; per-frame flame-shape numbers for frames 8 to 24 and the bake time."""
    evaluator = Evaluator(sim=simcache.SimCache(root=workdir / tag, memory_budget=1 << 30, disk_budget=8 << 30))
    with gpu_lock():
        started = time.perf_counter()
        evaluator.evaluate_raster(document, 'cache', frame=24, typed=True)
        bake_s = time.perf_counter() - started
    settings = _volume_settings(document['nodes']['render']['params'])
    frames = {}
    for frame in METRIC_FRAMES:
        with gpu_lock():
            volume = evaluator.evaluate_raster(document, 'cache', frame=frame, typed=True)
        frames[str(frame)] = flame_metrics.frame_metrics(volume, settings, floor_height=FUEL_TOP)
        frames[str(frame)]['colour'] = flame_metrics.colour_by_height(volume, settings, FUEL_TOP)
    return evaluator, bake_s, frames


def flame_sweep(args):
    """Field-only measurements of the solver controls on the 64-cell shot (no rendering): which setting moves the flame's profile,
    tongue count, smoke eddies and sway, and what each costs in bake time. The shaped shot's solver settings were picked from this."""
    results = {}
    for name, overrides in FLAME_SWEEP.items():
        overrides = dict(overrides)
        colour = overrides.pop('colour', 'follow')
        document = build_document(flame_shape=False, **overrides)
        if colour == 'follow':
            document['nodes']['render']['params'].update(SHAPED_RENDER)
        _, bake_s, frames = bake_and_measure(document, args.workdir, 'sweep-' + str(len(results)))
        results[name] = {'overrides': overrides, 'colour': colour, 'bake_seconds': round(bake_s, 2), **flame_metrics.summarise(frames)}
        print(name, json.dumps(results[name]), flush=True)
    benchmark = REPO / 'benchmarks' / 'fire_shot' / f'fire-shape-sweep-{args.tag}.json'
    benchmark.write_text(json.dumps({'date': args.date, 'cells': 64, 'frames': [METRIC_FRAMES[0], METRIC_FRAMES[-1]],
                                     'variants': results}, indent=2) + '\n')


def flame_time(args):
    """Warm frame times of one variant in a process of its own (`--flame-time before|after`), printed as JSON.

    Measured apart from the metrics pass: that pass reads every frame's volume and runs a lot of NumPy, and the frame times it leaves
    behind in the same process run 15 to 30 % slower than a clean process. Same sequence as the grain step's timing: one warm-up render,
    then five renders with new seeds (an identical request is answered from the image cache), median of the five."""
    document = build_document(args.width, args.height, flame_shape=args.flame_time == 'after')
    params = document['nodes']['render']['params']
    evaluator = Evaluator(sim=simcache.SimCache(root=args.workdir / ('shape-' + args.flame_time), memory_budget=1 << 30,
                                                disk_budget=8 << 30))
    out = {}
    with gpu_lock():
        evaluator.evaluate_raster(document, 'cache', frame=24, typed=True)
        for frame in FRAMES:
            params.update(render_output='rgba', pt_samples=128, pt_seed=1)
            evaluator.evaluate_raster(document, 'render', frame=frame)
            seconds = []
            for repeat in range(GRAIN_TIMING_REPEATS):
                params['pt_seed'] = 100 + repeat
                started = time.perf_counter()
                evaluator.evaluate_raster(document, 'render', frame=frame)
                seconds.append(time.perf_counter() - started)
            out[str(frame)] = {'seconds_median': round(float(np.median(seconds)), 3), 'seconds_min': round(min(seconds), 3),
                               'seconds_all': [round(x, 3) for x in seconds]}
    print(json.dumps(out))


def flame_shape(args):
    """The soft shot (grain step) against the shaped shot: flame-shape numbers, bake and frame times, stills and a contact sheet."""
    images_dir = REPO / 'docs' / 'images' / 'fire_shot'
    images_dir.mkdir(parents=True, exist_ok=True)
    runs, pictures = {}, {}
    for name, flame_shape_on in (('before', False), ('after', True)):
        document = build_document(args.width, args.height, flame_shape=flame_shape_on)
        params = document['nodes']['render']['params']
        evaluator, bake_s, frames = bake_and_measure(document, args.workdir, 'shape-' + name)
        run = {'bake_seconds': round(bake_s, 3), 'frames': frames, 'summary': flame_metrics.summarise(frames), 'render': {}}
        pictures[name] = {}
        for frame in FRAMES:
            with gpu_lock():
                params.update(render_output='rgba', pt_samples=128, pt_seed=1)
                pixels = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
                params['pt_seed'] = 2
                other = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
                params.update(render_output='emission', pt_seed=1)
                emission = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
                params.update(render_output='rgba', pt_seed=1)
            frames[str(frame)]['image'] = flame_metrics.image_rows(emission)
            run['render'][str(frame)] = {'grain_rms': float(np.sqrt(((pixels[..., :3] - other[..., :3]) ** 2).mean() / 2)),
                                         'mean_linear_rgb': float(pixels[..., :3].mean())}
            pictures[name][frame] = pixels
            write_png(images_dir / f'fire-shape-{name}-{frame:03d}.png', pixels)
        timing = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--workdir', str(args.workdir), '--width', str(args.width),
                                 '--height', str(args.height), '--flame-time', name], check=True, capture_output=True, text=True)
        for frame, numbers in json.loads(timing.stdout.strip().splitlines()[-1]).items():
            run['render'][frame].update(numbers)
        runs[name] = run
        print(name, json.dumps({'bake': run['bake_seconds'], **run['summary']}), flush=True)
    rows = [np.concatenate([downsample(pictures[n][f]) for f in FRAMES], axis=1) for n in ('before', 'after')]
    sheet_name = f'fire-shape-before-after-{args.tag}.png'
    write_png(images_dir / sheet_name, np.concatenate(rows, axis=0))
    with gpu_lock():
        device = gpu3d._state()['info'].get('device', 'unknown')
    result = {'date': args.date, 'device': device, 'resolution': [args.width, args.height], 'samples': 128, 'before': runs['before'],
              'after': runs['after'], 'contact_sheet': f'docs/images/fire_shot/{sheet_name}',
              'contact_sheet_rows': ['before', 'after'], 'contact_sheet_columns': list(FRAMES)}
    (REPO / 'benchmarks' / 'fire_shot' / f'fire-shape-{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True, help='disposable simulation cache outside the repository')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--samples', type=int, default=None, help='override the document sample count')
    parser.add_argument('--compare', action='store_true', help='render 32 and 64 cells at 32 and 128 samples')
    parser.add_argument('--grain', action='store_true', help='compare the old and new fire-emission estimators on the shot')
    parser.add_argument('--grain-estimator', choices=('old', 'new'), help=argparse.SUPPRESS)
    parser.add_argument('--smoke-grain', action='store_true', help='fire sampled as a light at smoke vertices against the previous renderer')
    parser.add_argument('--smoke-grain-estimator', choices=('old', 'new'), help=argparse.SUPPRESS)
    parser.add_argument('--reference-samples', type=int, default=2048)
    parser.add_argument('--flame', action='store_true', help='soft shot against the shaped flame: profile, tongues, eddies, times, stills')
    parser.add_argument('--flame-time', choices=('before', 'after'), help=argparse.SUPPRESS)
    parser.add_argument('--flame-sweep', action='store_true', help='field measurements of the solver controls on the 64-cell shot')
    parser.add_argument('--curl-sweep', action='store_true', help='render 64 cells with sideways curl strengths')
    parser.add_argument('--tag', default='2026-10-30')
    parser.add_argument('--date', default='2026-10-30')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.smoke_grain_estimator:
        args.samples = args.samples or 128
        smoke_grain_estimator(args)
        return
    if args.smoke_grain:
        args.samples = args.samples or 128
        smoke_grain(args)
        return
    if args.grain_estimator:
        args.samples = args.samples or 128
        grain_estimator(args)
        return
    if args.grain:
        args.samples = args.samples or 128
        grain(args)
        return
    if args.flame_time:
        flame_time(args)
        return
    if args.flame_sweep:
        flame_sweep(args)
        return
    if args.flame:
        flame_shape(args)
        return
    if args.curl_sweep:
        curl_sweep(args)
        return
    if args.compare:
        with gpu_lock():
            args.device = gpu3d._state()['info'].get('device', 'unknown')
        compare(args)
        return
    images = REPO / 'docs' / 'images' / 'fire_shot'
    images.mkdir(parents=True, exist_ok=True)
    document = load_document(ROOT / 'fire_shot.nbcomp')
    params = document['nodes']['render']['params']
    params.update(width=args.width, height=args.height, sampling='fixed')
    if args.samples:
        params['pt_samples'] = args.samples
    samples = int(params['pt_samples'])
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
        write_png(images / f'fire-look-{frame:03d}.png', pixels)
        with gpu_lock():
            volume = evaluator.evaluate_raster(document, 'cache', frame=frame, typed=True)
        frames[str(frame)] = {'seconds': round(seconds, 3), 'path': f'docs/images/fire_shot/fire-look-{frame:03d}.png',
                              'mean_linear_rgb': float(pixels[..., :3].mean()), 'flame': flame_occupancy(volume)}
        if frame == 16:
            on = pixels
            params['pt_seed'] = 2
            with gpu_lock():
                other = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
            params['pt_seed'] = 1
            frames[str(frame)]['grain_rms'] = float(np.sqrt(((pixels[..., :3] - other[..., :3]) ** 2).mean() / 2))
    off_document = load_document(ROOT / 'fire_shot.nbcomp')
    off_document['nodes']['render']['params'].update(width=args.width, height=args.height,
                                                     pt_samples=samples, sampling='fixed',
                                                     volume_fire_light_meshes=0.0)
    started = time.perf_counter()
    with gpu_lock():
        off = evaluator.evaluate_raster(off_document, 'render', frame=16).pixels.copy()
    off_s = time.perf_counter() - started
    write_png(images / 'fire-look-016-mesh-light-off.png', off)
    solver = document['nodes']['fire']['params']
    result = {'date': args.date, 'device': device, 'resolution': [args.width, args.height],
              'cells': round(2.0 / solver['division_size']), 'samples_per_pixel': samples, 'bake_seconds': round(bake_s, 3), 'frames': frames,
              'comparison_off_seconds': round(off_s, 3),
              'comparison_mean_linear_rgb_delta': float((on - off)[..., :3].mean()),
              'comparison_off_path': 'docs/images/fire_shot/fire-look-016-mesh-light-off.png'}
    benchmark = REPO / 'benchmarks' / 'fire_shot' / f'fire-look-shot-{args.tag}.json'
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
