"""Fluids 18 A: sweep the main fire simulation (no up-res) to find where the smoke's curl has to come from.

Each setting runs in a process of its own (`--one`), so the card's peak memory is that setting's alone; the driver (`--sweep`) polls
`nvidia-smi` while it runs. Per setting: bake time for frames 1 to 24, the flame and smoke measures of frames 8 to 24, the image-plane
curl of the velocity that carries the smoke (frame 16, `flame_metrics.projected_curl` on the main simulation's own grid), stills at
frames 8, 16 and 24 (two noise seeds, for grain and smoke structure against the shot as it is today) and the 128-sample frame time.

    python base_sweep.py --sweep --workdir <scratch dir> --variants shot v0.3 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

DIVISION = .03125      # the shot's 64 cells over the 2 metre box
UPRES_C = {'swirl': .5, 'swirl_scale': 6.0, 'swirl_life': 2.0}      # Fluids 17 C's best setting

# name -> overrides on the shot's solver (`make_shot.build_document(**fire_overrides)`). The shot today is vorticity 1.0 (`SHAPED_SOLVER`),
# 64 cells, MacCormack. The brief's vorticity list is 0.3 (the node's default), 0.6, 1.0 and 2.0; the resolution list is the shot's size and
# 1.5 and 2 times finer.
VARIANTS = {
    'shot': dict(auto_substeps=0),      # the shot as it was before Fluids 18 C: one substep a frame (a new node has the automatic rule on)
    'v0.3': dict(vorticity=.3),
    'v0.6': dict(vorticity=.6),
    'v2.0': dict(vorticity=2.0),
    'v3.0': dict(vorticity=3.0),
    'v4.0': dict(vorticity=4.0),
    'r1.5': dict(division_size=DIVISION / 1.5),
    'r2': dict(division_size=DIVISION / 2),
    'r1.5_v2.0': dict(division_size=DIVISION / 1.5, vorticity=2.0),
    'r2_v2.0': dict(division_size=DIVISION / 2, vorticity=2.0),
    'r2_v4.0': dict(division_size=DIVISION / 2, vorticity=4.0),
    'semi': dict(advection='semi_lagrangian'),
    # Beyond the brief's list: the solver's own turbulence control (existing knobs), at the shot's size, to see whether a mechanism that is already built
    # curls the smoke. Not a recommendation: the flame sweep of Fluids 13 B found turbulence changes the flame.
    # Fluids 17 C's best up-res swirl (factor 2, swirl 0.5, life 2) on top of the finer main grids
    'r2_upresC': dict(division_size=DIVISION / 2, smoke_upres=UPRES_C),
    'r1.5_upresC': dict(division_size=DIVISION / 1.5, smoke_upres=UPRES_C),
    # Fluids 18 B: more substeps per frame (the node's own `substeps`, 1 today) so the travel per step stays the shot's 64-cell value on finer grids
    's2': dict(substeps=2),
    's4': dict(substeps=4),
    'r1.5_s2': dict(division_size=DIVISION / 1.5, substeps=2),
    'r2_s2': dict(division_size=DIVISION / 2, substeps=2),
    'r2_s3': dict(division_size=DIVISION / 2, substeps=3),
    # Fluids 18 C: the automatic substep rule (`auto_substeps`) against the old one substep a frame, at the shot's three grid sizes
    'off96': dict(division_size=DIVISION / 1.5, auto_substeps=0), 'off128': dict(division_size=DIVISION / 2, auto_substeps=0),
    'auto64': dict(auto_substeps=1), 'auto96': dict(division_size=DIVISION / 1.5, auto_substeps=1), 'auto128': dict(division_size=DIVISION / 2, auto_substeps=1),
    # and the flame's own knobs on the automatic rule at 64 cells (burn rate 1.2 and flame lifespan 1 in the shot; cooling 1.0 per frame)
    'auto64_burn.6': dict(auto_substeps=1, burn_rate=.6), 'auto64_burn.3': dict(auto_substeps=1, burn_rate=.3),
    'auto64_life2': dict(auto_substeps=1, flame_lifespan=2.0), 'auto64_cool.5': dict(auto_substeps=1, cooling_rate=.5),
    'auto64_target3': dict(auto_substeps=1, substep_target=3.0), 'auto64_target4': dict(auto_substeps=1, substep_target=4.0),
    'auto64_burn.9': dict(auto_substeps=1, burn_rate=.9), 'auto64_burn.8': dict(auto_substeps=1, burn_rate=.8),
    'auto64_heat2': dict(auto_substeps=1, temperature_output=2.0), 'auto64_heat2.4': dict(auto_substeps=1, temperature_output=2.4),
    'auto64_exp2.5': dict(auto_substeps=1, burn_expansion=2.5), 'auto64_burn.9_exp2.5': dict(auto_substeps=1, burn_rate=.9, burn_expansion=2.5),
    # Fluids 18 C part 4: Fluids 17 C's carried swirl (factor 2, swirl 0.5, life 2) on the shot as it was and on the automatic rule
    'off64_upresC': dict(auto_substeps=0, smoke_upres=UPRES_C),
    'auto64_upresC': dict(auto_substeps=1, smoke_upres=UPRES_C), 'auto96_upresC': dict(division_size=DIVISION / 1.5, auto_substeps=1, smoke_upres=UPRES_C),
    'auto64_heat2.3': dict(auto_substeps=1, temperature_output=2.3), 'auto96_heat2.3': dict(division_size=DIVISION / 1.5, auto_substeps=1, temperature_output=2.3),
    'auto128_heat2.3': dict(division_size=DIVISION / 2, auto_substeps=1, temperature_output=2.3),
    'auto64h_upresC': dict(auto_substeps=1, temperature_output=2.3, smoke_upres=UPRES_C),
    'auto96h_upresC': dict(division_size=DIVISION / 1.5, auto_substeps=1, temperature_output=2.3, smoke_upres=UPRES_C),
    # the cap, not the target, binds in the shot (the fastest cell needs 17 to 33 substeps at 2 cells each): 16 and 32 at 64 cells, 16 at 128
    'auto64h_cap16': dict(auto_substeps=1, temperature_output=2.3, substep_cap=16), 'auto64h_cap32': dict(auto_substeps=1, temperature_output=2.3, substep_cap=32),
    'auto128h_cap16': dict(division_size=DIVISION / 2, auto_substeps=1, temperature_output=2.3, substep_cap=16),
    't1_s3': dict(turbulence=1.0, swirl_size=3.0),
    't3_s3': dict(turbulence=3.0, swirl_size=3.0),
    't3_s6': dict(turbulence=3.0, swirl_size=6.0),
}


def one(args):
    from nodebased import gpu3d, simcache
    from nodebased.imaging import Evaluator, write_png, _volume_settings
    import flame_metrics
    from make_shot import FUEL_TOP, build_document
    import render_evidence as re_

    # `col_<name>`: the same solver settings on a plain hot smoke column (no fire, dense smoke from the same source) instead of the fire, to see
    # whether the main solver rolls smoke that has a clear edge.
    column = args.name.startswith('col_')
    base_name = args.name[4:] if column else args.name
    overrides = dict(VARIANTS[base_name]) if base_name in VARIANTS else json.loads(base_name)
    document = build_document(args.width, args.height, **overrides)
    if column:
        document['nodes']['fuel']['params'].update(src_fuel=0.0, src_density=1.0, src_temperature=1.5)
        document['nodes']['fire']['params'].update(fire=0)
    params = document['nodes']['render']['params']
    workdir = args.workdir / ('base-' + args.name)
    evaluator = Evaluator(sim=simcache.SimCache(root=workdir, memory_budget=1 << 30, disk_budget=16 << 30))
    settings = _volume_settings(params)
    out = {'overrides': overrides}
    with re_.gpu_lock():
        started = time.perf_counter()
        evaluator.evaluate_raster(document, 'cache', frame=24, typed=True)
        out['bake_seconds'] = round(time.perf_counter() - started, 3)
        out['device'] = gpu3d._state()['info'].get('device', 'unknown')
    if 'upres' in document['nodes']:
        with re_.gpu_lock():
            started = time.perf_counter()
            evaluator.evaluate_raster(document, 'upres', frame=24, typed=True)
            out['upres_bake_seconds'] = round(time.perf_counter() - started, 3)
    if args.bake_only:
        print('RESULT ' + json.dumps({'overrides': overrides, 'bake_seconds': out['bake_seconds'], 'device': out['device']}), flush=True)
        return
    frames = {}
    for frame in re_.METRIC_FRAMES:
        with re_.gpu_lock():
            volume = evaluator.evaluate_raster(document, 'cache', frame=frame, typed=True)
        frames[str(frame)] = flame_metrics.frame_metrics(volume, settings, floor_height=FUEL_TOP)
        if frame == 16:
            density = np.asarray(volume.density if volume.density is not None else volume.to_sparse().to_dense()['density'])
            velocity = np.asarray(volume.velocity)
            camera = document['nodes']['camera']['params']
            out['projected_curl'] = flame_metrics.projected_curl(
                density, velocity, np.asarray(volume.origin, np.float64), float(volume.voxel_size),
                (camera['tx'], camera['ty'], camera['tz']), (0.0, camera['target_y'], 0.0))
            out['grid'] = list(density.shape)
    out['summary'] = flame_metrics.summarise(frames)
    images_dir = args.images
    images_dir.mkdir(parents=True, exist_ok=True)
    pictures = {}
    render = {}
    for frame in re_.FRAMES:
        with re_.gpu_lock():
            params.update(render_output='rgba', pt_samples=128, pt_seed=1)
            first = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
            params['pt_seed'] = 2
            second = evaluator.evaluate_raster(document, 'render', frame=frame).pixels.copy()
        pictures[frame] = (first, second)
        write_png(images_dir / f'fire-base-{args.name}-{frame:03d}.png', first)
        render[str(frame)] = {'grain_rms': float(np.sqrt(((first[..., :3] - second[..., :3]) ** 2).mean() / 2))}
    np.savez_compressed(workdir.parent / f'pics-{args.name}.npz',
                        **{f'{f}_{i}': pictures[f][i] for f in pictures for i in (0, 1)})
    reference_file = workdir.parent / 'pics-shot.npz'
    if reference_file.exists():
        reference = np.load(reference_file)
        for frame in re_.FRAMES:
            mask = flame_metrics.smoke_mask(reference[f'{frame}_0'])
            render[str(frame)]['smoke_image'] = flame_metrics.smoke_structure(*pictures[frame], mask)
            render[str(frame)]['smoke_image_own_mask'] = flame_metrics.smoke_structure(*pictures[frame], flame_metrics.smoke_mask(pictures[frame][0]))
    else:
        for frame in re_.FRAMES:
            render[str(frame)]['smoke_image'] = flame_metrics.smoke_structure(*pictures[frame], flame_metrics.smoke_mask(pictures[frame][0]))
    # 128-sample frame time: one warm-up, then five renders with new seeds, median.
    with re_.gpu_lock():
        for frame in re_.FRAMES:
            params.update(render_output='rgba', pt_samples=128, pt_seed=1)
            evaluator.evaluate_raster(document, 'render', frame=frame)
            seconds = []
            for repeat in range(5):
                params['pt_seed'] = 100 + repeat
                started = time.perf_counter()
                evaluator.evaluate_raster(document, 'render', frame=frame)
                seconds.append(time.perf_counter() - started)
            render[str(frame)].update(seconds_median=round(float(np.median(seconds)), 3), seconds_min=round(min(seconds), 3))
    out['render'] = render
    print('RESULT ' + json.dumps(out), flush=True)


def curl_only(args):
    """`--curl-only`: bake and measure the frame-16 curl two ways (the main grid's own, and Fluids 17's: velocity and smoke trilinearly put on a
    grid twice as fine, which is what the up-res measure sees), plus the field contrast, and save a mid-plane slice of the smoke density.
    `upresC` is Fluids 17 C's best (swirl 0.5, life 2) through the up-res pass, measured as that step measured it."""
    from nodebased import fluid3d, simcache
    from nodebased.imaging import Evaluator
    import flame_metrics
    from make_shot import build_document
    import render_evidence as re_
    upres = args.name.endswith('upresC')
    if upres:
        overrides = dict(VARIANTS[args.name]) if args.name in VARIANTS else {}
        document = build_document(args.width, args.height, **({'smoke_upres': UPRES_C} | overrides))
    else:
        overrides = dict(VARIANTS[args.name])
        document = build_document(args.width, args.height, **overrides)
    workdir = args.workdir / ('curl-' + args.name)
    evaluator = Evaluator(sim=simcache.SimCache(root=workdir, memory_budget=1 << 30, disk_budget=16 << 30))
    with re_.gpu_lock():
        started = time.perf_counter()
        evaluator.evaluate_raster(document, 'upres' if upres else 'cache', frame=24, typed=True)
        bake = time.perf_counter() - started
        volume = evaluator.evaluate_raster(document, 'cache', frame=16, typed=True)
    density = np.asarray(volume.density if volume.density is not None else volume.to_sparse().to_dense()['density'])
    velocity = np.asarray(volume.velocity)
    camera = document['nodes']['camera']['params']
    cam, target = (camera['tx'], camera['ty'], camera['tz']), (0.0, camera['target_y'], 0.0)
    out = {'overrides': {k: v for k, v in overrides.items()}, 'bake_seconds': round(bake, 3), 'grid': list(density.shape)}
    out['own_grid'] = flame_metrics.projected_curl(density, velocity, np.asarray(volume.origin, np.float64), float(volume.voxel_size), cam, target)
    if upres:
        out['fluids17_measure'] = re_.effective_curl(evaluator, document, UPRES_C)
        fine = evaluator.evaluate_raster(document, 'upres', frame=16, typed=True)
        grid = fine.sparse if fine.sparse is not None else fine.to_sparse()
        density = grid.to_dense()['density']
    else:
        # the main grid's fields on a grid twice as fine, cropped to where there is smoke (the same box rule the up-res measure uses)
        factor = 2
        live = np.argwhere(density > 1e-3 * density.max())
        lo, hi = np.maximum(live.min(0) - 2, 0), np.minimum(live.max(0) + 3, density.shape)
        axes = [np.arange(lo[a] * factor, hi[a] * factor, dtype=np.float32) for a in range(3)]
        ratio = [(density.shape[a] - 1) / (density.shape[a] * factor - 1) for a in range(3)]
        q = [np.clip(axes[a] * np.float32(ratio[a]), 0, density.shape[a] - 1) for a in range(3)]
        qx, qy, qz = np.meshgrid(*q, indexing='ij')
        fine_density = fluid3d.trilerp(density, qx, qy, qz)
        fine_velocity = np.stack([fluid3d.trilerp(velocity[..., a], qx, qy, qz) for a in range(3)], axis=-1)
        voxel = float(volume.voxel_size) / factor
        origin = np.asarray(volume.origin, np.float64) + lo * float(volume.voxel_size)
        out['fluids17_measure'] = flame_metrics.projected_curl(fine_density, fine_velocity, origin, voxel, cam, target)
    mid = density[:, :, density.shape[2] // 2]
    np.savez_compressed(args.workdir / f'slice-{args.name}.npz', density=mid.astype(np.float32))
    contrast = flame_metrics.frame_metrics(volume, __import__('nodebased.imaging', fromlist=['_volume_settings'])._volume_settings(
        document['nodes']['render']['params']), floor_height=.21)['eddies']['density_contrast']
    out['density_contrast_16'] = contrast
    print('RESULT ' + json.dumps(out), flush=True)
    import shutil
    shutil.rmtree(workdir, ignore_errors=True)


def poll_memory(stop, peak):
    while not stop.is_set():
        try:
            text = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                  timeout=5).stdout
            peak[0] = max(peak[0], int(text.split()[0]))
        except Exception:
            pass
        stop.wait(.4)


def sweep(args):
    results_file = REPO / 'benchmarks' / 'fire_shot' / f'fire-base-sweep-{args.tag}.json'
    results = json.loads(results_file.read_text())['variants'] if results_file.exists() and not args.fresh else {}
    idle = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True,
                              text=True).stdout.split()[0])
    for name in args.variants:
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        print(name, 'start; other compute apps on the card:', busy or 'none', flush=True)
        stop, peak = threading.Event(), [0]
        thread = threading.Thread(target=poll_memory, args=(stop, peak), daemon=True)
        thread.start()
        started = time.perf_counter()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--one', '--name', name, '--workdir', str(args.workdir),
                               '--images', str(args.images), '--width', str(args.width), '--height', str(args.height)] + (['--bake-only'] if args.bake_only else []),
                              capture_output=True, text=True)
        stop.set()
        thread.join()
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
        if not lines:
            print(name, 'FAILED', proc.stderr[-600:], flush=True)
            continue
        row = json.loads(lines[-1][7:])
        row['peak_gpu_mib'] = peak[0]
        row['peak_gpu_mib_over_idle'] = peak[0] - idle
        row['wall_seconds'] = round(time.perf_counter() - started, 1)
        results[name] = row
        if args.bake_only:
            print(name, 'bake-only', row['bake_seconds'], 'peak over idle', row['peak_gpu_mib_over_idle'], flush=True)
            continue
        brief = {k: row[k] for k in ('bake_seconds', 'projected_curl', 'peak_gpu_mib_over_idle')}
        brief['contrast'] = row['summary']['density_contrast_mean']
        brief['frame16'] = {k: row['render']['16'].get(k) for k in ('grain_rms', 'seconds_median')}
        brief['structure16'] = row['render']['16']['smoke_image']['structure']
        print(name, json.dumps(brief), flush=True)
        results_file.write_text(json.dumps({'date': args.date, 'resolution': [args.width, args.height], 'samples': 128,
                                            'idle_gpu_mib': idle, 'variants': results}, indent=2) + '\n')
        import shutil
        shutil.rmtree(args.workdir / ('base-' + name), ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--one', action='store_true')
    parser.add_argument('--curl-only', action='store_true', help='one setting: both curl measures and a density slice (see curl_only)')
    parser.add_argument('--name')
    parser.add_argument('--variants', nargs='*', default=list(VARIANTS))
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=REPO / 'docs' / 'images' / 'fire_shot')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--tag', default='2026-11-20')
    parser.add_argument('--date', default='2026-11-20')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--bake-only', action='store_true', help='bake and time only (warm repeat of the bake time and peak memory)')
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.curl_only:
        curl_only(args)
    elif args.one:
        one(args)
    else:
        sweep(args)


if __name__ == '__main__':
    main()
