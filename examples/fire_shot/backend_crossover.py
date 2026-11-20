"""Fluids 18 C: bake the fire shot at 32, 48, 64, 96 and 128 cells on the processor and on a card, to find where the card wins.

One process per adapter (`--adapter default|integrated|cpu`: the RTX 3080 Ti, the Radeon, llvmpipe), run under
`flock -x /tmp/nb-gpu.lock` by the caller (the lock is not taken here). Each setting is baked up to twice from an empty cache (frames 1 to 24,
one substep per frame, the shot's own solver settings) and the faster time is kept; a processor bake that takes over a minute is not repeated.

    flock -x /tmp/nb-gpu.lock python backend_crossover.py --adapter default --scratch <dir> --out <file.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CELLS = (32, 48, 64, 96, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='default', choices=('default', 'integrated', 'cpu'))
    parser.add_argument('--cells', type=int, nargs='*', default=list(CELLS))
    parser.add_argument('--backends', nargs='*', default=['cpu', 'resident'])
    parser.add_argument('--cpu-max', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--substeps', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()

    from nodebased import gpu3d, simcache
    original = gpu3d._state
    kind = args.adapter
    gpu3d._state = lambda choice=None: original(kind if (choice or 'default') == 'default' else choice)
    info = gpu3d._state().get('info', {})
    device = str(info.get('device') or info.get('description') or 'unnamed')
    print('adapter:', device, flush=True)

    from nodebased.imaging import Evaluator
    from make_shot import build_document

    rows = []
    for cells in args.cells:
        for backend in args.backends:
            if backend == 'cpu' and cells > args.cpu_max:
                continue
            best, every = None, []
            for repeat in range(args.repeats):
                document = build_document(320, 180, division_size=2.0 / cells, auto_substeps=0, substeps=args.substeps, pressure=backend)
                work = Path(tempfile.mkdtemp(prefix='xover-', dir=args.scratch))
                evaluator = Evaluator(sim=simcache.SimCache(root=work, memory_budget=1 << 30, disk_budget=8 << 30))
                started = time.perf_counter()
                evaluator.evaluate_raster(document, 'cache', frame=24, typed=True)
                seconds = time.perf_counter() - started
                shutil.rmtree(work, ignore_errors=True)
                every.append(round(seconds, 3))
                best = seconds if best is None else min(best, seconds)
                if seconds > 60:
                    break
            rows.append({'cells': cells, 'backend': backend, 'seconds': round(best, 3), 'all': every})
            print(json.dumps(rows[-1]), flush=True)
    args.out.write_text(json.dumps({'adapter': device, 'kind': kind, 'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
