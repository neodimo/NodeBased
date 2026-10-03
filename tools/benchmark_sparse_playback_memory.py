"""Measure process-resident field memory for dense and 8-cubed sparse plume caches."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def child(kind, size):
    from nodebased.sparsevol import SparseGrid, TILE

    def rss():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Linux /proc did not report resident memory")

    before = rss()
    n = int(size)
    lo, hi = int(round(n * 0.275)), int(round(n * 0.725))
    if kind == "dense":
        # Match SparseGrid.to_dense's former np.full allocation, which commits the complete dense frame.
        field = np.full((n, n, n), 0.0, np.float32)
        field[lo:hi, lo:hi, lo:hi] = 1.0
        retained = field
    else:
        tile_lo, tile_hi = lo // TILE, -(-hi // TILE)
        coords = np.asarray([(x, y, z) for x in range(tile_lo, tile_hi)
                             for y in range(tile_lo, tile_hi) for z in range(tile_lo, tile_hi)], np.int32)
        blocks = np.zeros((len(coords), TILE, TILE, TILE), np.float32)
        for i, coord in enumerate(coords):
            origin = coord * TILE
            a = np.maximum(lo - origin, 0)
            b = np.minimum(hi - origin, TILE)
            if np.all(b > a):
                blocks[(i, slice(a[0], b[0]), slice(a[1], b[1]), slice(a[2], b[2]))] = 1.0
        retained = SparseGrid((n, n, n), coords, {"density": blocks})
    gc.collect()
    after = rss()
    print(json.dumps({"kind": kind, "size": n, "rss_before": before, "rss_after": after,
                      "resident_delta": after - before,
                      "sparse_field_bytes": retained.nbytes if kind == "sparse" else None}))


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]))
        return
    results = []
    for size in (128, 256):
        for kind in ("dense", "sparse"):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", kind, str(size)],
                                    check=True, capture_output=True, text=True)
            results.append(json.loads(result.stdout))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
