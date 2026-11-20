"""Fluids 18 C: a contact sheet of base-sweep stills (rows = variants in the order given, columns = frames 8, 16 and 24) from the sweep's
`pics-<name>.npz` files (linear RGBA, two noise seeds; the first is used), written through the app's own PNG export.

    python contact_sheet.py --workdir <sweep workdir> --rows shot auto64 auto128 --out docs/images/fire_shot/<name>.png [--halve 1] [--crop x0 y0 x1 y1]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FRAMES = (8, 16, 24)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--rows', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--halve', type=int, default=1, help='times the stills are halved (2 by 2 averages)')
    parser.add_argument('--crop', type=float, nargs=4, default=None, help='fractions x0 y0 x1 y1 of each still, applied before halving')
    parser.add_argument('--gain', type=float, default=1.0, help='brighten by this factor (linear) to read faint smoke')
    parser.add_argument('--frames', type=int, nargs='*', default=list(FRAMES))
    args = parser.parse_args()
    from nodebased.imaging import write_png
    rows = []
    for name in args.rows:
        data = np.load(args.workdir / f'pics-{name}.npz')
        tiles = []
        for frame in args.frames:
            image = np.asarray(data[f'{frame}_0'], np.float32)
            if args.crop:
                h, w = image.shape[:2]
                image = image[int(args.crop[1] * h):int(args.crop[3] * h), int(args.crop[0] * w):int(args.crop[2] * w)]
            for _ in range(args.halve):
                h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
                image = image[:h, :w].reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3))
            image = image.copy()
            image[..., :3] *= args.gain
            tiles.append(image)
        rows.append(np.concatenate(tiles, axis=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.out, np.concatenate(rows, axis=0))
    print(args.out)


if __name__ == '__main__':
    main()
