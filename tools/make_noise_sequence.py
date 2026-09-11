"""Generate the moving-noise EXR test sequence used for NodeBased playback work.

The first version of this sequence was produced ad hoc and never written down,
which made the output impossible to reproduce or adjust. It also landed as
32-bit float: 93 MB a frame, 9.3 GB for 100 frames, which is absurd for test
footage. This script is the reproducible replacement.

Two choices do the heavy lifting on size:

* ``half`` instead of ``float``. Scene-linear test noise has nowhere near
  24 bits of meaningful mantissa, so the extra precision was pure cost.
* ``zips`` (single-scanline ZIP) instead of the writer default. ZIPS is the
  friendlier variant for a viewer that pulls individual scanlines, which is
  exactly what this footage exists to exercise.

The pattern is multi-octave value noise on a coarse lattice, smoothly
upsampled. That matters for more than looks: per-pixel white noise is
incompressible and would defeat the point of choosing a compression at all.
The time axis wraps, so frame 100 tiles back onto frame 1 and the sequence can
loop in a player without a visible seam.

Usage:
    .venv/bin/python tools/make_noise_sequence.py
    .venv/bin/python tools/make_noise_sequence.py --frames 24 --width 1920 --height 1080
"""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

SEED = 20260911
MID_GREY = 0.18
CONTRAST = 3.0
EXPOSURE_STOPS = 8.0
# Coarse -> fine. Each entry is (lattice_w, lattice_h, lattice_t, amplitude).
# The lattice stays 16:9-ish so the noise features do not come out stretched.
OCTAVES = [(8, 5, 4, 1.0), (16, 9, 8, 0.5), (32, 18, 16, 0.25),
           (64, 36, 24, 0.125), (128, 72, 32, 0.0625)]


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def upsample_weights(lattice, size):
    """Index/weight pairs that blow a wrapping lattice axis up to `size`.

    Computed once per octave and reused for every frame; recomputing this per
    frame was the obvious way to make a 100-frame render take all afternoon.
    """
    positions = (np.arange(size, dtype=np.float64) + 0.5) * lattice / size - 0.5
    low = np.floor(positions).astype(np.int64)
    weight = smoothstep(positions - low).astype(np.float32)
    return low % lattice, (low + 1) % lattice, weight


def value_noise(width, height, frames, channels=3):
    """Render the whole sequence as a generator of (frames, height, width, ch)."""
    rng = np.random.default_rng(SEED)
    layers = []
    for lat_w, lat_h, lat_t, amplitude in OCTAVES:
        # One independent lattice per channel gives R/G/B genuinely different
        # structure, so a Grade or channel-shuffle node has something to bite on.
        grid = rng.random((lat_t, lat_h, lat_w, channels), dtype=np.float32)
        layers.append((grid, amplitude,
                       upsample_weights(lat_w, width), upsample_weights(lat_h, height)))
    total = sum(amplitude for _, amplitude, _, _ in layers)

    for frame in range(frames):
        accumulated = np.zeros((height, width, channels), np.float32)
        for grid, amplitude, (x0, x1, wx), (y0, y1, wy) in layers:
            lat_t = grid.shape[0]
            # Wrapping time axis: the sequence loops cleanly at `frames`.
            position = frame / frames * lat_t
            z0 = int(np.floor(position)) % lat_t
            z1 = (z0 + 1) % lat_t
            wz = smoothstep(np.float32(position - np.floor(position)))
            plane = grid[z0] * (1.0 - wz) + grid[z1] * wz

            rows = plane[y0] * (1.0 - wy)[:, None, None] + plane[y1] * wy[:, None, None]
            cells = rows[:, x0] * (1.0 - wx)[None, :, None] + rows[:, x1] * wx[None, :, None]
            accumulated += cells * amplitude
        yield frame, accumulated / total


def write_half_exr(path, frame, compression):
    """Write RGBA half-float EXR.

    Deliberately not `nodebased.media.write_exr`: that one is the product's
    export path and is pinned to 32-bit float on purpose. Test footage should
    not be a reason to loosen an export guarantee.
    """
    import OpenImageIO as oiio
    from nodebased.media import WORKING

    path = Path(path)
    handle, temporary = tempfile.mkstemp(prefix='.' + path.stem, suffix='.exr', dir=path.parent)
    os.close(handle)
    writer = None
    try:
        spec = oiio.ImageSpec(frame.shape[1], frame.shape[0], 4, oiio.HALF)
        spec.channelnames = ['R', 'G', 'B', 'A']
        spec.attribute('oiio:ColorSpace', WORKING)
        spec.attribute('compression', compression)
        writer = oiio.ImageOutput.create(temporary)
        if writer is None or not writer.open(temporary, spec):
            raise ValueError('Cannot open EXR output: ' + oiio.geterror())
        if not writer.write_image(frame):
            raise ValueError('Cannot write EXR: ' + writer.geterror())
        if not writer.close():
            raise ValueError('Cannot finalize EXR: ' + writer.geterror())
        writer = None
        os.replace(temporary, path)
    finally:
        if writer:
            writer.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=3840)
    parser.add_argument('--height', type=int, default=2160)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--start', type=int, default=1, help='first frame number in the filename')
    parser.add_argument('--compression', default='zips', choices=['zips', 'zip'])
    parser.add_argument('--name', default='noise_test_4k')
    parser.add_argument('--outdir', default='.')
    arguments = parser.parse_args()

    outdir = Path(arguments.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    written = 0
    for index, rgb in value_noise(arguments.width, arguments.height, arguments.frames):
        # Summed octaves pile up around 0.5 by central limit, so the raw pattern is a
        # narrow grey band. Expand it first, then place it on a scene-linear exposure
        # ramp centred on 18% grey. The result spans EXPOSURE_STOPS stops, which is
        # what makes it worth anything as footage for exercising a Grade node.
        expanded = np.clip(0.5 + (rgb - 0.5) * CONTRAST, 0.0, 1.0)
        rgb = (MID_GREY * 2.0 ** ((expanded - 0.5) * EXPOSURE_STOPS)).astype(np.float32)
        frame = np.empty(rgb.shape[:2] + (4,), np.float32)
        frame[..., :3] = rgb
        frame[..., 3] = 1.0
        path = outdir / f'{arguments.name}.{arguments.start + index:04d}.exr'
        write_half_exr(path, frame, arguments.compression)
        written += 1
        if written % 10 == 0 or written == arguments.frames:
            print(f'{written}/{arguments.frames} {path.name}', flush=True)

    total = sum(f.stat().st_size for f in outdir.glob(f'{arguments.name}.*.exr'))
    print(f'{written} frames, {total / 2**30:.2f} GiB total, '
          f'{total / written / 2**20:.1f} MiB/frame, half float, {arguments.compression}')


if __name__ == '__main__':
    main()
