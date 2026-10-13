"""Render the 24-frame plate with NodeBased itself and check it against the committed hashes.

There is no filmed footage in the repository: the plate is a deterministic render of a small set
(textured ground, two set pieces, the mixed scene's cube and sphere, a key light, an environment
light) through Render3D, with a moving camera. Each frame is an EXR with beauty, normals, depth,
motion and object ids. The frames are not committed; this script regenerates them and compares each
frame's pixel SHA-256 (the EXR header carries a write time, so file bytes are not compared) with `plate.sha256.json`.

    python examples/demo_shot/make_plate.py [--out DIR] [--size WxH] [--frames N] [--update-hashes] [--k1 K1] [--k2 K2]

`--k1 -0.03` makes the lens plate instead: the same camera move through a lens with that radial distortion (the
Camera3D lens model, `nodebased/lensmodel.py`), at any size (`--size 960x540`, `1920x1080`). The scene is rendered
`shot.OVERSCAN` times larger through the ideal camera and bent into the format, so the corners show the set.
Beauty only: the data passes stay undistorted and are not written. Its hashes live in `plate_k1.sha256.json`.

`--distorted` is the plate of the lens round trip (`demo_shot_lens.nbcomp`, run_by_hand.py --plate distorted): 768 by
432, `shot.DISTORTED_K1` and a small `shot.DISTORTED_K2`. Its hashes live in `plate_distorted.sha256.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

HASHES = shot.HERE / "plate.sha256.json"
LENS_HASHES = shot.HERE / "plate_k1.sha256.json"
DISTORTED_HASHES = shot.HERE / "plate_distorted.sha256.json"


def frame_digest(path):
    """SHA-256 of a frame's channel names and float pixels (the EXR header carries a write time)."""
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    try:
        spec = source.spec()
        digest = hashlib.sha256(",".join(spec.channelnames).encode())
        digest.update(source.read_image("float").tobytes())
        return digest.hexdigest()
    finally:
        source.close()


def bilinear(image, sx, sy):
    """`image` (H, W, C) read at fractional pixel indices (sx, sy), clamped to the edge; the Redistort node's bilinear filter."""
    height, width = image.shape[:2]
    invalid = ~(np.isfinite(sx) & np.isfinite(sy))      # a point the lens model cannot invert reads as black
    sx, sy = np.clip(np.nan_to_num(sx), 0.0, width - 1.0), np.clip(np.nan_to_num(sy), 0.0, height - 1.0)
    x0, y0 = np.minimum(np.floor(sx).astype(int), width - 2), np.minimum(np.floor(sy).astype(int), height - 2)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    out = (top * (1 - fy) + bottom * fy).astype(np.float32)
    out[invalid] = 0.0
    return out


def distort_frame(ideal, size, k1, k2=0.0):
    """The plate of a lens with radial `k1` and `k2`: `ideal` (the scene through an ideal camera, centred, at least as large as
    `size`) read where the lens model of `nodebased.lensmodel` puts each plate pixel, which is what Redistort does."""
    from nodebased import lensmodel
    width, height = size
    ideal_height, ideal_width = ideal.shape[:2]
    gx, gy = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    ux, uy = lensmodel.undistort_pixels(lensmodel.Lens(k1=k1, k2=k2), gx, gy, width, height)
    return bilinear(ideal, ux + (ideal_width - width) / 2.0 - 0.5, uy + (ideal_height - height) / 2.0 - 0.5)


def lens_scene(size, overscan):
    """The scene commands for the ideal render of the lens plate: `overscan` times the plate in each direction, same lens."""
    from nodebased import filmback
    big = tuple(int(round(v * overscan / 2.0)) * 2 for v in size)
    commands = shot.scene_commands(*big)
    shot.animate_camera(commands)
    for command in commands:
        if command.get("id") == "camera" and command["op"] == "create":
            # the same focal length on a taller film back keeps the pixel size of the lens
            command["params"]["vaperture"] = filmback.DEFAULT_VAPERTURE * big[1] / size[1]
        if command.get("id") == "render" and command["op"] == "create":
            command["params"]["render_output"] = "rgba"
    return commands


def render_plate(folder, size=shot.PLATE_SIZE, frames=shot.FRAMES, k1=0.0, overscan=shot.OVERSCAN, k2=0.0):
    """Write plate.0001.exr .. to `folder`; return {frame number: pixel digest}. With `k1` (and `k2`) the lens plate (beauty only)."""
    from nodebased.core import Dispatcher
    from nodebased.imaging import Evaluator
    from nodebased.media import raster_layer_arrays, write_exr
    dispatcher = Dispatcher()
    if k1 or k2:
        commands = lens_scene(size, overscan)
    else:
        commands = shot.scene_commands(*size)
        shot.animate_camera(commands, frames=frames)
    dispatcher.execute({"op": "batch", "commands": commands})
    evaluator, folder, hashes = Evaluator(), Path(folder), {}
    folder.mkdir(parents=True, exist_ok=True)
    for frame in range(1, frames + 1):
        raster = evaluator.evaluate_raster(dispatcher.document, "render", frame=frame)
        target = folder / (shot.plate_pattern(".").split("/")[-1] % frame)
        if k1 or k2:
            write_exr(target, distort_frame(raster.to_display(), size, k1, k2), bits="float")
        else:
            # float keeps depth, motion and ids exact; a plate this size is about 3 MB a frame
            write_exr(target, raster.to_display(), bits="float", layers=raster_layer_arrays(raster))
        hashes[frame] = frame_digest(target)
    return hashes


def check(hashes, expected):
    return [frame for frame, digest in hashes.items() if expected.get(str(frame)) != digest]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="folder for plate.%%04d.exr (default: the NodeBased cache, demo_shot/plate)")
    parser.add_argument("--size", default="x".join(map(str, shot.PLATE_SIZE)))
    parser.add_argument("--frames", type=int, default=shot.FRAMES)
    parser.add_argument("--update-hashes", action="store_true", help="rewrite the hashes file from this run")
    parser.add_argument("--k1", type=float, default=0.0, help="radial lens distortion of the lens plate (e.g. -0.03); 0 is the plain plate")
    parser.add_argument("--k2", type=float, default=0.0, help="second radial term of the lens plate")
    parser.add_argument("--distorted", action="store_true",
                        help="the plate of the lens round trip: 960x540 with k1 and k2 of shot.DISTORTED_*; --out defaults to plate_distorted")
    args = parser.parse_args()
    if args.distorted:
        args.k1, args.k2, args.size = shot.DISTORTED_K1, shot.DISTORTED_K2, "x".join(map(str, shot.DISTORTED_SIZE))
    size = tuple(int(v) for v in args.size.lower().split("x"))
    folder = Path(args.out) if args.out else (shot.distorted_plate_dir() if args.distorted else shot.plate_dir())
    started = time.perf_counter()
    hashes = render_plate(folder, size, args.frames, k1=args.k1, k2=args.k2)
    lens = bool(args.k1 or args.k2)
    print(f"rendered {len(hashes)} frames at {size[0]}x{size[1]}" + (f" with k1 {args.k1}, k2 {args.k2}" if lens else "")
          + f" into {folder} in {time.perf_counter() - started:.1f} s")
    target = DISTORTED_HASHES if args.distorted else (LENS_HASHES if lens else HASHES)
    if args.update_hashes:
        target.write_text(json.dumps({"size": list(size), "k1": args.k1, "k2": args.k2,
                                      "frames": {str(k): v for k, v in hashes.items()}}, indent=1) + "\n")
        print(f"wrote {target.name}")
        return 0
    recorded = json.loads(target.read_text())
    if (recorded["size"] != list(size) or len(recorded["frames"]) != len(hashes)
            or recorded.get("k1", 0.0) != args.k1 or recorded.get("k2", 0.0) != args.k2):
        print(f"hashes are recorded for {recorded['size']}, {len(recorded['frames'])} frames, k1 "
              f"{recorded.get('k1', 0.0)} and k2 {recorded.get('k2', 0.0)}; nothing to compare at this size")
        return 0
    wrong = check(hashes, recorded["frames"])
    print("plate matches the committed hashes" if not wrong else f"HASH MISMATCH on frames {wrong}")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
