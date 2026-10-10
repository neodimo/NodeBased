"""Render the 24-frame plate with NodeBased itself and check it against the committed hashes.

There is no filmed footage in the repository: the plate is a deterministic render of a small set
(textured ground, two set pieces, the mixed scene's cube and sphere, a key light, an environment
light) through Render3D, with a moving camera. Each frame is an EXR with beauty, normals, depth,
motion and object ids. The frames are not committed; this script regenerates them and compares each
frame's pixel SHA-256 (the EXR header carries a write time, so file bytes are not compared) with `plate.sha256.json`.

    python examples/demo_shot/make_plate.py [--out DIR] [--size WxH] [--frames N] [--update-hashes]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shot  # noqa: E402

HASHES = shot.HERE / "plate.sha256.json"


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


def render_plate(folder, size=shot.PLATE_SIZE, frames=shot.FRAMES):
    """Write plate.0001.exr .. to `folder`; return {frame number: pixel digest}."""
    from nodebased.core import Dispatcher
    from nodebased.imaging import Evaluator
    from nodebased.media import raster_layer_arrays, write_exr
    dispatcher = Dispatcher()
    commands = shot.scene_commands(*size)
    shot.animate_camera(commands, frames=frames)
    dispatcher.execute({"op": "batch", "commands": commands})
    evaluator, folder, hashes = Evaluator(), Path(folder), {}
    folder.mkdir(parents=True, exist_ok=True)
    for frame in range(1, frames + 1):
        raster = evaluator.evaluate_raster(dispatcher.document, "render", frame=frame)
        target = folder / (shot.plate_pattern(".").split("/")[-1] % frame)
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
    parser.add_argument("--update-hashes", action="store_true", help="rewrite plate.sha256.json from this run")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    folder = Path(args.out) if args.out else shot.plate_dir()
    started = time.perf_counter()
    hashes = render_plate(folder, size, args.frames)
    print(f"rendered {len(hashes)} frames at {size[0]}x{size[1]} into {folder} in {time.perf_counter() - started:.1f} s")
    if args.update_hashes:
        HASHES.write_text(json.dumps({"size": list(size), "frames": {str(k): v for k, v in hashes.items()}},
                                     indent=1) + "\n")
        print(f"wrote {HASHES.name}")
        return 0
    recorded = json.loads(HASHES.read_text())
    if recorded["size"] != list(size) or len(recorded["frames"]) != len(hashes):
        print(f"hashes are recorded for {recorded['size']} and {len(recorded['frames'])} frames; "
              "nothing to compare at this size")
        return 0
    wrong = check(hashes, recorded["frames"])
    print("plate matches the committed hashes" if not wrong else f"HASH MISMATCH on frames {wrong}")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
