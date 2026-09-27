"""WriteVDB3D: bake a scene's fluid volume or liquid surface into OpenVDB `.vdb` files on request."""
from pathlib import Path

from . import vdbio
from .media import is_sequence, sequence_path
from .scene3d import Scene


def export_vdb(document, key, frames, evaluator=None):
    """Write the WriteVDB3D node `key` for each frame; returns the paths written.

    A path with a padded pattern (`smoke.%04d.vdb`) writes one file per frame; a plain path takes
    one frame only. Existing files are refused unless the node's overwrite knob is on, and the
    check covers every target before any file is written.
    """
    node = document["nodes"].get(key)
    if node is None or node["type"] != "WriteVDB3D":
        raise ValueError("VDB export requires a WriteVDB3D node")
    params = node["params"]
    path = str(params.get("vdb_write_path") or "").strip()
    if not path:
        raise ValueError("Set an output path on the WriteVDB3D node first")
    if Path(path).suffix.lower() != ".vdb":
        raise ValueError("WriteVDB3D writes .vdb files")
    upstream = node["inputs"].get("scene")
    if upstream is None or upstream not in document["nodes"]:
        raise ValueError("WriteVDB3D: connect an upstream scene")
    frames = list(frames)
    if len(frames) > 1 and not is_sequence(path):
        raise ValueError(f"{Path(path).name!r} is a single file, so a {len(frames)}-frame range would "
                         "overwrite it every frame. Use a padded pattern such as smoke.%04d.vdb.")
    targets = [Path(sequence_path(path, frame)).expanduser() for frame in frames]
    if not params.get("vdb_write_overwrite"):
        existing = [t.name for t in targets if t.exists()]
        if existing:
            raise ValueError(f"{existing[0]} already exists; turn on Overwrite to replace it")
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()
    compression = params.get("vdb_write_compression", "zip")
    written = []
    for frame, target in zip(frames, targets):
        scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
        if not isinstance(scene, Scene):
            raise ValueError("WriteVDB3D: connect an upstream scene")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f"Cannot write VDB {target}: {error}") from error
        try:
            vdbio.write_scene(target, scene, compression=compression, half=bool(params.get("vdb_write_half")),
                              narrow_band=float(params.get("vdb_write_narrow_band", 3.0)))
        except vdbio.VdbError as error:
            raise ValueError(str(error)) from None
        written.append(str(target))
    return written
