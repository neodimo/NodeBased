"""WriteVDB3D: bake each fluid in a scene into a named OpenVDB file."""
from pathlib import Path
import re

from . import vdbio
from .media import is_sequence, sequence_path
from .scene3d import Scene


def export_vdb(document, key, frames, evaluator=None):
    """Write every volume and liquid surface for each requested frame.

    Multiple scene fluids get distinct filenames, using their solver/object names. A padded
    pattern writes a frame sequence. The full output set is checked before any file is written.
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
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()

    jobs = []
    for frame in frames:
        scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
        if not isinstance(scene, Scene):
            raise ValueError("WriteVDB3D: connect an upstream scene")
        members = [("volume", volume, volume.name or f"fluid-{index + 1}")
                   for index, volume in enumerate(scene.volumes)]
        members.extend(("liquid", inst.surface, inst.node_key or f"liquid-{index + 1}")
                       for index, inst in enumerate(scene.particles) if inst.surface is not None)
        if not members:
            raise ValueError("WriteVDB3D: the scene has no volume and no liquid surface to write")
        frame_path = Path(sequence_path(path, frame)).expanduser()
        seen = set()
        for kind, member, label in members:
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_-") or kind
            base = safe
            serial = 2
            while safe.casefold() in seen:
                safe = f"{base}-{serial}"
                serial += 1
            seen.add(safe.casefold())
            target = frame_path
            if len(members) > 1:
                target = target.with_name(f"{target.stem}.{safe}{target.suffix}")
            single = Scene(volumes=(member,)) if kind == "volume" else Scene(
                particles=(next(inst for inst in scene.particles if inst.surface is member),))
            jobs.append((target, single, float(params.get("cache_resolution", 1.0)) if kind == "volume" else 1.0))

    if not params.get("vdb_write_overwrite"):
        existing = [target.name for target, _, _ in jobs if target.exists()]
        if existing:
            raise ValueError(f"{existing[0]} already exists; turn on Overwrite to replace it")
    written = []
    for target, scene, resolution in jobs:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f"Cannot write VDB {target}: {error}") from error
        try:
            vdbio.write_scene(target, scene, compression=params.get("vdb_write_compression", "zip"),
                              half=bool(params.get("vdb_write_half")),
                              narrow_band=float(params.get("vdb_write_narrow_band", 3.0)),
                              cache_resolution=resolution)
        except vdbio.VdbError as error:
            raise ValueError(str(error)) from None
        written.append(str(target))
    return written
