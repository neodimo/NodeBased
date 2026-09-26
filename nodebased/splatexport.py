"""WriteSplat3D: bake a scene's splats into one 3DGS PLY, independent of the desktop UI."""
from pathlib import Path

import numpy as np

from . import splats
from .media import is_sequence, sequence_path
from .scene3d import Scene


def scene_cloud(scene):
    """One SplatCloud holding every splat of `scene` in world space.

    Each instance's world matrix, SH degree cap, opacity scale and splat scale are baked in, so the
    file shows what the renderer shows. An untouched identity instance keeps its raw scale logs and
    opacity logits, so a read followed by a write reproduces the stored numbers exactly. Instances
    with fewer SH bands than the richest one are padded with zero bands.
    """
    pieces = []
    for instance in scene.splats:
        cloud = instance.cloud
        degree = cloud.sh_degree if instance.sh_degree is None else min(cloud.sh_degree, int(instance.sh_degree))
        raw_scale, raw_opacity = cloud.raw_scale_log, cloud.raw_opacity_logit
        scales, opacity = cloud.scales, cloud.opacity
        if instance.scale_scale != 1.0:
            scales, raw_scale = scales * np.float32(instance.scale_scale), None
        if instance.opacity_scale != 1.0:
            opacity, raw_opacity = np.clip(opacity * np.float32(instance.opacity_scale), 0, 1), None
        cloud = splats.SplatCloud(cloud.positions, scales, cloud.rotations, opacity,
                                  cloud.sh[:, :(degree + 1) ** 2], degree, raw_scale, raw_opacity,
                                  cloud.colorspace)
        matrix = np.asarray(instance.matrix, dtype=np.float64)
        if not np.array_equal(matrix, np.eye(4)):
            cloud = cloud.transformed(matrix)
        pieces.append(cloud)
    if not pieces:
        raise ValueError('WriteSplat3D: the scene has no splats to write')
    if len({p.colorspace for p in pieces}) > 1:
        raise ValueError('WriteSplat3D: splats in the scene use different colour spaces')
    if len(pieces) == 1:
        return pieces[0]
    degree = max(p.sh_degree for p in pieces)
    bands = (degree + 1) ** 2

    def padded(p):
        out = np.zeros((len(p), bands, 3), np.float32)
        out[:, :p.sh.shape[1]] = p.sh
        return out
    raw = all(p.raw_scale_log is not None and p.raw_opacity_logit is not None for p in pieces)
    return splats.SplatCloud(
        np.concatenate([p.positions for p in pieces]), np.concatenate([p.scales for p in pieces]),
        np.concatenate([p.rotations for p in pieces]), np.concatenate([p.opacity for p in pieces]),
        np.concatenate([padded(p) for p in pieces]), degree,
        np.concatenate([p.raw_scale_log for p in pieces]) if raw else None,
        np.concatenate([p.raw_opacity_logit for p in pieces]) if raw else None, pieces[0].colorspace)


def export_splats(document, key, frames, evaluator=None):
    """Write the WriteSplat3D node `key` for each frame; returns the paths written.

    A path with a padded pattern (splats.%04d.ply) writes one file per frame; a plain path takes
    one frame only. Existing files are refused unless the node's overwrite knob is on, and the
    check covers every target before any file is written.
    """
    node = document['nodes'].get(key)
    if node is None or node['type'] != 'WriteSplat3D':
        raise ValueError('Splat export requires a WriteSplat3D node')
    params = node['params']
    path = str(params.get('splat_write_path') or '').strip()
    if not path:
        raise ValueError('Set an output path on the WriteSplat3D node first')
    if Path(path).suffix.lower() != '.ply':
        raise ValueError('WriteSplat3D writes .ply files')
    upstream = node['inputs'].get('scene')
    if upstream is None or upstream not in document['nodes']:
        raise ValueError('WriteSplat3D: connect an upstream scene')
    frames = list(frames)
    if len(frames) > 1 and not is_sequence(path):
        raise ValueError(f'{Path(path).name!r} is a single file, so a {len(frames)}-frame range would '
                         'overwrite it every frame. Use a padded pattern such as splats.%04d.ply.')
    targets = [Path(sequence_path(path, frame)).expanduser() for frame in frames]
    if not params.get('splat_write_overwrite'):
        existing = [t.name for t in targets if t.exists()]
        if existing:
            raise ValueError(f'{existing[0]} already exists; turn on Overwrite to replace it')
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()
    written = []
    for frame, target in zip(frames, targets):
        scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
        if not isinstance(scene, Scene):
            raise ValueError('WriteSplat3D: connect an upstream scene')
        cloud = scene_cloud(scene)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f'Cannot write splats {target}: {error}') from error
        splats.write_ply(cloud, target)
        written.append(str(target))
    return written
