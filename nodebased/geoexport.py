"""Explicit OBJ sequences and USD stage export, independent of the desktop UI."""
from pathlib import Path

from . import usdio
from .media import is_sequence, sequence_path
from .scene3d import Scene, write_obj


def export_obj(document, key, frames, path, evaluator=None):
    """Export full-resolution scenes, dispatching by extension (legacy public name).

    Patterns write static per-frame files; an unpatterned USD range is one stage.
    """
    node = document['nodes'].get(key)
    if node is None or node['type'] != 'WriteGeo3D':
        raise ValueError('Geometry export requires a WriteGeo3D node')
    path = str(path).strip() if path is not None else ''
    if not path:
        raise ValueError('Set an output path on the WriteGeo3D node first')
    upstream = node['inputs'].get('scene')
    if upstream is None or upstream not in document['nodes']:
        raise ValueError('WriteGeo3D: connect an upstream scene')
    suffix = Path(path).suffix.lower()
    usd = suffix in ('.usd', '.usda', '.usdc', '.usdz')
    if not usd and suffix != '.obj':
        raise ValueError('Supported geometry extensions: .obj, .usd, .usda, .usdc, .usdz')
    frames = list(frames)
    if not usd and len(frames) > 1 and not is_sequence(path):
        raise ValueError(f'{Path(path).name!r} is a single file, so a {len(frames)}-frame '
                         'range would overwrite it every frame. Use a padded pattern '
                         'such as geo.%04d.obj.')
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()

    def write(scenes, target, samples=None):
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f'Cannot export geometry {target}: {error}') from error
        if usd:
            try:
                usdio.write_usd(scenes, target, frames=samples)
            except RuntimeError as error:
                raise ValueError(str(error)) from None
        else:
            write_obj(scenes, target)

    written = []
    sampled_scenes = []
    for frame in frames:
        scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
        if not isinstance(scene, Scene):
            raise ValueError('WriteGeo3D: connect an upstream scene')
        target = str(Path(sequence_path(path, frame)).expanduser())
        if usd and not is_sequence(path) and len(frames) > 1:
            sampled_scenes.append(scene)
        else:
            write(scene, target)
            written.append(target)
    if sampled_scenes:
        target = str(Path(path).expanduser())
        write(sampled_scenes, target, frames)
        written.append(target)
    return written
