"""Explicit OBJ sequence export, independent of the desktop UI."""
from pathlib import Path

from .media import is_sequence, sequence_path
from .scene3d import Scene, write_obj


def export_obj(document, key, frames, path, evaluator=None):
    """Evaluate a WriteGeo3D upstream scene at full resolution and write each requested frame."""
    node = document['nodes'].get(key)
    if node is None or node['type'] != 'WriteGeo3D':
        raise ValueError('OBJ export requires a WriteGeo3D node')
    path = str(path).strip() if path is not None else ''
    if not path:
        raise ValueError('Set an output path on the WriteGeo3D node first')
    upstream = node['inputs'].get('scene')
    if upstream is None or upstream not in document['nodes']:
        raise ValueError('WriteGeo3D: connect an upstream scene')
    frames = list(frames)
    if len(frames) > 1 and not is_sequence(path):
        raise ValueError(f'{Path(path).name!r} is a single file, so a {len(frames)}-frame '
                         'range would overwrite it every frame. Use a padded pattern '
                         'such as geo.%04d.obj.')
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()
    written = []
    for frame in frames:
        scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
        if not isinstance(scene, Scene):
            raise ValueError('WriteGeo3D: connect an upstream scene')
        target = str(Path(sequence_path(path, frame)).expanduser())
        try:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ValueError(f'Cannot export OBJ {target}: {error}') from error
        write_obj(scene, target)
        written.append(target)
    return written
