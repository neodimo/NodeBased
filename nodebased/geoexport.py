"""Explicit OBJ sequences and USD stage export, independent of the desktop UI."""
from pathlib import Path

import numpy as np

from . import usdio
from .media import is_sequence, sequence_path
from . import scene3d
from .scene3d import Scene, resolve_instances, write_obj


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
    scene_state = path.lower().endswith('.scene.json')
    usd = suffix in ('.usd', '.usda', '.usdc', '.usdz')
    if not scene_state and not usd and suffix != '.obj':
        raise ValueError('Supported geometry extensions: .obj, .usd, .usda, .usdc, .usdz, .scene.json')
    frames = list(frames)
    if not usd and not scene_state and len(frames) > 1 and not is_sequence(path):
        raise ValueError(f'{Path(path).name!r} is a single file, so a {len(frames)}-frame '
                         'range would overwrite it every frame. Use a padded pattern '
                         'such as geo.%04d.obj.')
    if evaluator is None:
        from .imaging import Evaluator
        evaluator = Evaluator()

    if scene_state:
        camera_key = node.get('inputs', {}).get('camera')
        if not camera_key or camera_key not in document['nodes']:
            raise ValueError("SceneState export needs a Camera3D connected to WriteGeo3D's camera input")
        from .scene_state import write_scene_state
        samples = {}
        for frame in frames:
            scene = evaluator.evaluate_raster(document, upstream, frame=frame, tier=1, typed=True)
            camera = evaluator.evaluate_raster(document, camera_key, frame=frame, tier=1, typed=True)
            if not isinstance(scene, Scene) or not isinstance(camera, scene3d.Camera):
                raise ValueError('SceneState export requires a scene and Camera3D connection')
            samples[int(frame)] = {'scene': scene, 'camera': camera}
            # The evaluator's rigid solvers retain their per-frame solved bodies while evaluating
            # the scene. Preserve physical state alongside the rendered geometry they produce.
            body_rows = []
            for solver in getattr(evaluator, '_rigid_solvers', {}).values():
                for index, body in enumerate(solver.bodies):
                    rotation = body.rotation_matrix()
                    transform = np.eye(4, dtype=np.float64)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = body.position
                    body_rows.append({
                        'id': index, 'shape': body.shape, 'transform': transform,
                        'linear_velocity': body.velocity.copy(),
                        'angular_velocity': body.angular_velocity.copy(),
                        'mass': float(body.effective_mass), 'sleeping': bool(body.sleeping),
                    })
            if body_rows:
                samples[int(frame)]['simulations'] = {'rigid_bodies': body_rows}
        time = document.get('time', {})
        width, height = 1920, 1080
        for candidate in document.get('nodes', {}).values():
            if candidate.get('type') == 'Render3D' and candidate.get('inputs', {}).get('scene') == key:
                width = int(candidate['params'].get('width', width))
                height = int(candidate['params'].get('height', height))
                break
        # Include the same supporting evidence the contract describes, generated from the exact
        # sampled scene/camera. The terminal frame uses the next evaluated frame for forward motion.
        from . import cryptomatte3d, motionblur
        control_samples = {}
        ordered_frames = sorted(samples)
        for frame in sorted(samples):
            scene, camera = samples[frame]['scene'], samples[frame]['camera']
            beauty, layers = scene3d.render_multichannel(
                scene, camera, width, height, passes='beauty,normals,depth')
            samples[frame]['passes'] = {
                'beauty': beauty,
                **layers,
                'object_id': scene3d.render(scene, camera, width, height, output='object_id'),
            }
            crypto_layers, crypto_metadata = cryptomatte3d.render_cryptomatte(
                scene, camera, width, height, samples=1)
            samples[frame]['passes']['cryptomatte'] = crypto_layers
            samples[frame]['passes']['cryptomatte_metadata'] = crypto_metadata
            def sample_at(sample_frame):
                if sample_frame in samples:
                    return samples[sample_frame]['scene'], samples[sample_frame]['camera']
                return (evaluator.evaluate_raster(document, upstream, frame=sample_frame,
                                                  tier=1, typed=True),
                        evaluator.evaluate_raster(document, camera_key, frame=sample_frame,
                                                  tier=1, typed=True))
            later_scene, later_camera = sample_at(frame + 1)
            previous_scene, previous_camera = sample_at(frame - 1)
            forward = motionblur.motion_vectors(scene, camera, later_scene, later_camera,
                                                width, height)
            backward = motionblur.motion_vectors(scene, camera, previous_scene, previous_camera,
                                                 width, height)
            samples[frame]['passes']['motion'] = forward
            samples[frame]['passes']['motion_backward'] = backward
            crypto_object = crypto_layers.get('CryptoObject00')
            if crypto_object is None:
                raise ValueError('ControlBundle export requires the primary Cryptomatte object layer')
            object_ids = crypto_object[..., 0].copy()
            object_ids[crypto_object[..., 1] <= 0] = 0
            control_samples[frame] = {
                'beauty': beauty, 'depth': layers['depth'], 'normals': layers['normals'],
                'motion_forward': forward, 'motion_backward': backward, 'object_ids': object_ids,
            }
        write_scene_state(path, samples, first_frame=min(samples), last_frame=max(samples),
                          fps=float(time.get('fps', 24.0)), resolution=(width, height))
        from .control_bundle import write_control_bundle
        control_manifest = write_control_bundle(
            path, Path(path).with_suffix('.controls'), control_samples,
            first_frame=min(samples), last_frame=max(samples))
        return [str(Path(path).expanduser()), str(Path(path).with_suffix('.npz').expanduser()),
                str(control_manifest.expanduser())]

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
        scene = resolve_instances(scene)
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
