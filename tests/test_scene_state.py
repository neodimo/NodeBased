import tempfile
import unittest
import json
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from nodebased import cryptomatte, cryptomatte3d, envlight, motionblur, scene3d
from nodebased.geoexport import export_obj
from nodebased.scene_state import SCHEMA_VERSION, read_scene_state, write_scene_state
from nodebased.pathtrace import camera_rays
from nodebased.control_bundle import (SceneStateVersionMismatch, read_control_bundle,
                                      write_control_bundle)


def _assert_same(test, actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        test.assertIsInstance(actual, type(expected))
        for field in fields(expected):
            _assert_same(test, getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            _assert_same(test, a, e)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            _assert_same(test, actual[key], expected[key])
    else:
        test.assertEqual(actual, expected)


def _geometry(name, x):
    return scene3d.Geometry(
        vertices=np.array(((-.4, -.4, 0), (.4, -.4, 0), (0, .4, 0)), np.float32),
        triangles=np.array(((0, 1, 2),), np.int32), color=(.7, .2, .1, 1),
        transform=scene3d.Transform3D(position=scene3d.Vec3(x, 0, 0)), name=name)


def _shot(frame):
    geoms = tuple(_geometry(f"mesh{i}", i + frame * .1) for i in range(3))
    source = _geometry("tree", 0)
    instances = scene3d.InstanceSet(
        (source,), np.array((np.eye(4), np.eye(4) * 1), np.float64),
        np.zeros(2, np.int32), ids=np.array((100, 101), np.int64))
    # The second instance matrix needs a valid homogeneous bottom row.
    matrices = instances.matrices.copy()
    matrices[1] = np.eye(4); matrices[1, 0, 3] = 1.5 + frame * .1
    instances = scene3d.InstanceSet(instances.sources, matrices, instances.variant, ids=instances.ids)
    particles = scene3d.ParticleInstance(
        positions=np.array(((0, .5, 0), (1, .5, 0)), np.float32) + (frame * .01),
        sizes=np.array((.2, .3), np.float32), colors=np.ones((2, 4), np.float32),
        velocities=np.ones((2, 3), np.float32) * frame, ids=np.array((12, 13), np.int64))
    volume = scene3d.Volume(np.full((2, 2, 2), frame, np.float32), velocity=np.ones((2, 2, 2, 3), np.float32))
    scene = scene3d.Scene(
        geometries=geoms,
        lights=(scene3d.Light(kind="Point", position=scene3d.Vec3(frame, 2, 1), shadows=True),
                scene3d.Light(kind="Spot", intensity=3, cone_angle=42, area_width=2)),
        instances=(instances,), particles=(particles,), volumes=(volume,),
        environments=(envlight.Environment(np.ones((2, 4, 3), np.float32) * frame,
                                             envlight.fingerprint_of(np.ones((2, 4, 3), np.float32) * frame),
                                             rotation=frame * 5, blur=.2),))
    camera = scene3d.Camera(
        transform=scene3d.Transform3D(position=scene3d.Vec3(frame, 0, 5), rotation=scene3d.Vec3(0, frame, 0)),
        fov=48, near=.2, far=200, haperture=36, vaperture=24, fstop=2.8,
        focus_distance=4, aperture_blades=7)
    return scene, camera


class SceneStateTests(unittest.TestCase):
    def test_animated_scene_round_trips_every_typed_value_and_arrays(self):
        samples = {f: {"scene": _shot(f)[0], "camera": _shot(f)[1], "resolution": (64, 48)}
                   for f in (1, 2, 3)}
        simulations = {f: {"rigid_bodies": [{"id": 7, "transform": np.eye(4) * f,
                                             "velocity": np.array((f, 2, 3), np.float32)}]}
                           for f in samples}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.scene.json"
            write_scene_state(path, samples, first_frame=1, last_frame=3, fps=24,
                              resolution=(64, 48), pixel_aspect=1.2,
                              simulations=simulations)
            result = read_scene_state(path)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["frame_range"], {"first": 1, "last": 3})
        self.assertEqual(result["coordinates"]["handedness"], "right-handed")
        for row in result["frames"]:
            frame = row["frame"]
            expected_scene, expected_camera = samples[frame]["scene"], samples[frame]["camera"]
            self.assertEqual(row["time_seconds"], (frame - 1) / 24)
            self.assertEqual(row["resolution"], [64, 48])
            _assert_same(self, row["scene"], expected_scene)
            _assert_same(self, row["camera"], expected_camera)
            _assert_same(self, row["simulations"], simulations[frame])
            env_ref = row["environment_images"][0]
            self.assertRegex(env_ref, r"\.npz#array_\d{6}$")

    def test_export_ids_match_render3d_cryptomatte_pass_ids(self):
        geom = scene3d.Geometry(
            vertices=np.array(((-1, -1, 0), (1, -1, 0), (0, 1, 0)), np.float32),
            triangles=np.array(((0, 1, 2),), np.int32), color=(1, 1, 1, 1), name="hero")
        source = _geometry("tree", 0)
        inst = scene3d.InstanceSet((source,), np.array((np.eye(4), np.eye(4)), np.float64),
                                   np.array((0, 0), np.int32), ids=np.array((8, 9), np.int64))
        scene = scene3d.Scene(geometries=(geom,), instances=(inst,))
        camera = scene3d.Camera()
        _, metadata = cryptomatte3d.render_cryptomatte(scene, camera, 24, 24, levels=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "same.scene.json"
            write_scene_state(path, {1: {"scene": scene, "camera": camera}}, resolution=(24, 24))
            exported = read_scene_state(path)["frames"][0]["objects"]
        manifest = metadata["CryptoObject"]["manifest"]
        for obj in exported:
            self.assertEqual(obj["id"], manifest[obj["name"]])
        self.assertEqual(exported[0]["id"], f"{cryptomatte.name_to_bits('hero'):08x}")
        self.assertEqual([item["type"] for item in exported], ["geometry", "instance", "instance"])

    def test_writegeo3d_scene_json_mode_samples_scene_and_camera_per_frame(self):
        class Evaluator:
            def evaluate_raster(self, document, target, frame, **_kwargs):
                scene, camera = _shot(frame)
                return scene if target == "scene" else camera

        document = {"time": {"first": 1, "last": 2, "fps": 30}, "nodes": {
            "write": {"type": "WriteGeo3D", "params": {},
                      "inputs": {"scene": "scene", "camera": "camera"}},
            "scene": {"type": "Scene3D", "params": {}, "inputs": {}},
            "camera": {"type": "Camera3D", "params": {}, "inputs": {}},
            "render": {"type": "Render3D", "params": {"width": 80, "height": 45},
                       "inputs": {"scene": "write"}},
        }}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "range.scene.json"
            written = export_obj(document, "write", range(1, 3), path, evaluator=Evaluator())
            result = read_scene_state(path)
            self.assertEqual([Path(p).suffix for p in written], [".json", ".npz", ".json"])
            self.assertEqual(Path(written[2]).name, "manifest.json")
            self.assertEqual([f["frame"] for f in result["frames"]], [1, 2])
            self.assertEqual(result["frames"][0]["resolution"], [80, 45])
            self.assertEqual(result["coordinates"]["fps"], 30)
            controls = read_control_bundle(written[2], path)
        self.assertEqual([frame.frame for frame in controls], [1, 2])
        self.assertEqual(controls[0].beauty.values.shape, (45, 80, 4))
        self.assertEqual(controls[0].object_names, {obj["id"]: obj["name"]
                                                    for obj in result["frames"][0]["objects"]})
        frame_one_scene, frame_one_camera = _shot(1)
        frame_two_scene, frame_two_camera = _shot(2)
        geom = frame_one_scene.geometries[0]
        matrix = geom.world_matrix()
        world_point = geom.vertices.mean(axis=0) @ matrix[:3, :3].T + matrix[:3, 3]
        pixel, distance = scene3d.project(frame_one_camera, 80, 45, world_point[None, :])
        x, y = np.floor(pixel[0]).astype(int)
        ray, origin, cosine, _, _ = camera_rays(frame_one_camera, 80, 45,
                                               np.array([x + .5]), np.array([y + .5]))
        plane_distance = (-origin[2] / ray[0, 2]) * cosine[0]
        np.testing.assert_allclose(controls[0].depth.values[y, x, 0], plane_distance, atol=.001)
        np.testing.assert_allclose(controls[0].normals.values[y, x, 2], 1.0, atol=1e-6)
        next_geom = frame_two_scene.geometries[0]
        next_matrix = next_geom.world_matrix()
        next_point = next_geom.vertices.mean(axis=0) @ next_matrix[:3, :3].T + next_matrix[:3, 3]
        next_pixel, _ = scene3d.project(frame_two_camera, 80, 45, next_point[None, :])
        vector = controls[0].motion_forward.values[y, x]
        np.testing.assert_allclose(vector, next_pixel[0] - pixel[0], atol=.1)
        object_id = cryptomatte.name_to_bits("mesh0")
        self.assertEqual(int(controls[0].object_ids.values[y, x]), object_id)

    def test_control_bundle_round_trips_typed_layers_ids_and_color_contract(self):
        scene = scene3d.Scene(geometries=(_geometry("hero", 0),))
        camera = scene3d.Camera(near=.25, far=80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "shot.scene.json"
            write_scene_state(state, {1: {"scene": scene, "camera": camera,
                                          "resolution": (4, 3)}}, resolution=(4, 3))
            ident = np.asarray(cryptomatte.name_to_bits("hero"), np.uint32).view(np.float32)
            ids = np.zeros((3, 4), np.float32); ids[1, 2] = ident
            samples = {1: {
                "beauty": np.ones((3, 4, 4), np.float32),
                "depth": np.full((3, 4), 5.0, np.float32),
                "normals": np.broadcast_to((0, 0, 1), (3, 4, 3)).astype(np.float32),
                "motion_forward": np.zeros((3, 4, 2), np.float32),
                "motion_backward": np.zeros((3, 4, 2), np.float32),
                "object_ids": ids,
            }}
            manifest_path = write_control_bundle(state, root / "bundle", samples)
            result = read_control_bundle(manifest_path, state)[0]
            self.assertEqual(result.frame, 1)
            self.assertEqual(result.depth.units, "metres")
            self.assertEqual(result.depth.coordinate_space,
                             "camera-space +Z distance; positive forward")
            self.assertEqual(result.normals.coordinate_space,
                             "camera-space; right-handed; +Y up")
            self.assertEqual(result.motion_forward.units, "pixels per frame")
            self.assertEqual(result.beauty.color_space,
                             "OCIO role scene_linear (ACEScg)")
            self.assertEqual(result.object_names[f"{cryptomatte.name_to_bits('hero'):08x}"], "hero")
            self.assertEqual(int(result.object_ids.values[1, 2]), cryptomatte.name_to_bits("hero"))
            np.testing.assert_array_equal(result.depth.values[..., 0], 5.0)

            doc = json.loads(manifest_path.read_text())
            doc["scene_state_schema_version"] = SCHEMA_VERSION + 1
            manifest_path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(SceneStateVersionMismatch, "SceneStateVersionMismatch"):
                read_control_bundle(manifest_path, state)
            doc["scene_state_schema_version"] = SCHEMA_VERSION
            manifest_path.write_text(json.dumps(doc))
            state_doc = json.loads(state.read_text())
            state_doc["working_color_space"] = "different-shot"
            state.write_text(json.dumps(state_doc))
            with self.assertRaisesRegex(ValueError, "SceneStateMismatch"):
                read_control_bundle(manifest_path, state)

    def test_control_bundle_export_is_content_addressed_across_wall_clock_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "shot.scene.json"
            write_scene_state(state, {1: {"scene": scene3d.Scene(), "camera": scene3d.Camera(),
                                          "resolution": (2, 2)}}, resolution=(2, 2))
            rgba = np.ones((2, 2, 4), np.float32)
            samples = {1: {"beauty": rgba, "depth": np.zeros((2, 2), np.float32),
                           "normals": np.zeros((2, 2, 3), np.float32),
                           "motion_forward": np.zeros((2, 2, 2), np.float32),
                           "motion_backward": np.zeros((2, 2, 2), np.float32),
                           "object_ids": np.zeros((2, 2), np.float32)}}
            first = write_control_bundle(state, root / "bundle", samples, return_artifact_id=True)[1]
            time.sleep(1.05)
            second = write_control_bundle(state, root / "bundle", samples, return_artifact_id=True)[1]
            self.assertEqual(second, first)


if __name__ == "__main__":
    unittest.main()
