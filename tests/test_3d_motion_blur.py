"""Motion blur and the motion pass (lane L4 step R5, part 2): Render3D across a shutter.

docs/3D_FOUNDATION.md "Motion blur". Images are small so the file runs in seconds; the GPU tests are guarded like every
other GPU test. Animation curves put the object in motion; positions are exact, so the blur length and the motion
vectors are compared with numbers worked out from the camera's projection.
"""
import dataclasses
import math
import unittest
from pathlib import Path

import numpy as np

from nodebased import core, motionblur, pathtrace as pt, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_pathtrace import gpu_ready, sphere

WIDTH, HEIGHT = 96, 64
FOCAL = 35.0
DISTANCE = 10.0
CUBE = 2.0
FIXTURE = Path(__file__).parent / "fixtures" / "abc" / "probe.abc"


def curve(frame_values):
    return {"interpolation": "linear", "keys": [{"frame": f, "value": v} for f, v in frame_values]}


def graph(*, speed=2.0, mode="pathtrace", motion=1, shutter=1.0, samples=16, subject="Cube3D", chain=None,
          camera_curves=None, **render):
    """One emissive cube (or `subject`) moving `speed` units per frame along x, seen by a camera `DISTANCE` away."""
    d = Dispatcher()
    nodes = [("box", subject, dict(emission=1.0)),
             ("camera", "Camera3D", dict(tz=DISTANCE, focal=FOCAL)), ("scene", "Scene3D", {}),
             ("render", "Render3D", dict(width=WIDTH, height=HEIGHT, render_mode=mode, pt_samples=32, max_bounces=0,
                                         motion_blur=motion, shutter=shutter, motion_samples=samples, **render))]
    for key, kind, params in nodes:
        d.execute(dict(op="create", id=key, type=kind, params=params))
    d.execute(dict(op="connect", id="scene", input="object0", source="box"))
    d.execute(dict(op="connect", id="render", input="scene", source="scene"))
    d.execute(dict(op="connect", id="render", input="camera", source="camera"))
    d.document["animation"]["curves"]["box"] = {"tx": curve([(-10, -10 * speed), (10, 10 * speed)])}
    if camera_curves:
        d.document["animation"]["curves"]["camera"] = camera_curves
    return d


def image(d, frame=0):
    return Evaluator().evaluate(d.document, "render", frame=frame)


def extent(picture, axis=0, threshold=0.02):
    """Pixels the coverage spans along columns (axis 0) or rows (axis 1)."""
    lit = np.flatnonzero((picture[..., 3] > threshold).any(axis=axis))
    return int(lit.max() - lit.min() + 1)


def pixels_per_unit(depth):
    """Pixels one scene unit spans at `depth` for the 35 mm lens on the default film back, HEIGHT pixels tall."""
    return HEIGHT * FOCAL / (18.672 * depth)


class ShutterTests(unittest.TestCase):
    def test_times_are_centred_by_default_and_end_on_the_shutter(self):
        low, high = motionblur.shutter_window(10, 0.5)
        self.assertEqual((low, high), (9.75, 10.25))
        self.assertEqual(motionblur.shutter_window(10, 0.5, "start"), (9.5, 10))
        self.assertEqual(motionblur.shutter_window(10, 0.5, "end"), (10, 10.5))
        self.assertEqual(motionblur.shutter_window(10, 1.0, "custom", 2.0), (11.5, 12.5))
        self.assertEqual(motionblur.shutter_times(9.75, 10.25, 5), [9.75, 9.875, 10.0, 10.125, 10.25])
        self.assertEqual(motionblur.shutter_times(9.75, 10.25, 1), [10.0])

    def test_the_knobs_are_registered_and_old_documents_load(self):
        params = core.SPECS["Render3D"]["params"]
        self.assertEqual((params["motion_blur"], params["shutter"], params["shutter_offset"], params["motion_samples"]),
                         (0, 0.5, "centred", 8))
        self.assertIn("motion", core.CHOICES["render_output"])
        self.assertIn("motion", s.MULTICHANNEL_PASSES)
        d = Dispatcher()
        d.execute(dict(op="create", id="render", type="Render3D", params={}))
        for name in core._MOTION_DEFAULTS:
            d.document["nodes"]["render"]["params"].pop(name)
        upgraded = core.upgrade_document(d.document)
        for name, default in core._MOTION_DEFAULTS.items():
            self.assertEqual(upgraded["nodes"]["render"]["params"][name], default)


class TranslationBlurTests(unittest.TestCase):
    def blur_growth(self, **fields):
        sharp = image(graph(motion=0, **fields))
        blurred = image(graph(**fields))
        return extent(blurred) - extent(sharp)

    def test_blur_length_is_speed_times_shutter_in_every_mode(self):
        front = DISTANCE - CUBE / 2                       # the cube's near face is the widest thing on screen
        for mode in ("pathtrace", "raytrace", "raster"):
            for speed, shutter in ((2.0, 1.0), (3.0, 0.5)):
                with self.subTest(mode=mode, speed=speed, shutter=shutter):
                    expected = speed * shutter * pixels_per_unit(front)
                    self.assertAlmostEqual(self.blur_growth(mode=mode, speed=speed, shutter=shutter), expected, delta=2.0)

    def test_blur_is_centred_on_the_frame(self):
        sharp, blurred = image(graph(motion=0)), image(graph())
        lit = np.flatnonzero((sharp[..., 3] > 0.02).any(axis=0))
        blur = np.flatnonzero((blurred[..., 3] > 0.02).any(axis=0))
        self.assertAlmostEqual((blur.min() + blur.max()) / 2, (lit.min() + lit.max()) / 2, delta=1.5)

    def test_shutter_zero_and_motion_blur_off_are_the_sharp_image(self):
        off = image(graph(motion=0))
        np.testing.assert_array_equal(image(graph(shutter=0.0)), off)
        for mode in ("raytrace", "raster"):
            np.testing.assert_array_equal(image(graph(shutter=0.0, mode=mode)), image(graph(motion=0, mode=mode)))

    def test_a_still_object_is_unchanged_by_the_shutter(self):
        blurred, sharp = image(graph(speed=0.0)), image(graph(speed=0.0, motion=0))
        self.assertEqual(extent(blurred), extent(sharp))
        # both are Monte Carlo edges drawn from different seeds; away from the edges they are the same
        self.assertLess(float(np.abs(blurred - sharp).mean()), 0.005)

    def test_motion_samples_change_the_smoothness_not_the_length(self):
        few, many = image(graph(samples=4)), image(graph(samples=32))
        self.assertAlmostEqual(extent(few), extent(many), delta=2)

    def test_an_animated_camera_blurs_a_still_scene(self):
        pan = {"tx": curve([(-10, -10.0), (10, 10.0)]), "target_x": curve([(-10, -10.0), (10, 10.0)])}
        sharp = image(graph(speed=0.0, motion=0, camera_curves=pan))
        blurred = image(graph(speed=0.0, camera_curves=pan))
        expected = 1.0 * pixels_per_unit(DISTANCE - CUBE / 2)
        self.assertAlmostEqual(extent(blurred) - extent(sharp), expected, delta=2.0)


class RotationBlurTests(unittest.TestCase):
    def test_a_rotating_object_blurs_in_an_arc(self):
        d = Dispatcher()
        for key, kind, params in (("dot", "Sphere3D", dict(sphere_radius=0.15, tx=1.5, emission=1.0)),
                                  ("spin", "Axis3D", {}),
                                  ("camera", "Camera3D", dict(tz=DISTANCE, focal=FOCAL)), ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=WIDTH, height=WIDTH, render_mode="pathtrace",
                                                              pt_samples=32, max_bounces=0, motion_blur=1, shutter=1.0,
                                                              motion_samples=16))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="spin", input="object", source="dot"))
        d.execute(dict(op="connect", id="scene", input="object0", source="spin"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        d.document["animation"]["curves"]["spin"] = {"rz": curve([(-10, -300.0), (10, 300.0)])}      # 30 degrees a frame
        blurred = Evaluator().evaluate(d.document, "render", frame=0)
        centre = WIDTH / 2
        scale = WIDTH * FOCAL / (18.672 * DISTANCE)
        radius_px = 1.5 * scale
        ys, xs = np.nonzero(blurred[..., 3] > 0.05)
        distance = np.hypot(xs + 0.5 - centre, ys + 0.5 - centre)
        # every lit pixel sits on the circle the dot travels, within the dot's own size ...
        self.assertLess(float(np.abs(distance - radius_px).max()), 0.15 * scale + 1.5)
        # ... and together they span the 30 degrees of the shutter (an arc, not a streak)
        angles = np.degrees(np.arctan2(-(ys + 0.5 - centre), xs + 0.5 - centre))
        span = float(angles.max() - angles.min())
        dot_degrees = math.degrees(0.15 / 1.5) * 2
        self.assertAlmostEqual(span, 30.0 + dot_degrees, delta=6.0)
        self.assertGreater(span, dot_degrees + 20.0)


@unittest.skipUnless(FIXTURE.exists(), "Alembic fixture missing")
class DeformationBlurTests(unittest.TestCase):
    def test_a_time_sampled_alembic_mesh_blurs(self):
        d = Dispatcher()
        for key, kind, params in (("abc", "ReadAlembic3D", dict(abc_path=str(FIXTURE))), ("scene", "Scene3D", {}),
                                  ("camera", "Camera3D", dict(tx=1.0, ty=1.5, tz=4.0, target_x=1.0, target_y=1.5,
                                                              target_z=-2.5, focal=35.0)),
                                  ("render", "Render3D", dict(width=WIDTH, height=HEIGHT, render_mode="raytrace",
                                                              ambient=1.0, motion_blur=1, shutter=4.0,
                                                              motion_samples=9))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="scene", input="object0", source="abc"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        blurred = Evaluator().evaluate(d.document, "render", frame=3)
        d.document["nodes"]["render"]["params"]["motion_blur"] = 0
        sharp = Evaluator().evaluate(d.document, "render", frame=3)
        # the apex moves up by half a unit a frame; over a four-frame shutter the mesh smears upward
        self.assertGreater(extent(blurred, axis=1), extent(sharp, axis=1) + 3)
        partial = (blurred[..., 3] > 0.05) & (blurred[..., 3] < 0.95)
        self.assertGreater(int(partial.sum()), int(((sharp[..., 3] > 0.05) & (sharp[..., 3] < 0.95)).sum()))


class VelocityTests(unittest.TestCase):
    def test_advect_scene_moves_what_carries_a_velocity(self):
        rng = np.random.default_rng(3)
        geometry = dataclasses.replace(sphere(0.5), velocities=np.tile(np.array((1.0, 0.0, -2.0), np.float32),
                                                                       (len(sphere(0.5).vertices), 1)))
        particles = s.ParticleInstance(positions=rng.random((5, 3)).astype(np.float32), sizes=np.ones(5, np.float32),
                                       colors=np.ones((5, 4), np.float32),
                                       velocities=np.tile(np.array((0.0, 3.0, 0.0), np.float32), (5, 1)))
        matrices = np.tile(np.eye(4), (2, 1, 1))
        instances = s.InstanceSet((sphere(0.1),), matrices, np.zeros(2, np.int32),
                                  velocities=np.array(((1.0, 0, 0), (0, 0, 4.0))))
        scene = s.Scene((geometry, sphere(0.5)), particles=(particles,), instances=(instances,))
        moved = motionblur.advect_scene(scene, 0.5)
        np.testing.assert_allclose(moved.geometries[0].vertices - geometry.vertices,
                                   np.tile([0.5, 0.0, -1.0], (len(geometry.vertices), 1)), atol=1e-6)
        np.testing.assert_array_equal(moved.geometries[1].vertices, scene.geometries[1].vertices)   # no velocity: still
        np.testing.assert_allclose(moved.particles[0].positions - particles.positions,
                                   np.tile([0.0, 1.5, 0.0], (5, 1)), atol=1e-6)
        np.testing.assert_allclose(moved.instances[0].matrices[:, :3, 3], [[0.5, 0, 0], [0, 0, 2.0]])
        self.assertIs(motionblur.advect_scene(scene, 0.0), scene)
        back = motionblur.advect_scene(moved, -0.5)
        np.testing.assert_allclose(back.geometries[0].vertices, geometry.vertices, atol=1e-6)

    def test_point_velocities_read_the_nearest_particles(self):
        positions = np.array([[0, 0, 0], [0.1, 0, 0], [3, 0, 0], [3.1, 0, 0]], float)
        velocities = np.array([[1, 0, 0], [1, 0, 0], [0, 5, 0], [0, 5, 0]], float)
        found = motionblur.point_velocities(np.array([[0.05, 0.05, 0], [3.05, 0, 0.05], [1.5, 0, 0]]), positions,
                                            velocities, 0.5)
        np.testing.assert_allclose(found[0], [1, 0, 0])
        np.testing.assert_allclose(found[1], [0, 5, 0])
        self.assertGreater(float(np.abs(found[2]).max()), 0.9)             # widened to the nearest cell that has one

    def test_particles_and_instances_blur_from_velocity_in_the_path_tracer(self):
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 10)), s.Vec3(0, 0, 0), 30.0)
        dot = sphere(0.15, (1, 1, 1, 1), emission=1.0)
        matrices = np.tile(np.eye(4), (1, 1, 1))
        group = s.InstanceSet((dot,), matrices, np.zeros(1, np.int32), velocities=np.array(((2.0, 0, 0),)))
        scene = s.Scene(instances=(group,))
        times = motionblur.shutter_times(-0.5, 0.5, 9)
        moments = [(motionblur.advect_scene(scene, t), camera) for t in times]
        blurred = pt.render_motion(moments, 64, 64, (0, 0, 0, 0), 0.0, "rgba", pt.PathSettings(samples=72, max_bounces=0))
        sharp = pt.render(scene, camera, 64, 64, (0, 0, 0, 0), 0.0, "rgba", pt.PathSettings(samples=8, max_bounces=0))
        scale = 64 / (2 * 10 * math.tan(math.radians(15)))
        self.assertAlmostEqual(extent(blurred) - extent(sharp), 2.0 * scale, delta=3.0)


class FluidSurfaceVelocityTests(unittest.TestCase):
    def test_surface_vertices_carry_the_velocity_of_the_particles_beside_them(self):
        from tests.test_liquid_nodes import at
        from tests.test_liquid_surface import tank
        d = tank()
        surface = at(Evaluator(), d, "sf", 3)
        particles = at(Evaluator(), d, "sol", 3)
        self.assertEqual(surface.velocities.shape, surface.vertices.shape)
        speed = np.linalg.norm(particles.velocities, axis=1)
        self.assertGreater(float(speed.max()), 0.0)
        self.assertLessEqual(float(np.linalg.norm(surface.velocities, axis=1).max()), float(speed.max()) * 1.001)
        # each vertex holds a mean of nearby particle velocities: it lies inside their range, per axis
        for axis in range(3):
            column = surface.velocities[:, axis]
            self.assertGreaterEqual(float(column.min()), float(particles.velocities[:, axis].min()) - 1e-5)
            self.assertLessEqual(float(column.max()), float(particles.velocities[:, axis].max()) + 1e-5)
        # a vertex sitting on a particle takes that particle's velocity when it is alone in its cell of a coarse grid
        one = motionblur.point_velocities(particles.positions[:1], particles.positions[:1], particles.velocities[:1], 0.1)
        np.testing.assert_allclose(one[0], particles.velocities[0], atol=1e-6)
        moved = motionblur.advect_scene(s.Scene((surface,)), 0.5).geometries[0]
        np.testing.assert_allclose(moved.vertices - surface.vertices, 0.5 * surface.velocities, atol=1e-5)


class MotionPassTests(unittest.TestCase):
    def test_translation_matches_the_analytic_screen_velocity(self):
        speed = 2.0
        vectors = image(graph(motion=0, speed=speed, render_output="motion"), frame=0)
        front = DISTANCE - CUBE / 2
        expected = speed * pixels_per_unit(front)
        covered = vectors[..., 3] > 0
        self.assertGreater(int(covered.sum()), 100)
        np.testing.assert_allclose(vectors[..., 0][covered], expected, rtol=0.02)
        np.testing.assert_allclose(vectors[..., 1][covered], 0.0, atol=0.02)
        self.assertEqual(float(np.abs(vectors[~covered]).max()), 0.0)

    def test_camera_motion_gives_the_opposite_vector(self):
        pan = {"tx": curve([(-10, -10.0), (10, 10.0)]), "target_x": curve([(-10, -10.0), (10, 10.0)])}                # the camera moves right one unit a frame
        vectors = image(graph(speed=0.0, motion=0, render_output="motion", camera_curves=pan), frame=0)
        covered = vectors[..., 3] > 0
        expected = -1.0 * pixels_per_unit(DISTANCE - CUBE / 2)
        np.testing.assert_allclose(vectors[..., 0][covered], expected, rtol=0.03)

    def test_rotation_vectors_are_tangent_and_scale_with_radius(self):
        d = Dispatcher()
        for key, kind, params in (("plate", "Card3D", dict(emission=1.0, card_width=4.0, card_height=4.0)),
                                  ("spin", "Axis3D", {}), ("camera", "Camera3D", dict(tz=DISTANCE, focal=FOCAL)),
                                  ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=WIDTH, height=WIDTH, render_output="motion",
                                                              render_mode="pathtrace"))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        d.execute(dict(op="connect", id="spin", input="object", source="plate"))
        d.execute(dict(op="connect", id="scene", input="object0", source="spin"))
        d.execute(dict(op="connect", id="render", input="scene", source="scene"))
        d.execute(dict(op="connect", id="render", input="camera", source="camera"))
        d.document["animation"]["curves"]["spin"] = {"rz": curve([(-10, -100.0), (10, 100.0)])}   # 10 degrees a frame
        vectors = Evaluator().evaluate(d.document, "render", frame=0)
        scale = WIDTH * FOCAL / (18.672 * DISTANCE)
        omega = math.radians(10.0)
        for px, py in ((60, 32), (48, 20), (36, 40)):
            x, y = (px + 0.5 - WIDTH / 2) / scale, -(py + 0.5 - WIDTH / 2) / scale       # world offsets from the axis
            # rotating counter-clockwise by omega moves (x, y) by omega * (-y, x); image y points down
            expected = np.array((-y * omega, -x * omega)) * scale
            np.testing.assert_allclose(vectors[py, px, :2], expected, atol=0.06 * scale * math.hypot(x, y) * omega + 0.15)

    def test_velocity_geometry_moves_along_its_velocity(self):
        card = s._card(2.0, 2.0, (1, 1, 1, 1), s.Transform3D())
        moving = dataclasses.replace(card, emission=1.0, velocities=np.tile(np.array((1.5, 0, 0), np.float32),
                                                                             (len(card.vertices), 1)))
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, DISTANCE)), s.Vec3(0, 0, 0),
                          s.camera_from_node({"params": {**core.SPECS["Camera3D"]["params"], "focal": FOCAL}}).fov)
        scene = s.Scene((moving,))
        later = motionblur.next_scene(scene, s.Scene((card,)))
        vectors = motionblur.motion_vectors(scene, camera, later, camera, WIDTH, HEIGHT)
        covered = vectors[..., 3] > 0
        np.testing.assert_allclose(vectors[..., 0][covered], 1.5 * pixels_per_unit(DISTANCE), rtol=0.02)

    def test_multichannel_carries_the_motion_pass_beside_a_sharp_beauty(self):
        d = graph(motion=0, render_output="multichannel", passes="beauty,motion", mode="raytrace")
        raster = Evaluator().evaluate_raster(d.document, "render", frame=0)
        self.assertIn("motion", raster.layers)
        motion = raster.layers["motion"].pixels
        self.assertGreater(float(motion[..., 0].max()), 10.0)
        again = Evaluator().evaluate(graph(motion=0, mode="raytrace").document, "render", frame=0)
        np.testing.assert_allclose(raster.pixels, again, atol=1e-6)

    def test_the_motion_pass_drives_vectorblur_like_the_real_blur(self):
        """The 2D blur of a sharp render by the motion pass reaches about as far as the 3D motion blur."""
        d = graph(motion=0, render_output="multichannel", passes="beauty,motion", mode="raytrace", shutter=1.0)
        sharp = Evaluator().evaluate_raster(d.document, "render", frame=0)
        vectors = sharp.layers["motion"].pixels
        speed_px = float(vectors[..., 0][vectors[..., 3] > 0].mean())
        three_d = extent(image(graph(mode="raytrace", shutter=1.0)))
        self.assertAlmostEqual(three_d - extent(sharp.pixels), speed_px, delta=3.0)


@unittest.skipUnless(gpu_ready(), "wgpu adapter unavailable for the path tracer")
class GpuMotionBlurTests(unittest.TestCase):
    def test_gpu_shutter_shares_match_the_cpu(self):
        camera = s.Camera(s.Transform3D(position=s.Vec3(0, 0, 10)), s.Vec3(0, 0, 0), 30.0)
        dot = sphere(0.4, (1, 1, 1, 1), emission=1.0)
        moments = [(s.Scene((dataclasses.replace(dot, transform=s.Transform3D(position=s.Vec3(x, 0, 0))),)), camera)
                   for x in np.linspace(-1.0, 1.0, 9)]
        settings = pt.PathSettings(samples=72, max_bounces=0)
        cpu = pt.render_motion(moments, 48, 48, (0, 0, 0, 0), 0.0, "rgba", settings)
        gpu = pt.render_motion(moments, 48, 48, (0, 0, 0, 0), 0.0, "rgba", settings, backend="gpu")
        self.assertAlmostEqual(extent(cpu), extent(gpu), delta=1)
        self.assertLess(float(np.abs(cpu - gpu)[..., 3].mean()), 0.01)


if __name__ == "__main__":
    unittest.main()
