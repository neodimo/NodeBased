"""GPU depth, position and object_id passes for scenes that contain splats (step R7a): the first splat whose
accumulated opacity reaches one half, in per-pixel plane-depth order, in front of the opaque mesh depth, held
against the CPU reference (`scene3d.render`). Hit coverage and object ids must be identical; depth and position
agree to float32 rounding. Run on every adapter (`force-adapter.py integrated|cpu`)."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from nodebased import gpu3d, gpusplat, scene3d as s, splats
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

W, H = 47, 35
CENTRE = (H // 2, W // 2)       # the pixel centred on the optical axis of the default camera (5 units away)


def one(position, opacity=.9, size=1.0, yaw=0.0, pitch=0.0, scales=None):
    """A flat splat facing the default camera (normal along z), optionally turned about y and x."""
    cy, sy, cp, sp = np.cos(yaw/2), np.sin(yaw/2), np.cos(pitch/2), np.sin(pitch/2)
    q = np.array([cy*cp, cy*sp, sy*cp, -sy*sp])          # about x (pitch) after y (yaw), w first
    scales = (size, size, .01) if scales is None else scales
    return splats.SplatCloud(np.array([position], float), np.array([scales], float), q[None],
                             np.array([opacity]), np.full((1, 1, 3), 1.0), 0, colorspace='linear')


def many(count, seed, spread=(1.8, 1.2, 1.0)):
    """A random overlapping cloud of tilted splats of mixed size and opacity."""
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, 1, 3)); sh[:, 0] = (rng.uniform(.1, .9, (count, 3))-.5)/splats.C0
    return splats.SplatCloud(rng.uniform(-1, 1, (count, 3))*spread, rng.uniform(.06, .4, (count, 3)) * (1, 1, .08),
                             rng.normal(size=(count, 4)), rng.uniform(.15, .95, count), sh, 0, colorspace='linear')


def card(z=0.0, yaw=0.0, size=(6.0, 4.5), x=0.0):
    return s._card(size[0], size[1], (.4, .5, .6, 1), s.Transform3D(s.Vec3(x, 0, z), s.Vec3(0, yaw, 0)))


def scene(*clouds, geometries=()):
    return s.Scene(geometries=tuple(geometries), splats=tuple(s.SplatInstance(c) for c in clouds))


def both(sc, output, width=W, height=H):
    cpu = s.render(sc, s.Camera(), width, height, output=output)
    gpu = gpu3d.render(sc, s.Camera(), width, height, output=output)
    return np.asarray(cpu), np.asarray(gpu)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Parity(unittest.TestCase):
    def check(self, sc, output, tol=1e-4, width=W, height=H):
        cpu, gpu = both(sc, output, width, height)
        np.testing.assert_array_equal(gpu[..., 3] > 0, cpu[..., 3] > 0, err_msg=f'{output} coverage')
        if output == 'object_id':
            np.testing.assert_array_equal(gpu[..., 0], cpu[..., 0])
        np.testing.assert_allclose(gpu, cpu, atol=tol, rtol=0)
        return cpu

    def test_every_output_against_the_cpu_with_and_without_meshes(self):
        cloud = many(400, 3)
        cases = {'splats only': scene(cloud),
                 'mesh behind': scene(cloud, geometries=(card(-1.2),)),
                 'mesh in front': scene(cloud, geometries=(card(1.5, size=(1.5, 1.2)),)),
                 'mesh cutting through': scene(cloud, geometries=(card(0, yaw=.6, x=.031),)),
                 'two clouds and a mesh': scene(cloud, many(250, 9), geometries=(card(.1, yaw=-.5, x=.031),))}
        for name, sc in cases.items():
            for output in ('depth', 'position', 'object_id'):
                with self.subTest(case=name, output=output):
                    self.check(sc, output, tol=5e-4 if output == 'position' else 1e-4)

    def test_overlap_and_alpha_threshold(self):
        # Opacity accumulates front to back: .3 then .3 gives .51 (the second splat is the hit), .29 then .29
        # gives .4959 (no hit), one splat of .4 is never enough, and a splat of .6 is a hit by itself.
        for name, (first, second), expect in (
                ('.3 then .3', (.3, .3), 2), ('.29 then .29', (.29, .29), 0), ('.4 alone', (.4, 0), 0),
                ('.6 alone', (.6, 0), 1)):
            with self.subTest(name):
                parts = [one((0, 0, 0), first)] + ([one((0, 0, -1), second)] if second else [])
                sc = scene(*parts)
                cpu, gpu = both(sc, 'object_id')
                self.assertEqual(gpu[CENTRE][0], expect)
                self.assertEqual(cpu[CENTRE][0], expect)
                self.assertEqual(gpu[CENTRE][3], 1.0 if expect else 0.0)
                _, gpu_depth = both(sc, 'depth')
                if expect:
                    self.assertAlmostEqual(gpu_depth[CENTRE][0], 5.0 + (expect - 1), places=4)
                self.check(sc, 'object_id')

    def test_the_hit_is_found_beyond_the_registers_that_hold_the_nearest(self):
        # Forty faint splats of .05 each: the hit is the fourteenth nearest (1-.95^14 = .512), past the twelve
        # fragments one pass of the shader keeps.
        sc = scene(*[one((0, 0, -.05*k), .05) for k in range(40)])
        cpu, gpu = both(sc, 'object_id')
        self.assertEqual(cpu[CENTRE][0], 14)
        self.assertEqual(gpu[CENTRE][0], 14)
        _, depth = both(sc, 'depth')
        self.assertAlmostEqual(depth[CENTRE][0], 5.0 + .05*13, places=4)
        self.check(sc, 'object_id')

    def test_depth_order_follows_the_plane_not_the_centre(self):
        # Two wide splats with the same centre depth, turned in opposite directions: each is nearer on one half of
        # the picture. The GPU must sort per pixel by plane depth like the CPU (and as a centre sort could not).
        # (Centred a quarter pixel off the axis: planes that meet exactly on a pixel centre tie, and a tie is
        # decided by float rounding.)
        sc = scene(one((.03, 0, 0), .95, size=1.6, yaw=.7), one((.03, 0, 0), .95, size=1.6, yaw=-.7))
        cpu, gpu = both(sc, 'object_id')
        left, right = gpu[CENTRE[0], CENTRE[1]-9][0], gpu[CENTRE[0], CENTRE[1]+9][0]
        self.assertEqual({left, right}, {1.0, 2.0})
        np.testing.assert_array_equal(cpu[..., 0], gpu[..., 0])
        self.check(sc, 'depth')

    def test_mesh_occludes_and_is_occluded(self):
        splat = one((0, 0, 0), .95, size=1.2)
        for name, z, expect_id, expect_depth in (('card behind the splat', -1.0, 2, 5.0),
                                                 ('card in front of the splat', 1.0, 1, 4.0)):
            with self.subTest(name):
                sc = scene(splat, geometries=(card(z),))
                cpu, gpu = both(sc, 'object_id')
                # The mesh is object 1; the splat comes after the scene's one geometry, so it is object 2.
                self.assertEqual(gpu[CENTRE][0], expect_id)
                np.testing.assert_array_equal(cpu[..., 0], gpu[..., 0])
                _, depth = both(sc, 'depth')
                self.assertAlmostEqual(depth[CENTRE][0], expect_depth, places=4)
        # A card turned through the splat's plane: the splat is in front where the card is behind it and the other
        # way round; both ids appear and the picture equals the CPU's.
        sc = scene(one((.03, 0, 0), .95, size=1.2), geometries=(card(0, yaw=.8, size=(6, 4), x=.03),))
        cpu, gpu = both(sc, 'object_id')
        shown = set(np.unique(gpu[..., 0][gpu[..., 3] > 0]))
        self.assertEqual(shown, {1.0, 2.0})
        np.testing.assert_array_equal(cpu[..., 0], gpu[..., 0])

    def test_a_splat_behind_the_mesh_is_not_counted_towards_the_threshold(self):
        # .3 + .3 reaches the threshold only if both count; the front one is hidden by an opaque card between them,
        # so the pixel keeps the card (id 1) and its depth.
        sc = scene(one((0, 0, 1.5), .3), one((0, 0, -1.5), .3), geometries=(card(0),))
        cpu, gpu = both(sc, 'object_id')
        self.assertEqual(gpu[CENTRE][0], 1)
        self.assertEqual(cpu[CENTRE][0], 1)
        self.check(sc, 'depth')

    def test_position_matches_the_world_point_of_the_hit(self):
        sc = scene(one((.3, -.2, -.4), .95, size=1.5))
        cpu, gpu = both(sc, 'position')
        self.assertEqual(gpu[CENTRE][3], 1.0)
        np.testing.assert_allclose(gpu[CENTRE][:3], cpu[CENTRE][:3], atol=5e-4)
        self.assertAlmostEqual(gpu[CENTRE][2], -.4, places=3)     # the splat's plane is z = -.4
        self.check(sc, 'position', tol=5e-4)

    def test_empty_frame_and_off_screen_splats(self):
        sc = scene(one((50, 0, 0), .9))
        cpu, gpu = both(sc, 'depth')
        np.testing.assert_array_equal(cpu, gpu)
        self.assertEqual(float(gpu[..., 3].max()), 0.0)
        behind = scene(one((0, 0, 20), .9))
        np.testing.assert_array_equal(*both(behind, 'object_id'))

    def test_timings_are_recorded(self):
        scene_ = scene(many(60, 1))
        gpu3d.render(scene_, s.Camera(), W, H, output='depth')
        for key in ('data_bin_ms', 'data_resolve_ms', 'data_tile_entries'):
            self.assertIn(key, gpusplat.last_timings)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Fallbacks(unittest.TestCase):
    def test_unsupported_data_outputs_and_modes_stay_on_the_cpu(self):
        sc = scene(many(30, 2))
        for output in ('normals', 'uv', 'splats'):
            with self.subTest(output=output):
                with self.assertRaisesRegex(gpu3d.Unsupported, 'splat data passes|splats output'):
                    gpu3d.render(sc, s.Camera(), W, H, output=output)
        with self.assertRaisesRegex(gpu3d.Unsupported, 'splat data passes'):
            gpu3d.render(sc, s.Camera(), W, H, output='depth', mode='raytrace')
        transparent = scene(many(30, 2), geometries=(replace(card(), color=(1, 1, 1, .5)),))
        with self.assertRaisesRegex(gpu3d.Unsupported, 'transparent meshes'):
            gpu3d.render(transparent, s.Camera(), W, H, output='depth')

    def test_a_missing_capability_is_reported_not_dropped(self):
        sc = scene(many(30, 2))
        with patch.object(gpusplat, 'check_data_capability', return_value='test capability reason'):
            with self.assertRaisesRegex(gpu3d.Unsupported, 'test capability reason'):
                gpu3d.render(sc, s.Camera(), W, H, output='depth')

    def test_the_tile_list_budget_refuses_instead_of_truncating(self):
        sc = scene(many(200, 4))
        with patch.object(gpusplat, 'GPU_SPLAT_MEMORY_CAP', 256):
            with self.assertRaisesRegex(ValueError, 'GPU splat render needs|tile lists need'):
                gpu3d.render(sc, s.Camera(), W, H, output='depth')


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class NodeGraph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = self.tmp.name + '/cloud.ply'; splats.write_ply(many(300, 5), path)
        self.d = Dispatcher()
        for key, kind, params in (
                ('read', 'ReadSplat3D', dict(splat_path=path, splat_colorspace='linear')),
                ('scene', 'Scene3D', {}), ('camera', 'Camera3D', {}),
                ('render', 'Render3D', dict(width=W, height=H, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        for key, slot, source in (('scene', 'object0', 'read'), ('render', 'scene', 'scene'),
                                  ('render', 'camera', 'camera')):
            self.d.execute(dict(op='connect', id=key, input=slot, source=source))

    def render(self, output, backend):
        for param, value in (('render_output', output), ('render_backend', backend)):
            self.d.execute(dict(op='set', id='render', param=param, value=value))
        return np.asarray(Evaluator().evaluate(self.d.document, 'render'))

    def test_render3d_depth_position_and_object_id_run_on_the_gpu_and_match(self):
        for output in ('depth', 'position', 'object_id'):
            with self.subTest(output=output):
                cpu = self.render(output, 'cpu')
                gpu = self.render(output, 'gpu')
                auto = self.render(output, 'auto')
                np.testing.assert_array_equal(gpu[..., 3] > 0, cpu[..., 3] > 0)
                np.testing.assert_allclose(gpu, cpu, atol=5e-4, rtol=0)
                np.testing.assert_array_equal(auto, gpu)
                self.assertGreater(float(cpu[..., 3].sum()), 0)

    def test_outputs_still_on_the_cpu_fall_back_unchanged(self):
        cpu = self.render('normals', 'cpu')
        np.testing.assert_array_equal(self.render('normals', 'auto'), cpu)
        with self.assertRaisesRegex(ValueError, 'unsupported.*data passes.*CPU-only'):
            self.render('normals', 'gpu')


if __name__ == '__main__':
    unittest.main()
