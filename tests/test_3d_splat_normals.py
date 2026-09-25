"""Splat normals: the blended `normals_blend` pass, camera-facing orientation, k-nearest smoothing."""
from dataclasses import replace
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import gpu3d, gpusplat as g, scene3d as s, splats, splatraster as r
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.splatshade import (estimated_normals, nearest_neighbours, normal_confidence,
                                  orient_to_eye, smooth_normals)
from tests import gpu_precision

SIZE = 33


def cloud(positions=((0, 0, 0),), scales=(1, 1, .01), rotations=None, opacity=.8, colour=(.6, .3, .04)):
    n = len(positions)
    sh = np.zeros((n, 1, 3))
    sh[:, 0] = (np.asarray(colour) - .5) / splats.C0
    return splats.SplatCloud(positions, np.broadcast_to(scales, (n, 3)),
                             np.tile((1, 0, 0, 0), (n, 1)) if rotations is None else rotations,
                             np.full(n, opacity), sh, 0, colorspace='linear')


def yaw(degrees):
    half = np.deg2rad(degrees) / 2
    return (np.cos(half), 0, np.sin(half), 0)


def noisy_plane(n=400, noise=.35, seed=3):
    """Splats on the z=0 plane whose normals are tilted at random: the estimate is noisy, the truth is +z."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(n, 3)); axis[:, 2] = 0
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.normal(0, noise, n)
    quaternions = np.column_stack((np.cos(angle / 2), axis * np.sin(angle / 2)[:, None]))
    positions = np.column_stack((rng.uniform(-1, 1, (n, 2)), np.zeros(n)))
    return cloud(positions, (.12, .12, .01), quaternions, opacity=.9)


class BlendPass(unittest.TestCase):
    def render(self, scene, camera=None, **kw):
        return s.render(scene, camera or s.Camera(), SIZE, SIZE, output='normals_blend', **kw)

    def test_flat_card_facing_the_camera_is_the_constant_vector(self):
        scene = s.Scene(splats=(s.SplatInstance(cloud()),))
        image = self.render(scene)
        for x, y in ((16, 16), (12, 20), (20, 12)):
            np.testing.assert_allclose(image[y, x, :3], (0, 0, 1), atol=1e-6)
        np.testing.assert_allclose(image[16, 16, 3], .8, atol=1e-6)
        back = replace(s.Camera(), transform=s.Transform3D(s.Vec3(0, 0, -5)))
        np.testing.assert_allclose(self.render(scene, back)[16, 16, :3], (0, 0, -1), atol=1e-6)

    def test_splat_plane_matches_mesh_plane(self):
        angle = 30
        transform = s.Transform3D(s.Vec3(0, 0, 0), s.Vec3(0, angle, 0))
        mesh = s.Scene((s._card(2, 2, (1, 1, 1, 1), transform),))
        splat = s.Scene(splats=(s.SplatInstance(cloud(scales=(.5, .5, .01), rotations=[yaw(angle)], opacity=.99)),))
        expected = s.render(mesh, s.Camera(), SIZE, SIZE, output='normals')
        actual = self.render(splat)
        inside = (expected[..., 3] > 0) & (actual[..., 3] > .5)
        self.assertGreater(inside.sum(), 40)
        np.testing.assert_allclose(actual[inside][:, :3], expected[inside][:, :3], atol=1e-3)
        np.testing.assert_allclose(expected[16, 16, :3], (np.sin(np.deg2rad(angle)), 0, np.cos(np.deg2rad(angle))), atol=1e-6)

    def test_overlapping_splats_blend_by_alpha(self):
        # Two coincident planes tilted +-20 degrees: the blend is the direction between them, the
        # first-hit `normals` pass reports only one of them.
        c = cloud(positions=((0, 0, 0), (0, 0, 0)), scales=(.6, .6, .01), rotations=[yaw(20), yaw(-20)], opacity=.5)
        scene = s.Scene(splats=(s.SplatInstance(c),))
        blend = self.render(scene)[16, 16, :3]
        # Front splat (input order on a depth tie) weighs .5, the one behind it .5 * .5.
        a, b = np.array((np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20)))), np.array((-np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20))))
        expected = .5 * a + .25 * b
        np.testing.assert_allclose(blend, expected / np.linalg.norm(expected), atol=1e-3)
        first = s.render(scene, s.Camera(), SIZE, SIZE, output='normals')[16, 16, :3]
        self.assertGreater(abs(first[0]), abs(blend[0]) + .1)

    def test_without_splats_it_is_the_normals_pass(self):
        mesh = s.Scene((s._sphere(.9, 12, (1, 1, 1, 1), s.Transform3D()),))
        np.testing.assert_array_equal(self.render(mesh), s.render(mesh, s.Camera(), SIZE, SIZE, output='normals'))

    def test_splat_over_a_mesh_blends_through_transparency(self):
        wall = s._card(4, 4, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, -1)))
        scene = s.Scene((wall,), splats=(s.SplatInstance(cloud(scales=(.5, .5, .01), rotations=[yaw(40)], opacity=.5)),))
        n = self.render(scene)[16, 16]
        self.assertAlmostEqual(n[3], 1, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(n[:3])), 1, places=5)
        self.assertGreater(n[0], .05)          # pulled toward the splat's tilt ...
        self.assertLess(n[0], np.sin(np.deg2rad(40)))  # ... but not all the way: the wall shows through
        # An opaque wall in front hides the splat completely.
        front = s.Scene((s._card(4, 4, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, 1))),), splats=scene.splats)
        np.testing.assert_allclose(self.render(front)[16, 16, :3], (0, 0, 1), atol=1e-6)

    def test_gpu_backend_reports_cpu_only(self):
        if not gpu3d.available():
            self.skipTest('No wgpu adapter')
        scene = s.Scene(splats=(s.SplatInstance(cloud()),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), SIZE, SIZE, output='normals_blend')


class Orientation(unittest.TestCase):
    def test_no_normal_faces_away_from_the_camera(self):
        rng = np.random.default_rng(5)
        c = cloud(rng.uniform(-3, 3, (500, 3)), (.2, .3, .05), rng.normal(size=(500, 4)))
        for eye in ((0, 0, 5), (4, -3, -2), (-6, 1, 1)):
            for k in (0, 5):
                n = estimated_normals(c, eye, k)
                oriented = orient_to_eye(n, c.positions, eye)
                facing = np.sum(oriented * (np.asarray(eye) - c.positions), axis=1)
                self.assertTrue((facing >= 0).all())
                if k:
                    # smoothed normals are already eye-facing, and unit length
                    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-6)
                    self.assertTrue((np.sum(n * (np.asarray(eye) - c.positions), axis=1) > -1e-6).mean() > .98)

    def test_blend_pass_never_faces_away(self):
        c = noisy_plane(120)
        for eye_z in (5, -5):
            camera = replace(s.Camera(), transform=s.Transform3D(s.Vec3(0, 0, eye_z)))
            image = s.render(s.Scene(splats=(s.SplatInstance(c, normal_smoothing=4),)), camera, SIZE, SIZE, output='normals_blend')
            covered = image[..., 3] > .3
            self.assertTrue(covered.any())
            self.assertTrue((image[covered][:, 2] * np.sign(eye_z) > 0).all())


class Smoothing(unittest.TestCase):
    def test_reduces_variance_on_a_noisy_cloud(self):
        c = noisy_plane()
        eye = (0, 0, 5)
        raw = orient_to_eye(c.normals(), c.positions, eye)
        smooth = estimated_normals(c, eye, 8)
        error = lambda n: np.linalg.norm(n - (0, 0, 1), axis=1).mean()
        self.assertLess(error(smooth), .5 * error(raw))
        again = estimated_normals(c, eye, 8)
        np.testing.assert_array_equal(smooth, again)      # deterministic

    def test_zero_is_todays_estimate_untouched(self):
        c = noisy_plane(60)
        n = c.normals()
        self.assertIs(smooth_normals(c.positions, n, c.scales, (0, 0, 5), 0), n)
        np.testing.assert_array_equal(estimated_normals(c, (0, 0, 5), 0), n)

    def test_round_blobs_do_not_drag_their_neighbours(self):
        positions = [(0, 0, 0), (.1, 0, 0), (0, .1, 0), (.1, .1, 0)]
        scales = np.array([(1, 1, .01)] * 3 + [(.3, .3, .3)])
        c = cloud(positions, scales)
        self.assertEqual(normal_confidence(c.scales)[3], 0)
        np.testing.assert_allclose(estimated_normals(c, (0, 0, 5), 3)[:3], [(0, 0, 1)] * 3, atol=1e-6)

    def test_grid_search_matches_brute_force(self):
        rng = np.random.default_rng(11)
        for points in (rng.uniform(size=(3000, 3)), np.column_stack((rng.uniform(size=(3000, 2)), np.zeros(3000)))):
            idx, valid = nearest_neighbours(points, 6)
            d = ((points[:, None] - points[None]) ** 2).sum(2)
            d[np.arange(len(points)), np.arange(len(points))] = np.inf
            expected = np.argsort(d, axis=1, kind='stable')[:, :6]
            self.assertTrue(valid.all())
            # Exact except where a neighbour lies beyond the 3x3x3 cell block: a handful in thousands.
            self.assertGreater((idx == expected).mean(), .998)

    def test_relight_uses_smoothed_normals_and_zero_is_unchanged(self):
        c = noisy_plane(200)
        light = s.Light(position=s.Vec3(0, 0, 1), intensity=1.0)
        def render(k):
            scene = s.Scene(lights=(light,), splats=(s.SplatInstance(c, relight=1, normal_smoothing=k),))
            return s.render(scene, s.Camera(), SIZE, SIZE, ambient=.1)
        base = s.render(s.Scene(lights=(light,), splats=(s.SplatInstance(c, relight=1),)), s.Camera(), SIZE, SIZE, ambient=.1)
        np.testing.assert_array_equal(render(0), base)
        smooth = render(8)
        self.assertFalse(np.array_equal(smooth, base))
        # A plane facing the light and camera lights evenly once its normals are smoothed.
        covered = smooth[..., 3] > .9
        self.assertLess(smooth[covered][:, 0].std(), base[covered][:, 0].std())

    def test_data_pass_uses_smoothed_normals(self):
        c = noisy_plane(300)
        def pass_(k):
            scene = s.Scene(splats=(s.SplatInstance(c, opacity_scale=5, normal_smoothing=k),))
            return s.render(scene, s.Camera(), SIZE, SIZE, output='normals')
        raw, smooth = pass_(0), pass_(8)
        covered = raw[..., 3] > 0
        error = lambda image: np.linalg.norm(image[covered][:, :3] - (0, 0, 1), axis=1).mean()
        self.assertLess(error(smooth), error(raw))


class Knob(unittest.TestCase):
    def test_node_knob_default_range_and_old_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'plane.ply')
            splats.write_ply(noisy_plane(80), path)
            d = Dispatcher()
            for key, kind, params in (('read', 'ReadSplat3D', dict(splat_path=str(path), splat_relight=1.0)),
                                      ('light', 'Light3D', dict(tz=5.0)),
                                      ('cam', 'Camera3D', dict(tz=5.0)), ('scene', 'Scene3D', {}),
                                      ('render', 'Render3D', dict(width=SIZE, height=SIZE, ambient=.1))):
                d.execute(dict(op='create', id=key, type=kind))
                for param, value in params.items():
                    d.execute(dict(op='set', id=key, param=param, value=value))
            d.execute(dict(op='connect', id='scene', input='object0', source='read'))
            d.execute(dict(op='connect', id='scene', input='object1', source='light'))
            d.execute(dict(op='connect', id='render', input='scene', source='scene'))
            d.execute(dict(op='connect', id='render', input='camera', source='cam'))
            self.assertEqual(0, d.document['nodes']['read']['params']['splat_normal_smoothing'])
            off = Evaluator().evaluate(d.document, target='render')
            d.execute(dict(op='set', id='read', param='splat_normal_smoothing', value=6))
            on = Evaluator().evaluate(d.document, target='render')
            self.assertFalse(np.array_equal(off, on))
            d.execute(dict(op='set', id='render', param='render_output', value='normals_blend'))
            blend = Evaluator().evaluate(d.document, target='render')
            self.assertGreater(blend[..., 3].max(), .5)
            with self.assertRaises(Exception):
                d.execute(dict(op='set', id='read', param='splat_normal_smoothing', value=65))
            d.execute(dict(op='set', id='read', param='splat_normal_smoothing', value=0))
            blend = Evaluator().evaluate(d.document, target='render')
            document = copy.deepcopy(d.document)
            del document['nodes']['read']['params']['splat_normal_smoothing']
            with tempfile.TemporaryDirectory() as other:
                target = Path(other, 'old.json')
                target.write_text(json.dumps(document))
                loaded = load_document(target)
            self.assertEqual(0, loaded['nodes']['read']['params']['splat_normal_smoothing'])
            np.testing.assert_array_equal(Evaluator().evaluate(loaded, target='render'), blend)


@unittest.skipUnless(gpu3d.available(), 'No wgpu adapter')
class GPUParity(unittest.TestCase):
    def test_relit_splats_with_smoothed_normals_match_the_cpu(self):
        state = gpu3d._state()
        camera = s.Camera()
        c = noisy_plane(300)
        lights = (s.Light(position=s.Vec3(1, 1, 1)),)
        for k in (0, 6):
            with self.subTest(smoothing=k):
                instance = s.SplatInstance(c, relight=1, normal_smoothing=k)
                actual = g.render_layer(state, [instance], camera, 64, 48, lighting=(lights, .15, None))
                prepared = r.prepare_splats([instance], camera, 64, 48, lighting=(lights, .15))
                expected = r.accumulate_splats(prepared)
                gpu_precision.note(self.id())
                for a, b in zip(actual, expected):
                    error = np.abs(a - b)
                    self.assertLessEqual(error.max(), gpu_precision.tolerance(2e-3))
                    self.assertLessEqual(error.mean(), gpu_precision.tolerance(2e-4))
        plain = g.render_layer(state, [replace(instance, normal_smoothing=0)], camera, 64, 48, lighting=(lights, .15, None))
        smooth = g.render_layer(state, [instance], camera, 64, 48, lighting=(lights, .15, None))
        self.assertGreater(np.abs(plain[0] - smooth[0]).max(), 1e-3)   # smoothing reaches the GPU picture


if __name__ == '__main__':
    unittest.main()
