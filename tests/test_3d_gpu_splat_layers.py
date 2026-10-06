"""Transparent meshes mixed with Gaussian splats on the GPU (Rendering 7 step S2): the ray tracer records every mesh
surface along each primary ray and `gpusplat.LayeredResolve` merges them with the splat fragments per pixel in
(depth, authored index) order, mesh surfaces first on equal depth, then composites front to back over the background,
as `splatraster.accumulate_splats` does on the CPU with `mesh_layers`. Held against the CPU reference
(`scene3d.render`) on mesh-in-front, mesh-behind and interleaved-depth fixtures, with pixel values worked out by hand
for the premultiplied colour and alpha. Run on every adapter (`force-adapter.py integrated|cpu`)."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from nodebased import gpu3d, gpurt_render, gpusplat, scene3d as s, splats
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

W, H = 47, 35
CENTRE = (H // 2, W // 2)       # the pixel centred on the optical axis of the default camera (5 units away)
TOL = 2e-3                      # the parity target of the splat renderer: float blending plus the 1e-4 stopping point


def disc(position, colour, opacity=.9, size=1.2, yaw=0.0, name=None):
    """A flat splat facing the default camera, of a given linear colour (turned about y by `yaw`)."""
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    sh = ((np.asarray(colour, float)-.5)/splats.C0)[None, None, :]
    return splats.SplatCloud(np.array([position], float), np.array([[size, size, .01]], float),
                             np.array([[cy, 0, sy, 0]]), np.array([opacity]), sh, 0, colorspace='linear')


def many(count, seed, spread=(1.8, 1.2, 1.0), opacity=(.15, .95)):
    """A random overlapping cloud of tilted splats of mixed size and opacity."""
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, 1, 3)); sh[:, 0] = (rng.uniform(.1, .9, (count, 3))-.5)/splats.C0
    return splats.SplatCloud(rng.uniform(-1, 1, (count, 3))*spread, rng.uniform(.06, .4, (count, 3))*(1, 1, .08),
                             rng.normal(size=(count, 4)), rng.uniform(*opacity, count), sh, 0, colorspace='linear')


def card(z=0.0, colour=(.2, .5, .9), alpha=.5, size=(6.0, 4.5), yaw=0.0, x=0.0, texture=None):
    return s._card(size[0], size[1], (*colour, alpha), s.Transform3D(s.Vec3(x, 0, z), s.Vec3(0, yaw, 0)), texture)


def scene(*clouds, geometries=(), **kwargs):
    return s.Scene(geometries=tuple(geometries), splats=tuple(s.SplatInstance(c) for c in clouds), **kwargs)


def both(sc, width=W, height=H, samples=1, background=(0, 0, 0, 0), mode='raster', ambient=0.0):
    cpu = s.render(sc, s.Camera(), width, height, background, samples=samples, mode=mode, ambient=ambient)
    gpu = gpu3d.render(sc, s.Camera(), width, height, background, samples=samples, mode=mode, ambient=ambient)
    return np.asarray(cpu), np.asarray(gpu)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Parity(unittest.TestCase):
    def check(self, sc, tol=TOL, **kwargs):
        cpu, gpu = both(sc, **kwargs)
        np.testing.assert_allclose(gpu, cpu, atol=tol, rtol=0)
        return cpu, gpu

    def test_mesh_in_front_behind_cutting_and_stacked(self):
        cloud = many(300, 3)
        cases = {'mesh behind': scene(cloud, geometries=(card(-1.2),)),
                 'mesh in front': scene(cloud, geometries=(card(1.5),)),
                 'mesh cutting through': scene(cloud, geometries=(card(0, yaw=.6),)),
                 'two cards around the cloud': scene(cloud, geometries=(card(1.0), card(-.3, colour=(.9, .3, .2)))),
                 'two clouds and a card': scene(cloud, many(250, 9), geometries=(card(.1, yaw=-.5),)),
                 'a nearly opaque card': scene(cloud, geometries=(card(.2, alpha=.998),)),
                 'a faint card': scene(cloud, geometries=(card(.2, alpha=.03),))}
        for name, sc in cases.items():
            for mode in ('raster', 'raytrace'):
                with self.subTest(case=name, mode=mode):
                    self.check(sc, mode=mode)

    def test_the_picture_is_not_a_coincidence_of_empty_frames(self):
        sc = scene(many(300, 3), geometries=(card(.2),))
        cpu, gpu = self.check(sc)
        self.assertGreater(cpu[..., 3].mean(), .3)
        # The card only: the splats change the picture.
        meshes = gpu3d.render(scene(geometries=(card(.2),)), s.Camera(), W, H)
        self.assertGreater(np.abs(gpu-np.asarray(meshes)).max(), .05)

    def test_background_and_supersampling(self):
        sc = scene(many(300, 5), geometries=(card(.2),))
        for background in ((0, 0, 0, 0), (.1, .2, .3, 1.0), (.2, .1, 0, .5)):
            for samples in (1, 2):
                with self.subTest(background=background, samples=samples):
                    self.check(sc, background=background, samples=samples)

    def test_lit_meshes_and_ambient(self):
        sc = scene(many(300, 5), geometries=(card(.2, yaw=.4), card(-.6, yaw=-.3, colour=(.8, .5, .1))),
                   lights=(s.Light(kind='Directional', position=s.Vec3(1, 1, 3), shadows=False),))
        self.check(sc, ambient=.2)
        shadowed = replace(sc, lights=(s.Light(kind='Directional', position=s.Vec3(1, 1, 3), shadows=True),))
        self.check(shadowed, ambient=.2)

    def test_textured_alpha_makes_a_mesh_transparent(self):
        # An opaque colour with a texture whose alpha is half on the left: only the left half is a layer.
        texture = np.ones((8, 8, 4), 'f4'); texture[:, :4] *= .5
        sc = scene(many(200, 2), geometries=(card(.3, alpha=1.0, texture=texture),))
        self.assertFalse(s._opaque_meshes(sc))
        self.check(sc)

    def test_mesh_surfaces_are_ordered_with_splats_by_depth_on_both_sides(self):
        # Splats in front of, between and behind two cards, one pixel each: the order matters because the
        # colours differ, so a wrong merge shows in the value (checked against the CPU and by hand below).
        sc = scene(disc((0, 0, 1.5), (1, 0, 0), .4), disc((0, 0, .5), (0, 1, 0), .4), disc((0, 0, -.5), (0, 0, 1), .4),
                   disc((0, 0, -1.5), (1, 1, 0), .4),
                   geometries=(card(1.0, alpha=.3), card(0.0, colour=(.9, .9, .9), alpha=.3), card(-1.0, alpha=.3)))
        self.check(sc, tol=1e-4)

    def test_no_splat_in_the_frame_is_the_meshes_alone(self):
        far = disc((0, 0, 40.0), (1, 0, 0), .9)                  # behind the camera's far plane
        sc = scene(far, geometries=(card(0.0),))
        cpu, gpu = self.check(sc, tol=1e-6)
        np.testing.assert_allclose(gpu, np.asarray(s.render(scene(geometries=(card(0.0),)), s.Camera(), W, H)), atol=1e-6)

    def test_many_faint_fragments_past_one_pass_and_a_mesh_between_them(self):
        # Forty faint splats (.05 each, 16 are kept per pass: three passes) with a card between the 20th and 21st.
        clouds = [disc((0, 0, -.05*k), (.2+.02*k, .3, .9-.02*k), .05, size=2.0) for k in range(40)]
        for z in (-.05*19.5, -.05*7.5, -.05*15.5, 1.0, -3.0):
            with self.subTest(card_z=z):
                self.check(scene(*clouds, geometries=(card(z, alpha=.4),)), tol=1e-4)
        passes = gpusplat.last_timings['beauty_passes']
        self.assertGreaterEqual(passes, 3)

    def test_dense_faint_cloud_and_a_card(self):
        rng = np.random.default_rng(21)
        cloud = replace(many(4000, 8), opacity=rng.uniform(.01, .09, 4000))
        self.check(scene(cloud, geometries=(card(-.4, alpha=.6),)), tol=TOL)
        self.check(scene(cloud, geometries=(card(.4, alpha=.6),)), tol=TOL)

    def test_banding_changes_nothing(self):
        sc = scene(many(300, 3), geometries=(card(.2, yaw=.3), card(-.7)))
        whole = np.asarray(gpu3d.render(sc, s.Camera(), 64, 70))
        per_ray = 64 + 16*gpusplat.LAYER_RECORD_VEC4
        with patch.object(gpurt_render, 'LAYER_BAND_BYTES', 64*per_ray*16):     # 16 rows (one tile row) per band
            banded = np.asarray(gpu3d.render(sc, s.Camera(), 64, 70))
        np.testing.assert_array_equal(banded, whole)
        cpu = np.asarray(s.render(sc, s.Camera(), 64, 70))
        np.testing.assert_allclose(whole, cpu, atol=TOL, rtol=0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Pixels(unittest.TestCase):
    """Values worked out by hand at the pixel on the optical axis. Unlit surfaces draw their authored colour,
    premultiplied by alpha; a splat's colour is its linear colour times its opacity at the pixel centre."""
    def centre(self, sc, **kwargs):
        cpu, gpu = both(sc, **kwargs)
        np.testing.assert_allclose(gpu[CENTRE], cpu[CENTRE], atol=1e-5)
        return gpu[CENTRE].astype(float)

    def test_card_in_front_of_a_splat(self):
        red = (1, 0, 0)
        for alpha in (.25, .5, .8):
            with self.subTest(alpha=alpha):
                got = self.centre(scene(disc((0, 0, 0), (0, .6, .2), .9), geometries=(card(1.0, red, alpha),)))
                expect = np.array((*(np.array(red)*alpha), alpha)) + (1-alpha)*np.array((0, .6*.9, .2*.9, .9))
                np.testing.assert_allclose(got, expect, atol=1e-5)

    def test_card_behind_a_splat(self):
        got = self.centre(scene(disc((0, 0, 0), (0, .6, .2), .9), geometries=(card(-1.0, (1, 0, 0), .5),)))
        expect = np.array((0, .6*.9, .2*.9, .9)) + .1*np.array((.5, 0, 0, .5))
        np.testing.assert_allclose(got, expect, atol=1e-5)

    def test_splat_between_two_cards(self):
        near, far = (1, 0, 0), (0, 0, 1)
        got = self.centre(scene(disc((0, 0, 0), (0, 1, 0), .6),
                                geometries=(card(1.0, near, .5), card(-1.0, far, .5))))
        front = np.array((.5, 0, 0, .5))
        splat = np.array((0, .6, 0, .6))
        back = np.array((0, 0, .5, .5))
        expect = front + .5*splat + .5*.4*back
        np.testing.assert_allclose(got, expect, atol=1e-5)

    def test_opaque_card_hides_the_splats_behind_it_and_not_those_in_front(self):
        got = self.centre(scene(disc((0, 0, 1.5), (0, 1, 0), .5), disc((0, 0, -1.5), (1, 0, 0), .9),
                                geometries=(card(0.0, (0, 0, 1), 1.0),)))
        np.testing.assert_allclose(got, np.array((0, .5, 0, .5)) + .5*np.array((0, 0, 1, 1)), atol=1e-5)

    def test_background_shows_through_what_is_left(self):
        got = self.centre(scene(disc((0, 0, 0), (0, .6, .2), .5), geometries=(card(1.0, (1, 0, 0), .5),)),
                          background=(.2, .4, .6, 1.0))
        covered = np.array((.5, 0, 0, .5)) + .5*np.array((0, .3, .1, .5))
        expect = covered + (1-covered[3])*np.array((.2, .4, .6, 1.0))
        np.testing.assert_allclose(got, expect, atol=1e-5)

    def test_a_card_in_front_of_the_splat_only_where_it_covers_the_picture(self):
        # A small card in front of the left of the picture: the splats there are seen through it, the rest not.
        sc = scene(disc((0, 0, 0), (0, .6, .2), .9, size=3.0), geometries=(card(1.0, (1, 0, 0), .5, size=(1.0, 1.0), x=-.9),))
        cpu, gpu = both(sc)
        np.testing.assert_allclose(gpu, cpu, atol=1e-5)
        inside, outside = gpu[CENTRE[0], W//2-9], gpu[CENTRE[0], W//2+9]
        self.assertGreater(inside[0], .2)
        self.assertEqual(outside[0], 0.0)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Routing(unittest.TestCase):
    def test_opaque_scenes_keep_their_old_path(self):
        opaque = scene(many(200, 3), geometries=(card(-1.0, alpha=1.0),))
        self.assertTrue(s._opaque_meshes(opaque))
        with patch.object(gpusplat, 'open_layered', side_effect=AssertionError('opaque scenes are not layered')):
            gpu3d.render(opaque, s.Camera(), W, H)
            gpu3d.render(opaque, s.Camera(), W, H, mode='raytrace')

    def test_transparent_scenes_use_the_layered_path_in_both_modes(self):
        sc = scene(many(200, 3), geometries=(card(-1.0),))
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode), patch.object(gpusplat, 'open_layered', wraps=gpusplat.open_layered) as spy:
                gpu3d.render(sc, s.Camera(), W, H, mode=mode)
                self.assertEqual(spy.call_count, 1)

    def test_what_stays_on_the_cpu_says_so(self):
        sc = scene(many(100, 3), geometries=(card(-1.0),))
        projected = replace(card(-1.0, alpha=1.0), projection=s.Projection(s.Camera(), np.ones((2, 2, 4), 'f4')))
        for mode in ('raster', 'raytrace'):
            with self.subTest(mode=mode), self.assertRaisesRegex(gpu3d.Unsupported, 'Camera-projected geometry'):
                gpu3d.render(scene(many(100, 3), geometries=(projected,)), s.Camera(), W, H, mode=mode)
            with self.subTest(mode=mode, output='depth'), self.assertRaisesRegex(
                    gpu3d.Unsupported, 'data passes.*CPU-only'):
                gpu3d.render(sc, s.Camera(), W, H, output='depth', mode=mode)
        with patch.object(gpusplat, 'check_layered_capability', return_value='test capability reason'), \
                self.assertRaisesRegex(gpu3d.Unsupported, 'test capability reason'):
            gpu3d.render(sc, s.Camera(), W, H)

    def test_too_many_mesh_surfaces_raise_the_cpu_error(self):
        stack = tuple(card(.1*k, alpha=.05) for k in range(gpusplat.MAX_MESH_LAYERS+1))
        sc = scene(many(100, 3), geometries=stack)
        with self.assertRaisesRegex(ValueError, 'MAX_MESH_LAYERS'):
            s.render(sc, s.Camera(), W, H)
        with self.assertRaisesRegex(ValueError, 'MAX_MESH_LAYERS'):
            gpu3d.render(sc, s.Camera(), W, H)
        # Sixteen are fine.
        both(replace(sc, geometries=stack[:gpusplat.MAX_MESH_LAYERS]))

    def test_the_pass_limit_is_a_fallback_not_a_wrong_picture(self):
        sc = scene(*[disc((0, 0, -.01*k), (.5, .5, .5), .02, size=2.0) for k in range(60)], geometries=(card(-1.0),))
        with patch.object(gpusplat, 'BEAUTY_MAX_PASSES', 2), self.assertRaisesRegex(gpu3d.Unsupported, 'CPU'):
            gpu3d.render(sc, s.Camera(), W, H)

    def test_a_cancelled_render_stops(self):
        import threading
        from nodebased.cancellation import Cancelled
        event = threading.Event(); event.set()
        with self.assertRaises(Cancelled):
            gpu3d.render(scene(many(100, 3), geometries=(card(-1.0),)), s.Camera(), W, H, cancel=event)


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class Graph(unittest.TestCase):
    """A Render3D node over a saved splat file and a transparent Card3D: `auto` takes the GPU, `gpu` agrees."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        path = self.tmp.name + '/cloud.ply'; splats.write_ply(many(250, 4), path)
        self.d = Dispatcher()
        for key, kind, params in (
                ('read', 'ReadSplat3D', dict(splat_path=path, splat_colorspace='linear')),
                ('card', 'Card3D', dict(card_width=6.0, card_height=4.5, red=.2, green=.5, blue=.9,
                                         alpha=.5, tz=.4)),
                ('scene', 'Scene3D', {}), ('camera', 'Camera3D', {}),
                ('render', 'Render3D', dict(width=W, height=H, samples=1))):
            self.d.execute(dict(op='create', id=key, type=kind, params=params))
        for slot, source in (('object0', 'read'), ('object1', 'card')):
            self.d.execute(dict(op='connect', id='scene', input=slot, source=source))
        for slot in ('scene', 'camera'):
            self.d.execute(dict(op='connect', id='render', input=slot, source=slot))

    def render(self, backend):
        self.d.execute(dict(op='set', id='render', param='render_backend', value=backend))
        return np.asarray(Evaluator().evaluate(self.d.document, 'render'))

    def test_auto_and_gpu_match_the_cpu_without_falling_back(self):
        expected = self.render('cpu')
        self.assertGreater(expected[..., 3].mean(), .2)
        with patch.object(s, 'render', side_effect=AssertionError('auto must not fall back')):
            automatic = self.render('auto')
            forced = self.render('gpu')
        np.testing.assert_allclose(automatic, expected, atol=TOL, rtol=0)
        np.testing.assert_array_equal(forced, automatic)
