"""Shadow bias, blur and samples on Light3D (lane L4 step B): CPU reference behaviour."""
import copy
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.core import Dispatcher, LIMITS, SPECS, upgrade_document, validate
from nodebased.scene3d import Light, Vec3

SIZE = (192, 144)
CAMERA = s.Camera(s.Transform3D(Vec3(0, 10, 1)))          # looking down at the ground


def ground():
    return s._card(12, 12, (1, 1, 1, 1), s.Transform3D(rotation=Vec3(-90, 0, 0)))


def blocker(height=1.0):
    return s._card(1, 1, (1, 1, 1, 1), s.Transform3D(Vec3(0, height, 0), Vec3(-90, 0, 0)))


def light(**kw):
    args = dict(kind="Point", position=Vec3(3, 4, 0), shadows=True)
    args.update(kw)
    return Light(**args)


def render(lt, size=SIZE, mode=None):
    scene = s.Scene((ground(), blocker()), (lt,))
    if mode is None:
        return s.render(scene, CAMERA, *size, ambient=.1)
    return gpu3d.render(scene, CAMERA, *size, ambient=.1, mode=mode)


def transition_pixels(image):
    """Pixels between the lit ground and the umbra, left of the darkest pixel on the centre row, that
    are strictly between 10% and 90% of the way from lit to dark."""
    row = image[image.shape[0] // 2, :, 0]
    edge = int(np.argmin(row))
    lit, dark = row[0], row[edge]
    part = row[:edge]
    return int(np.count_nonzero((part < lit - .1 * (lit - dark)) & (part > lit - .9 * (lit - dark))))


class KnobDefaultsTests(unittest.TestCase):
    def test_registered_with_defaults_and_limits(self):
        params = SPECS["Light3D"]["params"]
        self.assertEqual((params["shadow_bias"], params["shadow_blur"], params["shadow_samples"]),
                         (s.SHADOW_BIAS_DEFAULT, 0.0, 1))
        for name in ("shadow_bias", "shadow_blur", "shadow_samples"):
            self.assertIn(name, LIMITS)

    def test_light_from_node_reads_the_knobs(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "l", "type": "Light3D", "params": {
            "shadow_bias": .02, "shadow_blur": 3.0, "shadow_samples": 8}})
        lt = s.light_from_node(d.document["nodes"]["l"])
        self.assertEqual((lt.shadow_bias, lt.shadow_blur, lt.shadow_samples), (.02, 3.0, 8))

    def test_old_document_loads_with_defaults(self):
        d = Dispatcher()
        d.execute({"op": "create", "id": "l", "type": "Light3D", "params": {}})
        old = copy.deepcopy(d.document)
        for name in ("shadow_bias", "shadow_blur", "shadow_samples"):
            del old["nodes"]["l"]["params"][name]
        upgraded = upgrade_document(old)
        validate(upgraded)
        lt = s.light_from_node(upgraded["nodes"]["l"])
        self.assertEqual((lt.shadow_bias, lt.shadow_blur, lt.shadow_samples), (s.SHADOW_BIAS_DEFAULT, 0.0, 1))

    def test_default_knobs_render_the_hard_shadow_unchanged(self):
        base = render(light())
        # Samples without blur, and the explicit default bias, must not change a single bit.
        np.testing.assert_array_equal(base, render(light(shadow_samples=16)))
        np.testing.assert_array_equal(base, render(light(shadow_bias=s.SHADOW_BIAS_DEFAULT)))


class BiasTests(unittest.TestCase):
    def test_larger_bias_removes_a_constructed_acne_case(self):
        # A ground point with a card 0.005 above it (inside the default epsilon, well outside a tiny one).
        # A ray starting under the card is blocked: acne. The default bias lifts the origin over it.
        v0 = np.array([[-2, .005, -2], [-2, .005, 2]], float)
        e1 = np.array([[4, 0, 0], [4, 0, 0]], float)
        e2 = np.array([[0, 0, 4], [0, 0, -4]], float)
        prim = s.TriangleSet(v0, e1, e2, np.ones(2, np.float32))
        pos = np.array([[-.3, 0, -.2]], np.float32)
        normal = np.array([[0, 1, 0]], np.float32)
        args = (pos, normal, light(), np.array([0, 4, 0], np.float32), np.array([0, -1, 0], np.float32),
                v0, e1, e2, np.ones(2, np.float32), 1e-3 * 12, None)
        default = s._shadow_visibility(*args, triangles=prim)
        tiny = s._shadow_visibility(*(args[:2] + (light(shadow_bias=1e-4),) + args[3:]), triangles=prim)
        self.assertEqual(float(default[0]), 1.0)
        self.assertEqual(float(tiny[0]), 0.0)


class BlurTests(unittest.TestCase):
    def test_blur_widens_the_penumbra(self):
        widths = [transition_pixels(render(light(shadow_blur=blur, shadow_samples=32)))
                  for blur in (0.0, 4.0, 12.0)]
        self.assertLessEqual(widths[0], 2)
        self.assertGreaterEqual(widths[1], widths[0] + 1)
        self.assertGreaterEqual(widths[2], widths[1] + 2)

    def test_directional_blur_widens_the_penumbra_too(self):
        hard = transition_pixels(render(light(kind="Directional")))
        soft = transition_pixels(render(light(kind="Directional", shadow_blur=8.0,
                                              shadow_samples=32)))
        self.assertGreaterEqual(soft, hard + 2)

    def test_same_scene_renders_identically(self):
        lt = light(shadow_blur=4.0, shadow_samples=6)
        np.testing.assert_array_equal(render(lt), render(lt))

    def test_result_is_independent_of_chunking(self):
        lt = light(shadow_blur=4.0, shadow_samples=4)
        pts = np.random.default_rng(3).uniform(-2, 2, (300, 3)).astype(np.float32)
        pts[:, 1] = 0
        normal = np.tile(np.float32((0, 1, 0)), (300, 1))
        v0, e1, e2 = (np.array([[-.5, 1, -.5], [-.5, 1, .5]], float), np.array([[1, 0, 0]] * 2, float),
                      np.array([[0, 0, 1], [0, 0, -1]], float))
        alpha = np.ones(2, np.float32)
        args = (np.array([0, 4, 0], np.float32), np.array([0, -1, 0], np.float32),
                v0, e1, e2, alpha, 1.2e-2, None)
        whole = s._shadow_visibility(pts, normal, lt, *args)
        old = s._SHADOW_RAY_CHUNK
        try:
            s._SHADOW_RAY_CHUNK = 7
            chunked = s._shadow_visibility(pts, normal, lt, *args)
        finally:
            s._SHADOW_RAY_CHUNK = old
        np.testing.assert_array_equal(whole, chunked)

    def test_more_samples_are_smoother(self):
        # Inside the penumbra the visibility is a fraction; with one sample it is 0 or 1 per point.
        one = render(light(shadow_blur=6.0, shadow_samples=1))
        many = render(light(shadow_blur=6.0, shadow_samples=32))
        self.assertGreater(np.unique(many[..., 0]).size, np.unique(one[..., 0]).size)

    def test_hard_shadow_keeps_its_core(self):
        image = render(light(shadow_blur=3.0, shadow_samples=16))
        self.assertAlmostEqual(float(image[SIZE[1] // 2, :, 0].min()), .1, places=3)   # umbra stays at ambient


class CacheKeyTests(unittest.TestCase):
    def test_knob_change_changes_the_splat_shadow_key(self):
        base = light()
        keys = {s._light_key(base)}
        for change in (dict(shadow_bias=.01), dict(shadow_blur=2.0), dict(shadow_blur=2.0, shadow_samples=8)):
            key = s._light_key(replace(base, **change))
            self.assertNotIn(key, keys)
            keys.add(key)
        # Samples only matter once there is blur.
        self.assertEqual(s._light_key(base), s._light_key(replace(base, shadow_samples=9)))

    def test_splat_cache_invalidates_on_knob_change(self):
        from tests.test_3d_splat_shadow_cache import W, H, camera, lamp, shell
        s.clear_splat_shadow_cache()
        scene = s.Scene((), (lamp(),), (s.SplatInstance(shell(400), relight=1.0),))
        stat = lambda: s.splat_shadow_stats["rays_traced"]
        s.render(scene, camera(), W, H, ambient=.1)
        first = stat()
        s.render(scene, camera(), W, H, ambient=.1)
        self.assertEqual(first, stat())                          # warm: nothing traced
        for change in (dict(shadow_blur=3.0, shadow_samples=4), dict(shadow_bias=.02)):
            before = stat()
            soft = replace(scene, lights=(replace(lamp(), **change),))
            s.render(soft, camera(), W, H, ambient=.1)
            self.assertGreater(stat(), before, change)


@unittest.skipUnless(gpu3d.available(), "wgpu adapter unavailable")
class GpuShadowControlTests(unittest.TestCase):
    """The raster and ray-traced GPU paths apply bias, blur and samples like the CPU reference."""
    modes = ("raster", "raytrace")

    def both(self, lt, mode):
        try:
            return render(lt), render(lt, mode=mode)
        except gpu3d.Unsupported as exc:
            self.skipTest(str(exc))

    def test_default_hard_shadow_matches_cpu(self):
        for mode in self.modes:
            cpu, gpu = self.both(light(), mode)
            self.assertLess(float(np.abs(gpu - cpu).mean()), 5e-3, mode)

    def test_bias_matches_cpu_and_lifts_the_shadow_away(self):
        for mode in self.modes:
            cpu, gpu = self.both(light(shadow_bias=.3), mode)
            self.assertLess(float(np.abs(gpu - cpu).mean()), 5e-3, mode)
            # The origin is lifted above the blocker, so there is no umbra left on the ground.
            self.assertGreater(float(gpu[SIZE[1] // 2, :, 0].min()), .5, mode)
            self.assertLess(float(render(light(), mode=mode)[SIZE[1] // 2, :, 0].min()), .2, mode)

    def test_blur_matches_cpu_within_sampling_noise(self):
        for mode in self.modes:
            for kind in ("Point", "Directional"):
                lt = light(kind=kind, shadow_blur=8.0, shadow_samples=32)
                cpu, gpu = self.both(lt, mode)
                self.assertLess(float(np.abs(gpu - cpu).mean()), 5e-3, (mode, kind))
                self.assertLess(float(np.abs(gpu - cpu).max()), .15, (mode, kind))
                self.assertGreater(transition_pixels(gpu), transition_pixels(render(light(kind=kind), mode=mode)) + 1)

    def test_gpu_blur_is_reproducible(self):
        lt = light(shadow_blur=6.0, shadow_samples=8)
        for mode in self.modes:
            np.testing.assert_array_equal(render(lt, mode=mode), render(lt, mode=mode))


if __name__ == "__main__":
    unittest.main()
