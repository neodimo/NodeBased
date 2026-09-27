"""Lane L4 plan 3, step B: fire (blackbody emission), multiple scattering, the phase knob and the Pyro look controls."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d, volumerender
from nodebased.core import LIMITS, SPECS, Dispatcher, upgrade_document
from nodebased.imaging import Evaluator
from tests import gpu_precision
from tests.test_volume_scene import box, make, wire

SIZE = 48
CAMERA = scene3d.Camera()          # at (0, 0, 5) looking at the origin
BASE = volumerender.VolumeSettings(step_size=0.05, absorption=0.4, shadow_steps=16)
FIRE = replace(BASE, fire_intensity=1.0)
TOLERANCE = 5e-3 if gpu_precision.half_float_target() else 2e-4


def render(scene, settings=BASE, size=SIZE, camera=CAMERA, **kwargs):
    return scene3d.render(scene, camera, size, size, samples=1, volume=settings, **kwargs)


def hot_cold(temperature_top, temperature_bottom, n=16, density=1.5):
    """A cube (2 units a side, filling the frame) whose upper half (y > 0) carries `temperature_top` and lower half
    `temperature_bottom`."""
    volume = box(n, density, 0.0, size=2.0)
    temperature = np.zeros((n, n, n), np.float32)
    temperature[:, n // 2:, :] = temperature_top
    temperature[:, :n // 2, :] = temperature_bottom
    return replace(volume, temperature=temperature)


class BlackbodyTests(unittest.TestCase):
    def test_a_1500_kelvin_flame_is_redder_than_daylight(self):
        warm, day = volumerender.blackbody_chromaticity(1500.0), volumerender.blackbody_chromaticity(6500.0)
        self.assertGreater(warm[0] / warm[2], 10 * day[0] / day[2])
        self.assertGreater(warm[0], warm[1])
        self.assertGreater(warm[1], warm[2])
        self.assertLess(abs(day[0] - day[2]), .25 * day[1])       # 6500 K is close to neutral
        np.testing.assert_allclose(warm @ [.2722287168, .6740817658, .0537895174], 1.0, atol=.02)   # luminance 1 in AP1

    def test_the_colour_warms_smoothly_as_the_kelvin_falls(self):
        kelvin = np.linspace(800, 8000, 40)
        ratio = volumerender.blackbody_chromaticity(kelvin)
        ratio = ratio[:, 0] / np.maximum(ratio[:, 2], 1e-9)
        self.assertTrue((np.diff(ratio) < 0).all())

    def test_brightness_grows_with_the_fourth_power_of_the_kelvin(self):
        one, two = volumerender.blackbody_radiance(1500.0), volumerender.blackbody_radiance(3000.0)
        np.testing.assert_allclose(two[1] / one[1], 16.0 * volumerender.blackbody_chromaticity(3000.0)[1]
                                   / volumerender.blackbody_chromaticity(1500.0)[1], rtol=1e-6)

    def test_the_table_lookup_follows_the_curve_and_a_ramp_replaces_it(self):
        table = volumerender.fire_table(BASE)
        kelvin = np.array([700.0, 1500.0, 2600.0])
        np.testing.assert_allclose(volumerender.fire_radiance(table, kelvin), volumerender.blackbody_radiance(kelvin), rtol=.02)
        ramp = replace(BASE, fire_ramp="1000:0,0,1;2000:1,0,0")
        blue, red = volumerender.fire_radiance(volumerender.fire_table(ramp), np.array([900.0, 2500.0]))
        np.testing.assert_allclose(blue, (0, 0, 1), atol=1e-6)
        np.testing.assert_allclose(red, (1, 0, 0), atol=1e-6)
        halfway = volumerender.fire_radiance(volumerender.fire_table(ramp), np.array([1500.0]))[0]
        np.testing.assert_allclose(halfway, (.5, 0, .5), atol=.03)

    def test_a_bad_ramp_is_refused_with_the_stop_named(self):
        with self.assertRaisesRegex(ValueError, "0,1"):
            replace(BASE, fire_ramp="1000:0,1").validated()


class FireTests(unittest.TestCase):
    def test_emission_adds_energy_only_where_the_temperature_passes_the_threshold(self):
        # Top half 1500 K, bottom half 300 K (below the 600 K threshold): the bottom must not change at all.
        scene = scene3d.Scene(volumes=(hot_cold(1.0, 0.2),))
        emission = replace(FIRE, fire_light=0.0)      # the smoke's own glow; fire_light is the next test
        plain, fire = render(scene, BASE), render(scene, emission)
        added = (fire - plain)[..., :3].sum(axis=-1)
        rows = np.arange(SIZE)
        top = added[(rows > 10) & (rows < SIZE // 2 - 2)][:, SIZE // 4:3 * SIZE // 4]
        bottom = added[rows > SIZE // 2 + 2][:, SIZE // 4:3 * SIZE // 4]
        self.assertGreater(float(top.min()), 0.05)
        self.assertEqual(float(np.abs(bottom).max()), 0.0)
        self.assertTrue((fire[..., :3] >= plain[..., :3] - 1e-6).all())      # fire never subtracts light
        # Raising the threshold above the top's kelvin switches the top off too.
        np.testing.assert_array_equal(render(scene, replace(FIRE, fire_threshold=1600.0)), plain)
        # ... and lowering it lights the bottom.
        low = render(scene, replace(emission, fire_threshold=100.0))
        self.assertGreater(float((low - fire)[rows > SIZE // 2 + 2].max()), 0.0)

    def test_a_hot_plume_glows_red_orange_and_hotter_is_whiter(self):
        def glow(temperature, settings=FIRE):
            scene = scene3d.Scene(volumes=(hot_cold(temperature, temperature),))
            return (render(scene, settings) - render(scene, BASE))[SIZE // 2, SIZE // 2, :3]      # the emission alone
        image = glow(1.0)
        self.assertGreater(image[0], image[1])
        self.assertGreater(image[1], image[2])
        cooler = glow(.6)
        self.assertGreater(image[2] / image[0], cooler[2] / cooler[0])          # 1500 K is bluer than 900 K
        self.assertGreater(image[0], cooler[0])                                  # and brighter
        scale = glow(1.0, replace(FIRE, temperature_scale=3000.0))
        self.assertGreater(scale[2] / scale[0], image[2] / image[0])

    def test_fire_intensity_scales_the_emission_and_zero_is_the_old_render(self):
        scene = scene3d.Scene(volumes=(hot_cold(1.0, 1.0),))
        plain, one, two = render(scene, BASE), render(scene, FIRE), render(scene, replace(FIRE, fire_intensity=2.0))
        np.testing.assert_array_equal(render(scene, replace(BASE, fire_intensity=0.0)), plain)
        # Emission is linear in the intensity (alpha is untouched by fire).
        np.testing.assert_allclose((two - plain)[..., :3], 2 * (one - plain)[..., :3], rtol=2e-5, atol=1e-6)
        np.testing.assert_array_equal(one[..., 3], plain[..., 3])

    def test_a_volume_without_temperature_does_not_glow(self):
        scene = scene3d.Scene(volumes=(box(16, 1.5),))
        np.testing.assert_array_equal(render(scene, FIRE), render(scene, BASE))

    def test_fire_lights_the_smoke_above_it(self):
        # Only the bottom is hot; the top is cold smoke. `fire_light` is the emission-to-scattering term.
        scene = scene3d.Scene(volumes=(hot_cold(0.0, 1.0),))
        plain = render(scene, BASE)
        dark = render(scene, replace(FIRE, fire_light=0.0))
        lit = render(scene, FIRE)
        rows = np.arange(SIZE)
        top = (rows > 8) & (rows < SIZE // 2 - 6)
        column = slice(SIZE // 4, 3 * SIZE // 4)
        gain = (lit - dark)[top][:, column, :3]
        np.testing.assert_array_equal(dark[top][:, column], plain[top][:, column])      # cold smoke, no fire light: as before
        self.assertGreater(float(gain.sum(axis=-1).max()), 0.02)
        self.assertGreater(float(gain[..., 0].sum()), 1.5 * float(gain[..., 2].sum()))          # orange light
        stronger = render(scene, replace(FIRE, fire_light=2.0))
        np.testing.assert_allclose((stronger - dark)[top][:, column, :3], 2 * gain, rtol=2e-4, atol=1e-6)
        # Nearer the fire is brighter than farther from it.
        near, far = (float((lit - dark)[r, SIZE // 2, :3].sum()) for r in (SIZE // 2 - 8, SIZE // 2 - 16))
        self.assertGreater(near, far)
        # No fire, no light: intensity 0 leaves the smoke unlit.
        np.testing.assert_array_equal(render(scene, replace(BASE, fire_light=5.0)), render(scene, BASE))

    def test_the_fire_light_grid_is_bounded_and_conserves_the_emission_it_blurs(self):
        volume = hot_cold(1.0, 1.0, n=20)
        grid, cell = volumerender.fire_light_grid(volume, replace(FIRE, fire_light=1.0))
        self.assertLessEqual(max(grid.shape[:3]), volumerender.FIRE_LIGHT_CELLS)
        self.assertEqual(grid.shape[-1], 3)
        self.assertTrue((grid >= 0).all())
        self.assertIsNone(volumerender.fire_light_grid(volume, replace(FIRE, fire_light=0.0)))
        self.assertIsNone(volumerender.fire_light_grid(box(8, 1.0), FIRE))


class PhaseTests(unittest.TestCase):
    def cube(self):
        return box(16, 1.2, size=1.6)

    def test_zero_anisotropy_is_isotropic_and_exact(self):
        self.assertEqual(volumerender._henyey_greenstein(0.3, 0.0), 1.0)
        n = 4001
        c = np.linspace(-1, 1, n)
        for g in (.3, .8, -.5):      # normalised: the mean over the sphere of P is 1
            self.assertAlmostEqual(float(np.trapezoid(volumerender._henyey_greenstein(c, g), c) / 2), 1.0, delta=2e-3)

    def light(self, z):
        return scene3d.Light("Directional", position=scene3d.Vec3(0.0, 0.0, z), target=scene3d.Vec3())

    def test_forward_scattering_brightens_the_backlit_side_and_dims_the_front(self):
        for z, brighter in ((-6.0, True), (6.0, False)):       # the camera sits at +z
            scene = scene3d.Scene(volumes=(self.cube(),), lights=(self.light(z),))
            iso = render(scene, replace(BASE, shadow_density=0.0), ambient=0.0)
            fwd = render(scene, replace(BASE, shadow_density=0.0, anisotropy=0.8), ambient=0.0)
            centre = (slice(SIZE // 2 - 4, SIZE // 2 + 4),) * 2
            ratio = float(fwd[centre][..., :3].sum() / iso[centre][..., :3].sum())
            self.assertEqual(ratio > 1.0, brighter, f'light at z={z}: ratio {ratio}')
            if brighter:
                self.assertGreater(ratio, 3.0)
            back = render(scene, replace(BASE, shadow_density=0.0, anisotropy=-0.8), ambient=0.0)
            self.assertEqual(float(back[centre][..., :3].sum()) < float(iso[centre][..., :3].sum()), brighter)

    def test_the_ambient_term_ignores_the_phase(self):
        scene = scene3d.Scene(volumes=(self.cube(),), lights=(scene3d.Light(intensity=0.0),))
        np.testing.assert_array_equal(render(scene, replace(BASE, anisotropy=0.8), ambient=.3), render(scene, BASE, ambient=.3))


class MultiScatterTests(unittest.TestCase):
    def scene(self):
        light = scene3d.Light("Directional", position=scene3d.Vec3(6, 0, 0), target=scene3d.Vec3())
        return scene3d.Scene(volumes=(box(16, 4.0),), lights=(light,))

    def test_zero_multi_scatter_is_the_single_scattering_render_bit_for_bit(self):
        scene = self.scene()
        reference = render(scene, volumerender.VolumeSettings(absorption=0.4), ambient=0.1)
        for blur in (0.0, 0.5, 1.0):
            image = render(scene, volumerender.VolumeSettings(absorption=0.4, multi_scatter=0.0, multi_scatter_blur=blur),
                           ambient=0.1)
            np.testing.assert_array_equal(image, reference)

    def test_multiple_scattering_fills_the_shadow_and_never_exceeds_the_unshadowed_light(self):
        scene = self.scene()
        single = render(scene, BASE, ambient=0.0)
        multi = render(scene, replace(BASE, multi_scatter=1.0), ambient=0.0)
        unshadowed = render(scene, replace(BASE, shadow_density=0.0), ambient=0.0)
        row = SIZE // 2
        far_side = (row, SIZE // 2 - 6)
        self.assertGreater(float(multi[far_side][:3].sum()), 1.3 * float(single[far_side][:3].sum()))
        self.assertTrue((multi[..., :3] <= unshadowed[..., :3] + 1e-6).all())      # energy stays bounded by the light
        half = render(scene, replace(BASE, multi_scatter=0.5), ambient=0.0)
        self.assertTrue((half[..., :3] >= single[..., :3] - 1e-6).all() and (half[..., :3] <= multi[..., :3] + 1e-6).all())
        self.assertEqual(float(np.abs(multi[..., 3] - single[..., 3]).max()), 0.0)      # extinction is unchanged

    def test_the_blur_knob_sets_how_far_the_light_leaks(self):
        scene = self.scene()
        sharp = render(scene, replace(BASE, multi_scatter=1.0, multi_scatter_blur=0.0), ambient=0.0)
        soft = render(scene, replace(BASE, multi_scatter=1.0, multi_scatter_blur=1.0), ambient=0.0)
        px = (SIZE // 2, SIZE // 2 - 6)
        self.assertGreater(float(soft[px][:3].sum()), float(sharp[px][:3].sum()))

    def test_it_costs_no_extra_density_lookups(self):
        scene = self.scene()
        calls = []
        original = volumerender._trilinear
        volumerender._trilinear = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        try:
            render(scene, BASE, size=16, ambient=0.0)
            single = len(calls)
            calls.clear()
            render(scene, replace(BASE, multi_scatter=0.7, multi_scatter_blur=0.3), size=16, ambient=0.0)
        finally:
            volumerender._trilinear = original
        self.assertEqual(len(calls), single)


class QualityPresetTests(unittest.TestCase):
    def test_presets_set_the_step_and_shadow_counts_over_the_volume_diagonal(self):
        scene = scene3d.Scene(volumes=(box(16, 1.0, size=2.0),))
        diagonal = float(np.linalg.norm([2.0, 2.0, 2.0]))
        got = {name: replace(BASE, quality=name).resolved(scene.volumes) for name in ("preview", "medium", "final")}
        self.assertLess(got["final"].step_size, got["medium"].step_size)
        self.assertLess(got["medium"].step_size, got["preview"].step_size)
        self.assertLess(got["preview"].shadow_steps, got["medium"].shadow_steps)
        self.assertLess(got["medium"].shadow_steps, got["final"].shadow_steps)
        for name, (steps, shadow, octaves) in volumerender.QUALITY_PRESETS.items():
            self.assertAlmostEqual(got[name].step_size * steps, diagonal, places=6)
            self.assertEqual((got[name].shadow_steps, got[name].octaves), (shadow, octaves))
        self.assertIs(BASE.resolved(scene.volumes), BASE)                       # custom keeps the knobs
        self.assertEqual(got["final"].resolved(scene.volumes), got["final"])    # idempotent

    def test_presets_change_how_many_samples_a_render_takes(self):
        scene = scene3d.Scene(volumes=(box(16, 2.0),), lights=(scene3d.Light(),))
        counts = {}
        original = volumerender._trilinear
        for name in ("custom", "preview", "final"):
            calls = []
            volumerender._trilinear = lambda *a, calls=calls, **k: (calls.append(1), original(*a, **k))[1]
            try:
                render(scene, replace(BASE, quality=name), size=16, ambient=0.0)
            finally:
                volumerender._trilinear = original
            counts[name] = len(calls)
        self.assertLess(counts["preview"], counts["final"])
        self.assertNotEqual(counts["custom"], counts["preview"])

    def test_all_presets_agree_on_a_smooth_volume(self):
        scene = scene3d.Scene(volumes=(box(16, 1.5),), lights=(scene3d.Light(),))
        images = {name: render(scene, replace(BASE, quality=name), ambient=.1) for name in ("preview", "final")}
        np.testing.assert_allclose(images["preview"], images["final"], atol=.06)


@unittest.skipUnless(gpu3d.available(), "no wgpu adapter")
class GPUParityTests(unittest.TestCase):
    CAM = scene3d.Camera(scene3d.Transform3D(scene3d.Vec3(0.3, 0.6, 3.2)), scene3d.Vec3(0, 0.5, 0))
    SUN = scene3d.Light("Directional", position=scene3d.Vec3(4, 5, 3), target=scene3d.Vec3())
    BACK = scene3d.Light("Point", color=(1.0, 0.6, 0.3), intensity=3.0, position=scene3d.Vec3(-1.5, 1.2, -1.0),
                         target=scene3d.Vec3(0, .5, 0), falloff_type="Quadratic")
    SETTINGS = volumerender.VolumeSettings(step_size=0.03, density_scale=8.0, shadow_steps=12)

    def check(self, lights, **knobs):
        scene = scene3d.Scene(volumes=(scene3d.analytic_plume(32, 0),), lights=lights)
        settings = replace(self.SETTINGS, **knobs)
        expected = scene3d.render(scene, self.CAM, 128, 96, volume=settings, ambient=.15)
        actual = gpu3d.render(scene, self.CAM, 128, 96, volume=settings, ambient=.15)
        self.assertLess(float(np.abs(actual - expected).max()), TOLERANCE * 10)
        self.assertLess(float(np.abs(actual - expected).mean()), TOLERANCE)
        return expected, actual

    def test_fire_matches_the_cpu_reference_lit_and_unlit(self):
        for lights in ((self.SUN,), ()):
            expected, _ = self.check(lights, fire_intensity=1.5)
            plain = scene3d.render(scene3d.Scene(volumes=(scene3d.analytic_plume(32, 0),), lights=lights), self.CAM, 128, 96,
                                   volume=self.SETTINGS, ambient=.15)
            self.assertGreater(float((expected - plain)[..., :3].max()), 0.2, 'the plume must glow')

    def test_a_custom_ramp_and_fire_light_off_match(self):
        self.check((self.SUN,), fire_intensity=1.0, fire_ramp="500:0.2,0,0;1500:2,0.5,0;3000:4,3,1")
        self.check((self.SUN,), fire_intensity=1.0, fire_light=0.0)

    def test_multiple_scattering_and_the_phase_match(self):
        self.check((self.SUN, self.BACK), multi_scatter=0.8, multi_scatter_blur=0.3, anisotropy=0.6)
        self.check((self.SUN,), anisotropy=-0.4)
        self.check((self.SUN,), multi_scatter=0.5, quality="final")

    def test_everything_together_and_the_presets_match(self):
        self.check((self.SUN, self.BACK), fire_intensity=2.0, multi_scatter=0.6, anisotropy=0.3, quality="medium")
        self.check((), fire_intensity=1.0, quality="preview")


class Render3DKnobTests(unittest.TestCase):
    def graph(self, **render):
        d = Dispatcher()
        make(d, p=("Plume3D", {"plume_resolution": 12}), s=("Scene3D", {}), c=("Camera3D", {"ty": .5, "tz": 3, "target_y": .5}),
             l=("Light3D", {"tx": -3.0, "ty": 2.0, "tz": 4.0}),
             r=("Render3D", {"width": 32, "height": 32, "samples": 1, "volume_density_scale": 4.0, **render}))
        wire(d, "s", "object0", "p")
        wire(d, "s", "object1", "l")
        wire(d, "r", "scene", "s")
        wire(d, "r", "camera", "c")
        return d

    def image(self, d):
        return Evaluator().evaluate(dict(d.document, view="r"), frame=1)

    def test_the_new_knobs_exist_with_neutral_defaults_and_limits(self):
        params = SPECS["Render3D"]["params"]
        self.assertEqual((params["volume_anisotropy"], params["volume_multi_scatter"], params["volume_fire_intensity"],
                          params["volume_quality"], params["volume_fire_ramp"]), (0.0, 0.0, 0.0, "custom", ""))
        self.assertEqual(LIMITS["volume_anisotropy"], (-0.99, 0.99))
        settings = volumerender.VolumeSettings()
        self.assertEqual((settings.anisotropy, settings.multi_scatter, settings.fire_intensity, settings.quality),
                         (0.0, 0.0, 0.0, "custom"))

    def test_old_documents_render_as_before_with_the_knobs_at_their_defaults(self):
        d = self.graph()
        doc = d.document
        new = ("volume_anisotropy", "volume_multi_scatter", "volume_multi_scatter_blur", "volume_fire_intensity",
               "volume_temperature_scale", "volume_fire_threshold", "volume_fire_light", "volume_fire_ramp", "volume_quality")
        for name in new:
            del doc["nodes"]["r"]["params"][name]
        upgraded = upgrade_document(doc)
        self.assertEqual(upgraded["nodes"]["r"]["params"]["volume_quality"], "custom")
        self.assertEqual(upgraded["nodes"]["r"]["params"]["volume_temperature_scale"], 1500.0)
        old = Evaluator().evaluate(dict(upgraded, view="r"), frame=1)
        np.testing.assert_array_equal(old, self.image(self.graph()))
        self.assertGreater(float(old[..., 3].max()), .05)

    def test_the_knobs_reach_the_raymarch(self):
        base = self.image(self.graph())
        fire = self.image(self.graph(volume_fire_intensity=1.0))
        self.assertGreater(float(fire[..., :3].sum()), float(base[..., :3].sum()))
        ramp = self.image(self.graph(volume_fire_intensity=1.0, volume_fire_ramp="500:0,0,3;9000:0,0,3"))
        self.assertEqual(float(ramp[..., 0].sum()), float(base[..., 0].sum()))     # a blue ramp adds no red
        self.assertGreater(float(ramp[..., 2].sum()), float(base[..., 2].sum()))
        for name, value in (("volume_multi_scatter", 1.0), ("volume_anisotropy", 0.7), ("volume_quality", "preview")):
            with self.subTest(knob=name):
                self.assertFalse(np.array_equal(self.image(self.graph(**{name: value})), base))


class ReferenceRenderTests(unittest.TestCase):
    def test_the_reference_render_is_committed_small_and_documented(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        image = root / "docs" / "images" / "volume_fire_smoke.png"
        self.assertTrue(image.exists())
        self.assertLess(image.stat().st_size, 200_000)
        self.assertIn("images/volume_fire_smoke.png", (root / "docs" / "3D_FOUNDATION.md").read_text(encoding="utf-8"))
