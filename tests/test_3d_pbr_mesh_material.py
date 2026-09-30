"""Physically based (metal/roughness) mesh materials (docs/3D_FOUNDATION.md "Materials", plan
"Production look" R1): the same Cook-Torrance GGX BRDF splats already use, CPU raster, CPU ray
trace and the wgpu rasterizer's CPU fallback; `material` "standard" (old documents) is untouched.
"""
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import gpu3d, scene3d as s, splatshade as sh
from nodebased.core import CHOICES, Dispatcher, LIMITS, load_document
from nodebased.knobs import knob_layout
from tests.test_3d_environment_light import env_of
from tests.test_3d_raytrace_render import interior


class LegacyUnchangedTests(unittest.TestCase):
    """material="standard" (every old document) must never read the new pbr_* fields."""

    def test_standard_material_ignores_pbr_fields(self):
        card = s._card(4, 4, (.5, .3, .8, 1), s.Transform3D())
        light = s.Light(position=s.Vec3(1, 2, 3))
        baseline = s.render(s.Scene((card,), (light,)), s.Camera(), 40, 40, ambient=.1)
        self.assertEqual(card.material, "standard")
        for metallic, roughness, pbr_specular in ((0.0, .5, .5), (1.0, .9, .1), (.3, .02, .9)):
            varied = replace(card, metallic=metallic, pbr_roughness=roughness, pbr_specular=pbr_specular)
            image = s.render(s.Scene((varied,), (light,)), s.Camera(), 40, 40, ambient=.1)
            np.testing.assert_array_equal(image, baseline)

    def test_old_document_gets_defaults_and_stays_standard(self):
        d = Dispatcher()
        for key, kind, params in [('card', 'Card3D', {}), ('light', 'Light3D', {}),
                ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
                ('render', 'Render3D', dict(width=16, height=16, samples=1))]:
            d.execute(dict(op='create', id=key, type=kind, params=params))
        for slot, source in [('object0', 'card'), ('object1', 'light')]:
            d.execute(dict(op='connect', id='scene', input=slot, source=source))
        for slot in ('scene', 'camera'):
            d.execute(dict(op='connect', id='render', input=slot, source=slot))
        for key in ('metallic', 'pbr_roughness', 'pbr_specular', 'material'):
            d.document['nodes']['card']['params'].pop(key, None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps(d.document))
            loaded = load_document(path)
        params = loaded['nodes']['card']['params']
        self.assertEqual((params['material'], params['metallic'], params['pbr_roughness'], params['pbr_specular']),
                         ("standard", 0.0, 0.5, 0.5))
        geometry = s.geometry_from_node(loaded['nodes']['card'])
        self.assertEqual((geometry.material, geometry.metallic, geometry.pbr_roughness, geometry.pbr_specular),
                         ("standard", 0.0, 0.5, 0.5))


class KnobTests(unittest.TestCase):
    def test_pbr_choice_and_knobs_on_every_geometry_kind(self):
        self.assertEqual(CHOICES["material"], ["standard", "pbr", "liquid"])
        for name in ("metallic", "pbr_roughness", "pbr_specular"):
            self.assertEqual(LIMITS[name], (0.0, 1.0))
        for kind in ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D"):
            labels = {(g.kind, tuple(g.params)) for g in knob_layout(kind)}
            self.assertIn(("float_slider", ("metallic",)), labels)
            self.assertIn(("float_slider", ("pbr_roughness",)), labels)
            self.assertIn(("float_slider", ("pbr_specular",)), labels)


class EnergyTests(unittest.TestCase):
    """White furnace: a rough dielectric or metal sphere under a uniform environment of 1 stays at 1."""

    def test_white_furnace(self):
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0))
        sphere = s._sphere(1, 24, (1, 1, 1, 1), s.Transform3D())
        env = env_of(np.ones((16, 32, 3), np.float32))
        yy, xx = np.mgrid[:32, :32]
        inside = ((xx - 15.5) ** 2 + (yy - 15.5) ** 2) < 8 ** 2
        for metallic in (0.0, 0.5, 1.0):
            for roughness in (0.05, 0.4, 1.0):
                geo = replace(sphere, material="pbr", metallic=metallic, pbr_roughness=roughness)
                image = s.render(s.Scene((geo,), environments=(env,)), camera, 32, 32)
                rgb = image[..., :3][inside] / np.maximum(image[..., 3][inside], 1e-6)[:, None]
                np.testing.assert_allclose(rgb.mean(axis=0), 1.0, atol=0.05,
                                           err_msg=f"metallic={metallic} roughness={roughness}")

    def test_a_darker_dielectric_never_exceeds_the_light(self):
        camera = s.Camera(s.Transform3D(s.Vec3(0, 0, 4)), s.Vec3(0, 0, 0))
        sphere = replace(s._sphere(1, 24, (.5, .5, .5, 1), s.Transform3D()), material="pbr", pbr_roughness=.4)
        env = env_of(np.ones((16, 32, 3), np.float32))
        image = s.render(s.Scene((sphere,), environments=(env,)), camera, 32, 32)
        self.assertTrue(np.all(image[..., :3] <= image[..., 3:4] * 1.03))


class TintTests(unittest.TestCase):
    def test_metal_tints_by_base_color_dielectric_stays_white(self):
        card = replace(s._card(4, 4, (1.0, 0.15, 0.15, 1), s.Transform3D()), material="pbr", pbr_roughness=.3)
        light = s.Light("Point", (1, 1, 1), 3.0, s.Vec3(1, 1, 3))

        def peak_specular(metallic):
            image = s.render(s.Scene((replace(card, metallic=metallic),), (light,)),
                             s.Camera(), 33, 33, output="specular")
            return image[16, 16, :3]
        dielectric = peak_specular(0.0)
        metal = peak_specular(1.0)
        self.assertGreater(float(dielectric.min()), 0)
        self.assertLess(float(np.std(dielectric)), 0.02 * float(dielectric.mean()))
        ratio = metal / max(float(metal.sum()), 1e-9)
        base = np.array(card.color[:3]) / sum(card.color[:3])
        np.testing.assert_allclose(ratio, base, atol=0.03)


class RoughnessTests(unittest.TestCase):
    def test_roughness_widens_the_highlight_monotonically(self):
        card = replace(s._card(6, 6, (1, 1, 1, 1), s.Transform3D()), material="pbr", metallic=0.0, pbr_specular=.5)
        light = s.Light("Point", (1, 1, 1), 4.0, s.Vec3(1.4, 0, 3))
        camera = s.Camera()

        def highlight_area(roughness):
            # The lobe's footprint (pixels above 5% of its own peak), not a fixed pixel: a fixed
            # off-axis point rises then falls as an energy-conserving lobe both widens and flattens.
            image = s.render(s.Scene((replace(card, pbr_roughness=roughness),), (light,)),
                             camera, 65, 65, output="specular")
            channel = image[..., 0]
            return int((channel > 0.05 * channel.max()).sum())
        areas = [highlight_area(r) for r in (0.08, 0.25, 0.5, 0.85)]
        self.assertTrue(all(a <= b for a, b in zip(areas, areas[1:])), areas)
        self.assertGreater(areas[-1], areas[0] * 3)


class SplatMatchTests(unittest.TestCase):
    """A mesh and a splat sharing one metallic/roughness/base colour under one light must match:
    both call `splatshade._cook_torrance`."""

    def test_mesh_and_splat_match_under_one_light(self):
        position = np.zeros((1, 3))
        normal = np.array([[0.0, 0.0, 1.0]])
        eye = np.array([0.0, 0.0, 5.0])
        toward_eye = eye - position
        light = s.Light("Directional", (1, .8, .6), 1.3, s.Vec3(0, 0, 0), s.Vec3(1, 2, 3))
        lights = ((light, *light.world()),)
        base_rgb = np.array([[0.6, 0.3, 0.8]])
        for metallic in (0.0, 1.0):
            for roughness in (0.15, 0.6):
                diffuse, specular = s._shade_pbr_mesh(position, normal, toward_eye, base_rgb, lights, 0.05,
                                                       (), metallic, roughness, 0.04, None, True)
                mesh_colour = (base_rgb * diffuse + specular)[0]
                splat_colour = sh.shade_splats(np.zeros((1, 3)), base_rgb, position, normal, np.ones(1), eye,
                                               (light,), 0.05, 1.0, roughness=np.full(1, roughness),
                                               occlusion=np.ones(1), metallic=metallic)[0]
                np.testing.assert_allclose(mesh_colour, splat_colour, atol=1e-8, rtol=1e-6,
                                           err_msg=f"metallic={metallic} roughness={roughness}")


class RaytraceParityTests(unittest.TestCase):
    def test_raster_and_raytrace_agree(self):
        sphere = replace(s._sphere(.9, 20, (.7, .5, .2, 1), s.Transform3D()),
                         material="pbr", metallic=.6, pbr_roughness=.35)
        ground = s._card(6, 6, (.4, .4, .4, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)))
        lights = (s.Light(position=s.Vec3(-2, 3, 4)), s.Light("Point", (.3, .6, .9), .8, s.Vec3(2, 1, 3)))
        scene = s.Scene((sphere, ground), lights)
        a = s.render(scene, s.Camera(), 48, 36, ambient=.05)
        b = s.render(scene, s.Camera(), 48, 36, ambient=.05, mode="raytrace")
        ids = s.render(scene, s.Camera(), 48, 36, output="object_id")
        mask = interior(np.concatenate((ids, a[..., 3:4]), axis=2))
        self.assertGreater(mask.sum(), 10)
        np.testing.assert_allclose(b[mask], a[mask], atol=1e-4, rtol=0)


class GPUFallbackTests(unittest.TestCase):
    def test_gpu_ray_tracer_falls_back_to_cpu(self):
        # Y3 of 3, part 1: the GPU raster path now shades plain `pbr` metallic/roughness factors
        # itself instead of refusing (see tests/test_3d_gpu.py's `test_pbr_material_matches_the_cpu_reference`
        # and `EnvironmentRefusalBoundaries` for the raster path's remaining refusals: an Environment
        # or any texture map together with `pbr`). The GPU ray tracer has no `pbr` material table at
        # all yet and still refuses unconditionally; this check needs no real adapter either way.
        card = replace(s._card(3, 3, (.4, .6, .3, 1), s.Transform3D()), material="pbr", metallic=.4,
                       pbr_roughness=.4)
        scene = s.Scene((card,), (s.Light(),))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(scene, s.Camera(), 24, 24, mode="raytrace")
        cpu = s.render(scene, s.Camera(), 24, 24)
        self.assertGreater(float(cpu[..., 3].sum()), 0)


if __name__ == "__main__":
    unittest.main()
