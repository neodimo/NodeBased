"""PBR texture maps in the CPU path tracer (lane L4, "Plan Rendering 3", step X1 of 2).

Metallic-roughness, normal, occlusion and emissive maps on `scene3d.Geometry`, sampled per hit in
`pathtrace.py`. docs/3D_FOUNDATION.md "Materials" and the GPU refusal in `gpupathtrace.pack`.
"""
import dataclasses
import math
import unittest

import numpy as np

from nodebased import gpu3d, gpupathtrace, pathtrace as pt, scene3d as s
from tests.test_3d_pathtrace import FRONT, card, center, trace, uniform_env


def rgba(*channels):
    return np.array(((channels,),), np.float32)  # a 1x1 texture, broadcast by _sample's edge clamp


class MetallicRoughnessTests(unittest.TestCase):
    """A per-UV metallic-roughness texture overrides the shape's own scalar knobs (materials 3)."""

    def test_the_texture_darkens_a_metal_half_the_scalar_knobs_alone_would_not(self):
        # G is roughness, B is metallic (glTF's packing): the left half of the texture is a mirror
        # (roughness 0, metallic 1), the right half stays dielectric and rough like the scalar default.
        mr = np.zeros((1, 2, 4), np.float32)
        mr[0, 0] = (0, 0.0, 1.0, 1)   # left column (low u): mirror
        mr[0, 1] = (0, 1.0, 0.0, 1)   # right column (high u): rough dielectric
        textured = card(4, 4, (0.6, 0.6, 0.6, 1), (0, 0, 0), material="pbr", metallic=0.0, pbr_roughness=1.0,
                       metallic_roughness_texture=mr)
        # the light sits well off to one side: a mirror pixel's delta reflection almost never finds it
        light = s.Light("Point", (1, 1, 1), 6.0, s.Vec3(6, 0, 4), s.Vec3(0, 0, 0), shadows=False)
        scene = s.Scene((textured,), lights=(light,))
        img = trace(scene, FRONT, (32, 16), 64, max_bounces=1)
        left_half, right_half = img[:, :16, :3], img[:, 16:, :3]
        self.assertLess(float(left_half.mean()), float(right_half.mean()) * 0.4)
        # without the texture (the uniform, fully dielectric scalar), a light straight ahead lights
        # both halves alike by symmetry
        symmetric_light = s.Light("Point", (1, 1, 1), 6.0, s.Vec3(0, 0, 6), s.Vec3(0, 0, 0), shadows=False)
        flat = trace(s.Scene((dataclasses.replace(textured, metallic_roughness_texture=None),),
                             lights=(symmetric_light,)), FRONT, (32, 16), 64, max_bounces=1)
        self.assertAlmostEqual(float(flat[:, :16, :3].mean()) / float(flat[:, 16:, :3].mean()), 1.0, delta=0.05)

    def test_old_documents_with_no_texture_are_unaffected(self):
        ball = card(4, 4, (0.5, 0.5, 0.5, 1), (0, 0, 0), material="pbr", metallic=0.3, pbr_roughness=0.4)
        scene = s.Scene((ball,), environments=(uniform_env(),))
        a = trace(scene, samples=16, seed=1)
        b = trace(scene, samples=16, seed=1)
        np.testing.assert_array_equal(a, b)


class NormalMapTests(unittest.TestCase):
    """A tangent-space normal map tilts the shading normal by exactly what its texel encodes."""

    def test_tilting_the_normal_toward_the_light_brightens_it_by_the_predicted_cosine_ratio(self):
        # the card's UV layout (scene3d._card) makes its tangent world +X and bitangent world +Y, so a
        # texel encoding local normal (sin45, 0, cos45) tilts the shading normal exactly onto `wi` below.
        wi = np.array((1.0, 0.0, 1.0)) / math.sqrt(2)
        flat_cos = wi[2]  # N.wi for the untouched normal (0, 0, 1)
        texel = ((wi[0] + 1) / 2, 0.5, (wi[2] + 1) / 2, 1.0)
        normal_map = np.array(((texel,),), np.float32)
        # a directional light travelling from (1, 0, 1) toward the origin: wi = -direction = (1, 0, 1)/root 2
        light = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(1, 0, 1), s.Vec3(0, 0, 0))
        base_card = card(4, 4, (0.7, 0.7, 0.7, 1), (0, 0, 0))
        flat_scene = s.Scene((base_card,), lights=(light,))
        tilted_scene = s.Scene((dataclasses.replace(base_card, normal_texture=normal_map),), lights=(light,))
        flat = center(trace(flat_scene, FRONT, (16, 16), 96, output="diffuse", max_bounces=1)).mean()
        tilted = center(trace(tilted_scene, FRONT, (16, 16), 96, output="diffuse", max_bounces=1)).mean()
        predicted = 1.0 / flat_cos   # N.wi becomes 1 once the normal points straight at the light
        self.assertAlmostEqual(float(tilted) / float(flat), predicted, delta=0.06)

    def test_old_documents_with_no_normal_map_are_unaffected(self):
        light = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(1, 2, 3), s.Vec3(0, 0, 0))
        scene = s.Scene((card(4, 4, (0.7, 0.7, 0.7, 1), (0, 0, 0)),), lights=(light,))
        a = trace(scene, FRONT, (16, 16), 8, seed=3)
        b = trace(scene, FRONT, (16, 16), 8, seed=3)
        np.testing.assert_array_equal(a, b)


class EmissiveTests(unittest.TestCase):
    """`emissive_color`/`emissive_texture` add light independently of the base colour (materials 3)."""

    def test_a_black_surface_glows_the_emissive_colour(self):
        black = card(2, 2, (0.0, 0.0, 0.0, 1), (0, 0, 0), emissive_color=(0.4, 0.1, 0.9))
        img = trace(s.Scene((black,)), FRONT, (16, 16), 4, output="emission")
        np.testing.assert_allclose(center(img), np.broadcast_to((0.4, 0.1, 0.9), center(img).shape), atol=1e-5)

    def test_an_emissive_texture_multiplies_the_emissive_colour_per_texel(self):
        tex = rgba(0.5, 0.5, 0.5, 1.0)
        card_g = card(2, 2, (0, 0, 0, 1), (0, 0, 0), emissive_color=(1.0, 1.0, 1.0), emissive_texture=tex)
        img = trace(s.Scene((card_g,)), FRONT, (16, 16), 4, output="emission")
        np.testing.assert_allclose(center(img), np.broadcast_to((0.5, 0.5, 0.5), center(img).shape), atol=1e-5)


class OcclusionTests(unittest.TestCase):
    def test_the_occlusion_texture_darkens_the_diffuse_response(self):
        light = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(0, 0, 1), s.Vec3(0, 0, 0))
        plain = card(4, 4, (0.6, 0.6, 0.6, 1), (0, 0, 0))
        occluded = dataclasses.replace(plain, occlusion_texture=rgba(0.2, 0.2, 0.2, 1.0))
        bright = center(trace(s.Scene((plain,), lights=(light,)), FRONT, (16, 16), 32, output="diffuse")).mean()
        dark = center(trace(s.Scene((occluded,), lights=(light,)), FRONT, (16, 16), 32, output="diffuse")).mean()
        self.assertAlmostEqual(float(dark) / float(bright), 0.2, delta=0.03)


class GpuRefusalTests(unittest.TestCase):
    """The GPU path tracer refuses a scene it cannot yet shade correctly (like the base texture refusal)."""

    def test_each_new_pbr_map_is_refused_on_the_gpu(self):
        base = card(2, 2, (0.5, 0.5, 0.5, 1), (0, 0, 0), material="pbr")
        one_by_one = rgba(1, 1, 1, 1)
        for field in ("metallic_roughness_texture", "normal_texture", "occlusion_texture", "emissive_texture"):
            with self.subTest(field=field):
                geometry = dataclasses.replace(base, **{field: one_by_one})
                scene = s.Scene((geometry,), environments=(uniform_env(),))
                with self.assertRaises(gpu3d.Unsupported):
                    gpupathtrace.pack(pt.build_scene(scene))
        with self.subTest(field="emissive_color"):
            geometry = dataclasses.replace(base, emissive_color=(1.0, 0.0, 0.0))
            scene = s.Scene((geometry,), environments=(uniform_env(),))
            with self.assertRaises(gpu3d.Unsupported):
                gpupathtrace.pack(pt.build_scene(scene))


if __name__ == "__main__":
    unittest.main()
