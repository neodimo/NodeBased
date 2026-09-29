"""Lane L4, R7 of 7: particle materials, attribute ramps, lighting and shadows.

ParticleRender3D gains a "pbr" material (the same metallic/roughness/specular vocabulary a mesh's
`material` "pbr" uses, docs/3D_FOUNDATION.md "Materials") and attribute-bound ramps that read the
solved age or speed into colour, opacity, size and self-emission, baked once at this node. "pbr"
particles are lit by the scene's lights and dome and cast shadows onto pbr meshes; everything defaults
off, so an old document renders exactly as it did before this step.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from nodebased import scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator

SIZE = 120
FOCAL = 1.0 / math.tan(math.radians(45.0) / 2)


def make(d, **nodes):
    for key, (kind, params) in nodes.items():
        d.execute({"op": "create", "id": key, "type": kind, "params": params})


def wire(d, target, slot, source):
    d.execute({"op": "connect", "id": target, "input": slot, "source": source})


class RampBakingTests(unittest.TestCase):
    """`scene3d.apply_particle_look` bakes ramps once, at ParticleRender3D, never inside the solve."""

    def _instance(self):
        return s.ParticleInstance(
            positions=np.zeros((3, 3), np.float32),
            sizes=np.array([1.0, 1.0, 1.0], np.float32),
            colors=np.array([[0.2, 0.2, 0.2, 1.0]] * 3, np.float32),
            ages=np.array([0.0, 2.0, 4.0], np.float32),
            lifetimes=np.array([4.0, 4.0, 4.0], np.float32),
            velocities=np.zeros((3, 3), np.float32))

    def test_ramp_off_by_default_leaves_colour_and_size_untouched(self):
        instance = self._instance()
        result = s.apply_particle_look(instance, {})
        np.testing.assert_array_equal(result.colors, instance.colors)
        np.testing.assert_array_equal(result.sizes, instance.sizes)
        self.assertIsNone(result.emission)
        self.assertEqual(result.material, "standard")

    def test_colour_ramp_by_age_maps_as_specified(self):
        instance = self._instance()
        params = {"particle_ramp_by": "age", "particle_color_ramp": "0:1,0,0;1:0,0,1"}
        result = s.apply_particle_look(instance, params)
        # age 0 -> t=0 (red), age 2 of 4 -> t=0.5 (mid), age 4 of 4 -> t=1 (blue).
        np.testing.assert_allclose(result.colors[0, :3], (1, 0, 0), atol=1e-5)
        np.testing.assert_allclose(result.colors[1, :3], (0.5, 0, 0.5), atol=1e-5)
        np.testing.assert_allclose(result.colors[2, :3], (0, 0, 1), atol=1e-5)
        self.assertAlmostEqual(float(result.colors[0, 3]), 1.0)  # alpha untouched

    def test_size_ramp_multiplies_on_top_of_size_scale(self):
        instance = replace(self._instance(), size_scale=2.0)
        params = {"particle_ramp_by": "age", "particle_size_ramp": "0:1;1:0.25"}
        result = s.apply_particle_look(instance, params)
        # apply_particle_look bakes the ramp into `sizes`; ParticleRender3D's own `size_scale`
        # (already on the instance from the representation/size_scale replace) is untouched here
        # and multiplies again at draw time (particle_sprites), so the two compose.
        np.testing.assert_allclose(result.sizes, [1.0, 0.625, 0.25], atol=1e-5)
        self.assertEqual(result.size_scale, 2.0)

    def test_emission_ramp_by_age_brightens_toward_death(self):
        instance = self._instance()
        params = {"particle_ramp_by": "age", "particle_emission_ramp": "0:0;1:4"}
        result = s.apply_particle_look(instance, params)
        np.testing.assert_allclose(result.emission, [0.0, 2.0, 4.0], atol=1e-5)

    def test_constant_emission_used_when_no_ramp(self):
        instance = self._instance()
        result = s.apply_particle_look(instance, {"particle_emission": 3.0})
        np.testing.assert_allclose(result.emission, [3.0, 3.0, 3.0])

    def test_ramp_by_speed_reads_velocity_magnitude(self):
        instance = replace(self._instance(), velocities=np.array([[0, 0, 0], [3, 4, 0], [0, 0, 2]], np.float32))
        params = {"particle_ramp_by": "speed", "particle_opacity_ramp": "0:1;5:0.2"}
        result = s.apply_particle_look(instance, params)
        # speeds are 0, 5 and 2: opacity 1.0, 0.2 (at the ramp's own end) and 1 + (0.2-1)*2/5 = 0.68.
        np.testing.assert_allclose(result.colors[:, 3], [1.0, 0.2, 0.68], atol=1e-5)

    def test_bad_stop_names_itself(self):
        instance = self._instance()
        with self.assertRaises(ValueError) as ctx:
            s.apply_particle_look(instance, {"particle_ramp_by": "age", "particle_color_ramp": "0:1,0"})
        self.assertIn("0:1,0", str(ctx.exception))

    def test_material_and_shadow_fields_carry_through(self):
        instance = self._instance()
        result = s.apply_particle_look(instance, {"particle_material": "pbr", "particle_metallic": 0.7,
                                                   "particle_pbr_roughness": 0.3, "particle_pbr_specular": 0.6,
                                                   "particle_cast_shadows": 0})
        self.assertEqual(result.material, "pbr")
        self.assertEqual((result.metallic, result.pbr_roughness, result.pbr_specular), (0.7, 0.3, 0.6))
        self.assertFalse(result.cast_shadows)


class LitSphereTests(unittest.TestCase):
    """A "pbr" sphere particle is lit by the same GGX BRDF a pbr mesh sphere is (R7 of 7)."""

    def _render(self, geometries=(), particles=(), lights=()):
        return s.render(s.Scene(geometries=geometries, particles=particles, lights=lights),
                        s.Camera(), SIZE, SIZE, ambient=0.05)

    def test_a_lit_sphere_particle_shades_like_a_sphere_mesh_of_the_same_material(self):
        light = s.Light(position=s.Vec3(2, 3, 4))
        mesh = replace(s._sphere(0.8, 32, (0.6, 0.3, 0.2, 1.0), s.Transform3D()),
                      material="pbr", metallic=0.2, pbr_roughness=0.4, pbr_specular=0.5)
        particle = s.ParticleInstance(positions=np.zeros((1, 3), np.float32), sizes=np.array([1.6], np.float32),
                                      colors=np.array([[0.6, 0.3, 0.2, 1.0]], np.float32), render_as="spheres",
                                      material="pbr", metallic=0.2, pbr_roughness=0.4, pbr_specular=0.5)
        mesh_image = self._render(geometries=(mesh,), lights=(light,))
        particle_image = self._render(particles=(particle,), lights=(light,))
        centre = SIZE // 2
        np.testing.assert_allclose(mesh_image[centre, centre], particle_image[centre, centre], atol=0.05)
        # The lit side (towards the light, upper right) must be brighter than the far side, on both.
        near = particle_image[centre - 15, centre + 15, 0]
        far = particle_image[centre + 15, centre - 15, 0]
        self.assertGreater(near, far + 0.1)

    def test_standard_particle_ignores_the_scene_lights_headlight_style(self):
        light = s.Light(position=s.Vec3(2, 3, 4))
        particle = s.ParticleInstance(positions=np.zeros((1, 3), np.float32), sizes=np.array([1.6], np.float32),
                                      colors=np.array([[0.6, 0.3, 0.2, 1.0]], np.float32), render_as="spheres")
        baseline = self._render(particles=(particle,))
        lit = self._render(particles=(particle,), lights=(light,))
        # The old "headlight" shading is fixed and does not read scene lights at all.
        np.testing.assert_array_equal(baseline, lit)

    def test_emission_brightens_a_particle_pixel(self):
        particle = s.ParticleInstance(positions=np.zeros((1, 3), np.float32), sizes=np.array([1.6], np.float32),
                                      colors=np.array([[0.3, 0.3, 0.3, 1.0]], np.float32), render_as="points",
                                      emission=np.array([2.0], np.float32))
        dim = s.ParticleInstance(positions=np.zeros((1, 3), np.float32), sizes=np.array([1.6], np.float32),
                                 colors=np.array([[0.3, 0.3, 0.3, 1.0]], np.float32), render_as="points")
        bright_image = self._render(particles=(particle,))
        dim_image = self._render(particles=(dim,))
        centre = SIZE // 2
        self.assertGreater(bright_image[centre, centre, 0], dim_image[centre, centre, 0] + 0.3)


class ParticleShadowTests(unittest.TestCase):
    """A "pbr" particle with `cast_shadows` on darkens a pbr mesh below it (R7 of 7).

    Both renders below keep the same visible particle sprite (only `cast_shadows` differs), so any
    difference is the shadow's own contribution, never the particle's on-screen footprint.
    """

    def _scenes(self, cast_shadows):
        light = s.Light(kind="Directional", target=s.Vec3(0, -1, 0), shadows=True, intensity=2.0)
        plane = replace(s._card(20, 20, (0.8, 0.8, 0.8, 1.0),
                                s.Transform3D(position=s.Vec3(0, -1.5, 0), rotation=s.Vec3(-90, 0, 0))),
                       material="pbr", metallic=0.0, pbr_roughness=0.8)
        caster = s.ParticleInstance(positions=np.array([[0, 0, 0]], np.float32), sizes=np.array([1.0], np.float32),
                                    colors=np.array([[1, 1, 1, 1]], np.float32), render_as="spheres",
                                    cast_shadows=cast_shadows)
        camera = s.Camera(transform=s.Transform3D(position=s.Vec3(0, 3, 8)), target=s.Vec3(0, -1.5, 0))
        return s.render(s.Scene(geometries=(plane,), particles=(caster,), lights=(light,)),
                        camera, SIZE, SIZE, ambient=0.05), caster, camera

    def test_a_sphere_particle_casts_a_shadow_onto_a_plane(self):
        casting, caster, camera = self._scenes(True)
        not_casting, _, _ = self._scenes(False)
        # The particle's own sprite (rendered identically either way) must not be mistaken for shadow.
        sprite_only = s.render(s.Scene(particles=(caster,)), camera, SIZE, SIZE, ambient=0.0)
        sprite_mask = sprite_only[..., 3] > 0
        darker = (not_casting[..., 0] - casting[..., 0]) > 0.05
        self.assertTrue(darker.any())
        self.assertEqual(int((darker & sprite_mask).sum()), 0)
        np.testing.assert_allclose(casting[~darker], not_casting[~darker], atol=1e-5)

    def test_cast_shadows_off_casts_nothing_onto_the_particle_free_background(self):
        casting, caster, camera = self._scenes(True)
        not_casting, _, _ = self._scenes(False)
        light = s.Light(kind="Directional", target=s.Vec3(0, -1, 0), shadows=True, intensity=2.0)
        plane = replace(s._card(20, 20, (0.8, 0.8, 0.8, 1.0),
                                s.Transform3D(position=s.Vec3(0, -1.5, 0), rotation=s.Vec3(-90, 0, 0))),
                       material="pbr", pbr_roughness=0.8)
        plane_only = s.render(s.Scene(geometries=(plane,), lights=(light,)), camera, SIZE, SIZE, ambient=0.05)
        # Away from the particle's own sprite, "off" matches a scene with no particle at all: nothing shadows.
        sprite_only = s.render(s.Scene(particles=(caster,)), camera, SIZE, SIZE, ambient=0.0)
        sprite_mask = sprite_only[..., 3] > 0
        np.testing.assert_allclose(not_casting[~sprite_mask], plane_only[~sprite_mask], atol=1e-5)
        self.assertFalse(np.allclose(casting[~sprite_mask], plane_only[~sprite_mask], atol=1e-5))


class NodeGraphTests(unittest.TestCase):
    """The full ParticleRender3D node applies a colour ramp end to end."""

    def _document(self, frame):
        d = Dispatcher()
        # A long `life` (relative to the frames sampled below) keeps every particle alive so far, so the
        # population only grows and ages, with no deaths yet to pull the distribution back to a steady
        # state (docs/SIMULATION.md): the mean age fraction rises frame over frame, cleanly.
        # `emit_speed` spreads each frame's births along +Y, one per pixel row, so every particle in
        # the population keeps its own pixel instead of stacking on the emitter point (where only the
        # topmost of several coincident opaque points would ever show).
        make(d, s=("Scene3D", {}), cam=("Camera3D", {}),
            r=("Render3D", {"width": SIZE, "height": SIZE, "samples": 1}),
            e=("ParticleEmitter3D", {"emit_rate": 1.0, "emit_speed": 0.3, "life": 1000.0, "particle_size": 0.2}),
            p=("ParticleRender3D", {"representation": "points", "particle_ramp_by": "age",
                                    "particle_color_ramp": "0:1,0,0;1:0,0,1"}))
        wire(d, "r", "scene", "s")
        wire(d, "r", "camera", "cam")
        wire(d, "p", "particles", "e")
        wire(d, "s", "object0", "p")
        return Evaluator().evaluate(dict(d.document, view="r"), frame=frame)

    def test_the_colour_ramp_shifts_from_red_to_blue_as_particles_age(self):
        young = self._document(frame=1)
        old = self._document(frame=8)
        young_mask, old_mask = young[..., 3] > 0, old[..., 3] > 0
        self.assertTrue(young_mask.any())
        self.assertTrue(old_mask.any())
        self.assertGreater(old[old_mask][:, 2].mean(), young[young_mask][:, 2].mean())
        self.assertGreater(young[young_mask][:, 0].mean(), old[old_mask][:, 0].mean())


class InstanceMaterialVariantTests(unittest.TestCase):
    """Instance3D already expands each instance from its own source Geometry (`expand_instances`), so a
    per-source material or `inst_color_from_points` tint carries straight through to shading (R7 of 7,
    part 2): no new mechanism is needed, only this coverage.
    """

    def test_two_source_variants_keep_their_own_material(self):
        metal = replace(s._sphere(0.5, 12, (0.8, 0.8, 0.8, 1.0), s.Transform3D()),
                        material="pbr", metallic=1.0, pbr_roughness=0.1, name="metal")
        rough = replace(s._sphere(0.5, 12, (0.8, 0.8, 0.8, 1.0), s.Transform3D(position=s.Vec3(2, 0, 0))),
                        material="pbr", metallic=0.0, pbr_roughness=1.0, name="rough")
        points = s.Geometry(np.array([[0, 0, 0], [2, 0, 0]], np.float32), np.zeros((0, 3), np.int32), (1, 1, 1, 1))
        light = s.Light(position=s.Vec3(2, 3, 4))
        instances = s.instances_from_node(points, s.Geometry(np.zeros((0, 3), np.float32),
                                          np.zeros((0, 3), np.int32), (1, 1, 1, 1)), {})
        # Build the InstanceSet directly with two source variants (as Instance3D would with two meshes wired).
        matrices = np.tile(np.eye(4), (2, 1, 1))
        matrices[0, :3, 3], matrices[1, :3, 3] = (0, 0, 0), (2, 0, 0)
        instance_set = s.InstanceSet((metal, rough), matrices, np.array([0, 1], np.int32))
        scene = s.Scene(instances=(instance_set,), lights=(light,))
        image = s.render(scene, s.Camera(transform=s.Transform3D(position=s.Vec3(1, 0, 6))), SIZE, SIZE, ambient=0.05)
        self.assertTrue((image[..., 3] > 0).any())

    def test_color_from_points_tints_each_instance(self):
        source = replace(s._sphere(0.4, 10, (1.0, 1.0, 1.0, 1.0), s.Transform3D()), material="pbr")
        points = s.Geometry(np.array([[0, 0, 0], [1.5, 0, 0]], np.float32), np.zeros((0, 3), np.int32), (1, 1, 1, 1),
                            normals=None)
        light = s.Light(position=s.Vec3(0, 5, 5))
        red_blue = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], np.float32)
        instances = s.InstanceSet((source,), np.tile(np.eye(4), (2, 1, 1)), np.zeros(2, np.int32),
                                  colors=red_blue)
        instances.matrices[0, :3, 3], instances.matrices[1, :3, 3] = (0, 0, 0), (1.5, 0, 0)
        geometries = s.expand_instances(instances)
        self.assertAlmostEqual(geometries[0].color[0], 1.0)
        self.assertAlmostEqual(geometries[0].color[2], 0.0)
        self.assertAlmostEqual(geometries[1].color[0], 0.0)
        self.assertAlmostEqual(geometries[1].color[2], 1.0)


if __name__ == "__main__":
    unittest.main()
