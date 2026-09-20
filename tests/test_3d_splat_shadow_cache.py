"""Per-splat shadow visibility is kept between renders. It depends on the casters, the mesh
occluders and where each light is; never on the camera, a light's colour or the Relight amount.
Every warm result here is compared with a cold render of the same scene, byte for byte."""
from dataclasses import replace
import gc
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s
from nodebased.cancellation import Cancelled
from tests.test_3d_splat_relight import cloud, light

W, H = 48, 32


def shell(n=900, seed=5):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return replace(cloud(p, scales=(.09, .09, .02), rotations=rng.normal(size=(n, 4))),
                   opacity=rng.uniform(.2, .9, n))


def lamp(**kwargs):
    return light((3, 1, 3), shadows=True, **kwargs)


def camera(x=0.0, y=0.0, z=4.0):
    return s.Camera(s.Transform3D(s.Vec3(x, y, z)))


def cold(scene, cam, **kwargs):
    s.clear_splat_shadow_cache()
    return s.render(scene, cam, W, H, **kwargs)


def traced(action):
    before = s.splat_shadow_stats["rays_traced"]
    result = action()
    return result, s.splat_shadow_stats["rays_traced"] - before


class SplatShadowCacheTests(unittest.TestCase):
    def setUp(self):
        s.clear_splat_shadow_cache()
        self.inst = s.SplatInstance(shell(), relight=1.0)
        self.scene = s.Scene((), (lamp(),), (self.inst,))

    def test_the_same_render_again_traces_nothing_and_matches(self):
        first, rays = traced(lambda: s.render(self.scene, camera(), W, H, ambient=.1))
        self.assertGreater(rays, 0)
        builds = s.splat_shadow_stats["caster_builds"]
        again, rays = traced(lambda: s.render(self.scene, camera(), W, H, ambient=.1))
        self.assertEqual(0, rays)
        self.assertEqual(builds, s.splat_shadow_stats["caster_builds"])
        np.testing.assert_array_equal(first, again)

    def test_a_camera_move_reuses_what_it_saw_and_equals_a_cold_render(self):
        moved = camera(1.6, .7, 3.2)
        expected, cold_rays = traced(lambda: cold(self.scene, moved, ambient=.1))
        s.clear_splat_shadow_cache()
        s.render(self.scene, camera(), W, H, ambient=.1)
        warm, warm_rays = traced(lambda: s.render(self.scene, moved, W, H, ambient=.1))
        # Rays are traced in different batches on the warm path, so this also proves a ray's
        # result does not depend on which other rays share its batch.
        np.testing.assert_array_equal(expected, warm)
        self.assertLess(warm_rays, cold_rays)
        for mode in ("raster", "raytrace"):
            np.testing.assert_array_equal(cold(self.scene, moved, mode=mode),
                                          s.render(self.scene, moved, W, H, mode=mode))

    def test_colour_intensity_ambient_and_relight_are_not_part_of_the_key(self):
        s.render(self.scene, camera(), W, H, ambient=.1)
        variants = {
            "colour": (replace(self.scene, lights=(lamp(color=(1, .6, .3)),)), .1),
            "intensity": (replace(self.scene, lights=(lamp(intensity=2.5),)), .1),
            "ambient": (self.scene, .4),
            "relight": (replace(self.scene, splats=(replace(self.inst, relight=.35),)), .1),
            "same direction, light further away": (replace(self.scene, lights=(light((6, 2, 6), shadows=True),)), .1),
        }
        for name, (scene, ambient) in variants.items():
            warm, rays = traced(lambda: s.render(scene, camera(), W, H, ambient=ambient))
            self.assertEqual(0, rays, name)
            s_stats = dict(s.splat_shadow_stats)
            np.testing.assert_array_equal(cold(scene, camera(), ambient=ambient), warm, name)
            self.assertGreater(s.splat_shadow_stats["rays_traced"], s_stats["rays_traced"], name)
            s.clear_splat_shadow_cache()
            s.render(self.scene, camera(), W, H, ambient=.1)

    def test_whatever_changes_a_shadow_is_traced_again(self):
        card = s._card(1.2, 1.2, (1, 1, 1, 1), s.Transform3D(s.Vec3(1.5, .5, 1.5)))
        matrix = np.eye(4)
        matrix[:3, 3] = (.4, 0, 0)
        variants = {
            "light direction": replace(self.scene, lights=(light((-3, 1, 3), shadows=True),)),
            "point light position": replace(self.scene, lights=(light((3, 1, 3), shadows=True, kind="Point"),)),
            "node transform": replace(self.scene, splats=(replace(self.inst, matrix=matrix),)),
            "splat scale": replace(self.scene, splats=(replace(self.inst, scale_scale=1.7),)),
            "splat opacity": replace(self.scene, splats=(replace(self.inst, opacity_scale=.4),)),
            "a mesh occluder": replace(self.scene, geometries=(card,)),
            "a second cloud": replace(self.scene, splats=(self.inst, s.SplatInstance(shell(200, 9), matrix))),
        }
        base = s.render(self.scene, camera(), W, H, ambient=.1)
        for name, scene in variants.items():
            s.clear_splat_shadow_cache()
            s.render(self.scene, camera(), W, H, ambient=.1)
            warm, rays = traced(lambda: s.render(scene, camera(), W, H, ambient=.1))
            self.assertGreater(rays, 0, name)
            np.testing.assert_array_equal(cold(scene, camera(), ambient=.1), warm, name)
            self.assertFalse(np.array_equal(base, warm), name)

    def test_a_moved_or_faded_mesh_occluder_is_a_different_key(self):
        def scene(z, alpha):
            card = s._card(1.2, 1.2, (1, 1, 1, alpha), s.Transform3D(s.Vec3(1.5, .5, z)))
            return replace(self.scene, geometries=(card,))
        s.render(scene(1.5, 1.0), camera(), W, H, ambient=.1)
        for name, other in {"moved": scene(1.9, 1.0), "faded": scene(1.5, .5)}.items():
            warm, rays = traced(lambda: s.render(other, camera(), W, H, ambient=.1))
            self.assertGreater(rays, 0, name)
            np.testing.assert_array_equal(cold(other, camera(), ambient=.1), warm, name)
            s.clear_splat_shadow_cache()
            s.render(scene(1.5, 1.0), camera(), W, H, ambient=.1)

    def test_an_equal_copy_of_a_cloud_is_safe_and_dead_clouds_leave_no_token(self):
        s.render(self.scene, camera(), W, H, ambient=.1)
        twin = replace(self.scene, splats=(s.SplatInstance(shell(), relight=1.0),))
        warm, rays = traced(lambda: s.render(twin, camera(), W, H, ambient=.1))
        self.assertGreater(rays, 0)   # identity, not content: a copy is traced again, never wrong
        np.testing.assert_array_equal(cold(twin, camera(), ambient=.1), warm)
        tokens = len(s._splat_cloud_tokens)
        del twin, warm
        s.clear_splat_shadow_cache()
        gc.collect()
        self.assertLess(len(s._splat_cloud_tokens), tokens)

    def test_unlit_instances_store_nothing_and_shadows_off_never_builds(self):
        unlit = replace(self.scene, splats=(replace(self.inst, relight=0.0),))
        _, rays = traced(lambda: s.render(unlit, camera(), W, H))
        self.assertEqual(0, rays)
        self.assertEqual({}, dict(s._splat_visibility))
        builds = s.splat_shadow_stats["caster_builds"]
        s.render(self.scene, camera(), W, H, shadows=False)
        self.assertEqual(builds, s.splat_shadow_stats["caster_builds"])

    def test_a_cancelled_render_leaves_only_valid_values(self):
        class CancelAfter:
            def __init__(self, polls):
                self.polls = polls
            def is_set(self):
                self.polls -= 1
                return self.polls < 0
        cancelled = 0
        for polls in (3, 8, 20):
            try:
                s.render(self.scene, camera(), W, H, ambient=.1, cancel=CancelAfter(polls))
            except Cancelled:
                cancelled += 1
        self.assertGreater(cancelled, 0)
        np.testing.assert_array_equal(cold(self.scene, camera(), ambient=.1),
                                      s.render(self.scene, camera(), W, H, ambient=.1))

    def test_old_caster_sets_and_their_visibility_are_dropped(self):
        scenes = []
        for index in range(s._SPLAT_CASTER_SETS + 2):
            matrix = np.eye(4)
            matrix[0, 3] = index * .1
            scenes.append(replace(self.scene, splats=(replace(self.inst, matrix=matrix),)))
            s.render(scenes[-1], camera(), W, H, ambient=.1)
        self.assertEqual(s._SPLAT_CASTER_SETS, len(s._splat_casters))
        self.assertEqual(set(s._splat_casters), {key[0] for key in s._splat_visibility})
        with patch.object(s, "_SPLAT_CASTER_BYTES", 1):
            s.render(scenes[1], camera(), W, H, ambient=.1)
        self.assertEqual(1, len(s._splat_casters))   # over the byte cap only the newest set stays
        self.assertGreater(next(iter(s._splat_casters.values())).nbytes, 900 * 100)
        with patch.object(s, "_SPLAT_VISIBILITY_BYTES", 1):
            s.render(replace(self.scene, lights=(light((-3, 1, 3), shadows=True),)), camera(), W, H)
        self.assertEqual(1, len(s._splat_visibility))
        _, rays = traced(lambda: s.render(scenes[0], camera(), W, H, ambient=.1))
        self.assertGreater(rays, 0)
        np.testing.assert_array_equal(cold(scenes[0], camera(), ambient=.1),
                                      s.render(scenes[0], camera(), W, H, ambient=.1))


if __name__ == "__main__":
    unittest.main()
