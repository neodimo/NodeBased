"""Catch shadows: meshes darken a capture's OWN colours without relighting it. The multiplier is
the scene's light with the occluders over the same light without them; splats never occlude here."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nodebased import scene3d as s, splats
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.splatraster import prepare_splats
from nodebased.splatshade import shadow_catch
from tests.test_3d_splat_relight import cloud, light

W, H = 40, 40
GRID = [(x, y, 0.0) for y in np.linspace(-1.5, 1.5, 7) for x in np.linspace(-1.5, 1.5, 7)]


def field():
    """A flat field of splats facing +z. The light comes in at 45 degrees from +x, and a card
    hovering beside the field (so it never hides the field from the camera) shadows x > 0.75."""
    return replace(cloud(GRID, scales=(.12, .12, .02)), opacity=np.full(len(GRID), .9))


def card(x=3.2, alpha=1.0, z=1.5):
    return s._card(1.9, 4.0, (1, 1, 1, alpha), s.Transform3D(s.Vec3(x, 0, z)))


def colours(scene, ambient=0.0):
    """Per-splat colours exactly as the rasterizer receives them, in input order."""
    provider = s._SplatShadows(scene.splats, scene.lights, *mesh_parts(scene))
    provider.relit_shadows = any(i.relight > 0 for i in scene.splats)
    prepared = prepare_splats(scene.splats, camera(), W, H, lighting=(scene.lights, ambient, provider))
    return np.asarray(prepared.splats[4])


def mesh_parts(scene):
    from nodebased.raytrace import Bvh, TriangleSet
    triangles = []
    for g in scene.geometries:
        m = g.world_matrix()
        world = (m[:3, :3] @ g.vertices.T + m[:3, 3:4]).T
        triangles.append(world[g.triangles])
    if not triangles:
        return None, None, .001
    t = np.concatenate(triangles).astype(np.float64)
    alphas = np.concatenate([np.full(len(g.triangles), g.color[3], np.float32) for g in scene.geometries])
    primitives = TriangleSet(t[:, 0], t[:, 1]-t[:, 0], t[:, 2]-t[:, 0], alphas)
    return primitives, Bvh.build(*primitives.aabbs()), .001


def camera():
    return s.Camera(s.Transform3D(s.Vec3(0, 0, 6)))


def sun(**kwargs):
    return light((1, 0, 1), shadows=True, **kwargs)


class ShadowCatchTests(unittest.TestCase):
    def setUp(self):
        s.clear_splat_shadow_cache()
        self.cloud = field()
        self.shadowed = np.array([p[0] > .9 for p in GRID])
        self.lit = np.array([p[0] < .6 for p in GRID])

    def scene(self, catch=1.0, relight=0.0, lights=None, geometries=None):
        return s.Scene((card(),) if geometries is None else geometries, (sun(),) if lights is None else lights,
                       (s.SplatInstance(self.cloud, relight=relight, shadow_catch=catch),))

    def test_off_is_byte_identical_and_traces_nothing(self):
        plain = s.Scene((card(),), (sun(),), (s.SplatInstance(self.cloud),))
        before = dict(s.splat_shadow_stats)
        for mode in ("raster", "raytrace"):
            for output in ("rgba", "splats"):
                np.testing.assert_array_equal(
                    s.render(plain, camera(), W, H, mode=mode, output=output),
                    s.render(self.scene(catch=0.0), camera(), W, H, mode=mode, output=output))
        self.assertEqual(before, dict(s.splat_shadow_stats))

    def test_shadowed_splats_go_dark_and_the_rest_keep_their_exact_colour(self):
        baked = colours(self.scene(catch=0.0))
        caught = colours(self.scene(catch=1.0))
        np.testing.assert_array_equal(baked[self.lit], caught[self.lit])
        np.testing.assert_array_equal(np.zeros_like(baked[self.shadowed]), caught[self.shadowed])
        self.assertEqual((14, 35), (int(self.shadowed.sum()), int(self.lit.sum())))

    def test_ambient_unshadowed_lights_strength_and_card_alpha_dilute_the_shadow(self):
        baked = colours(self.scene(catch=0.0))[self.shadowed]
        cases = {
            "ambient": (dict(), .25, .25 / 1.25),
            "fill light without shadows": (dict(lights=(sun(), light((0, 0, 1), intensity=3.0))), 0.0, 3 / 4),
            "half strength": (dict(catch=.5), 0.0, .5),
            "half transparent card": (dict(geometries=(card(alpha=.5),)), 0.0, .5),
            "coloured sun": (dict(lights=(sun(color=(1, 0, 0)), light((0, 0, 1), color=(0, 1, 0)))), 0.0,
                             .7152 / (.2126 + .7152)),
        }
        for name, (kwargs, ambient, factor) in cases.items():
            caught = colours(self.scene(**kwargs), ambient)[self.shadowed]
            np.testing.assert_allclose(caught, baked * factor, rtol=2e-6, atol=0, err_msg=name)

    def test_the_multiplier_function_on_its_own(self):
        lights = (sun(intensity=2.0), light((1, 0, 0)))
        visibility = np.array([[0., 1.], [.5, 1.], [1., 1.]])
        np.testing.assert_allclose(shadow_catch(lights, .5, visibility, 1.0), [1.5/3.5, 2.5/3.5, 1.0])
        np.testing.assert_allclose(shadow_catch(lights, .5, visibility, .4), [1-.4*(2/3.5), 1-.4*(1/3.5), 1.0])
        np.testing.assert_array_equal(shadow_catch((), 0.0, visibility, 1.0), np.ones(3))

    def test_relight_blends_the_caught_capture_with_the_relit_result(self):
        baked = colours(self.scene(catch=0.0))
        caught = colours(self.scene(catch=1.0))
        relit = colours(self.scene(catch=0.0, relight=1.0))
        mixed = colours(self.scene(catch=1.0, relight=.5))
        np.testing.assert_allclose(mixed, .5 * caught + .5 * relit, rtol=2e-6, atol=1e-9)
        # Fully relit splats already carry traced shadows, so catching changes nothing.
        np.testing.assert_array_equal(relit, colours(self.scene(catch=1.0, relight=1.0)))
        self.assertFalse(np.array_equal(baked, mixed))

    def test_splats_never_occlude_a_caught_shadow_and_no_splat_bvh_is_built(self):
        above = replace(cloud([(x, y, .6) for x, y, _ in GRID], scales=(.3, .3, .02)), opacity=np.full(len(GRID), .99))
        scene = s.Scene((), (sun(),), (s.SplatInstance(self.cloud, shadow_catch=1.0), s.SplatInstance(above)))
        builds = s.splat_shadow_stats["caster_builds"]
        plain = replace(scene, splats=(s.SplatInstance(self.cloud), s.SplatInstance(above)))
        np.testing.assert_array_equal(s.render(plain, camera(), W, H), s.render(scene, camera(), W, H))
        with_card = replace(scene, geometries=(card(),))
        image = s.render(with_card, camera(), W, H, output="splats")
        self.assertFalse(np.array_equal(image, s.render(replace(plain, geometries=(card(),)), camera(), W, H, output="splats")))
        self.assertEqual(builds, s.splat_shadow_stats["caster_builds"])

    def test_the_render_shows_the_shadow_in_both_modes_and_leaves_data_passes_alone(self):
        scene, plain = self.scene(), self.scene(catch=0.0)
        small = scene
        for mode in ("raster", "raytrace"):
            caught = s.render(small, camera(), W, H, mode=mode, output="splats")
            free = s.render(replace(small, splats=plain.splats), camera(), W, H, mode=mode, output="splats")
            right, left = caught[:, W*3//4:, :3].sum(), caught[:, :W//4, :3].sum()
            self.assertLess(right, .2 * free[:, W*3//4:, :3].sum(), mode)
            np.testing.assert_array_equal(free[:, :W//4], caught[:, :W//4], mode)
            np.testing.assert_array_equal(caught[..., 3], free[..., 3], mode)  # alpha untouched
            self.assertGreater(left, 0)
        np.testing.assert_array_equal(s.render(small, camera(), W, H, output="splats"),
                                      s.render(small, camera(), W, H, output="splats", mode="raytrace"))
        for output in ("depth", "normals", "position", "object_id"):
            np.testing.assert_array_equal(s.render(plain, camera(), W, H, output=output),
                                          s.render(scene, camera(), W, H, output=output), output)
        np.testing.assert_array_equal(s.render(plain, camera(), W, H, shadows=False),
                                      s.render(scene, camera(), W, H, shadows=False))

    def test_caught_shadows_are_cached_and_follow_the_card_the_light_and_the_node(self):
        scene = self.scene()
        first = s.render(scene, camera(), W, H)
        traced = s.splat_shadow_stats["rays_traced"]
        self.assertGreater(traced, 0)
        np.testing.assert_array_equal(first, s.render(scene, s.Camera(s.Transform3D(s.Vec3(0, 0, 6))), W, H))
        s.render(replace(scene, lights=(sun(color=(1, .5, .2), intensity=3.0),)), camera(), W, H)
        self.assertEqual(traced, s.splat_shadow_stats["rays_traced"])
        matrix = np.eye(4)
        matrix[0, 3] = .8
        moved = {"card": replace(scene, geometries=(card(x=-1.0),)),
                 "light": replace(scene, lights=(light((-1, 0, 1), shadows=True),)),
                 "node": replace(scene, splats=(replace(scene.splats[0], matrix=matrix),))}
        for name, other in moved.items():
            s.clear_splat_shadow_cache()
            s.render(scene, camera(), W, H)   # every variant starts from the base scene's warm cache
            before = s.splat_shadow_stats["rays_traced"]
            warm = s.render(other, camera(), W, H)
            self.assertGreater(s.splat_shadow_stats["rays_traced"], before, name)
            s.clear_splat_shadow_cache()
            np.testing.assert_array_equal(s.render(other, camera(), W, H), warm, name)
            self.assertFalse(np.array_equal(first, warm), name)

    def test_point_lights_stop_at_the_light_and_the_budget_refuses(self):
        lamp = light((0, 0, .5), shadows=True, kind="Point")   # below the card at z = 1.5
        baked = colours(self.scene(catch=0.0, lights=(lamp,)))
        np.testing.assert_array_equal(baked, colours(self.scene(lights=(lamp,))))
        from unittest.mock import patch
        with patch.object(s, "SPLAT_SHADOW_BUDGET", 0):
            # The splats output keeps mesh-receiving shadows out of it, so only catching is budgeted.
            with self.assertRaisesRegex(ValueError, "Splat shadow-catch rays exceed"):
                s.render(self.scene(), camera(), W, H, output="splats")
            s.render(self.scene(catch=0.0), camera(), W, H, output="splats")

    def test_the_node_has_the_knob_and_old_documents_load_with_it_off(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "field.ply")
            splats.write_ply(self.cloud, path)
            d = Dispatcher()
            for key, kind, params in (("read", "ReadSplat3D", dict(splat_path=str(path))),
                                      ("card", "Card3D", dict(card_width=1.9, card_height=4.0, tx=3.2, tz=1.5)),
                                      ("sun", "Light3D", dict(tx=1.0, ty=0.0, tz=1.0, shadows="on")),
                                      ("cam", "Camera3D", dict(tz=6.0)), ("scene", "Scene3D", {}),
                                      ("render", "Render3D", dict(width=W, height=H))):
                d.execute(dict(op="create", id=key, type=kind))
                for param, value in params.items():
                    d.execute(dict(op="set", id=key, param=param, value=value))
            for slot, source in (("object0", "read"), ("object1", "card"), ("object2", "sun")):
                d.execute(dict(op="connect", id="scene", input=slot, source=source))
            d.execute(dict(op="connect", id="render", input="scene", source="scene"))
            d.execute(dict(op="connect", id="render", input="camera", source="cam"))
            self.assertEqual(0.0, d.document["nodes"]["read"]["params"]["splat_shadow_catch"])
            off = Evaluator().evaluate(d.document, target="render")
            d.execute(dict(op="set", id="read", param="splat_shadow_catch", value=1.0))
            on = Evaluator().evaluate(d.document, target="render")
            self.assertLess(on[..., :3].sum(), off[..., :3].sum())
            with self.assertRaises(Exception):
                d.execute(dict(op="set", id="read", param="splat_shadow_catch", value=1.5))
            old = Path(folder, "old.nbcomp")
            d.execute(dict(op="set", id="read", param="splat_shadow_catch", value=0.0))
            import copy
            document = copy.deepcopy(d.document)
            del document["nodes"]["read"]["params"]["splat_shadow_catch"]
            import json
            old.write_text(json.dumps(document))
            loaded = load_document(old)
            self.assertEqual(0.0, loaded["nodes"]["read"]["params"]["splat_shadow_catch"])
            np.testing.assert_array_equal(off, Evaluator().evaluate(loaded, target="render"))
        knob = next(g for g in knob_layout("ReadSplat3D") if "splat_shadow_catch" in g.params)
        self.assertEqual(("float_slider", "Catch shadows", (0, 1)), (knob.kind, knob.label, tuple(knob.soft_range)))


if __name__ == "__main__":
    unittest.main()
