"""Cast shadows on/off per ReadSplat3D: a capture's sky shell or room walls would otherwise block
every light in the scene. Off removes the instance from the shadow casters only; it still renders,
still receives shadows when relit and still catches mesh shadows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nodebased import scene3d as s, splats
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from nodebased.raytrace import Bvh
from tests.test_3d_splat_relight import cloud, light

W, H = 40, 40


def ground():
    return s._card(4, 4, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, 0)))


def sun():
    return light((1, 0, 1), shadows=True)


def camera():
    return s.Camera(s.Transform3D(s.Vec3(0, 0, 6)))


def blocker(**kwargs):
    """A splat up the light's path and outside the camera's view: it only shows up as a shadow."""
    return s.SplatInstance(cloud(((4, 0, 4),), scales=(.4, .4, .4)), **kwargs)


def render(splat_instances, mode="raster", **kwargs):
    s.clear_splat_shadow_cache()
    return s.render(s.Scene((ground(),), (sun(),), tuple(splat_instances)), camera(), W, H,
                    ambient=.1, mode=mode, **kwargs)


class SplatCastToggleTests(unittest.TestCase):
    def test_on_is_the_default_and_matches_the_old_constructor(self):
        self.assertTrue(blocker().cast_shadows)
        old = s.SplatInstance(blocker().cloud, np.eye(4), 3, 1.0, 1.0, 0.0, 0.0)
        np.testing.assert_array_equal(render((old,)), render((blocker(cast_shadows=True),)))

    def test_off_leaves_the_mesh_exactly_as_if_the_splat_were_not_there(self):
        for mode in ("raster", "raytrace"):
            bare = render((), mode)
            on = render((blocker(),), mode)
            off = render((blocker(cast_shadows=False),), mode)
            self.assertGreater(int((on[..., 0] < bare[..., 0] - 1e-3).sum()), 100, mode)
            np.testing.assert_array_equal(off, bare, mode)

    def test_off_builds_no_splat_bvh_and_costs_no_shadow_budget(self):
        for instance, caster_sets, bvhs in ((blocker(), 1, 1), (blocker(cast_shadows=False), 0, 0)):
            s.clear_splat_shadow_cache()
            before = s.splat_shadow_stats["caster_builds"]
            with patch.object(Bvh, "build", wraps=Bvh.build) as build:
                s.render(s.Scene((ground(),), (sun(),), (instance,)), camera(), W, H)
            self.assertEqual(caster_sets, s.splat_shadow_stats["caster_builds"] - before)
            self.assertEqual(bvhs, build.call_count)
        with patch.object(s, "SPLAT_SHADOW_BUDGET", 1):
            with self.assertRaisesRegex(ValueError, "Splat shadow rays"):
                render((blocker(),))
            render((blocker(cast_shadows=False),))
        # A relit splat next to a 50-splat shell: the shell only counts while it casts.
        relit = s.SplatInstance(cloud(((0, 0, 0),)), relight=1)
        shell = cloud([(x, 3, 0) for x in np.linspace(-2, 2, 50)], scales=(.05, .05, .05))
        with patch.object(s, "SPLAT_SHADOW_BUDGET", 5000):
            with self.assertRaisesRegex(ValueError, "Splat shadow rays"):
                render((relit, s.SplatInstance(shell)))
            render((relit, s.SplatInstance(shell, cast_shadows=False)))

    def test_a_relit_splat_ignores_a_non_caster_and_still_takes_the_mesh_shadow(self):
        lamp = light(shadows=True)
        relit = s.SplatInstance(cloud(((.13, .07, 0),)), relight=1)
        above = cloud(((.13, .07, 1.5),))
        shadowed = s._splat_shadow_visibility((relit, s.SplatInstance(above)), (lamp,))[0]
        self.assertLess(shadowed[0, 0], .5)
        s.clear_splat_shadow_cache()
        free = s._splat_shadow_visibility((relit, s.SplatInstance(above, cast_shadows=False)), (lamp,))
        np.testing.assert_array_equal(free[0], [[1.]])
        # The non-caster itself still receives: relit, under an opaque card, it goes black.
        from tests.test_3d_splat_shadows import SplatShadowTests
        card = s._card(3, 3, (0, 0, 0, 1), s.Transform3D(s.Vec3(0, 0, 2)))
        receiver = s.SplatInstance(cloud(((.13, .07, 0),)), relight=1, cast_shadows=False)
        s.clear_splat_shadow_cache()
        np.testing.assert_array_equal(
            SplatShadowTests.visibility(self, (receiver,), (lamp,), card)[0], [[0.]])

    def test_only_the_instances_left_on_cast(self):
        lamp = light(shadows=True)
        relit = s.SplatInstance(cloud(((0, 0, 0),)), relight=1)
        near, far = cloud(((0, 0, 1.5),)), cloud(((.2, 0, 3.0),))
        s.clear_splat_shadow_cache()
        expected = s._splat_shadow_visibility((relit, s.SplatInstance(near)), (lamp,))[0]
        s.clear_splat_shadow_cache()
        both = s._splat_shadow_visibility((relit, s.SplatInstance(near), s.SplatInstance(far)), (lamp,))[0]
        self.assertLess(both[0, 0], expected[0, 0])
        s.clear_splat_shadow_cache()
        mixed = s._splat_shadow_visibility(
            (relit, s.SplatInstance(near), s.SplatInstance(far, cast_shadows=False)), (lamp,))[0]
        np.testing.assert_array_equal(mixed, expected)

    def test_the_toggle_is_part_of_the_cache_key(self):
        s.clear_splat_shadow_cache()
        scene = lambda **kw: s.Scene((ground(),), (sun(),), (blocker(**kw),))
        on = s.render(scene(), camera(), W, H, ambient=.1)
        off = s.render(scene(cast_shadows=False), camera(), W, H, ambient=.1)
        again = s.render(scene(), camera(), W, H, ambient=.1)
        self.assertFalse(np.array_equal(on, off))
        np.testing.assert_array_equal(on, again)
        lamp = light(shadows=True)
        relit = s.SplatInstance(cloud(((0, 0, 0),)), relight=1)
        above = cloud(((0, 0, 1.5),))
        s.clear_splat_shadow_cache()
        warm = s._splat_shadow_visibility((relit, s.SplatInstance(above)), (lamp,))[0]
        flipped = s._splat_shadow_visibility((relit, s.SplatInstance(above, cast_shadows=False)), (lamp,))[0]
        self.assertLess(warm[0, 0], 1.)
        np.testing.assert_array_equal(flipped, [[1.]])

    def test_the_node_has_the_knob_and_old_documents_load_with_it_on(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "blocker.ply")
            splats.write_ply(blocker().cloud, path)
            d = Dispatcher()
            for key, kind, params in (("read", "ReadSplat3D", dict(splat_path=str(path))),
                                      ("card", "Card3D", dict(card_width=4.0, card_height=4.0)),
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
            self.assertEqual("on", d.document["nodes"]["read"]["params"]["splat_cast_shadows"])
            on = Evaluator().evaluate(d.document, target="render")
            d.execute(dict(op="set", id="read", param="splat_cast_shadows", value="off"))
            scene = Evaluator().evaluate_raster(d.document, "scene", typed=True)
            self.assertFalse(scene.splats[0].cast_shadows)
            off = Evaluator().evaluate(d.document, target="render")
            self.assertGreater(off[..., :3].sum(), on[..., :3].sum())
            with self.assertRaises(Exception):
                d.execute(dict(op="set", id="read", param="splat_cast_shadows", value="maybe"))
            document = copy.deepcopy(d.document)
            del document["nodes"]["read"]["params"]["splat_cast_shadows"]
            old = Path(folder, "old.nbcomp")
            old.write_text(json.dumps(document))
            self.assertEqual("on", load_document(old)["nodes"]["read"]["params"]["splat_cast_shadows"])
        labels = [group.label for group in knob_layout("ReadSplat3D") if "splat_cast_shadows" in group.params]
        self.assertEqual(["Cast shadows"], labels)


if __name__ == "__main__":
    unittest.main()
