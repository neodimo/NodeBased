"""Kept specular: a capture's view-dependent highlights (SH beyond DC) survive relighting and
shadow catching. `Keep specular` = 0 is the old behaviour byte for byte."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nodebased import scene3d as s, splats
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from tests.test_3d_splat_relight import light
from tests.test_3d_splat_shadow_catch import W, H, GRID, card, camera, sun, colours

LUMA = np.array((.2126, .7152, .0722))


def shiny(highlight_x=0.0, degree=1):
    """A flat field facing +z, DC colour plus a Gaussian highlight lobe in the SH z term.

    The capture's highlight is a bump centred on `highlight_x`; it is the only view-dependent part.
    """
    positions = np.array(GRID, np.float64)
    n = len(positions)
    sh = np.zeros((n, 4, 3))
    sh[:, 0] = (np.array((.5, .3, .1)) - .5) / splats.C0
    bump = 1.6 * np.exp(-((positions[:, 0] - highlight_x) ** 2 + positions[:, 1] ** 2) / .5)
    sh[:, 2] = -(bump / splats.C1)[:, None]
    c = splats.SplatCloud(positions, np.full((n, 3), (.12, .12, .02)), np.tile((1, 0, 0, 0), (n, 1)),
                          np.full(n, .9), sh[:, :1] if degree == 0 else sh, degree)
    return c


def dc_only(c):
    return splats.SplatCloud(c.positions, c.scales, c.rotations, c.opacity, c.sh[:, :1], 0)


def render(cloud, *, relight=0.0, specular=0.0, catch=0.0, lights=(), extra=(), ambient=1.0):
    instance = s.SplatInstance(cloud, relight=relight, shadow_catch=catch, specular=specular)
    scene = s.Scene(geometries=tuple(extra), lights=tuple(lights), splats=(instance,))
    return s.render(scene, camera(), W, H, ambient=ambient)


def luminance(image):
    return image[..., :3] @ LUMA


class KeptSpecularTests(unittest.TestCase):
    def test_relighting_drops_the_highlight_today_and_keeps_it_when_asked(self):
        """The loss reproduced: the direct render has a highlight, the relit one (specular off) is flat."""
        c = shiny()
        direct = render(c)
        flat = render(dc_only(c))
        highlight = luminance(direct) - luminance(flat)
        peak = np.unravel_index(highlight.argmax(), highlight.shape)
        self.assertGreater(highlight[peak], .05)
        # Pinned reference (documented in docs/3D_FOUNDATION.md "Specular (splats)").
        self.assertAlmostEqual(luminance(direct).max(), 3.3619, delta=1e-3)
        self.assertAlmostEqual(luminance(render(c, relight=1)).max(), .0765, delta=1e-3)
        # Ambient 1, no lights: relit = albedo x 1, so the direct render is the reference.
        lost = render(c, relight=1)
        np.testing.assert_allclose(lost, flat, atol=1e-6)
        self.assertLess(luminance(lost)[peak], luminance(direct)[peak] - .05)
        kept = render(c, relight=1, specular=1)
        self.assertAlmostEqual(luminance(kept)[peak], luminance(direct)[peak], delta=1e-4)
        np.testing.assert_allclose(kept, direct, atol=1e-4)

    def test_specular_is_a_knob_between_off_and_kept(self):
        c = shiny()
        flat = luminance(render(dc_only(c)))
        highlight = lambda k: (luminance(render(c, relight=1, specular=k)) - flat).max()
        self.assertLess(highlight(0), 1e-6)
        self.assertAlmostEqual(highlight(.5), .5 * highlight(1), delta=1e-4)
        self.assertGreater(highlight(1), .05)

    def test_relit_highlight_rides_on_top_of_the_new_lighting(self):
        c = shiny()
        lamp = (light((0, 0, 1), intensity=1),)
        lit_off = render(c, relight=1, lights=lamp, ambient=.2)
        lit_on = render(c, relight=1, specular=1, lights=lamp, ambient=.2)
        expected = luminance(render(c, relight=1, specular=1, lights=(), ambient=1.0)) - \
            luminance(render(dc_only(c), relight=1, lights=(), ambient=1.0))
        np.testing.assert_allclose(luminance(lit_on) - luminance(lit_off), expected, atol=1e-4)

    def test_catcher_darkens_the_diffuse_and_leaves_the_highlight(self):
        c = shiny(highlight_x=1.2)          # highlight inside the card's shadow (x > .75)
        blocker = (card(),)
        common = dict(catch=1.0, lights=(sun(),), extra=blocker, ambient=0.0)
        direct_highlight = luminance(render(c, ambient=0.0)) - luminance(render(dc_only(c), ambient=0.0))
        peak = np.unravel_index(direct_highlight.argmax(), direct_highlight.shape)

        def contrast(specular):
            return (luminance(render(c, specular=specular, **common))
                    - luminance(render(dc_only(c), specular=specular, **common)))

        off, kept = contrast(0.0), contrast(1.0)
        self.assertLess(off[peak], .8 * direct_highlight[peak])            # multiplied along with the diffuse
        self.assertAlmostEqual(kept[peak], direct_highlight[peak], delta=1e-4)
        # The diffuse is still darkened under the card either way.
        shadowed = luminance(render(dc_only(c), specular=1.0, **common))
        plain = luminance(render(dc_only(c), ambient=0.0))
        self.assertLess(shadowed[peak], plain[peak] * .9)

    def test_off_and_dc_only_clouds_are_unchanged(self):
        c = shiny()
        lamp = (light((0, 0, 1)),)
        np.testing.assert_array_equal(render(c, relight=1, lights=lamp), render(c, relight=1, specular=0, lights=lamp))
        d = dc_only(c)
        np.testing.assert_array_equal(render(d, relight=1, lights=lamp, specular=1),
                                      render(d, relight=1, lights=lamp))
        np.testing.assert_array_equal(render(c), render(c, specular=1))   # nothing to keep without relight or catch

    def test_sh_degree_clamp_limits_what_is_kept(self):
        c = shiny()
        instance = s.SplatInstance(c, relight=1, specular=1, sh_degree=0)
        scene = s.Scene(splats=(instance,))
        np.testing.assert_allclose(s.render(scene, camera(), W, H, ambient=1.0),
                                   render(dc_only(c)), atol=1e-6)

    def test_gpu_matches_cpu(self):
        from nodebased import gpu3d
        if not gpu3d.available():
            self.skipTest("no wgpu adapter")
        from tests import gpu_precision
        c = shiny()
        scene = s.Scene(lights=(light((0, 0, 1)),),
                        splats=(s.SplatInstance(c, relight=.8, specular=1),))
        cpu = s.render(scene, camera(), W, H, ambient=.3)
        gpu = gpu3d.render(scene, camera(), W, H, ambient=.3)
        np.testing.assert_allclose(gpu, cpu, atol=gpu_precision.tolerance(3e-3), rtol=0)
        self.assertGreater(np.abs(cpu - s.render(replace(scene, splats=(replace(scene.splats[0], specular=0),)),
                                                 camera(), W, H, ambient=.3)).max(), .01)


class ShinySphereRelightTests(unittest.TestCase):
    """The mesh path already carries specular per light; pin that the highlight survives the Relight node."""

    def graph(self, relight_specular):
        d = Dispatcher()
        for key, kind, params in (
                ("ball", "Sphere3D", dict(red=.5, green=.3, blue=.2, spec_amount=1.0, spec_shininess=40,
                                          rows=24, columns=48)),
                ("lamp", "Light3D", dict(tx=1.0, ty=1.0, tz=2.0, intensity=1.0)),
                ("cam", "Camera3D", dict(tz=4.0)), ("scene", "Scene3D", {}),
                ("beauty", "Render3D", dict(width=48, height=48, samples=1, ambient=.2)),
                ("bundle", "Render3D", dict(width=48, height=48, samples=1, ambient=.2, render_output="relight")),
                ("relight", "Relight", dict(red=.2, green=.2, blue=.2, specular=relight_specular))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        for target, slot, source in (("scene", "object0", "ball"), ("scene", "object1", "lamp"),
                                     ("beauty", "scene", "scene"), ("beauty", "camera", "cam"),
                                     ("bundle", "scene", "scene"), ("bundle", "camera", "cam"),
                                     ("relight", "image", "bundle"), ("relight", "light0", "lamp")):
            d.execute(dict(op="connect", id=target, input=slot, source=source))
        return d

    def peaks(self, specular):
        d = self.graph(specular)
        direct = Evaluator().evaluate(d.document, target="beauty")
        relit = Evaluator().evaluate(d.document, target="relight")
        return luminance(direct).max(), luminance(relit).max(), direct, relit

    def test_highlight_survives_and_the_knob_scales_it(self):
        direct_peak, relit_peak, direct, relit = self.peaks(1.0)
        self.assertAlmostEqual(relit_peak, direct_peak, delta=2e-3)
        np.testing.assert_allclose(relit[..., :3], direct[..., :3], atol=2e-3)
        _, off_peak, _, _ = self.peaks(0.0)
        _, half_peak, _, _ = self.peaks(.5)
        self.assertLess(off_peak, direct_peak - .1)            # 0 removes the highlight
        self.assertAlmostEqual(half_peak, (off_peak + direct_peak) / 2, delta=2e-2)


class KnobTests(unittest.TestCase):
    def test_the_node_has_the_knob_and_old_documents_load_with_it_off(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "shiny.ply")
            splats.write_ply(shiny(), path)
            d = Dispatcher()
            for key, kind, params in (("read", "ReadSplat3D", dict(splat_path=str(path), splat_relight=1.0)),
                                      ("cam", "Camera3D", dict(tz=6.0)), ("scene", "Scene3D", {}),
                                      ("render", "Render3D", dict(width=W, height=H))):
                d.execute(dict(op="create", id=key, type=kind))
                for param, value in params.items():
                    d.execute(dict(op="set", id=key, param=param, value=value))
            d.execute(dict(op="connect", id="scene", input="object0", source="read"))
            d.execute(dict(op="connect", id="render", input="scene", source="scene"))
            d.execute(dict(op="connect", id="render", input="camera", source="cam"))
            self.assertEqual(0.0, d.document["nodes"]["read"]["params"]["splat_specular"])
            off = Evaluator().evaluate(d.document, target="render")
            d.execute(dict(op="set", id="read", param="splat_specular", value=1.0))
            on = Evaluator().evaluate(d.document, target="render")
            self.assertGreater(luminance(on).max() - luminance(off).max(), .05)
            with self.assertRaises(Exception):
                d.execute(dict(op="set", id="read", param="splat_specular", value=1.5))
            d.execute(dict(op="set", id="read", param="splat_specular", value=0.0))
            document = copy.deepcopy(d.document)
            del document["nodes"]["read"]["params"]["splat_specular"]
            old = Path(folder, "old.nbcomp")
            old.write_text(json.dumps(document))
            loaded = load_document(old)
            self.assertEqual(0.0, loaded["nodes"]["read"]["params"]["splat_specular"])
            np.testing.assert_array_equal(off, Evaluator().evaluate(loaded, target="render"))
        knob = next(g for g in knob_layout("ReadSplat3D") if "splat_specular" in g.params)
        self.assertEqual(("float_slider", "Keep specular", (0, 1)),
                         (knob.kind, knob.label, tuple(knob.soft_range)))


if __name__ == "__main__":
    unittest.main()
