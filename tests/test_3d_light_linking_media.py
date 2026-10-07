"""Light linking for liquid glints, whitewater and particles, and smoke (lane L4, plan "Rendering 8", step T1 of 3, part 2).

Rendering 6 left these out: a liquid's glints and its refracted solids' lighting, `pbr` particles (spheres, points, whitewater
foam and spray lit like a mesh) and the shadows they throw, and smoke (a `Volume` had no link). Each now has `light_link`
(ParticleRender3D, Plume3D, ReadVDB3D and FluidCache3D carry the knobs) and every renderer that draws it reads it. The property
every test states: **an excluded light leaves no mark on the object and gets no shadow from it**, which in these scenes means
the render with the object excluded from a light equals the render with that light only where the object is not."""
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, scene3d as s, volumerender
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from tests.test_3d_gpu_volumes import CAMERA as SMOKE_CAMERA, H as SMOKE_H, SETTINGS as SMOKE_SETTINGS, W as SMOKE_W
from tests.test_liquid_render import CAMERA as LIQUID_CAMERA, board

LINKED = ("exclude", ("fill",))
NOBODY = ("include", ())


# --- liquid ------------------------------------------------------------------------------------------------------------

KEY = s.Light("Point", (1, 1, 1), 2.0, s.Vec3(2, 3, 4), name="key")
FILL = s.Light("Point", (1, .4, .4), 2.0, s.Vec3(-3, 2, 4), name="fill")
BACKGROUND = (0.3, 0.5, 0.9, 1)


def liquid_scene(link, lights=(KEY, FILL)):
    sphere = replace(s._sphere(1.0, 32, (1, 1, 1, 1), s.Transform3D()), material="liquid", reflection=1.0,
                     roughness=0.3, light_link=link)
    return s.Scene((sphere, board()), lights)


def liquid(scene, mode, gpu=False):
    return (gpu3d.render if gpu else s.render)(scene, LIQUID_CAMERA, 64, 64, BACKGROUND, ambient=.1, mode=mode)


class LiquidTests(unittest.TestCase):
    def test_an_excluded_light_leaves_no_glint_on_the_liquid(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                plain = liquid(liquid_scene(s.LIGHT_LINK_ALL), mode)
                excluded = liquid(liquid_scene(LINKED), mode)
                included = liquid(liquid_scene(("include", ("key",))), mode)
                self.assertGreater(float(np.abs(plain - excluded).max()), .03)        # the glint was there
                np.testing.assert_array_equal(excluded, included)                      # the two ways to say it agree
                np.testing.assert_array_equal(liquid(liquid_scene(("exclude", ("nobody",))), mode), plain)

    def test_a_liquid_excluded_from_every_light_has_no_glint_but_still_refracts(self):
        for mode in ("raster", "raytrace"):
            dark = liquid(liquid_scene(NOBODY), mode)
            lit = liquid(liquid_scene(s.LIGHT_LINK_ALL), mode)
            self.assertGreater(float((lit - dark)[..., :3].max()), .03)                # the glints were there
            self.assertGreater(float((lit - dark)[..., :3].min()), -1e-6)              # and they only ever add light
            self.assertGreater(float(dark[32, 32, 3]), .99)                            # still opaque, still drawn

    def test_a_solid_seen_through_the_liquid_keeps_its_own_link(self):
        """The board behind the sphere is what the refracted ray lights: it is excluded from `fill`, the liquid is not."""
        lit_board = liquid(liquid_scene(s.LIGHT_LINK_ALL), "raytrace")
        dim_board = liquid(replace(liquid_scene(s.LIGHT_LINK_ALL), geometries=(
            liquid_scene(s.LIGHT_LINK_ALL).geometries[0], replace(board(), light_link=LINKED))), "raytrace")
        self.assertGreater(float(np.abs(lit_board - dim_board)[24:40, 24:40, :3].max()), .005)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuLiquidTests(unittest.TestCase):
    def test_the_ray_traced_gpu_matches_the_cpu_for_every_link(self):
        for link in (s.LIGHT_LINK_ALL, LINKED, ("include", ("key",)), NOBODY):
            with self.subTest(link=link):
                cpu = liquid(liquid_scene(link), "raytrace")
                gpu = liquid(liquid_scene(link), "raytrace", gpu=True)
                self.assertLessEqual(float(np.abs(cpu - gpu).max()), 2e-3)

    def test_the_gpu_shows_the_glint_only_for_the_linked_light(self):
        plain = liquid(liquid_scene(s.LIGHT_LINK_ALL), "raytrace", gpu=True)
        excluded = liquid(liquid_scene(LINKED), "raytrace", gpu=True)
        self.assertGreater(float(np.abs(plain - excluded).max()), .03)

    def test_a_linked_solid_behind_the_liquid_is_lit_by_its_own_link_on_the_gpu(self):
        scene = replace(liquid_scene(s.LIGHT_LINK_ALL), geometries=(liquid_scene(s.LIGHT_LINK_ALL).geometries[0],
                                                                    replace(board(), light_link=LINKED)))
        cpu, gpu = liquid(scene, "raytrace"), liquid(scene, "raytrace", gpu=True)
        self.assertLessEqual(float(np.abs(cpu - gpu).max()), 2e-3)


# --- particles ---------------------------------------------------------------------------------------------------------

SIZE = 96
SUN = s.Light("Directional", (1, 1, 1), 2.0, s.Vec3(), s.Vec3(0, -1, 0), name="sun", shadows=True)
SKY = s.Light("Point", (.4, .6, 1), 2.0, s.Vec3(2, 3, 4), name="sky", shadows=True)


def sphere_particle(link=s.LIGHT_LINK_ALL, **fields):
    base = dict(positions=np.zeros((1, 3), np.float32), sizes=np.array([1.6], np.float32),
                colors=np.array([[0.6, 0.3, 0.2, 1.0]], np.float32), render_as="spheres", material="pbr",
                metallic=0.2, pbr_roughness=0.4, pbr_specular=0.5, light_link=link)
    base.update(fields)
    return s.ParticleInstance(**base)


def draw(scene, mode, **kwargs):
    return s.render(scene, s.Camera(), SIZE, SIZE, ambient=0.05, mode=mode, **kwargs)


class ParticleTests(unittest.TestCase):
    def test_a_lit_particle_set_is_lit_only_by_its_included_lights(self):
        for mode in ("raster", "raytrace"):
            with self.subTest(mode=mode):
                both = draw(s.Scene(particles=(sphere_particle(),), lights=(KEY, FILL)), mode)
                excluded = draw(s.Scene(particles=(sphere_particle(LINKED),), lights=(KEY, FILL)), mode)
                key_only = draw(s.Scene(particles=(sphere_particle(),), lights=(KEY,)), mode)
                self.assertGreater(float(np.abs(both - excluded).max()), .03)
                np.testing.assert_allclose(excluded, key_only, atol=1e-6)

    def test_whitewater_particles_follow_the_link_like_any_lit_set(self):
        foam = lambda link: replace(sphere_particle(link, positions=np.array([[0, 0, 0], [.8, .2, 0]], np.float32),
                                                    sizes=np.array([1.0, .6], np.float32),
                                                    colors=np.ones((2, 4), np.float32)),
                                    whitewater_type=np.array([0, 1], np.uint8), render_as="foam")
        look = lambda link: s.apply_particle_look(foam(link), {"particle_material": "pbr", "light_link": link[0],
                                                              "light_link_list": ",".join(link[1])})
        self.assertEqual(look(LINKED).light_link, LINKED)
        both = draw(s.Scene(particles=(look(s.LIGHT_LINK_ALL),), lights=(KEY, FILL)), "raytrace")
        excluded = draw(s.Scene(particles=(look(LINKED),), lights=(KEY, FILL)), "raytrace")
        key_only = draw(s.Scene(particles=(look(s.LIGHT_LINK_ALL),), lights=(KEY,)), "raytrace")
        self.assertGreater(float(np.abs(both - excluded).max()), .005)
        np.testing.assert_allclose(excluded, key_only, atol=1e-6)

    def plane(self):
        return replace(s._card(20, 20, (0.8, 0.8, 0.8, 1.0),
                               s.Transform3D(position=s.Vec3(0, -1.5, 0), rotation=s.Vec3(-90, 0, 0))),
                       material="pbr", pbr_roughness=0.8)

    def shadowed_plane(self, link):
        caster = replace(sphere_particle(link), sizes=np.array([1.0], np.float32), material="standard")
        camera = s.Camera(transform=s.Transform3D(position=s.Vec3(0, 3, 8)), target=s.Vec3(0, -1.5, 0))
        scene = s.Scene(geometries=(self.plane(),), particles=(caster,), lights=(SUN,))
        return s.render(scene, camera, SIZE, SIZE, ambient=.05), s.render(replace(scene, geometries=()), camera, SIZE, SIZE)

    def test_a_particle_set_excluded_from_the_sun_casts_no_shadow_from_it(self):
        camera = s.Camera(transform=s.Transform3D(position=s.Vec3(0, 3, 8)), target=s.Vec3(0, -1.5, 0))
        bare = s.render(s.Scene(geometries=(self.plane(),), lights=(SUN,)), camera, SIZE, SIZE, ambient=.05)
        plain, sprite = self.shadowed_plane(s.LIGHT_LINK_ALL)
        linked, _ = self.shadowed_plane(("exclude", ("sun",)))
        beside = sprite[..., 3] <= 0
        self.assertGreater(float(np.abs(plain - bare)[beside].max()), .05)               # it did shadow the plane
        np.testing.assert_allclose(linked[beside], bare[beside], atol=1e-6)               # now it does not
        listed, _ = self.shadowed_plane(("include", ("sky",)))                           # lit by another light only
        np.testing.assert_allclose(listed[beside], bare[beside], atol=1e-6)

    def test_the_path_tracer_composites_lit_particles_with_the_link(self):
        from nodebased import pathtrace as pt
        path = pt.PathSettings(samples=4, max_bounces=1, diffuse_bounces=1)
        draw_pt = lambda scene: pt.render(scene, s.Camera(), 48, 48, (0, 0, 0, 0), 0.05, "rgba", path)
        both = draw_pt(s.Scene(particles=(sphere_particle(),), lights=(KEY, FILL)))
        excluded = draw_pt(s.Scene(particles=(sphere_particle(LINKED),), lights=(KEY, FILL)))
        key_only = draw_pt(s.Scene(particles=(sphere_particle(),), lights=(KEY,)))
        self.assertGreater(float(np.abs(both - excluded).max()), .03)
        np.testing.assert_allclose(excluded, key_only, atol=1e-6)

    def test_an_old_particle_set_has_no_link_and_links_every_light(self):
        self.assertEqual(sphere_particle().light_link, s.LIGHT_LINK_ALL)
        self.assertEqual(s.apply_particle_look(sphere_particle(), {}).light_link, s.LIGHT_LINK_ALL)


# --- smoke -------------------------------------------------------------------------------------------------------------

KEY_SUN = s.Light("Directional", position=s.Vec3(4, 5, 3), target=s.Vec3(), name="key")
FILL_LAMP = s.Light("Point", color=(1.0, 0.6, 0.3), intensity=3.0, position=s.Vec3(-1.5, 1.2, 1.0),
                    target=s.Vec3(0, .5, 0), falloff_type="Quadratic", name="fill")
OVERHEAD = s.Light("Directional", position=s.Vec3(0, 6, 0.01), target=s.Vec3(), shadows=True, name="overhead")
SMOKE_TOLERANCE = 2e-3


def smoke_scene(link, lights=(KEY_SUN, FILL_LAMP), geometries=()):
    return s.Scene(geometries, lights, volumes=(replace(s.analytic_plume(32, 0), light_link=link),))


def smoke(scene, gpu=False, ambient=.1):
    return (gpu3d.render if gpu else s.render)(scene, SMOKE_CAMERA, SMOKE_W, SMOKE_H, ambient=ambient, volume=SMOKE_SETTINGS)


def floor():
    return s._card(4, 4, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -0.05, 0), s.Vec3(-90, 0, 0)))


class SmokeTests(unittest.TestCase):
    gpu = False

    def test_a_plume_excluded_from_a_light_shows_no_in_scattering_from_it(self):
        plain = smoke(smoke_scene(s.LIGHT_LINK_ALL), self.gpu)
        excluded = smoke(smoke_scene(LINKED), self.gpu)
        key_only = smoke(s.Scene(lights=(KEY_SUN,), volumes=(s.analytic_plume(32, 0),)), self.gpu)
        self.assertGreater(int((plain[..., 3] > .02).sum()), 400)                          # a plume is on screen
        self.assertGreater(float(np.abs(plain - excluded).max()), .05)                     # the lamp lit it
        np.testing.assert_allclose(excluded, key_only, atol=SMOKE_TOLERANCE, rtol=0)       # and now only the key does

    def test_a_plume_excluded_from_every_light_keeps_only_its_ambient(self):
        dark = smoke(smoke_scene(NOBODY), self.gpu)
        unlit_but_ambient = smoke(s.Scene(lights=(), volumes=(s.analytic_plume(32, 0),)), self.gpu, ambient=.1)
        lit = smoke(smoke_scene(s.LIGHT_LINK_ALL), self.gpu)
        self.assertLess(float(dark[..., :3].sum()), float(lit[..., :3].sum()) * .5)
        np.testing.assert_allclose(dark[..., 3], unlit_but_ambient[..., 3], atol=SMOKE_TOLERANCE, rtol=0)

    def test_a_plume_excluded_from_the_overhead_light_throws_no_shadow_on_the_floor(self):
        scene = smoke_scene(("exclude", ("overhead",)), lights=(OVERHEAD,), geometries=(floor(),))
        shadowed = smoke(smoke_scene(s.LIGHT_LINK_ALL, lights=(OVERHEAD,), geometries=(floor(),)), self.gpu)
        linked = smoke(scene, self.gpu)
        clear = smoke(scene, self.gpu) if False else (gpu3d.render if self.gpu else s.render)(
            scene, SMOKE_CAMERA, SMOKE_W, SMOKE_H, ambient=.1, volume=replace(SMOKE_SETTINGS, shadow_density=0.0))
        bare = smoke(s.Scene(geometries=(floor(),), lights=(OVERHEAD,)), self.gpu)
        floor_only = (smoke(s.Scene(volumes=scene.volumes), self.gpu)[..., 3] < .002) & (bare[..., 3] > .99)
        self.assertGreater(int(floor_only.sum()), 200)
        self.assertGreater(float(np.abs(shadowed - clear)[floor_only].max()), .08)        # unlinked smoke shadows the floor
        np.testing.assert_allclose(linked[floor_only], clear[floor_only], atol=SMOKE_TOLERANCE, rtol=0)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuSmokeTests(SmokeTests):
    gpu = True

    def test_the_gpu_matches_the_cpu_for_every_link(self):
        for link in (s.LIGHT_LINK_ALL, LINKED, ("include", ("fill",)), NOBODY):
            with self.subTest(link=link):
                cpu, gpu = smoke(smoke_scene(link)), smoke(smoke_scene(link), True)
                self.assertLess(float(np.abs(cpu - gpu).max()), 2e-3)
                self.assertLess(float(np.abs(cpu - gpu).mean()), 2e-4)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class ViewportSmokeTests(unittest.TestCase):
    """The viewport draws smoke with the same shader; its light table lists every light, not just the point-like ones."""
    BACKGROUND = (0.02, 0.02, 0.03, 1.0)
    VIEW = s.Camera(s.Transform3D(s.Vec3(0.3, 0.6, 3.2)), s.Vec3(0, 0.5, 0), 45.0, 0.07, 1000.0)

    def setUp(self):
        from nodebased import viewportgpu
        self.gpu = viewportgpu.renderer()
        self.assertIsNotNone(self.gpu, viewportgpu.failure())

    def draw(self, link, lights=(KEY_SUN, FILL_LAMP)):
        scene = s.Scene((), lights, volumes=(replace(s.analytic_plume(40), light_link=link),))
        return self.gpu.render(scene, self.VIEW, 160, 120, self.BACKGROUND, headlight=False, ambient=.1).astype(int)

    def test_a_plume_excluded_from_a_light_shows_no_in_scattering_from_it_in_the_viewport(self):
        plain, linked = self.draw(s.LIGHT_LINK_ALL), self.draw(LINKED)
        key_only = self.draw(s.LIGHT_LINK_ALL, lights=(KEY_SUN,))
        self.assertGreater(int(np.abs(plain - linked).max()), 20)
        self.assertLessEqual(int(np.abs(linked - key_only).max()), 1)


# --- smoke in the path tracers ------------------------------------------------------------------------------------------

from nodebased import gpupathtrace, pathtrace as pt           # noqa: E402
from tests.test_volume_scene import box as smoke_box           # noqa: E402

PT_CAMERA = s.Camera()                                         # at (0, 0, 5) looking at the origin
PT_SMOKE = volumerender.VolumeSettings(absorption=0.4, scattering=0.6, step_size=0.05, shadow_steps=16)
PT_KEY = s.Light("Directional", (1, 1, 1), 1.0, s.Vec3(), s.Vec3(0.3, -0.2, -1.0), name="key")
PT_FILL = s.Light("Point", (1, .5, .3), 4.0, s.Vec3(-2.0, 1.5, 3.0), name="fill")
PT_OVERHEAD = s.Light("Directional", (1, 1, 1), 1.5, s.Vec3(), s.Vec3(0, -1, 0.01), name="overhead")


def traced(scene, backend="cpu", size=12, samples=96, volume=PT_SMOKE, **settings):
    settings.setdefault("max_bounces", 6)
    settings.setdefault("diffuse_bounces", 6)
    return pt.render(scene, PT_CAMERA, size, size, (0, 0, 0, 0), 0.0, "rgba", pt.PathSettings(samples=samples, **settings),
                     volume=volume, backend=backend)


def pt_plume(link, lights=(PT_KEY, PT_FILL), geometries=()):
    return s.Scene(geometries, lights, volumes=(replace(smoke_box(24, 2.0), light_link=link),))


class PathTracedSmokeTests(unittest.TestCase):
    backend = "cpu"

    def test_a_volume_excluded_from_a_light_is_not_lit_by_it(self):
        plain = traced(pt_plume(s.LIGHT_LINK_ALL), self.backend)
        excluded = traced(pt_plume(LINKED), self.backend)
        key_only = traced(s.Scene((), (PT_KEY,), volumes=(smoke_box(24, 2.0),)), self.backend)
        self.assertGreater(float(np.abs(plain - excluded).max()), .05)               # the point light lit it
        np.testing.assert_allclose(excluded, key_only, atol=2e-3, rtol=0)             # and now it is lit by the key alone

    def test_a_volume_excluded_from_every_light_is_dark(self):
        dark = traced(pt_plume(NOBODY), self.backend)
        self.assertLess(float(dark[..., :3].max()), 1e-6)
        self.assertGreater(float(dark[6, 6, 3]), .1)                                  # still there: it still absorbs

    def test_a_volume_excluded_from_the_overhead_light_casts_no_shadow_on_the_floor(self):
        floor_card = s._card(8, 8, (.8, .8, .8, 1), s.Transform3D(s.Vec3(0, -1.2, 0), s.Vec3(-90, 0, 0)))
        camera = s.Camera(s.Transform3D(s.Vec3(0, 3, 6)), s.Vec3(0, -.6, 0))
        draw = lambda scene, **kw: pt.render(scene, camera, 12, 12, (0, 0, 0, 0), 0.0, "rgba",
                                             pt.PathSettings(samples=96, max_bounces=1, diffuse_bounces=1),
                                             volume=kw.get("volume", PT_SMOKE), backend=self.backend)
        scene = lambda link: pt_plume(link, (PT_OVERHEAD,), (floor_card,))
        shadowed = draw(scene(s.LIGHT_LINK_ALL))
        linked = draw(scene(("exclude", ("overhead",))))
        clear = draw(scene(("exclude", ("overhead",))), volume=replace(PT_SMOKE, shadow_density=0.0))
        floor = draw(s.Scene((floor_card,), (PT_OVERHEAD,)))[..., 3] > .99
        pixels = floor & (draw(s.Scene(volumes=(smoke_box(24, 2.0),)))[..., 3] < .01)
        self.assertGreater(int(pixels.sum()), 10)
        self.assertGreater(float(np.abs(shadowed - clear)[pixels].max()), .05)        # unlinked smoke shadows the floor
        np.testing.assert_allclose(linked[pixels], clear[pixels], atol=2e-3, rtol=0)


@unittest.skipUnless(gpu3d.available(), "no GPU adapter")
class GpuPathTracedSmokeTests(PathTracedSmokeTests):
    backend = "gpu"

    def setUp(self):
        if not gpupathtrace.soft_supported(gpu3d._state()):
            self.skipTest("this adapter does not support GPU smoke path tracing")

    def test_the_gpu_agrees_with_the_cpu_reference(self):
        for link in (s.LIGHT_LINK_ALL, LINKED):
            with self.subTest(link=link):
                cpu, gpu = traced(pt_plume(link), "cpu", samples=256), traced(pt_plume(link), "gpu", samples=256)
                self.assertLess(float(np.abs(cpu - gpu).mean()), .02)


# --- the knobs ---------------------------------------------------------------------------------------------------------

class NodeTests(unittest.TestCase):
    """Plume3D, ReadVDB3D, FluidCache3D and ParticleRender3D carry the knobs; old documents without them link everything."""

    def document(self, **plume):
        d = Dispatcher()
        for key, kind, params in (("plume", "Plume3D", dict(plume_resolution=16, **plume)),
                                  ("key", "Light3D", dict(light_type="Directional")),
                                  ("fill", "Light3D", dict(light_type="Point", tx=-1.5, ty=1.2, tz=1.0, intensity=3.0)),
                                  ("camera", "Camera3D", dict(tz=3.2, ty=.6)),
                                  ("scene", "Scene3D", {}),
                                  ("render", "Render3D", dict(width=48, height=36, samples=1, render_output="rgba"))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        for slot, source in (("object0", "plume"), ("object1", "key"), ("object2", "fill")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        return d

    def render(self, d):
        return Evaluator().evaluate_raster(d.document, "render").pixels

    def test_the_knobs_exist_with_defaults_that_link_everything(self):
        from nodebased.core import SPECS
        from nodebased.knobs import knob_layout
        for kind in ("Plume3D", "ReadVDB3D", "FluidCache3D", "ParticleRender3D"):
            params = SPECS[kind]["params"]
            self.assertEqual((params["light_link"], params["light_link_list"]), ("all", ""), kind)
            laid_out = sorted(p for group in knob_layout(kind) for p in group.params)
            self.assertEqual(laid_out, sorted(params), kind)

    def test_plume_link_reaches_the_render_and_old_documents_render_as_before(self):
        d = self.document()
        plain = self.render(d)
        name = d.document["nodes"]["fill"]["name"]
        d.document["nodes"]["plume"]["params"].update(light_link="exclude", light_link_list=name)
        linked = self.render(d)
        self.assertGreater(float(np.abs(plain - linked).max()), 0.0)
        for name in ("light_link", "light_link_list"):
            d.document["nodes"]["plume"]["params"].pop(name)
        np.testing.assert_array_equal(self.render(d), plain)

    def test_the_cache_node_gives_a_solved_volume_its_link(self):
        from tests.test_fluid3d_nodes import at, plume
        d = plume()
        evaluator = Evaluator()
        self.assertEqual(at(evaluator, d, "c", 2).light_link, s.LIGHT_LINK_ALL)
        d.document["nodes"]["c"]["params"].update(light_link="include", light_link_list="key")
        volume = at(evaluator, d, "c", 2)
        self.assertEqual(volume.light_link, ("include", ("key",)))
        self.assertGreater(float(volume.density.max()), 0.0)

    def test_particle_render_node_gives_its_set_the_link(self):
        link = s.light_link_from_params({"light_link": "exclude", "light_link_list": "fill; key"})
        self.assertEqual(link, ("exclude", ("fill", "key")))
        look = s.apply_particle_look(sphere_particle(), {"light_link": "exclude", "light_link_list": "sun"})
        self.assertEqual(look.light_link, ("exclude", ("sun",)))


if __name__ == "__main__":
    unittest.main()
