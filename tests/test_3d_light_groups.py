"""Light groups and Render3D's per-light layers (lane L4, plan "Rendering 6", step R2 of 3, part 1).

Every light has a `light_group`; Render3D's `lights` pass writes one `light.<group>` RGB layer per group. The layers
add up to the beauty minus what no light touches, in the raster, ray-traced and path-traced modes (the path tracer on
the CPU reference and on the GPU, on whichever adapter the test run forces: see run/force-adapter.py)."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from nodebased import core, envlight, gpu3d, pathtrace as pt, scene3d as s
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.media import read_media_raster, write_exr

SIZE = (40, 28)
CAMERA = s.Camera(s.Transform3D(position=s.Vec3(0, 1.6, 5.0)), s.Vec3(0, -0.3, 0), 40.0)
BLACK = (0.0, 0.0, 0.0, 0.0)


def sun_map(width=64, height=32, sky=0.15, sun=12.0):
    d, _ = envlight.direction_grid(width, height)
    rgb = np.full((height, width, 3), sky, np.float32)
    rgb[d[..., 0] > 0.9] = sun
    return rgb


def environment(group="sky", **kw):
    rgb = sun_map()
    return envlight.Environment(rgb, envlight.fingerprint_of(rgb), 1.0, light_group=group, name="sky", **kw)


def ground(y=-1.0):
    return s.Geometry(np.array([[-8, y, 6], [8, y, 6], [8, y, -12], [-8, y, -12]], "f4"),
                      np.array([[0, 1, 2], [0, 2, 3]], "i4"), (.5, .5, .5, 1))


def ball(**kw):
    return replace(s._sphere(0.9, 24, (.7, .35, .25, 1), s.Transform3D(position=s.Vec3(0, -0.1, 0))),
                   material="pbr", metallic=.2, pbr_roughness=.4, emission=0.05, **kw)


def y1_scene(environments=True, key_shadows=True):
    """The Y1 scene (a PBR ball on a floor under an environment with a hard sun) with a shadowing key, a fill of two
    point lights and a rim, so there are three groups besides the sky."""
    lights = (
        s.Light("Directional", (1, .92, .8), 1.4, s.Vec3(3, 4, 2), s.Vec3(), shadows=key_shadows, name="key", light_group="key"),
        s.Light("Point", (.5, .6, 1), 2.0, s.Vec3(-3, 2, 3), name="fill_left", light_group="fill"),
        s.Light("Point", (1, .8, .6), 1.0, s.Vec3(3, 1.5, 3), name="fill_right", light_group="fill"),
        s.Light("Point", (1, 1, 1), 1.5, s.Vec3(0, 2, -3), name="rim"),
    )
    return s.Scene((ball(), ground()), lights, environments=(environment(),) if environments else ())


def no_light_part(scene, mode, **kw):
    """Whatever no light touches, rendered the way the beauty is: the emission of every surface."""
    if mode == "pathtrace":
        return pt.render(replace(scene, lights=(), environments=()), CAMERA, *SIZE, BLACK, 0.0, "rgba", kw["path"],
                         backend=kw.get("backend", "cpu"))
    return s.render(scene, CAMERA, *SIZE, output="emission", mode=mode)


def multichannel(scene, mode, ambient=0.1, **kw):
    return s.render_multichannel(scene, CAMERA, *SIZE, BLACK, passes="beauty,lights", ambient=ambient, mode=mode, **kw)


class GroupNameTests(unittest.TestCase):
    def test_a_light_without_a_group_is_in_the_default_group(self):
        self.assertEqual(s.light_group_name(s.Light()), "default")
        self.assertEqual(s.light_group_name(s.Light(light_group="  key ")), "key")
        self.assertEqual(s.light_group_name(environment("sky")), "sky")

    def test_layers_follow_the_groups_in_order_of_first_appearance_and_ambient_comes_last(self):
        _, layers = multichannel(y1_scene(), "raster")
        self.assertEqual(list(layers), ["light.key", "light.fill", "light.default", "light.sky", "light.ambient"])
        _, no_ambient = multichannel(y1_scene(), "raster", ambient=0.0)
        self.assertNotIn("light.ambient", no_ambient)

    def test_group_names_become_safe_layer_names_and_never_collide(self):
        lights = (s.Light("Point", (1, 1, 1), 1, s.Vec3(0, 2, 2), light_group="key light"),
                  s.Light("Point", (1, 1, 1), 1, s.Vec3(1, 2, 2), light_group="key-light"),
                  s.Light("Point", (1, 1, 1), 1, s.Vec3(2, 2, 2), light_group="ambient"))
        _, layers = multichannel(s.Scene((ball(),), lights), "raster")
        self.assertEqual(list(layers), ["light.key_light", "light.key_light_2", "light.ambient", "light.ambient_2"])

    def test_a_scene_without_lights_has_no_light_layers_and_the_pass_is_optional(self):
        _, layers = multichannel(s.Scene((ball(),)), "raster", ambient=0.0)
        self.assertEqual(layers, {})
        beauty, layers = s.render_multichannel(y1_scene(), CAMERA, *SIZE, BLACK, passes="depth")
        self.assertEqual(list(layers), ["depth"])
        self.assertEqual(float(beauty.max()), 0.0)

    def test_the_pass_name_is_parsed(self):
        self.assertEqual(s.parse_passes("lights,beauty"), ("beauty", "lights"))


class RasterSumTests(unittest.TestCase):
    def check(self, mode):
        scene = y1_scene()
        beauty, layers = multichannel(scene, mode)
        self.assertEqual(len(layers), 5)
        total = sum(layer[..., :3] for layer in layers.values()) + no_light_part(scene, mode)[..., :3]
        np.testing.assert_allclose(total, beauty[..., :3], atol=2e-5)
        for name, layer in layers.items():
            self.assertGreater(float(layer[..., :3].max()), 0.01, name)
        return layers

    def test_raster_layers_add_up_to_the_beauty_minus_emission(self):
        self.check("raster")

    def test_ray_traced_layers_add_up_to_the_beauty_minus_emission(self):
        self.check("raytrace")

    def test_a_group_moves_with_its_own_lights_only(self):
        scene = y1_scene()
        _, base = multichannel(scene, "raster")
        dimmer = replace(scene, lights=tuple(replace(l, intensity=l.intensity * 0.5) if l.light_group == "fill" else l
                                             for l in scene.lights))
        _, changed = multichannel(dimmer, "raster")
        np.testing.assert_allclose(changed["light.fill"][..., :3], base["light.fill"][..., :3] * 0.5, atol=2e-5)
        for name in ("light.key", "light.default", "light.sky", "light.ambient"):
            np.testing.assert_array_equal(changed[name], base[name])

    def test_a_shadow_belongs_to_the_group_that_casts_it(self):
        scene = y1_scene(environments=False)
        _, lit = multichannel(scene, "raytrace")
        _, unshadowed = multichannel(y1_scene(environments=False, key_shadows=False), "raytrace")
        self.assertGreater(float(np.abs(lit["light.key"] - unshadowed["light.key"]).max()), 0.1)
        for name in ("light.fill", "light.default", "light.ambient"):
            np.testing.assert_allclose(lit[name], unshadowed[name], atol=1e-6)


class PathTracedSumTests(unittest.TestCase):
    PATH = pt.PathSettings(samples=48, max_bounces=3, diffuse_bounces=2)

    def check(self, backend):
        scene = y1_scene()
        beauty, layers = multichannel(scene, "pathtrace", path=self.PATH, backend=backend)
        self.assertEqual(len(layers), 5)
        total = sum(layer[..., :3] for layer in layers.values()) + no_light_part(
            scene, "pathtrace", path=self.PATH, backend=backend)[..., :3]
        error = np.abs(total - beauty[..., :3])
        # The layers and the beauty are separate renders (each light's random numbers differ between them): the sums
        # agree to Monte Carlo noise, a few percent of the picture's own level.
        self.assertLess(float(error.mean()), 0.03 * float(beauty[..., :3].mean()))
        self.assertGreater(float(beauty[..., :3].mean()), 0.2)

    def test_cpu_path_tracer_layers_add_up_to_the_beauty(self):
        self.check("cpu")

    @unittest.skipUnless(gpu3d.available(), "no GPU adapter")
    def test_gpu_path_tracer_layers_add_up_to_the_beauty(self):
        self.check("gpu")

    def test_point_and_directional_light_groups_add_up_exactly(self):
        # without an area light or an environment no random number depends on the light set: the sum is exact
        scene = y1_scene(environments=False)
        beauty, layers = multichannel(scene, "pathtrace", path=self.PATH)
        total = sum(layer[..., :3] for layer in layers.values()) + no_light_part(scene, "pathtrace", path=self.PATH)[..., :3]
        np.testing.assert_allclose(total, beauty[..., :3], atol=1e-4)


class GraphTests(unittest.TestCase):
    def graph(self, **render):
        d = Dispatcher()
        for key, kind, params in (
                ("ball", "Sphere3D", dict(sphere_radius=.8)),
                ("key", "Light3D", dict(light_type="Directional", light_group="key", shadows="on")),
                ("fill", "Light3D", dict(light_type="Point", tx=-3.0, ty=2.0, tz=3.0, light_group="fill")),
                ("plain", "Light3D", dict(light_type="Point", tx=3.0, ty=2.0, tz=3.0)),
                ("camera", "Camera3D", {}), ("scene", "Scene3D", {}),
                ("render", "Render3D", dict(width=32, height=24, samples=1, render_output="multichannel",
                                            passes="beauty,lights", **render)),
                ("write", "Write", dict(bit_depth="float"))):
            d.execute(dict(op="create", id=key, type=kind, params=params))
        for slot, source in (("object0", "ball"), ("object1", "key"), ("object2", "fill"), ("object3", "plain")):
            d.execute(dict(op="connect", id="scene", input=slot, source=source))
        for slot in ("scene", "camera"):
            d.execute(dict(op="connect", id="render", input=slot, source=slot))
        d.execute(dict(op="connect", id="write", input="image", source="render"))
        return d

    def test_light3d_has_a_light_group_knob_and_it_reaches_the_renderer(self):
        self.assertEqual(core.SPECS["Light3D"]["params"]["light_group"], "")
        d = self.graph()
        raster = Evaluator().evaluate_raster(d.document, "render")
        self.assertEqual(list(raster.layers), ["light.key", "light.fill", "light.default", "light.ambient"])
        light = s.light_from_node({"params": d.document["nodes"]["fill"]["params"], "name": "fill"})
        self.assertEqual((light.name, light.light_group), ("fill", "fill"))

    def test_an_old_light3d_without_the_knob_is_in_the_default_group(self):
        d = self.graph()
        del d.document["nodes"]["plain"]["params"]["light_group"]
        light = s.light_from_node({"params": d.document["nodes"]["plain"]["params"], "name": "plain"})
        self.assertEqual(s.light_group_name(light), "default")

    def test_light_layers_write_to_an_exr_as_light_dot_group_and_read_back(self):
        d = self.graph()
        raster = Evaluator().evaluate_raster(d.document, "render")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lights.exr"
            write_exr(path, raster.pixels, bits="float", layers={name: layer.pixels for name, layer in raster.layers.items()})
            back = read_media_raster(path)
        self.assertEqual(sorted(back.layers), sorted(raster.layers))
        for name, layer in raster.layers.items():
            np.testing.assert_allclose(back.layers[name].pixels[..., :3], layer.pixels[..., :3], atol=1e-6)

    def test_motion_blurred_multichannel_carries_the_layers_too(self):
        from nodebased import motionblur
        scene = y1_scene(environments=False)
        moved = replace(scene, geometries=(replace(ball(), transform=s.Transform3D(position=s.Vec3(0.4, -0.1, 0))), ground()))
        beauty, layers = motionblur.multichannel(
            [(scene, CAMERA), (moved, CAMERA)], *SIZE, BLACK, passes="beauty,lights", ambient=0.1, samples=1, cancel=None,
            mode="raster", progress=None, volume=None, backend="cpu", path=None)
        self.assertEqual(list(layers), ["light.key", "light.fill", "light.default", "light.ambient"])
        a = multichannel(scene, "raster")[1]
        b = multichannel(moved, "raster")[1]
        np.testing.assert_allclose(layers["light.key"], (a["light.key"] + b["light.key"]) / 2, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
