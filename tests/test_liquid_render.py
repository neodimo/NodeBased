"""Rendering plan 3 step D, part 1: the liquid material in the ray-traced modes (CPU reference).

A glass-like cube of liquid over a checkerboard: the board is refracted by the computed offset; reflection is
stronger at grazing angles; absorption darkens thick liquid more than thin; a thin sheet is not black; the render
is deterministic; documents without the new parameters render as before.
"""
import math
import unittest
from dataclasses import replace

import numpy as np

from nodebased import gpu3d, gpurt_render, scene3d as s
from nodebased.core import SCHEMA_VERSION, SPECS, upgrade_document
from nodebased.liquid_render import fresnel, sigma_of

SIZE = 64
BOARD_Z, BOARD_WIDTH, SQUARE = -4.0, 6.0, 0.75
CUBE = 3.0      # cross-section of the block of liquid
DEPTH = 5.0     # along the view axis: the thicker the block, the larger the refraction offset
IOR = 1.5
CAMERA = s.Camera(fov=70.0)      # a wide lens: rays leave the axis far enough to refract visibly


def checker_texture():
    texels = np.zeros((64, 64, 4), np.float32)
    row, col = np.indices((64, 64))
    even = ((row // 8) + (col // 8)) % 2 == 0
    texels[even] = (1.0, 0.1, 0.1, 1.0)
    texels[~even] = (0.1, 0.1, 1.0, 1.0)
    return texels


def board():
    return s._card(BOARD_WIDTH, BOARD_WIDTH, (1, 1, 1, 1), s.Transform3D(s.Vec3(0, 0, BOARD_Z)), checker_texture())


def board_colour(x, y):
    """Texture colour of the board at world (x, y) by the same mapping the renderer uses."""
    u, v = (x + BOARD_WIDTH / 2) / BOARD_WIDTH, (y + BOARD_WIDTH / 2) / BOARD_WIDTH
    even = (int((1 - v) * 64) // 8 + int(u * 64) // 8) % 2 == 0
    return np.array((1.0, 0.1, 0.1) if even else (0.1, 0.1, 1.0))


def board_margin(x, y):
    """Distance (world units) from (x, y) to the nearest square edge."""
    fx, fy = (x + BOARD_WIDTH / 2) % SQUARE, (y + BOARD_WIDTH / 2) % SQUARE
    return min(fx, SQUARE - fx, fy, SQUARE - fy)


def liquid_cube(size=CUBE, depth=None, **fields):
    base = dict(material="liquid", reflection=0.0, absorption_color=(1.0, 1.0, 1.0))
    base.update(fields)
    cube = s._cube(size, (1, 1, 1, 1), s.Transform3D())
    if depth is not None:
        cube = replace(cube, vertices=cube.vertices * np.array((1, 1, depth / size), np.float32))
    return replace(cube, **base)


def render(scene, mode="raytrace", size=SIZE, **kwargs):
    return s.render(scene, CAMERA, size, size, mode=mode, **kwargs)


def view_ray(px, py, size=SIZE):
    focal = 1 / math.tan(math.radians(CAMERA.fov) / 2)
    d = np.array(((2 * (px + .5) / size - 1) / focal, (1 - 2 * (py + .5) / size) / focal, -1.0))
    return np.array((0, 0, 5.0)), d / np.linalg.norm(d)


def predicted_board_point(px, py, ior, half=DEPTH / 2):
    """Where the pixel's ray reaches the board through a slab of liquid with faces at z = +/- half."""
    origin, d = view_ray(px, py)
    entry = origin + d * ((half - origin[2]) / d[2])
    ratio = 1 / ior
    cos_i = -d[2]
    cos_t = math.sqrt(1 - ratio ** 2 * (1 - cos_i ** 2))
    t = ratio * d + (ratio * cos_i - cos_t) * np.array((0, 0, 1.0))
    exit_point = entry + t * ((-half - half) / t[2])
    end = exit_point + d * ((BOARD_Z + half) / d[2])
    return end, exit_point


class LiquidRenderTests(unittest.TestCase):
    def test_board_is_refracted_by_the_computed_offset(self):
        scene = s.Scene((liquid_cube(depth=DEPTH, ior=IOR), board()))
        image = render(scene)
        straight = render(s.Scene((board(),)))
        checked = 0
        for py in range(12, 52):
            for px in range(12, 52):
                origin, d = view_ray(px, py)
                end, exit_point = predicted_board_point(px, py, IOR)
                plain = origin + d * ((BOARD_Z - origin[2]) / d[2])
                entry = origin + d * ((DEPTH / 2 - origin[2]) / d[2])
                if max(abs(entry[0]), abs(entry[1]), abs(exit_point[0]), abs(exit_point[1])) > CUBE / 2 - 0.1 or max(abs(end[:2]).max(), abs(plain[:2]).max()) > 2.9:
                    continue        # a side face of the cube, or off the board
                if min(board_margin(*end[:2]), board_margin(*plain[:2])) < 0.15:
                    continue
                if np.allclose(board_colour(*end[:2]), board_colour(*plain[:2])):
                    self.assertTrue(np.allclose(image[py, px, :3], board_colour(*end[:2]), atol=0.02))
                    continue
                # the refracted colour, not the straight-through one
                np.testing.assert_allclose(image[py, px, :3], board_colour(*end[:2]), atol=0.02)
                self.assertFalse(np.allclose(image[py, px, :3], straight[py, px, :3], atol=0.05))
                checked += 1
        self.assertGreaterEqual(checked, 6)

    def test_output_is_opaque_and_deterministic(self):
        scene = s.Scene((liquid_cube(), board()))
        a, b = render(scene), render(scene)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a[24:40, 24:40, 3] == 1))

    def test_reflection_is_stronger_at_grazing_angles(self):
        f_head, _, _ = fresnel(np.array([1.0]), 1.0, 1.333)
        f_graze, _, _ = fresnel(np.array([0.05]), 1.0, 1.333)
        self.assertAlmostEqual(float(f_head[0]), ((1.333 - 1) / (1.333 + 1)) ** 2, places=4)
        self.assertGreater(float(f_graze[0]), 0.5)
        # In the render: white surroundings, everything transmitted is absorbed, so the pixel is the reflection.
        sphere = replace(s._sphere(1.0, 32, (1, 1, 1, 1), s.Transform3D()), material="liquid", reflection=1.0,
                         absorption_color=(0.001, 0.001, 0.001), absorption_distance=0.05)
        image = render(s.Scene((sphere,)), size=96, background=(1, 1, 1, 1))
        centre = float(image[48, 48, 0])
        rim = float(np.max(image[48, 48:, 0][image[48, 48:, 0] < 0.999]))
        self.assertLess(centre, 0.1)
        self.assertGreater(rim, 0.15)
        self.assertGreater(rim, 3 * centre)

    def test_absorption_darkens_thick_liquid_more_than_thin(self):
        colour = (0.5, 0.8, 0.95)
        values = {}
        for name, size in (("thin", 1.0), ("thick", 2.0)):
            cube = liquid_cube(size, absorption_color=colour, absorption_distance=1.0)
            image = render(s.Scene((cube,)), background=(1, 1, 1, 1))
            values[name] = image[SIZE // 2, SIZE // 2, :3]
        np.testing.assert_allclose(values["thin"], colour, atol=0.02)
        np.testing.assert_allclose(values["thick"], np.square(colour), atol=0.02)
        self.assertTrue(np.all(values["thick"] < values["thin"]))
        np.testing.assert_allclose(np.exp(-sigma_of(colour, 1.0)), colour)

    def test_thin_sheet_is_not_black_and_does_not_bend(self):
        slab = liquid_cube(4.0, reflection=1.0, absorption_color=(0.5, 0.8, 0.95), absorption_distance=1.0)
        slab = replace(slab, vertices=slab.vertices * np.array((1, 1, 0.004), np.float32))   # 0.008 thick
        image = render(s.Scene((slab, board())))
        straight = render(s.Scene((board(),)))
        self.assertTrue(np.all(image[28:36, 28:36, :3].max(axis=2) > 0.3))
        # no bend: the board behind the sheet matches the bare board wherever the sheet is
        np.testing.assert_allclose(image[28:36, 28:36, :3], straight[28:36, 28:36, :3] * 0.98, atol=0.08)

    def test_total_internal_reflection_from_inside(self):
        f, tir, _ = fresnel(np.array([0.3]), 1.333, 1.0)
        self.assertTrue(bool(tir[0]))
        self.assertEqual(float(f[0]), 1.0)

    def test_standard_material_and_old_documents_are_unchanged(self):
        plain = s.Scene((replace(s._cube(2, (.7, .5, .3, 1), s.Transform3D()), specular=.3), board()))
        explicit = s.Scene((replace(plain.geometries[0], material="standard", ior=1.5, reflection=0.2), board()))
        np.testing.assert_array_equal(render(plain), render(explicit))
        old = {"version": SCHEMA_VERSION, "settings": {}, "nodes": {"c": {"type": "Cube3D", "name": "c", "inputs": {},
                                                             "params": {"cube_size": 2.0}, "pos": [0, 0]}}}
        upgraded = upgrade_document(old)
        params = upgraded["nodes"]["c"]["params"]
        self.assertEqual(params["material"], "standard")
        fields = s.material_fields({"cube_size": 2.0})
        self.assertEqual(fields["material"], "standard")

    def test_node_registration(self):
        for kind in ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D"):
            self.assertEqual(SPECS[kind]["params"]["material"], "standard")
            self.assertAlmostEqual(SPECS[kind]["params"]["ior"], 1.333)
        self.assertEqual(SPECS["FluidSurface3D"]["params"]["material"], "liquid")


class LiquidRasterTests(unittest.TestCase):
    """The raster mode's screen-space approximation: Fresnel reflection plus the picture behind, shifted by the normal."""

    def sphere(self, **fields):
        base = dict(material="liquid", reflection=1.0, absorption_color=(0.8, 0.9, 1.0))
        base.update(fields)
        return replace(s._sphere(1.0, 48, (1, 1, 1, 1), s.Transform3D()), **base)

    def test_pixel_is_the_shifted_board_tinted_plus_fresnel_reflection(self):
        size, bg = 96, np.array((0.3, 0.5, 0.9))
        camera = s.Camera(fov=45.0)
        moved = replace(board(), transform=s.Transform3D(s.Vec3(0.37, 0.37, BOARD_Z)))   # squares, not a corner, behind the sphere
        board_only = s.render(s.Scene((moved,)), camera, size, size, (*bg, 1), mode="raster")
        image = s.render(s.Scene((self.sphere(), moved)), camera, size, size, (*bg, 1), mode="raster")
        focal = 1 / math.tan(math.radians(45.0) / 2)
        checked = 0
        for px, py in ((48, 48), (52, 50), (45, 47), (50, 53), (47, 44), (55, 46)):
            d = np.array(((2 * (px + .5) / size - 1) / focal, (1 - 2 * (py + .5) / size) / focal, -1.0))
            d /= np.linalg.norm(d)
            eye = np.array((0, 0, 5.0))
            b = eye @ d
            disc = b * b - (eye @ eye - 1.0)
            if disc <= 0:
                continue
            point = eye + d * (-b - math.sqrt(disc))          # sphere at the origin, radius 1
            normal = point                                    # outward unit normal
            v = -d
            cos = float(normal @ v)
            f0 = ((1.333 - 1) / (1.333 + 1)) ** 2
            fresnel_weight = f0 + (1 - f0) * (1 - cos) ** 5
            tilt = np.array((normal[0], normal[1])) * (1.333 - 1) * 0.1 * size    # view basis is the world axes here
            sx = int(np.clip(round(px + tilt[0]), 0, size - 1))
            sy = int(np.clip(round(py - tilt[1]), 0, size - 1))
            tint = np.array((0.8, 0.9, 1.0)) ** (1 / max(cos, 0.25))
            # the interpolated normal differs a hair from the exact sphere's, which can move the sample by one pixel
            errors = [np.abs(image[py, px, :3] - ((1 - fresnel_weight) * tint * board_only[
                          int(np.clip(sy + dy, 0, size - 1)), int(np.clip(sx + dx, 0, size - 1)), :3]
                                                   + fresnel_weight * bg)).max()
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            self.assertLess(min(errors), 0.03, (px, py))
            checked += 1
        self.assertGreaterEqual(checked, 4)

    def test_background_shows_through_and_grazing_reflects_more(self):
        size = 96
        image = s.render(s.Scene((self.sphere(absorption_color=(1, 1, 1)), board())), s.Camera(), size, size,
                         (1, 1, 1, 1), mode="raster")
        self.assertTrue(np.all(image[48, 48, 3] == 1))
        # white surroundings: the pixel brightens toward the rim where the reflection takes over
        centre = float(image[48, 48, :3].mean())
        rim = float(image[48, 48 + 20:48 + 21, :3].mean())
        self.assertNotEqual(round(centre, 3), round(rim, 3))

    def test_closed_liquid_draws_its_front_faces_only(self):
        from nodebased.liquid_render import is_closed
        self.assertTrue(is_closed(self.sphere()))
        self.assertFalse(is_closed(s._card(1, 1, (1, 1, 1, 1), s.Transform3D())))
        ordinary = s._cube(2, (1, 1, 1, 1), s.Transform3D())
        self.assertTrue(is_closed(ordinary))

    def test_raster_liquid_is_deterministic_and_data_passes_are_unchanged(self):
        scene = s.Scene((self.sphere(), board()))
        a = s.render(scene, s.Camera(), 64, 64, mode="raster")
        np.testing.assert_array_equal(a, s.render(scene, s.Camera(), 64, 64, mode="raster"))
        standard = s.Scene((replace(scene.geometries[0], material="standard"), board()))
        for output in ("depth", "normals", "object_id"):
            np.testing.assert_array_equal(s.render(scene, s.Camera(), 64, 64, output=output, mode="raster"),
                                          s.render(standard, s.Camera(), 64, 64, output=output, mode="raster"))

    def test_gpu_raster_hands_liquid_to_the_cpu(self):
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((self.sphere(), board())), s.Camera(), 32, 32, (0, 0, 0, 0), 0.0)


@unittest.skipUnless(gpu3d.available(), "wgpu adapter unavailable")
class LiquidGpuTests(unittest.TestCase):
    """The GPU ray tracer follows the same reflection and refraction tree as the CPU reference."""

    @classmethod
    def setUpClass(cls):
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter.limits["max-storage-buffers-per-shader-stage"] < 8:
            raise unittest.SkipTest("eight storage bindings unavailable")
        device = adapter.request_device_sync(required_limits={"max-storage-buffers-per-shader-stage": 8})
        cls.state = dict(wgpu=wgpu, device=device)

    def compare(self, scene, background=(0, 0, 0, 0), ambient=0.0, tolerance=2e-3):
        gpu = gpurt_render.render_beauty(self.state, scene, CAMERA, SIZE, SIZE, background, ambient)
        cpu = render(scene, background=background, ambient=ambient)
        error = float(np.abs(gpu - cpu).max())
        print(f"{self._testMethodName}: max error {error:.3g}", flush=True)
        self.assertLessEqual(error, tolerance)
        return gpu

    def test_refracted_board_matches_the_cpu(self):
        gpu = self.compare(s.Scene((liquid_cube(depth=DEPTH, ior=IOR), board())))
        self.assertGreater(float(gpu[SIZE // 2, SIZE // 2, 3]), 0.99)

    def test_fresnel_absorption_and_background_match_the_cpu(self):
        cube = liquid_cube(depth=DEPTH, ior=IOR, reflection=1.0, absorption_color=(0.5, 0.8, 0.95))
        self.compare(s.Scene((cube, board())), background=(0.3, 0.5, 0.9, 1))

    def test_lit_sphere_with_glints_matches_the_cpu(self):
        sphere = replace(s._sphere(1.0, 32, (1, 1, 1, 1), s.Transform3D()), material="liquid", reflection=1.0,
                         roughness=0.3)
        scene = s.Scene((sphere, board()), (s.Light(position=s.Vec3(2, 3, 4)),))
        self.compare(scene, background=(0.3, 0.5, 0.9, 1), ambient=0.1)

    def test_thin_sheet_matches_the_cpu_and_is_not_black(self):
        slab = liquid_cube(4.0, reflection=1.0, absorption_color=(0.5, 0.8, 0.95))
        slab = replace(slab, vertices=slab.vertices * np.array((1, 1, 0.004), np.float32))
        gpu = self.compare(s.Scene((slab, board())))
        self.assertTrue(np.all(gpu[28:36, 28:36, :3].max(axis=2) > 0.3))

    def test_through_the_dispatcher(self):
        scene = s.Scene((liquid_cube(depth=DEPTH, ior=IOR), board()))
        image = gpu3d.render(scene, CAMERA, SIZE, SIZE, (0, 0, 0, 0), 0.0, mode="raytrace")
        np.testing.assert_allclose(image, render(scene), atol=2e-3)


if __name__ == "__main__":
    unittest.main()
