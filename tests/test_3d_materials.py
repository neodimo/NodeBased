"""Analytic material probes and CPU/wgpu parity."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nodebased import gpu3d, scene3d as s
from nodebased.core import Dispatcher, SPECS, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout
from tests.test_3d_gpu_shadows import boundary, dilate
from tests import test_3d_shadows as shadow_reference


def unit(v):
    return v / max(np.linalg.norm(v), 1e-8)


def plane_point(camera, x, y, size=65):
    eye, view = s._view_basis(camera)
    f = 1 / np.tan(np.deg2rad(camera.fov / 2))
    ray = view.T @ np.array(((2*(x+.5)/size-1)/f, (1-2*(y+.5)/size)/f, -1))
    return eye - ray * eye[2] / ray[2]


class MaterialTests(unittest.TestCase):
    camera = s.Camera()
    card = s._card(8, 8, (.2, .4, .6, 1), s.Transform3D())
    light = s.Light(position=s.Vec3(0, 0, 5), intensity=2, color=(1, .8, .5))

    def render(self, card=None, light=None, camera=None, **kwargs):
        return s.render(s.Scene((card or self.card,), (light or self.light,)),
                        camera or self.camera, 65, 65, **kwargs)

    def analytic(self, card, light, camera, x, y):
        p = plane_point(camera, x, y)
        lp, direction = light.world()
        toward = unit(lp-p) if light.kind == 'Point' else -direction
        half = unit(toward + unit(camera.transform.position.array()-p))
        return (np.array(card.color[:3])*max(toward[2], 0) +
                card.specular*max(half[2], 0)**card.shininess*(toward[2] > 0)) * np.array(light.color)*light.intensity*card.color[3]

    def test_gpu_vertex_material_packing_without_device(self):
        card = replace(self.card, specular=.6, shininess=73, emission=2.5)
        _, _, vertices, _, _ = gpu3d._prepare(s.Scene((card,)), self.camera, 65, 65, None)
        packed = np.concatenate(vertices)
        self.assertEqual(packed.shape[1], 19)
        self.assertEqual(packed.strides[0], 76)
        np.testing.assert_array_equal(packed[:, 11:15], np.tile(card.color, (len(packed), 1)).astype('f4'))
        np.testing.assert_array_equal(packed[:, 16:19], np.tile((.6, 73, 2.5), (len(packed), 1)).astype('f4'))

    def test_backlight_and_ambient_add_no_specular(self):
        card = replace(self.card, specular=1)
        light = replace(self.light, position=s.Vec3(0, 0, -5))
        np.testing.assert_array_equal(self.render(card, light, ambient=.3),
                                      self.render(self.card, light, ambient=.3))
        disabled = replace(light, intensity=0)
        np.testing.assert_array_equal(self.render(card, disabled), self.render(self.card, disabled))

    def test_peak_and_shininess_falloff(self):
        for shininess in (8, 128):
            card = replace(self.card, specular=.7, shininess=shininess)
            image = self.render(card)
            np.testing.assert_allclose(image[32, 32, :3],
                (np.array(card.color[:3])+.7)*np.array(self.light.color)*2, atol=2e-6)
            np.testing.assert_allclose(image[30, 52, :3], self.analytic(card, self.light, self.camera, 52, 30), atol=2e-6)
        self.assertGreater(self.render(replace(self.card, specular=.7, shininess=8))[30, 52, 0],
                           self.render(replace(self.card, specular=.7, shininess=128))[30, 52, 0])

    def test_default_diffuse_and_data_outputs(self):
        image = self.render()
        expected = np.empty_like(image)
        expected[..., :3] = np.array(self.card.color[:3], 'f4') * (np.array(self.light.color, 'f4')*2)
        expected[..., 3] = 1
        np.testing.assert_array_equal(image, expected)
        material = replace(self.card, specular=.8, emission=2)
        for output in ('depth', 'normals'):
            np.testing.assert_array_equal(self.render(output=output), self.render(material, output=output))
        np.testing.assert_array_equal(self.render(shade=True), self.render(replace(self.card, specular=1), shade=True))

    def test_point_light_eye_and_alpha(self):
        light = replace(self.light, kind='Point', position=s.Vec3(2, 1, 4))
        card = replace(self.card, specular=.6, shininess=16, color=(.2, .4, .6, .5))
        values = []
        for camera in (self.camera, s.Camera(s.Transform3D(s.Vec3(2, 0, 5)))):
            value = self.render(card, light, camera)[31, 34]
            np.testing.assert_allclose(value[:3], self.analytic(card, light, camera, 34, 31), atol=2e-6)
            self.assertEqual(value[3], .5)
            values.append(value[0])
        self.assertGreater(abs(values[0]-values[1]), .01)

    def test_emission_unlit_textured_projected_and_alpha(self):
        texture = np.ones((8, 8, 4), 'f4')
        texture[:, :4, :3] = (.1, .3, .8)
        texture[:, 4:, :3] = (.7, .2, .1)
        texture *= .5
        for alpha in (1, .5):
            for mode in ('plain', 'texture', 'projection'):
                card = replace(self.card, color=(.2, .4, .6, alpha),
                    texture=texture if mode == 'texture' else None,
                    projection=s.Projection(self.camera, texture) if mode == 'projection' else None)
                base = s.render(s.Scene((card,)), self.camera, 65, 65)
                glow = s.render(s.Scene((replace(card, emission=2),)), self.camera, 65, 65)
                np.testing.assert_allclose(glow[..., :3], base[..., :3]*3, atol=2e-7)
                np.testing.assert_array_equal(glow[..., 3], base[..., 3])
                if mode == 'plain':
                    np.testing.assert_allclose(glow[30, 32, :3], np.array(card.color[:3])*alpha*3, atol=2e-7)
                else:
                    self.assertGreater(abs(glow[30, 15, 0]-glow[30, 50, 0]), .01)

    def test_shadow_suppresses_specular_not_emission(self):
        ref = shadow_reference.ShadowTests()
        ground = replace(ref.ground, specular=1, shininess=4, emission=.3)
        scene = s.Scene((ground, ref.blocker()), (ref.light(),))
        image = s.render(scene, ref.camera, 64, 48, ambient=.1)
        np.testing.assert_allclose(ref.pixel(image, (0, 0, 0)), (.4, .4, .4, 1), atol=1e-6)
        self.assertGreater(ref.pixel(image, (2, 0, 0))[0], 1.4)

    def graph(self, **material):
        d = Dispatcher()
        for key, kind, params in [('card', 'Card3D', material), ('light', 'Light3D', {}),
                ('camera', 'Camera3D', {}), ('scene', 'Scene3D', {}),
                ('render', 'Render3D', dict(width=32, height=32, samples=1))]:
            d.execute(dict(op='create', id=key, type=kind, params=params))
        for slot, source in [('object0', 'card'), ('object1', 'light')]:
            d.execute(dict(op='connect', id='scene', input=slot, source=source))
        for slot in ('scene', 'camera'):
            d.execute(dict(op='connect', id='render', input=slot, source=slot))
        return d

    def test_graph_materials_and_old_document(self):
        for params in ({}, dict(spec_amount=.7, spec_shininess=12., emission=.3)):
            d = self.graph(**params)
            scene = Evaluator().evaluate_raster(d.document, 'scene', typed=True)
            direct = replace(s._card(2, 2, (.8, .8, .8, 1), s.Transform3D()),
                specular=params.get('spec_amount', 0), shininess=params.get('spec_shininess', 32), emission=params.get('emission', 0))
            expected = s.render(replace(scene, geometries=(direct,)), self.camera, 32, 32, ambient=.1)
            np.testing.assert_array_equal(Evaluator().evaluate(d.document, 'render'), expected)
        d = self.graph()
        expected = Evaluator().evaluate(d.document, 'render')
        for kind in ('Cube3D', 'Sphere3D', 'ReadGeo3D'):
            d.execute(dict(op='create', id=kind, type=kind))
        for node in d.document['nodes'].values():
            for key in ('spec_amount', 'spec_shininess', 'emission'):
                node['params'].pop(key, None)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.json'
            path.write_text(json.dumps(d.document))
            loaded = load_document(path)
        np.testing.assert_array_equal(Evaluator().evaluate(loaded, 'render'), expected)
        for key in ('card', 'Cube3D', 'Sphere3D', 'ReadGeo3D'):
            for param, value in [('spec_amount', 0.), ('spec_shininess', 32.), ('emission', 0.)]:
                self.assertEqual(loaded['nodes'][key]['params'][param], value)
        old_card = s.geometry_from_node(d.document['nodes']['card'])
        self.assertEqual((old_card.specular, old_card.shininess, old_card.emission), (0, 32, 0))

    def test_knobs_and_all_geometry_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'mesh.obj'
            path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            for kind in ('Card3D', 'Cube3D', 'Sphere3D', 'ReadGeo3D'):
                params = dict(SPECS[kind]['params'], spec_amount=.4, spec_shininess=64., emission=2.)
                if kind == 'ReadGeo3D':
                    params['geo_path'] = str(path)
                g = s.geometry_from_node(dict(type=kind, params=params))
                self.assertEqual((g.specular, g.shininess, g.emission), (.4, 64, 2))
                groups = knob_layout(kind)
                surface = next(i for i, group in enumerate(groups) if group.kind == 'color')
                self.assertEqual([(g.kind, g.label) for g in groups[surface+1:surface+4]],
                    [('float_slider', 'Specular'), ('float_slider', 'Shininess'), ('float_slider', 'Emission')])


@unittest.skipUnless(gpu3d.available(), 'no wgpu adapter')
class GPUMaterials(unittest.TestCase):
    def compare(self, scene, peak_check=True):
        cpu = s.render(scene, s.Camera(), 64, 64, ambient=.1, samples=2)
        gpu = gpu3d.render(scene, s.Camera(), 64, 64, ambient=.1, samples=2)
        interior = (cpu[..., 3] > 0) & ~dilate(boundary(cpu), 2)
        self.assertGreater(interior.sum(), 10)
        self.assertLess(float(np.abs(cpu[interior]-gpu[interior]).mean()), 5e-3)
        if not peak_check:
            return
        baseline = s.render(replace(scene, geometries=tuple(replace(g, specular=0) for g in scene.geometries)),
                            s.Camera(), 64, 64, ambient=.1, samples=2)
        peak = np.unravel_index(np.argmax((cpu-baseline)[..., :3].sum(axis=2)), cpu.shape[:2])
        np.testing.assert_allclose(cpu[peak], gpu[peak], atol=1e-2, rtol=0)

    def test_sphere_material_shadows(self):
        sphere = replace(s._sphere(1, 32, (.2, .4, .6, 1), s.Transform3D()), specular=.8, shininess=24, emission=.2)
        ground = s._card(8, 8, (.3, .3, .3, 1), s.Transform3D(s.Vec3(0, -1, 0), s.Vec3(-90, 0, 0)))
        self.compare(s.Scene((sphere, ground), (s.Light(position=s.Vec3(2, 4, 5), shadows=True),)))

    def test_textured_emissive_card(self):
        # Smooth gradient: CPU (per-triangle mip) and GPU (trilinear) sample a hard texel edge
        # differently, which would make a single-pixel peak comparison test the sampler, not the material.
        ramp = np.linspace(.2, .8, 8, dtype='f4')
        texture = np.ones((8, 8, 4), 'f4')
        texture[..., 0], texture[..., 1], texture[..., 2] = ramp[None], ramp[None] * .5, ramp[None] * .25
        texture *= .5
        card = replace(s._card(3, 3, (.4, .7, .8, .5), s.Transform3D(), texture), emission=2, specular=.5)
        light = (s.Light(position=s.Vec3(0, 0, 5)),)
        # The CPU reference blends the two triangles of a transparent card twice on their shared diagonal
        # (no top-left fill rule), so the transparent card only gets the interior mean comparison; the
        # single-pixel specular peak is compared on an opaque variant.
        self.compare(s.Scene((card,), light), peak_check=False)
        opaque = np.array(texture); opaque[..., 3] = 1
        self.compare(s.Scene((replace(card, texture=opaque, color=(.4, .7, .8, 1)),), light))
        with self.assertRaises(gpu3d.Unsupported):
            gpu3d.render(s.Scene((replace(card, projection=s.Projection(s.Camera(), texture)),)), s.Camera(), 32, 32)
