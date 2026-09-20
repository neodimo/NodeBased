"""Analytic contracts for the shared per-centre splat relighting approximation."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nodebased import scene3d as s, splats
from nodebased.splatshade import (splat_albedo, normal_confidence, shade_splats,
                                  instance_colors, instance_geometry)
from nodebased.splatraster import prepare_splats
from nodebased.core import Dispatcher, load_document
from nodebased.imaging import Evaluator
from nodebased.knobs import knob_layout


def cloud(positions=((0, 0, 0),), scales=(.5, .5, .02), rotations=None):
    n = len(positions)
    sh = np.zeros((n, 1, 3))
    sh[:, 0] = (np.array((.6, .3, .04)) - .5) / splats.C0
    return splats.SplatCloud(positions, np.broadcast_to(scales, (n, 3)),
                            np.tile((1, 0, 0, 0), (n, 1)) if rotations is None else rotations,
                            np.full(n, .8), sh, 0)


def light(direction=(0, 0, 1), **kwargs):
    return s.Light(position=s.Vec3(*direction), **kwargs)


def shade(c, lights=(), ambient=.1, mix=1, eye=(0, 0, 5), visibility=None):
    return shade_splats(np.full((len(c), 3), .7), splat_albedo(c), c.positions,
                        c.normals(), normal_confidence(c.scales), eye, lights,
                        ambient, mix, visibility)


class RelightTests(unittest.TestCase):
    def test_zero_exact_and_default_render(self):
        baked = np.array([[np.nan, -0., .123]], dtype=np.float32)
        before = baked.tobytes()
        self.assertIs(shade_splats(baked, None, None, None, None, None, None, None, 0), baked)
        self.assertEqual(baked.tobytes(), before)
        scene = s.Scene(splats=(s.SplatInstance(cloud()),))
        baseline = s.render(scene, s.Camera(), 33, 33)
        for inst in (scene.splats[0], replace(scene.splats[0], relight=0)):
            actual = s.render(replace(scene, splats=(inst,), lights=(light(intensity=10),)),
                              s.Camera(), 33, 33, ambient=2)
            self.assertEqual(actual.tobytes(), baseline.tobytes())

    def test_dc_and_directional_centre_analytic(self):
        c = cloud()
        srgb = np.maximum(.5 + splats.C0*c.sh[:, 0].astype(float), 0)
        expected = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
        np.testing.assert_allclose(splat_albedo(c), expected, atol=1e-7)
        cases = [((), .2), ((light(),), 1.2), ((light((0, 0, -1)),), .2),
                 ((light((np.sqrt(3), 0, 1)),), .7),
                 ((light(intensity=2, color=(.2, .4, .8)),), np.array((.6, 1., 1.8))),
                 ((light(), light(intensity=2)), 3.2), ((light(intensity=0),), .2)]
        for lights, radiance in cases:
            with self.subTest(lights=lights):
                scene = s.Scene(lights=lights, splats=(s.SplatInstance(c, relight=1),))
                image = s.render(scene, s.Camera(), 33, 33, ambient=.2)
                np.testing.assert_allclose(image[16, 16, :3], expected[0]*radiance*.8, atol=1e-5)
                np.testing.assert_array_equal(image, s.render(scene, s.Camera(), 33, 33, ambient=.2, mode='raytrace'))
                np.testing.assert_allclose(s.render(scene, s.Camera(), 33, 33, ambient=.2, output='splats'), image, atol=1e-7)

    def test_albedo_ignores_view_dependent_sh_and_clamps_dc(self):
        c = cloud()
        sh = np.zeros((1, 4, 3), np.float32)
        sh[:, 0] = c.sh[:, 0]
        sh[:, 1:] = 4
        detailed = replace(c, sh=sh, sh_degree=1)
        np.testing.assert_array_equal(splat_albedo(detailed), splat_albedo(c))
        for eye in ((0, 0, 5), (2, 0, 5)):
            baked = splats.to_linear_color(splats.eval_sh(detailed, (0, 0, -1)))
            actual = shade_splats(baked, splat_albedo(detailed), detailed.positions,
                                  detailed.normals(), normal_confidence(detailed.scales),
                                  eye, (), .3, 1)
            np.testing.assert_allclose(actual, splat_albedo(c)*.3)
        dark = replace(c, sh=np.full((1, 1, 3), -10.))
        np.testing.assert_array_equal(splat_albedo(dark), np.zeros((1, 3)))

    def test_blob_rotation_and_interpolation(self):
        a = cloud(scales=(.1, .1, .1))
        b = replace(a, rotations=[[np.sqrt(.5), 0, np.sqrt(.5), 0]])
        lamp = light((1, 0, 1))
        np.testing.assert_array_equal(shade(a, (lamp,)), shade(b, (lamp,)))
        np.testing.assert_allclose(shade(a, (lamp,)), splat_albedo(a)*(.1+1/np.sqrt(2)), atol=1e-7)
        c = cloud(scales=(.5, .5, .25), rotations=b.rotations)
        confidence = normal_confidence(c.scales)
        np.testing.assert_allclose(confidence, [.5])
        v = np.array((0, 0, 1))
        n = c.normals()[0]
        if n @ v < 0:
            n = -n
        effective = .5*n+.5*v
        effective /= np.linalg.norm(effective)
        np.testing.assert_allclose(shade(c, (lamp,)), splat_albedo(c)*(.1+max(0, effective @ (np.array((1, 0, 1))/np.sqrt(2)))), atol=1e-7)
        np.testing.assert_array_equal(normal_confidence([[0, 0, 0], [.02, .5, .5]]), [0, .96])

    def test_point_mix_visibility_and_facing(self):
        c = cloud(((0, 0, 0), (1, 0, 0)))
        lamp = light((0, 0, 2), kind='Point')
        # Fully confident normals isolate the point direction from the view blend.
        baked = np.full((2, 3), .7)
        args = (baked, splat_albedo(c), c.positions, -c.normals(), np.ones(2), (0, 0, 5), (lamp,), .1)
        lit = shade_splats(*args, 1)
        np.testing.assert_allclose(lit, splat_albedo(c)*(np.array([1, 2/np.sqrt(5)])[:, None]+.1), atol=1e-7)
        np.testing.assert_allclose(shade_splats(*args, .5), (baked+lit)/2)
        np.testing.assert_allclose(shade_splats(*args, 1, visibility=np.zeros((2, 1))), splat_albedo(c)*.1)
        # Visibility indices include skipped lights.
        np.testing.assert_allclose(shade(c, (light(intensity=0), lamp), visibility=np.array([[1, 0], [1, 0]])), splat_albedo(c)*.1)

    def test_fibonacci_sphere(self):
        i = np.arange(200)
        z = 1-2*(i+.5)/200
        phi = i*np.pi*(3-np.sqrt(5))
        p = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
        # Quaternion taking +Z to the radial shortest axis.
        q = np.column_stack((1+z, -p[:, 1], p[:, 0], np.zeros(200)))
        q /= np.linalg.norm(q, axis=1)[:, None]
        c = cloud(p, (.2, .2, .02), q)
        v = (np.array((0, 0, 5))-p)
        v /= np.linalg.norm(v, axis=1)[:, None]
        facing = np.where((np.sum(p*v, axis=1) < 0)[:, None], -p, p)
        effective = .9*facing+.1*v
        effective /= np.linalg.norm(effective, axis=1)[:, None]
        rgb = shade(c, (light((1, 0, 0)),), ambient=0)
        np.testing.assert_allclose(rgb, splat_albedo(c)*np.maximum(effective[:, :1], 0), atol=2e-7)
        front = z > .3
        self.assertGreater(rgb[front & (p[:, 0] > 0)].mean(), rgb[front & (p[:, 0] < 0)].mean())
        images = [s.render(s.Scene(lights=(light((x, 0, 0)),), splats=(s.SplatInstance(c, relight=1),)), s.Camera(), 65, 65) for x in (1, -1)]
        self.assertGreater(images[0][:, 33:, :3].mean(), images[0][:, :32, :3].mean())
        self.assertLess(images[1][:, 33:, :3].mean(), images[1][:, :32, :3].mean())
        # Fibonacci samples are not exact mirror pairs; allow their sampling error.
        self.assertLess(np.abs(images[0][..., :3]-images[1][:, ::-1, :3]).mean(), .012)

    def test_layered_and_opaque_mesh_composites(self):
        c = cloud()
        lit = splat_albedo(c)[0]*1.2
        for alpha in (.5, 1.):
            card = s._card(30, 30, (0, 0, 0, alpha), s.Transform3D(s.Vec3(0, 0, 1)))
            scene = s.Scene((card,), (light(),), (s.SplatInstance(c, relight=1),))
            images = [s.render(scene, s.Camera(), 33, 33, ambient=.2, mode=mode)
                      for mode in ('raster', 'raytrace')]
            np.testing.assert_array_equal(*images)
            np.testing.assert_allclose(images[0][16, 16, :3], (1-alpha)*.8*lit, atol=1e-6)
            splat_pass = s.render(scene, s.Camera(), 33, 33, ambient=.2, output='splats')
            np.testing.assert_allclose(splat_pass[16, 16], [*((1-alpha)*.8*lit), (1-alpha)*.8], atol=1e-6)

    def test_world_transform_and_data_unchanged(self):
        matrix = s.Transform3D(rotation=s.Vec3(20, 35, 0), scale=s.Vec3(2, 1, .5)).matrix()
        inst = s.SplatInstance(cloud(), matrix, relight=1)
        world = inst.cloud.transformed(matrix)
        prepared = prepare_splats((inst,), s.Camera(), 33, 33, lighting=((light(),), .2))
        np.testing.assert_allclose(prepared.splats[4], shade(world, (light(),), ambient=.2), atol=1e-7)
        for output in ('depth', 'normals', 'position', 'object_id', 'uv'):
            scene = s.Scene(lights=(light(),), splats=(inst,))
            np.testing.assert_array_equal(s.render(scene, s.Camera(), 17, 17, output=output), s.render(replace(scene, splats=(replace(inst, relight=0),)), s.Camera(), 17, 17, output=output))

    def test_graph_upgrade_and_knob(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cloud.ply'
            splats.write_ply(cloud(), path)
            d = Dispatcher()
            for key, kind, params in [('read', 'ReadSplat3D', dict(splat_path=str(path), splat_relight=1.)), ('lamp', 'Light3D', {}), ('scene', 'Scene3D', {}), ('camera', 'Camera3D', dict(tz=5)), ('render', 'Render3D', dict(width=33, height=33, samples=1))]:
                d.execute(dict(op='create', id=key, type=kind, params=params))
            for key, slot, source in [('scene', 'object0', 'read'), ('scene', 'object1', 'lamp'), ('render', 'scene', 'scene'), ('render', 'camera', 'camera')]:
                d.execute(dict(op='connect', id=key, input=slot, source=source))
            e = Evaluator()
            scene = e.evaluate_raster(d.document, 'scene', typed=True)
            camera = e.evaluate_raster(d.document, 'camera', typed=True)
            self.assertEqual(scene.splats[0].relight, 1)
            np.testing.assert_array_equal(e.evaluate(d.document, 'render'), s.render(scene, camera, 33, 33, ambient=.1))
            d.execute(dict(op='set', id='read', param='splat_relight', value=0.))
            np.testing.assert_array_equal(e.evaluate(d.document, 'render'), s.render(replace(scene, lights=(), splats=(replace(scene.splats[0], relight=0),)), camera, 33, 33))
            old = copy.deepcopy(d.document)
            del old['nodes']['read']['params']['splat_relight']
            docpath = Path(tmp)/'old.nbcomp'
            docpath.write_text(json.dumps(old))
            self.assertEqual(load_document(docpath)['nodes']['read']['params']['splat_relight'], 0.)
        knob = next(g for g in knob_layout('ReadSplat3D') if 'splat_relight' in g.params)
        self.assertEqual((knob.kind, knob.label, knob.soft_range), ('float_slider', 'Relight', (0, 1)))


class InstanceHelpersTests(unittest.TestCase):
    def instance(self, **kwargs):
        c = cloud(((-.3, 0, 0), (.4, .2, -.5)))
        sh = np.random.default_rng(17).normal(0, .1, (2, 16, 3))
        sh[:, 0] = c.sh[:, 0]
        c = replace(c, sh=sh, sh_degree=3)
        matrix = s.Transform3D(rotation=s.Vec3(20, 35, 10),
                               scale=s.Vec3(1.2, .8, .7)).matrix()
        matrix[0, 1] += .15  # Include shear in the world transform.
        return s.SplatInstance(c, matrix, **kwargs)

    def test_baked_degree_colorspace_and_dtype(self):
        eye = np.array((.2, -.1, 5.))
        for colorspace in ('srgb', 'linear'):
            for degree in (None, -2, 0, 1, 2, 3, 9):
                inst = self.instance(sh_degree=degree)
                inst = replace(inst, cloud=replace(inst.cloud, colorspace=colorspace))
                world = inst.cloud.transformed(inst.matrix)
                dirs = world.positions.astype(np.float64) - eye
                dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-30)
                clamp = 3 if degree is None else max(0, min(degree, 3))
                expected = splats.to_linear_color(
                    splats.eval_sh(world.sh[:, :(clamp+1)**2], dirs), colorspace)
                actual = instance_colors(inst, eye)
                self.assertEqual(actual.shape, (2, 3))
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_relit_and_visibility(self):
        inst = self.instance(relight=.65, sh_degree=2)
        eye = (0, 0, 5)
        lights = (light(intensity=0), light(), light((1, 2, 3), kind='Point'))
        visibility = np.array([[1, 0, .25], [0, .5, 1]])
        world = inst.cloud.transformed(inst.matrix)
        baked = instance_colors(replace(inst, relight=0), eye)
        expected = shade_splats(baked, splat_albedo(world), world.positions,
                               world.normals(), normal_confidence(world.scales),
                               eye, lights, .2, inst.relight, visibility)
        actual = instance_colors(inst, eye, lights, .2, visibility)
        self.assertEqual(actual.tobytes(), expected.astype(np.float32).tobytes())
        self.assertFalse(np.array_equal(actual, instance_colors(inst, eye, lights, .2)))

    def test_geometry_and_empty(self):
        for opacity_scale in (0, .4, 2):
            inst = self.instance(opacity_scale=opacity_scale, scale_scale=1.7)
            world = inst.cloud.transformed(inst.matrix)
            geometry = instance_geometry(inst)
            for key, shape in [('positions', (2, 3)), ('rotations', (2, 4)),
                               ('scales', (2, 3)), ('opacity', (2,))]:
                self.assertEqual(geometry[key].shape, shape)
                self.assertEqual(geometry[key].dtype, np.float32)
            np.testing.assert_array_equal(geometry['positions'], world.positions)
            np.testing.assert_array_equal(geometry['rotations'], world.rotations)
            np.testing.assert_array_equal(geometry['scales'], world.scales*1.7)
            np.testing.assert_array_equal(geometry['opacity'], np.clip(world.opacity*opacity_scale, 0, 1))
        empty = splats.SplatCloud(np.empty((0, 3)), np.empty((0, 3)),
                                 np.empty((0, 4)), np.empty(0), np.empty((0, 1, 3)), 0)
        inst = s.SplatInstance(empty, relight=1)
        self.assertEqual(instance_colors(inst, (0, 0, 5)).shape, (0, 3))
        self.assertEqual(instance_geometry(inst)['scales'].shape, (0, 3))

    def test_prepare_colors_exact_and_render(self):
        from nodebased.splatraster import accumulate_splats
        camera = s.Camera()
        eye, _ = s._view_basis(camera)
        lights = (light(intensity=0), light())
        visibility = np.array([[1, .25], [0, .75]])
        for mix in (0, .65, 1):
            inst = self.instance(relight=mix)
            for source in (visibility, lambda index: visibility):
                prepared = prepare_splats((inst,), camera, 33, 33,
                                          lighting=(lights, .2, source))
                expected = instance_colors(inst, eye, lights, .2, visibility)
                self.assertEqual(prepared.splats[4].dtype, np.float32)
                self.assertEqual(prepared.splats[4].tobytes(), expected.tobytes())
                rgb, alpha = accumulate_splats(prepared)
                self.assertEqual(rgb.shape, (33, 33, 3))
                self.assertGreater(alpha.max(), 0)


if __name__ == '__main__':
    unittest.main()
