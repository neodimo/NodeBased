"""Relight parity and premultiplied channel contracts; no GPU device needed."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
import numpy as np
from nodebased import scene3d as s
from nodebased.core import Dispatcher, CHOICES
from nodebased.imaging import Evaluator
from nodebased.raster import Raster

BASE = ('albedo', 'normals', 'position', 'diffuse', 'specular', 'emission')

def cases():
    tex = np.ones((8,8,4),np.float32)
    tex[:,:,:3] = np.linspace(.1,.8,8)[None,:,None]
    tex[::2,:, :] *= .5
    for i in range(20):
        t = s.Transform3D(s.Vec3(.1*(i%3),-.1,0),s.Vec3(i*3,i*7,4))
        color=(.3,.6,.8,.45 if i%4==0 else 1.)
        if i%3==0: g=s._card(3,3,color,t)
        elif i%3==1: g=s._cube(1.7,color,t)
        else: g=s._sphere(1,8,color,t)
        g=replace(g,specular=.4,shininess=12,emission=.2)
        if i%2: g=replace(g,texture=tex)
        if i%5==0: g=replace(g,projection=s.Projection(s.Camera(fov=30),tex))
        if i%5==1: g=replace(g,parent=s.Transform3D(s.Vec3(.2,.1,.3),s.Vec3(10,20,0)).matrix() @ s.Transform3D(rotation=s.Vec3(0,0,15)).matrix())
        camera=s.Camera(near=4.7 if i%5==2 else .1)
        ground=s._card(7,7,(.7,.4,.2,1),s.Transform3D(s.Vec3(0,-1.2,0),s.Vec3(-90,0,0)))
        lights=() if i%4==0 else (s.Light(shadows=i%2==0),s.Light('Point',(.2,.6,.9),.7,s.Vec3(-2,3,4)))
        yield s.Scene((g,ground),lights),camera

class RelightTests(unittest.TestCase):
    def test_single_output_parity(self):
        # The standalone paths are unchanged and serve as independent references.
        for number, (scene, camera) in enumerate(cases()):
            _, layers = s.render(scene, camera, 24, 20, output='relight', ambient=.13)
            for name in BASE:
                with self.subTest(scene=number, channel=name):
                    expected = s.render(scene, camera, 24, 20, output=name, ambient=.13)
                    textured = any(g.texture is not None or g.projection is not None for g in scene.geometries)
                    # Texture sampling has float64 intermediates in the reference;
                    # the bundle's mandated float32 fragment arrays round earlier.
                    if name in ('diffuse', 'specular') or (textured and name in ('albedo', 'emission')):
                        np.testing.assert_allclose(layers[name], expected, atol=1e-5, rtol=0)
                    else:
                        np.testing.assert_array_equal(layers[name], expected)
                    self.assertEqual(layers[name].dtype, np.float32)
                    self.assertTrue(layers[name].flags.c_contiguous)

    def test_per_light_and_beauty_identities(self):
        # A single translucent surface permits unpremultiplication of its unitless
        # responses. Layered surfaces with different albedos cannot be factored this way.
        card = replace(s._card(4, 4, (.4,.6,.8,.5), s.Transform3D()), specular=.6, emission=.2)
        lights = (s.Light(shadows=True), s.Light('Point', (.3,.5,.9), .7, s.Vec3(-2,1,4)),
                  s.Light(intensity=0))
        scene = s.Scene((card,), lights)
        rgba, layers = s.render(scene, s.Camera(), 32, 24, output='relight', ambient=.13,
                                background=(1,1,1,1))
        albedo = layers['albedo'][..., :3]
        alpha = layers['albedo'][..., 3:4]
        straight = np.divide(albedo, alpha, out=np.zeros_like(albedo), where=alpha > 0)
        diffuse = .13 * albedo
        specular = np.zeros_like(albedo)
        for i, light in enumerate(lights[:2]):
            colour = np.array(light.color)*light.intensity
            diffuse += straight * layers[f'diffuse_L{i}'][..., :3] * colour
            specular += layers[f'specular_L{i}'][..., :3] * colour
        np.testing.assert_allclose(diffuse, layers['diffuse'][..., :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(specular, layers['specular'][..., :3], atol=1e-5, rtol=0)
        self.assertNotIn('diffuse_L2', layers)
        # RGB adds, alpha is shared. Direct rgba with opaque background also has
        # a background term, so it is deliberately not asserted equal here.
        np.testing.assert_allclose(rgba[..., :3], sum(layers[k][..., :3] for k in
                                   ('diffuse','specular','emission')), atol=1e-5, rtol=0)
        for k in ('albedo','diffuse','specular','emission','diffuse_L0','specular_L1'):
            np.testing.assert_array_equal(rgba[..., 3], layers[k][..., 3])

    def test_limits_and_samples(self):
        scene, camera = next(cases())
        for kwargs, message in ((dict(mode='raytrace'), 'raster-only'),
                                (dict(return_depth=True), 'return_depth=True')):
            with self.assertRaisesRegex(ValueError, message):
                s.render(scene, camera, 8, 8, output='relight', **kwargs)
        with self.assertRaisesRegex(ValueError, 'does not support scenes with splats'):
            s.render(replace(scene, splats=(object(),)), camera, 8, 8, output='relight')
        a, la = s.render(scene, camera, 24, 20, output='relight', samples=1)
        b, lb = s.render(scene, camera, 24, 20, output='relight', samples=4)
        np.testing.assert_array_equal(a, b)
        for name in la:
            np.testing.assert_array_equal(la[name], lb[name])
        # Actual existing unlit semantics: ambient is ignored, diffuse == albedo.
        np.testing.assert_array_equal(la['diffuse'], la['albedo'])
        self.assertEqual(set(la), set(BASE))
        empty, layers = s.render(s.Scene(), camera, 8, 8, output='relight')
        np.testing.assert_array_equal(empty, 0)
        self.assertEqual(set(layers), set(BASE))

    def test_shadow_visibility_folded_into_response(self):
        ground = s._card(12,12,(1,1,1,1),s.Transform3D(rotation=s.Vec3(-90,0,0)))
        blocker = s._card(1,1,(1,1,1,1),s.Transform3D(s.Vec3(0,1,0),s.Vec3(-90,0,0)))
        scene = s.Scene((ground,blocker),(s.Light(position=s.Vec3(0,4,0),shadows=True),))
        camera = s.Camera(s.Transform3D(s.Vec3(4,5,7)))
        _, on = s.render(scene,camera,64,48,output='relight')
        _, off = s.render(scene,camera,64,48,output='relight',shadows=False)
        self.assertGreater(float(np.max(off['diffuse_L0'][...,:3]-on['diffuse_L0'][...,:3])), .5)

    def test_graph_cache_backends_and_legacy(self):
        d, evaluator = Dispatcher(), Evaluator()
        for key, kind, params in (('card','Card3D',{}),('camera','Camera3D',{}),
                ('scene','Scene3D',{}),('render','Render3D',dict(width=24,height=20,samples=4,
                                                           render_output='relight',render_backend='auto'))):
            d.execute(dict(op='create',id=key,type=kind,params=params))
        d.execute(dict(op='connect',id='scene',input='object0',source='card'))
        for slot in ('scene','camera'):
            d.execute(dict(op='connect',id='render',input=slot,source=slot))
        with patch('nodebased.gpu3d.available', side_effect=AssertionError('GPU attempted')):
            value = evaluator.evaluate_raster(d.document,'render')
            self.assertIs(evaluator.evaluate_raster(d.document,'render'),value)
        self.assertIsInstance(value,Raster)
        self.assertEqual(set(value.layers),set(BASE))
        scene = evaluator.evaluate_raster(d.document,'scene',typed=True)
        np.testing.assert_array_equal(value.layers['albedo'].pixels,
                                     s.render(scene,s.Camera(),24,20,output='albedo'))
        self.assertIn('relight',CHOICES['render_output'])
        d.execute(dict(op='set',id='render',param='render_backend',value='gpu'))
        with self.assertRaisesRegex(ValueError,'GPU Render3D unsupported: the relight bundle output is CPU-only'):
            evaluator.evaluate_raster(d.document,'render')
        old = json.loads(json.dumps(d.document))
        del old['nodes']['render']['params']['render_output']
        old['nodes']['render']['params']['render_backend']='cpu'
        # Evaluator supplies defaults for absent knobs. The existing loader rejects
        # absent render_output; changing that migration is outside this lane.
        result=Evaluator().evaluate_raster(old,'render')
        self.assertIsNone(result.layers)
        np.testing.assert_array_equal(result.pixels,
                                     s.render(scene,s.Camera(),24,20,samples=4,ambient=.1))

    def test_raster_layers_are_optional_and_not_propagated(self):
        pixels=np.zeros((2,3,4),np.float32)
        layers={'albedo':Raster.of(pixels)}
        raster=Raster(pixels,layers=layers)
        self.assertIs(raster.layers,layers)
        for plain in (Raster.of(pixels),raster.with_pixels(pixels),raster.aligned(raster.data)):
            self.assertIsNone(plain.layers)

if __name__ == '__main__':
    unittest.main()
