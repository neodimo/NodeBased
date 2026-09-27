"""Lane 8 F2 filter pixel checks, layered depth behavior and tile/bypass parity."""
import unittest
import numpy as np
from nodebased.core import Dispatcher
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

class FilterF2Tests(unittest.TestCase):
    def graph(self, kind, params):
        d = Dispatcher()
        d.execute({"op":"create", "id":"src", "type":"Checker", "params":{"width":27,"height":23,"size":1}})
        d.execute({"op":"create", "id":"fx", "type":kind, "params":params})
        d.execute({"op":"connect", "id":"fx", "input":"image", "source":"src"})
        return d

    def both_paths(self, kind, params):
        d = self.graph(kind, params)
        doc = dict(d.document, view="fx")
        full = Evaluator().evaluate(doc)
        ex = TileExecutor(evaluator=Evaluator(), tile_edge=7)
        self.assertTrue(ex.supports_tiled(doc, "fx"))
        region = ex.canvas_region(doc, "fx", 1, 1)
        tiled = ex.compose_region(doc, "fx", region, 1, 1).pixels
        np.testing.assert_allclose(tiled, full, atol=1e-6)
        d.execute({"op":"disable", "id":"fx", "value":True})
        np.testing.assert_array_equal(Evaluator().evaluate(dict(d.document, view="fx")), Evaluator().evaluate(dict(d.document, view="src")))

    def test_bilateral_smooths_noise_and_keeps_hard_step(self):
        rng=np.random.default_rng(8); image=np.zeros((25,25,4),np.float32)
        image[:,:12,:3]=0.15; image[:,12:,:3]=0.85; image[...,3]=1
        image[:,:,:3]+=rng.normal(0,0.025,(25,25,3)).astype(np.float32)
        out=Evaluator._bilateral(image,{"spatial_size":3,"colour_sigma":0.12})
        self.assertLess(float(np.var(out[4:21,4:10,0])), float(np.var(image[4:21,4:10,0])))
        self.assertLess(abs(float(out[12,11,0]-out[12,12,0])-float(image[12,11,0]-image[12,12,0])),0.01)
        self.both_paths("Bilateral",{"spatial_size":3,"colour_sigma":0.1})

    def test_denoise_halves_flat_noise_variance_and_preserves_step(self):
        rng=np.random.default_rng(9); image=np.full((31,31,4),0.4,np.float32)
        image[:,:,:3]+=rng.normal(0,0.08,(31,31,3)).astype(np.float32)
        image[:,16:,0:3]+=0.35; image[...,3]=1
        out=Evaluator._bilateral(image,{"spatial_size":3,"colour_sigma":0.2})
        self.assertLess(np.var(out[4:12,4:12,0]),np.var(image[4:12,4:12,0])*0.5)
        actual=Evaluator._filtered_pixels("Denoise",{"denoise_strength":0.2},__import__("nodebased.raster",fromlist=["Raster"]).Raster.of(image),__import__("nodebased.tiers",fromlist=["Region"]).Region(0,0,31,31))
        self.assertLess(np.var(actual[4:12,4:12,0]),np.var(image[4:12,4:12,0])*0.5)
        self.assertGreater(float(out[15,16,0]-out[15,15,0]),0.2)
        self.both_paths("Denoise",{"denoise_strength":0.2})

    def test_degrain_amounts_are_per_channel(self):
        image=np.zeros((11,11,4),np.float32); image[5,5,:3]=1; image[...,3]=1
        out=Evaluator._degrain_simple(image,{"red_amount":0,"green_amount":2,"blue_amount":1})
        self.assertEqual(float(out[5,5,0]),1.0); self.assertLess(out[5,5,1],1.0); self.assertLess(out[5,5,2],1.0)
        self.both_paths("DegrainSimple",{"red_amount":0.5,"green_amount":1,"blue_amount":2})

    def test_zdefocus_focal_plane_and_far_radius(self):
        image=np.zeros((21,21,4),np.float32); image[10,7]=1
        far=np.ones((21,21),np.float32)*3
        out=Evaluator._zdefocus(image,far,{"focal_plane":1,"depth_of_field":1,"max_size":3,"depth_math":"depth"})
        self.assertAlmostEqual(float(out[10,7,0]),1/29,places=5)
        self.assertGreater(float(out[10,4,0]),0)
        self.assertEqual(float(out[10,3,0]),0)
        focused=Evaluator._zdefocus(image,np.ones((21,21),np.float32),{"focal_plane":1,"depth_of_field":1,"max_size":3})
        np.testing.assert_array_equal(focused,image)
        np.testing.assert_array_equal(Evaluator._zdefocus(image,far,{"focal_plane":1,"depth_of_field":1,"max_size":0}),image)

    def test_temporal_denoise_averages_neighbor_frames(self):
        d=Dispatcher()
        d.execute({"op":"create","id":"src","type":"Checker","params":{"width":19,"height":17,"size":1}})
        d.execute({"op":"set_key","id":"src","param":"size","frame":1,"value":1})
        d.execute({"op":"set_key","id":"src","param":"size","frame":2,"value":2})
        d.execute({"op":"set_key","id":"src","param":"size","frame":3,"value":4})
        for key,temporal in (("spatial",0),("temporal",1)):
            d.execute({"op":"create","id":key,"type":"Denoise","params":{"denoise_strength":0.2,"temporal":temporal}})
            d.execute({"op":"connect","id":key,"input":"image","source":"src"})
        evaluator=Evaluator(); doc=dict(d.document,view="temporal")
        temporal_pixels=evaluator.evaluate(doc,frame=2)
        spatial_pixels=evaluator.evaluate(dict(d.document,view="spatial"),frame=2)
        self.assertGreater(float(np.max(np.abs(temporal_pixels-spatial_pixels))),1e-4)

    def test_zdefocus_wired_depth_and_kernel_graph(self):
        d=Dispatcher()
        for key,kind,params in (("src","Constant",{"width":13,"height":13,"red":1,"alpha":1}),
                                ("depth","Constant",{"width":13,"height":13,"red":2,"alpha":1}),
                                ("kernel","Constant",{"width":3,"height":3,"red":1,"alpha":1}),
                                ("fx","ZDefocus",{"focal_plane":1,"depth_of_field":1,"max_size":2,"bokeh_shape":"image"})):
            d.execute({"op":"create","id":key,"type":kind,"params":params})
        d.execute({"op":"connect","id":"fx","input":"image","source":"src"})
        d.execute({"op":"connect","id":"fx","input":"depth","source":"depth"})
        d.execute({"op":"connect","id":"fx","input":"kernel","source":"kernel"})
        out=Evaluator().evaluate(dict(d.document,view="fx"))
        self.assertEqual(float(out[6,6,0]),1.0)

    def test_temporal_denoise_uses_full_frame_fallback(self):
        d=self.graph("Denoise",{"denoise_strength":0.2,"temporal":1})
        doc=dict(d.document,view="fx")
        executor=TileExecutor(evaluator=Evaluator())
        self.assertFalse(executor.supports_tiled(doc,"fx"))
        result=Evaluator().evaluate(doc,frame=1)
        self.assertEqual(result.shape,(23,27,4))

    def test_zdefocus_disc_and_image_kernel(self):
        image=np.zeros((13,13,4),np.float32); image[6,6]=1
        depth=np.ones((13,13),np.float32)*2
        p={"focal_plane":1,"depth_of_field":1,"max_size":2,"depth_math":"depth","bokeh_shape":"disc"}
        disc=Evaluator._zdefocus(image,depth,p)
        kernel=np.ones((3,3),np.float32)
        image_bokeh=Evaluator._zdefocus(image,depth,{**p,"bokeh_shape":"image"},kernel)
        self.assertAlmostEqual(float(disc[6,6,0]),1/13,places=5)
        self.assertAlmostEqual(float(image_bokeh[6,6,0]),1/25,places=5)

    def test_blade_bokeh_has_configured_corner_directions(self):
        image=np.zeros((31,31,4),np.float32); image[15,15]=1
        depth=np.ones((31,31),np.float32)*4
        out=Evaluator._zdefocus(image,depth,{"focal_plane":1,"depth_of_field":1,"max_size":6,"bokeh_shape":"blades","blade_count":5})[...,0]
        # The polygonal disc has five radial tip directions; each is visible in a small angular cone.
        ys,xs=np.nonzero(out); angles=np.mod(np.arctan2(ys-15,xs-15),2*np.pi)
        tips=np.mod(np.arange(5)*2*np.pi/5,2*np.pi)
        for tip in tips:
            self.assertTrue(np.any(np.abs(np.angle(np.exp(1j*(angles-tip))))<0.18))

if __name__ == "__main__": unittest.main()
