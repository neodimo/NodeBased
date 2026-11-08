import unittest
import numpy as np
from nodebased.ops2d_markerremoval import marker_removal
from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

class MarkerRemovalKernelTests(unittest.TestCase):
    def test_horizontal_gradient_mask_square(self):
        x=np.linspace(0,1,31,dtype=np.float32)
        image=np.repeat(x[None,:,None],9,axis=0).repeat(4,axis=2)
        mask=np.zeros((9,31,1),np.float32); mask[3:6,12:19]=1
        out=marker_removal(image,mask)
        np.testing.assert_allclose(out[4,12:19,0],x[12:19],atol=1e-4)
    def test_outside_mask_is_bit_identical_and_empty_is_passthrough(self):
        rng=np.random.default_rng(4); image=rng.random((10,12,4),dtype=np.float32)
        mask=np.zeros((10,12,1),np.float32); mask[4:6,4:7]=1
        out=marker_removal(image,mask)
        np.testing.assert_array_equal(out[~mask[...,0].astype(bool)],image[~mask[...,0].astype(bool)])
        np.testing.assert_array_equal(marker_removal(image,np.zeros_like(mask)),image)
    def test_control_channel_offset_moves_the_region(self):
        image=np.zeros((5,9,4),np.float32)
        image[...,0]=np.arange(9,dtype=np.float32)[None,:]
        image[...,3]=0
        image[2,3,3]=1
        params={"mode":"bilinear","linearsearch":20,"ctrlMask":"rgba.alpha","x_offset":1,"y_offset":0}
        moved=Evaluator._kernel("MarkerRemoval",params,[image])
        self.assertEqual(float(moved[2,2,0]),float(image[2,2,0]))
        self.assertAlmostEqual(float(moved[2,4,0]),4.0,places=5)
    def test_named_control_layer_is_supported(self):
        image=np.zeros((3,7,4),np.float32)
        image[...,0]=np.arange(7,dtype=np.float32)[None,:]
        mask=np.zeros((3,7,1),np.float32); mask[:,3]=1
        params={"mode":"bilinear","linearsearch":10,"ctrlMask":"paint.control","x_offset":0,"y_offset":0}
        out=Evaluator._kernel("MarkerRemoval",params,[image],layer_array=lambda name: mask if name=="paint.control" else None)
        np.testing.assert_allclose(out[:,3,0],3.0,atol=1e-5)
    def test_disabled_control_channel_is_passthrough(self):
        image=np.arange(28,dtype=np.float32).reshape(2,7,2).repeat(2,axis=2)
        params={"enabled":0,"mode":"bilinear","linearsearch":100,"ctrlMask":"rgba.alpha","x_offset":0,"y_offset":0}
        np.testing.assert_array_equal(Evaluator._kernel("MarkerRemoval",params,[image]),image)
    def test_search_range_limits_column_fill(self):
        image=np.repeat(np.arange(12,dtype=np.float32)[None,:,None],7,axis=0).repeat(4,axis=2)
        mask=np.zeros((7,12,1),np.float32); mask[1:6,4:8]=1
        short=marker_removal(image,mask,linearsearch=1)
        long=marker_removal(image,mask,linearsearch=10)
        self.assertFalse(np.array_equal(short[3,5],long[3,5]))
    def test_radial_beats_bilinear_on_masked_disc_in_radial_ramp(self):
        yy,xx=np.mgrid[-20:21,-20:21]; r=np.sqrt(xx*xx+yy*yy)
        image=np.repeat((r/20)[...,None],4,axis=2).astype(np.float32)
        mask=(r<=7).astype(np.float32)[...,None]
        radial=marker_removal(image,mask,"radial")
        bilinear=marker_removal(image,mask,"bilinear")
        err_rad=np.abs(radial[20,20,0]-image[20,20,0])
        err_bil=np.abs(bilinear[20,20,0]-image[20,20,0])
        self.assertLess(err_rad,err_bil)
    def test_evaluator_enable_and_tile_path(self):
        d=Dispatcher(empty_document())
        d.execute({"op":"create","id":"plate","type":"Constant","params":{"width":24,"height":20,"red":.3,"green":.2,"blue":.1,"alpha":1}})
        d.execute({"op":"create","id":"node","type":"MarkerRemoval","params":{}})
        d.execute({"op":"connect","id":"node","input":"image","source":"plate"})
        ev=Evaluator(); full=ev.evaluate_raster(d.document,"node").pixels
        exe=TileExecutor(evaluator=Evaluator(),tile_edge=8); doc=dict(d.document,view="node")
        tile=exe.canvas_region(doc,"node",frame=1,tier=1)
        self.assertTrue(exe.supports_tiled(doc,"node"))
        np.testing.assert_array_equal(exe.compose_region(doc,"node",tile,frame=1,tier=1).pixels,full)

if __name__=='__main__': unittest.main()
