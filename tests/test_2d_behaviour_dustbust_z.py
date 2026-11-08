import unittest
import numpy as np
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication, QDoubleSpinBox
from nodebased.app import Window, STYLE
from nodebased.core import Dispatcher, empty_document
from nodebased.imaging import Evaluator
from nodebased.tileexec import TileExecutor

APP=QApplication.instance() or QApplication([]); APP.setStyle("Fusion"); APP.setStyleSheet(STYLE)


class DustBustTests(unittest.TestCase):
    def make_doc(self):
        d=Dispatcher(empty_document()); d.execute({"op":"time","first":1,"last":2,"current":2})
        d.execute({"op":"create","id":"a","type":"Constant","params":{"width":24,"height":24,"red":0,"green":0,"blue":0,"alpha":1}})
        d.execute({"op":"set_curve","id":"a","param":"red","curve":{"interpolation":"linear","keys":[{"frame":1,"value":0.2},{"frame":2,"value":0.8}]}})
        d.execute({"op":"create","id":"b","type":"DustBust","params":{"frame_offset":-1,"edge_hardness":1}})
        d.execute({"op":"set_dustbust_spots","id":"b","spots":[{"frame":2,"x":12,"y":12,"radius":4}]})
        d.execute({"op":"connect","id":"b","input":"image","source":"a"})
        return d
    def test_temporal_replacement_hardness_view_source_and_frame_gate(self):
        d=self.make_doc(); ev=Evaluator()
        source=ev.evaluate_raster(d.document,"a",frame=1).pixels
        out=ev.evaluate_raster(d.document,"b",frame=2).pixels
        np.testing.assert_array_equal(out[12,12],source[12,12])
        np.testing.assert_array_equal(ev.evaluate_raster(d.document,"b",frame=1).pixels,ev.evaluate_raster(d.document,"a",frame=1).pixels)
        d.execute({"op":"set","id":"b","param":"view_source","value":1})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(d.document,"b",frame=2).pixels,source)
    def test_disabled_node_bypasses_the_image(self):
        d=self.make_doc(); source=Evaluator().evaluate_raster(d.document,"a",frame=2).pixels
        d.execute({"op":"disable","id":"b","value":True})
        np.testing.assert_array_equal(Evaluator().evaluate_raster(d.document,"b",frame=2).pixels,source)
    def test_frame_offset_invalidates_and_tile_equals_full(self):
        d=self.make_doc(); exe=TileExecutor(evaluator=Evaluator(),tile_edge=8)
        doc=dict(d.document,view="b"); full=Evaluator().evaluate_raster(doc,"b",frame=2).pixels
        tile=exe.canvas_region(doc,"b",frame=2,tier=1)
        np.testing.assert_array_equal(exe.compose_region(doc,"b",tile,frame=2,tier=1).pixels,full)
        before=Evaluator().evaluate_raster(d.document,"b",frame=2,return_digest=True)[1]
        d.execute({"op":"set","id":"b","param":"frame_offset","value":1})
        after=Evaluator().evaluate_raster(d.document,"b",frame=2,return_digest=True)[1]
        self.assertNotEqual(before,after)
        pixels_after=Evaluator().evaluate_raster(d.document,"b",frame=2).pixels
        self.assertNotEqual(float(full[12,12,0]),float(pixels_after[12,12,0]))
    def test_edge_hardness_and_output_mask_channel(self):
        d=self.make_doc()
        d.execute({"op":"set","id":"b","param":"edge_hardness","value":0.0})
        soft=Evaluator().evaluate_raster(d.document,"b",frame=2).pixels
        d.execute({"op":"set","id":"b","param":"edge_hardness","value":1.0})
        hard=Evaluator().evaluate_raster(d.document,"b",frame=2).pixels
        self.assertFalse(np.array_equal(soft[12,15],hard[12,15]))
        d.execute({"op":"set","id":"b","param":"maskchannel","value":"dust.coverage"})
        result=Evaluator().evaluate_raster(d.document,"b",frame=2)
        self.assertIn("dust.coverage",result.layers)
        self.assertGreater(float(result.layers["dust.coverage"].pixels[12,12,0]),0.99)
    def test_viewer_click_adds_one_spot_and_undoes_as_one_edit(self):
        d=Dispatcher(empty_document()); d.execute({"op":"create","id":"plate","type":"Constant","params":{"width":24,"height":24}})
        d.execute({"op":"create","id":"dust","type":"DustBust"}); d.execute({"op":"connect","id":"dust","input":"image","source":"plate"}); d.execute({"op":"view","id":"dust"})
        window=Window(d.document); window.show(); window.graph.items_by_id["dust"].setSelected(True); APP.processEvents()
        try:
            panel=window.findChild(QDoubleSpinBox,"dustbust-brush-size"); self.assertIsNotNone(panel)
            panel.setValue(7.0)
            self.assertTrue(window.viewer.hit_release(QPointF(12,12),Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier))
            self.assertEqual(len(window.dispatcher.document["node_data"]["dust"]["spots"]),1)
            self.assertEqual(window.dispatcher.document["node_data"]["dust"]["spots"][0]["radius"],7)
            window.dispatcher.execute({"op":"undo"})
            self.assertNotIn("dust",window.dispatcher.document["node_data"])
        finally:
            window.saved_document=window.dispatcher.document; window.close(); APP.processEvents()


if __name__ == "__main__": unittest.main()
