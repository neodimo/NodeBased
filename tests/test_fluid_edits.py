"""The artist-facing classification for every fluid-node knob."""
import unittest
import hashlib

from nodebased.core import SPECS
from nodebased.fluid_edits import FLUID_KINDS, edit_scope, knob_class, status_text


class FluidEditClassificationTests(unittest.TestCase):
    # Intentional schema snapshot: when any fluid-node knob is added, update the
    # classification policy in fluid_edits.py and this gate together.
    PARAM_SIGNATURES = {
        "ReadVDB3D": "319861cf6426294a0e41c89571401b34cc537a20b8f05faa9f9af9d94c15d894",
        "FluidSource3D": "ac9ecf3b4382a92bd2f2a804763f0e4f2c4295a1d7c1f72eca8e39e6ecf4c4f0",
        "FluidForce3D": "5cdda340c49d9ab74b6b99a3ce482533aa69806d9bc05f429a0531d53958d335",
        "FluidCollide3D": "44ae8f26b0979a8a9df7f86b1c0a786f314974ade37cc3ccd9cfa7c4ade23d47",
        "FluidSolver3D": "e18166f865d76f841fa8ccb2c918ce3ba628d86a52446f59a01a75099491575a",
        "FluidLiquidSolver3D": "a88dce83744225e1422632b96c062ac48ae96dd58d7557726afc8e5e439872d0",
        "FluidSurface3D": "884498fa7bc40634be1485bc736f7fede9b69a70911df974203393cbe9c46428",
        "FluidFoam3D": "e880f124076259386d054695634a1f3f617ca3fd0732de2fc6ee279c5fe692c5",
        "FluidWhitewater3D": "0a7bcc97a0e496b5b288efee90d24d99a2fb708fbf0bc735c98cb62702843dab",
        "FluidCache3D": "2d7f028be81f6e2233d23c5a7991643b348a258076561dafc34dcf836c0036c1",
        "FluidUpres3D": "8f9d9d872b3d7fbc6ca386c95803f421a010ad2bbc7bda5ac4278842be807757",
        "WriteVDB3D": "68d114011b3c3e1050b598f6d9a20ee5950efa58d7aea3b03918550dfa34e804",
    }

    def test_every_fluid_node_parameter_has_a_class(self):
        self.assertEqual(FLUID_KINDS, set(self.PARAM_SIGNATURES))
        for kind in FLUID_KINDS:
            signature = hashlib.sha256("\n".join(sorted(SPECS[kind]["params"])).encode()).hexdigest()
            self.assertEqual(signature, self.PARAM_SIGNATURES[kind],
                             f"fluid knob list changed for {kind}; classify additions explicitly")
            for knob in SPECS[kind]["params"]:
                self.assertIn(knob_class(kind, knob), {"look", "from-frame", "from-start"},
                              f"unclassified {kind}.{knob}")

    def test_post_solve_surface_edits_are_look_changes(self):
        self.assertEqual(knob_class("FluidSurface3D", "smoothing"), "look")
        self.assertEqual(knob_class("FluidSurface3D", "material"), "look")
        doc = {"nodes": {"surface": {"type": "FluidSurface3D", "params": {}}}}
        command = {"op": "set", "id": "surface", "param": "smoothing"}
        self.assertEqual(edit_scope(doc, command), ("look", None))
        self.assertEqual(status_text(doc, command), "LOOK UPDATE · simulation cache reused")

    def test_keyed_source_change_reports_the_prior_checkpoint(self):
        doc = {"nodes": {"source": {"type": "FluidSource3D", "params": {}}}}
        command = {"op": "set_key", "id": "source", "param": "src_center_x", "frame": 60}
        self.assertEqual(edit_scope(doc, command), ("from-frame", 60))
        self.assertEqual(status_text(doc, command), "SIM UPDATE · keyed at frame 60 · checkpoint before 60")

    def test_static_solver_change_reports_restart_from_start(self):
        doc = {"nodes": {"solver": {"type": "FluidSolver3D", "params": {}}}}
        command = {"op": "set", "id": "solver", "param": "vorticity"}
        self.assertEqual(edit_scope(doc, command), ("from-start", None))
        self.assertEqual(status_text(doc, command), "SIM UPDATE · re-bake from start")

    def test_non_fluid_edit_has_no_fluid_status(self):
        doc = {"nodes": {"light": {"type": "Light3D", "params": {}}}}
        self.assertEqual(status_text(doc, {"op": "set", "id": "light", "param": "intensity"}), "")


if __name__ == "__main__":
    unittest.main()
