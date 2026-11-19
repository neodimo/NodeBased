"""Lane 2 step X4: Reconcile3D's 2D transform tab has the seven resampling knobs of Nuke 17.0's Reference Guide.

Reference: Nuke 17.0 Reference Guide, Transform nodes > Reconcile3D (learn.foundry.com/nuke/17.0/content/reference_guide/transform_nodes/reconcile3d.html),
tab "2D transform": track, offset, stabilize, filter (default Parzen), clamp (off), black outside (on), motionblur (0), shutter (0.5),
shutter offset (start), custom offset (0).

Reconcile3D outputs the transform as a track, not the moved picture, so these knobs change no pixel here: the evaluated track carries them as
`transform_settings`. Older documents render the same.
"""
import copy
import json
import os
import unittest

import numpy as np

from nodebased import core, knobs, track3dknobs
from nodebased.core import SPECS, upgrade_document, validate
from tests.test_2d_behaviour_transform_n import reconcile, track_rig

NAMES = ("filter", "clamp", "black_outside", "motionblur", "shutter", "shutter_offset", "custom_offset")
GUIDE_DEFAULTS = {"filter": "parzen", "clamp": 0, "black_outside": 1, "motionblur": 0, "shutter": 0.5, "shutter_offset": "start", "custom_offset": 0.0}


class Reconcile3DResampleKnobs(unittest.TestCase):
    def test_defaults_are_the_guides(self):
        for name, value in GUIDE_DEFAULTS.items():
            self.assertEqual(SPECS["Reconcile3D"]["params"][name], value, name)

    def test_choices_and_limits(self):
        self.assertIn("parzen", core.CHOICES["filter"])
        self.assertEqual(core.CHOICES["shutter_offset"], ["start", "centred", "end", "custom"])
        for name in ("clamp", "black_outside", "motionblur", "shutter", "custom_offset"):
            self.assertIn(name, core.LIMITS)

    def test_the_knobs_follow_the_guides_order_after_stabilize(self):
        order = [g.params[0] for g in knobs.KNOB_LAYOUT["Reconcile3D"] if g.params]
        names = [n for n in order if n in ("track_x", "offset_x", "stabilize", *NAMES)]
        self.assertEqual(names, ["track_x", "offset_x", "stabilize", "filter", "clamp", "black_outside", "motionblur", "shutter", "shutter_offset"])

    def test_the_track_carries_the_settings_and_the_transform_is_unchanged(self):
        base = reconcile(track_rig(reconcile_params={"track_x": 10.0})[0])
        self.assertEqual(base["transform_settings"], GUIDE_DEFAULTS)
        changed_params = {"filter": "mitchell", "clamp": 1, "black_outside": 0, "motionblur": 8, "shutter": 1.5, "shutter_offset": "custom", "custom_offset": -0.25}
        other = reconcile(track_rig(reconcile_params={"track_x": 10.0, **changed_params})[0])
        self.assertEqual(other["transform_settings"], changed_params)
        # No pixel and no position changes: the node's output is the same track.
        np.testing.assert_allclose(other["output"], base["output"], atol=0)
        np.testing.assert_allclose(other["transform_value"], base["transform_value"], atol=0)
        self.assertEqual(other["track"]["x"]["curve"], base["track"]["x"]["curve"])

    def test_the_mapping_pairs_them_and_says_they_change_no_pixel(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "parity", "nodebased_to_nuke.json")
        entry = json.load(open(path))["pairs"]["Reconcile3D"]
        pairs = {e["nuke"][0]: e for e in entry["knobs"] if e.get("nuke")}
        for nuke, mine in (("filter", "filter"), ("clamp", "clamp"), ("black_outside", "black_outside"), ("motionblur", "motionblur"), ("shutter", "shutter"),
                           ("shutteroffset", "shutter_offset"), ("shuttercustomoffset", "custom_offset")):
            self.assertEqual(pairs[nuke]["nodebased"], [mine])
            self.assertIn("change no pixel", entry["default_notes"][nuke])

    def test_an_older_document_loads_with_the_defaults_and_the_same_output(self):
        doc, _, _ = track_rig(reconcile_params={"track_x": 4.0, "offset_y": 2.0})
        before = reconcile(doc)
        for name in NAMES:
            doc["nodes"]["rec"]["params"].pop(name)
        migrated = upgrade_document(copy.deepcopy(doc))
        validate(migrated)
        for name, value in GUIDE_DEFAULTS.items():
            self.assertEqual(migrated["nodes"]["rec"]["params"][name], value, name)
        after = reconcile(migrated)
        np.testing.assert_allclose(after["transform_value"], before["transform_value"], atol=0)
        np.testing.assert_allclose(after["output"], before["output"], atol=0)

    def test_transform_settings_reads_defaults_for_a_params_dict_without_them(self):
        self.assertEqual(track3dknobs.transform_settings({}), GUIDE_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
