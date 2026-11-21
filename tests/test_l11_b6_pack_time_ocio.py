"""Lane 11 step B6 (finish): the cases that needed a fractional frame, a moving plate, the fixture OCIO config or a named white render in Nuke.

These run where Nuke is not installed: they read the scripts' text and the plate frames NodeBased writes. What the scripts do in Nuke was
read in Nuke 17.0v6 on 11/21: a TimeWarp whose filter is none and whose lookup is 0.5 asks its input for frame 0.5 (a frame-stamped plate
reads 0.5, and NoTimeBlur under it rounds 2.4 the way its menu says); the root's colorManagement OCIO with OCIO_config custom loads the
fixture config, whose colourspaces then fill the OCIO nodes' menus.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_nuke_reference_pack as pack  # noqa: E402
import nuke_fidelity_cases as cases  # noqa: E402
from nuke17_knob_table import load_table  # noqa: E402


def _case(case_id):
    return next(c for c in cases.CASES if c.id == case_id)


class FractionalFrameTests(unittest.TestCase):
    def setUp(self):
        self.table = load_table()

    def test_a_fractional_frame_renders_frame_one_through_a_timewarp_at_that_frame(self):
        case = _case("notimeblur_rint_at_2.4")
        self.assertEqual(pack.unrenderable(case, self.table), "")
        script = pack.py_script(case, self.table)
        self.assertIn("warp['lookup'].setValue(2.4)", script)
        self.assertIn("warp['filter'].setValue('none')", script)
        self.assertIn("warp.setInput(0, node)", script)
        self.assertIn("crop.setInput(0, warp)", script)
        self.assertIn("nuke.execute(write, 1, 1)", script)

    def test_a_whole_frame_has_no_timewarp(self):
        script = pack.py_script(_case("timewarp_lookup_key_at_5"), self.table)
        self.assertNotIn("warp = nuke.nodes.TimeWarp()", script)
        self.assertIn("crop.setInput(0, node)", script)


class OcioConfigTests(unittest.TestCase):
    def setUp(self):
        self.table = load_table()

    def test_an_ocio_case_loads_its_own_config_and_names_its_colourspaces(self):
        case = _case("ocio_display_mon_bright")
        self.assertEqual(pack.unrenderable(case, self.table), "")
        script = pack.py_script(case, self.table)
        self.assertIn("nuke.root()['OCIO_config'].setValue('custom')", script)
        self.assertIn("os.path.join(fixtures, 'fixture-config.ocio')", script)
        self.assertIn("node['view'].setValue('Bright')", script)
        self.assertIn("node['display'].setValue('Mon')", script)

    def test_the_config_brings_the_files_beside_it(self):
        files = pack.case_files(_case("ocio_look_on"))
        self.assertIn("fixture-config.ocio", files)
        self.assertIn("bent.cube", files)

    def test_a_menu_knob_without_a_config_keeps_nukes_menus(self):
        case = _case("colorspace_illuminant_keeps_xyz")
        self.assertIsNone(pack.ocio_config(case))
        knob, value = case.nuke_knobs[0]
        self.assertEqual(pack.case_knob_statement(case, knob, value, self.table),
                         pack.knob_statement("Colorspace", "illuminant_out", value, self.table))


class NamedWhiteTests(unittest.TestCase):
    def test_d60_is_nukes_aces_white_and_dci_its_dci_p3_white(self):
        table = load_table()
        self.assertIn("setValue(14)  # ACES", pack.py_script(_case("colorspace_illuminant_keeps_xyz"), table))
        self.assertIn("# DCI-P3", pack.py_script(_case("colorspace_illuminant_keeps_xyz_dci"), table))


class MovingPlateTests(unittest.TestCase):
    def test_the_moving_plate_is_a_sequence_nodebased_writes(self):
        case = _case("kronos_motion_method_half_translation")
        table = load_table()
        self.assertTrue(pack.sequence_plate(case))
        self.assertEqual(pack.unrenderable(case, table), "")
        script = pack.py_script(case, table)
        first, last = pack.plate_range(case)
        self.assertIn(f"'kronos_motion_method_half_translation_plate.####.exr'", script)
        self.assertIn(f"first={first}, last={last}", script)
        self.assertIn("node['input.first'].setValue(1)", script)       # the project's range, as NodeBased's Kronos reads it
        self.assertIn("node['input.last'].setValue(8)", script)

    def test_plate_frames_move_by_the_chains_translate(self):
        case = _case("kronos_motion_method_half_translation")
        frames = pack.plate_sequences(case)
        first, last = pack.plate_range(case)
        self.assertEqual(len(frames), last - first + 1)
        two, three = frames[f"{case.id}_plate.2.exr"], frames[f"{case.id}_plate.3.exr"]
        self.assertEqual(two.shape, (64, 96, 4))
        # the chain's Transform moves the noise 4 pixels right per frame with a nearest filter
        np.testing.assert_allclose(three[:, 30:60], two[:, 26:56], atol=1e-6)
        self.assertGreater(float(np.abs(three - two).max()), 0.01)

    def test_extra_inputs_stay_unrenderable_with_the_reason(self):
        reason = pack.unrenderable(_case("kronos_resampling_bilinear_half_pixel_step"), load_table())
        self.assertIn("FgVecs", reason)


class GeometryWiringTests(unittest.TestCase):
    """The settings and wiring Nuke's Transform-group nodes need from a script (each probed in Nuke 17.0v6 on 11/21)."""

    def setUp(self):
        self.table = load_table()

    def test_nodebaseds_filter_reaches_nuke(self):
        self.assertIn("node['filter'].setValue(0)  # impulse", pack.py_script(_case("transform_scale_two_about_center"), self.table))
        self.assertIn("# impulse", pack.py_script(_case("reformat_fit_letterboxes"), self.table))
        self.assertIn("# linear", pack.py_script(_case("stmap_identity_map_returns_the_source"), self.table))

    def test_cornerpin_writes_its_from_corners(self):
        script = pack.py_script(_case("cornerpin_from_points_choose_the_part_that_is_pinned"), self.table)
        self.assertIn("node['from1'].setValue([4.0, 3.0])", script)
        self.assertIn("node['from4'].setValue([12.0, 9.0])", script)

    def test_stmap_takes_the_source_on_input_zero(self):
        script = pack.py_script(_case("stmap_identity_map_returns_the_source"), self.table)
        self.assertIn("node.setInput(0, read_A)", script)
        self.assertIn("node.setInput(1, read_UV)", script)

    def test_idistort_reads_its_vectors_from_a_layer_on_the_source(self):
        script = pack.py_script(_case("idistort_u_and_v_channels_choose_what_is_read"), self.table)
        self.assertIn("vectors.setInput(0, read_A)", script)
        self.assertIn("vectors.setInput(1, read_VS)" if "read_VS" in script else "vectors.setInput(1, read_", script)
        self.assertIn("node.setInput(0, vectors)", script)
        self.assertIn("node['uv'].setValue('nbv.green nbv.red')", script)


if __name__ == "__main__":
    unittest.main()
