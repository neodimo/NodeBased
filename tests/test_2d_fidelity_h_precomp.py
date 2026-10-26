"""Fidelity step H, part 3: Precomp takes Nuke's use_output, read file for output (reading, missing frames, reload), and the Open,
Render and Reload buttons, each of which does what it says."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import precompnode
from nodebased.core import SPECS, Dispatcher, atomic_save, empty_document, load_document, upgrade_document
from nodebased.imaging import Evaluator

SIZE = 8


def reference(path, *, colour=(1.0, 0.0, 0.0), out_path=None, last=3):
    """A document with a red Constant (`plate`) -> Grade (`graded`, viewed) and, with `out_path`, a Write (`w`) on `graded`."""
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "plate", "type": "Constant",
               "params": {"width": SIZE, "height": SIZE, "red": colour[0], "green": colour[1], "blue": colour[2]}})
    d.execute({"op": "create", "id": "graded", "type": "Grade", "params": {"multiply": 0.5}})
    d.execute({"op": "connect", "id": "graded", "input": "image", "source": "plate"})
    if out_path is not None:
        d.execute({"op": "create", "id": "w", "type": "Write", "params": {"path": str(out_path), "bit_depth": "float"}})
        d.execute({"op": "connect", "id": "w", "input": "image", "source": "graded"})
    d.execute({"op": "view", "id": "graded"})
    d.execute({"op": "time", "first": 1, "last": last, "current": 1})
    atomic_save(str(path), d.document)
    return d.document


def host(**params):
    d = Dispatcher(empty_document())
    d.execute({"op": "create", "id": "pre", "type": "Precomp", "params": params})
    return d


def render(d, frame=1):
    return Evaluator().evaluate_raster(d.document, "pre", frame=frame)


class UseOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "ref.nbcomp"
        reference(self.path)

    def test_off_takes_the_documents_view_and_ignores_the_named_node(self):
        graded = render(host(file=str(self.path))).pixels
        np.testing.assert_allclose(graded[0, 0, :3], [0.5, 0.0, 0.0], atol=1e-6)
        named_but_off = render(host(file=str(self.path), output_node="plate")).pixels
        np.testing.assert_array_equal(named_but_off, graded)

    def test_on_takes_the_named_node(self):
        plate = render(host(file=str(self.path), output_node="plate", use_output=1)).pixels
        np.testing.assert_allclose(plate[0, 0, :3], [1.0, 0.0, 0.0], atol=1e-6)

    def test_on_with_no_node_named_falls_back_to_the_view_and_a_wrong_name_is_refused(self):
        np.testing.assert_allclose(render(host(file=str(self.path), use_output=1)).pixels[0, 0, 0], 0.5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "no output node 'nope'"):
            render(host(file=str(self.path), output_node="nope", use_output=1))

    def test_a_precomp_saved_before_the_switch_keeps_its_output_node(self):
        d = host(file=str(self.path), output_node="plate", use_output=1)
        old = copy.deepcopy(d.document)
        del old["nodes"]["pre"]["params"]["use_output"]
        loaded = upgrade_document(old)
        self.assertEqual(loaded["nodes"]["pre"]["params"]["use_output"], 1)
        np.testing.assert_array_equal(Evaluator().evaluate_raster(loaded, "pre", frame=1).pixels, render(d).pixels)
        plain = copy.deepcopy(host(file=str(self.path)).document)
        del plain["nodes"]["pre"]["params"]["use_output"]
        self.assertEqual(upgrade_document(plain)["nodes"]["pre"]["params"]["use_output"], 0)

    def test_the_new_params_have_their_defaults(self):
        params = SPECS["Precomp"]["params"]
        self.assertEqual((params["use_output"], params["reading"], params["missing_frames"], params["reload_write"]), (0, 0, "error", 0))


class ReadFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "ref.nbcomp"
        self.files = self.dir / "out" / "f.%04d.exr"
        reference(self.path, out_path=self.files)

    def render_files(self):
        return precompnode.render({"file": str(self.path)}, Evaluator())

    def test_render_writes_the_write_over_the_documents_frame_range(self):
        written = self.render_files()
        self.assertEqual([Path(p).name for p in written], ["f.0001.exr", "f.0002.exr", "f.0003.exr"])
        self.assertTrue(all(Path(p).is_file() for p in written))
        document = load_document(str(self.path))
        expected = Evaluator().evaluate_raster(document, "graded", frame=2).pixels
        got = render(host(file=str(self.path), reading=1), 2).pixels
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_reading_shows_the_file_not_the_document(self):
        self.render_files()
        reference(self.path, colour=(0.0, 0.0, 1.0), out_path=self.files)           # the document changes, the files do not
        reading = render(host(file=str(self.path), reading=1)).pixels
        live = render(host(file=str(self.path))).pixels
        np.testing.assert_allclose(reading[0, 0, :3], [0.5, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(live[0, 0, :3], [0.0, 0.0, 0.5], atol=1e-6)

    def test_a_missing_frame_follows_missing_frames(self):
        self.render_files()
        Path(str(self.files) % 2).unlink()
        with self.assertRaisesRegex(ValueError, "Missing frame"):
            render(host(file=str(self.path), reading=1), 2)
        black = render(host(file=str(self.path), reading=1, missing_frames="black"), 2).pixels
        self.assertEqual(float(black.max()), 0.0)
        board = render(host(file=str(self.path), reading=1, missing_frames="checkerboard"), 2).pixels
        self.assertGreater(float(board.max()), 0.0)
        fallback = render(host(file=str(self.path), reading=1, missing_frames="read input"), 2).pixels
        np.testing.assert_allclose(fallback[0, 0, :3], [0.5, 0.0, 0.0], atol=1e-6)      # the document, evaluated
        there = render(host(file=str(self.path), reading=1, missing_frames="read input"), 1).pixels
        np.testing.assert_allclose(there[0, 0, :3], [0.5, 0.0, 0.0], atol=1e-6)

    def test_reading_with_no_write_or_no_file_says_so(self):
        reference(self.path)
        with self.assertRaisesRegex(ValueError, "no enabled Write"):
            render(host(file=str(self.path), reading=1))
        with self.assertRaisesRegex(ValueError, "no enabled Write"):
            precompnode.render({"file": str(self.path)}, Evaluator())
        reference(self.path, out_path="")
        with self.assertRaisesRegex(ValueError, "no file to read"):
            render(host(file=str(self.path), reading=1))

    def test_reload_write_makes_the_node_read_the_files_again(self):
        self.render_files()
        d = host(file=str(self.path), reading=1)
        evaluator = Evaluator()
        first = evaluator.evaluate_raster(d.document, "pre", frame=1).pixels
        np.testing.assert_allclose(first[0, 0, :3], [0.5, 0.0, 0.0], atol=1e-6)
        reference(self.path, colour=(0.0, 0.0, 1.0), out_path=self.files)
        self.render_files()                                                          # new files over the old names
        d.execute({"op": "set", "id": "pre", "param": "reload_write", "value": 1})
        again = evaluator.evaluate_raster(d.document, "pre", frame=1).pixels
        np.testing.assert_allclose(again[0, 0, :3], [0.0, 0.0, 0.5], atol=1e-6)

    def test_a_single_file_write_for_several_frames_is_refused(self):
        reference(self.path, out_path=self.dir / "once.exr")
        with self.assertRaisesRegex(ValueError, "single file"):
            precompnode.render({"file": str(self.path)}, Evaluator())

    def test_render_follows_the_writes_frame_limit_and_names_it_with_the_use_output_write(self):
        d = Dispatcher(load_document(str(self.path)))
        d.execute({"op": "set", "id": "w", "param": "use_limit", "value": 1})
        d.execute({"op": "set", "id": "w", "param": "first", "value": 2})
        d.execute({"op": "set", "id": "w", "param": "last", "value": 3})
        atomic_save(str(self.path), d.document)
        written = precompnode.render({"file": str(self.path), "output_node": "w", "use_output": 1}, Evaluator())
        self.assertEqual([Path(p).name for p in written], ["f.0002.exr", "f.0003.exr"])


class ButtonTests(unittest.TestCase):
    def setUp(self):
        from tests.test_desktop import APP, Window, release_window
        self.app = APP
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "ref.nbcomp"
        self.files = self.dir / "out" / "f.%04d.exr"
        reference(self.path, out_path=self.files)
        self.window = Window()
        self.addCleanup(release_window, self)
        self.window.show()
        APP.processEvents()
        self.window.dispatcher.execute({"op": "create", "id": "pre", "type": "Precomp", "pos": [0, 0],
                                        "params": {"file": str(self.path)}})

    def button(self, name):
        from PySide6.QtWidgets import QPushButton
        self.panel = self.window.build_node_panel("pre")             # kept, or Qt deletes the buttons with it
        found = self.panel.findChild(QPushButton, name)
        self.assertIsNotNone(found, name)
        return found

    def params(self):
        return self.window.graph_nodes()["pre"]["params"]

    def test_the_panel_has_the_four_buttons(self):
        for name in ("precomp-reload", "precomp-open", "precomp-render", "precomp-reload-write"):
            self.button(name)

    def test_reload_buttons_count_up_one_undo_step_each(self):
        self.button("precomp-reload").click()
        self.app.processEvents()
        self.assertEqual(self.params()["reload"], 1)
        self.button("precomp-reload-write").click()
        self.app.processEvents()
        self.assertEqual(self.params()["reload_write"], 1)

    def test_render_writes_the_files(self):
        self.button("precomp-render").click()
        self.app.processEvents()
        self.assertTrue(all(Path(str(self.files) % n).is_file() for n in (1, 2, 3)))
        self.assertEqual(self.params()["reload_write"], 1)               # so a Precomp that reads the files sees the new ones

    def test_render_reports_a_document_without_a_write(self):
        reference(self.path)
        self.assertEqual(self.window.render_precomp("pre"), [])
        self.assertIn("no enabled Write", self.window.node_errors.get("pre", ""))

    def test_open_loads_the_referenced_document_into_the_window(self):
        self.window.saved_document = copy.deepcopy(self.window.dispatcher.document)          # nothing unsaved to ask about
        self.button("precomp-open").click()
        self.app.processEvents()
        self.assertEqual(self.window.project_path, str(self.path))
        self.assertEqual(set(self.window.dispatcher.document["nodes"]), {"plate", "graded", "w"})
        self.assertEqual(self.window.dispatcher.document["view"], "graded")

    def test_open_of_a_missing_document_changes_nothing(self):
        self.window.dispatcher.execute({"op": "set", "id": "pre", "param": "file", "value": str(self.dir / "gone.nbcomp")})
        from unittest import mock
        with mock.patch("nodebased.app.QMessageBox.warning") as warning:
            self.assertFalse(self.window.open_precomp("pre"))
        warning.assert_called_once()
        self.assertIn("pre", self.window.dispatcher.document["nodes"])


if __name__ == "__main__":
    unittest.main()
