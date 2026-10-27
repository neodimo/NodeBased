"""Fidelity step F, part 4: Read takes localization, a proxy, reload and context variables; Write takes read-back (Read file with
on_error and reload), render order and context variables. Round trips through real files, the tile path against the evaluator, bypass
and old documents."""
import copy
import os
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from nodebased import localize, pathcontext, readknobs, readproxy, writenode
from nodebased.core import SPECS, Dispatcher, bypass_slot, empty_document, upgrade_document, validate
from nodebased.imaging import Evaluator, read_image_raster
from nodebased.media import write_exr
from nodebased.tileexec import TileExecutor

W, H = 24, 16


def picture(red, green=0.25, blue=0.5, width=W, height=H):
    image = np.zeros((height, width, 4), np.float32)
    image[..., 0], image[..., 1], image[..., 2], image[..., 3] = red, green, blue, 1.0
    return image


class FolderCase(unittest.TestCase):
    def setUp(self):
        # Resolved: a Read stores the resolved path, and Windows hands out the temp folder under its short name.
        self.dir = Path(tempfile.mkdtemp(prefix="nb-f4-")).resolve()
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))
        self.cache = self.dir / "cache"
        patcher = unittest.mock.patch.dict(os.environ, {localize.CACHE_ENV: str(self.cache), localize.FROM_ENV: ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def exr(self, name, red, **kw):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_exr(str(path), picture(red, **kw), bits="float")
        return path

    def read_document(self, **params):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Read", "params": params})
        d.execute({"op": "view", "id": "r"})
        return d.document


def replace_keeping_size_and_time(path, red):
    """Write a different picture over `path` that has the old file's size and modified time exactly (a padding attribute evens the
    size out), so nothing but the bytes says the file changed."""
    stamp = Path(path).stat()
    for pad in range(1, 400):
        write_exr(str(path), picture(red), bits="float", metadata={"pad": "x" * pad})
        if Path(path).stat().st_size == stamp.st_size:
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            return
    raise AssertionError("could not match the size")


def red_of(doc, evaluator=None, frame=1, tier=1, target="r"):
    raster = (evaluator or Evaluator()).evaluate_raster(doc, target=target, frame=frame, tier=tier)
    return float(raster.pixels[0, 0, 0]), raster


class DefaultsTests(unittest.TestCase):
    def test_read_and_write_carry_the_new_knobs_at_their_old_values(self):
        read, write = SPECS["Read"]["params"], SPECS["Write"]["params"]
        self.assertEqual((read["localizationPolicy"], read["reload"], read["proxy"], read["proxy_format"]),
                         ("from auto-localize path", 0, "", "file"))
        self.assertEqual((write["reading"], write["on_error"], write["reload"], write["render_order"]), (0, "error", 0, 1))
        for name in pathcontext.PARAMS:
            self.assertEqual((read[name], write[name]), ("", ""))
        self.assertFalse(readknobs.active(read))
        self.assertFalse(writenode.active(write))

    def test_an_old_read_and_write_load_with_the_defaults_and_read_the_same(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "r", "type": "Read", "params": {}})
        d.execute({"op": "create", "id": "w", "type": "Write", "params": {}})
        old = copy.deepcopy(d.document)
        for kind, names in (("r", list(readknobs.DEFAULTS)[11:]), ("w", ["reading", "on_error", "reload", "render_order", *pathcontext.PARAMS])):
            for name in names:
                old["nodes"][kind]["params"].pop(name, None)
        loaded = upgrade_document(old)
        validate(loaded)
        self.assertEqual(loaded["nodes"]["r"]["params"]["localizationPolicy"], "from auto-localize path")
        self.assertEqual(loaded["nodes"]["w"]["params"]["render_order"], 1)
        self.assertEqual(loaded["nodes"]["w"]["params"]["reading"], 0)


class ContextVariableTests(FolderCase):
    def test_pairs_fill_the_variables_of_a_path(self):
        p = {"key1": "SHOT", "value1": "sh010", "key2": "SEQ", "value2": "ab", "key3": "", "value3": "ignored"}
        self.assertEqual(pathcontext.substitute("/shots/%{SEQ}/%{SHOT}/plate.%04d.exr", p), "/shots/ab/sh010/plate.%04d.exr")
        self.assertEqual(pathcontext.substitute("/shots/%{OTHER}/x.exr", p), "/shots/%{OTHER}/x.exr")        # unnamed variables stay
        self.assertEqual(pathcontext.substitute("/plain.exr", p), "/plain.exr")
        self.assertEqual(pathcontext.substitute("%{A}%{A}", {"key1": "A", "value1": "1", "key2": "A", "value2": "2"}), "11")

    def test_a_read_reads_the_file_its_variables_name_and_keeps_the_stored_path(self):
        self.exr("sh010/plate.exr", 0.3)
        self.exr("sh020/plate.exr", 0.6)
        doc = self.read_document(path=str(self.dir / "%{SHOT}" / "plate.exr"), key1="SHOT", value1="sh010")
        self.assertAlmostEqual(red_of(doc)[0], 0.3, places=6)
        self.assertEqual(doc["nodes"]["r"]["params"]["path"], str(self.dir / "%{SHOT}" / "plate.exr"))
        doc["nodes"]["r"]["params"]["value1"] = "sh020"
        self.assertAlmostEqual(red_of(doc)[0], 0.6, places=6)

    def test_the_tile_path_reads_the_same_file(self):
        self.exr("sh010/plate.exr", 0.3)
        doc = self.read_document(path=str(self.dir / "%{SHOT}" / "plate.exr"), key1="SHOT", value1="sh010")
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        region = executor.canvas_region(doc, "r", frame=1, tier=1)
        got = executor.compose_region(doc, "r", region, frame=1, tier=1).pixels
        np.testing.assert_allclose(got, red_of(doc)[1].pixels, atol=1e-6)

    def test_a_variable_no_pair_names_is_reported_in_the_error(self):
        doc = self.read_document(path=str(self.dir / "%{NOPE}" / "plate.exr"))
        with self.assertRaises((ValueError, OSError)) as caught:
            red_of(doc)
        self.assertIn("%{NOPE}", str(caught.exception))

    def test_a_write_renders_to_the_path_its_variables_name(self):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "w", "type": "Write", "params": {"path": str(self.dir / "%{OUT}" / "w.exr"), "key1": "OUT", "value1": "final"}})
        self.assertEqual(pathcontext.substitute(d.document["nodes"]["w"]["params"]["path"], d.document["nodes"]["w"]["params"]),
                         str(self.dir / "final" / "w.exr"))


class ReloadTests(FolderCase):
    def test_reload_reads_a_file_whose_size_and_time_did_not_change(self):
        path = self.exr("plate.exr", 0.75)
        stamp = path.stat()
        evaluator = Evaluator()
        doc = self.read_document(path=str(path))
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.75, places=6)
        replace_keeping_size_and_time(path, 0.25)                                   # a different picture in a file of the same size and time
        after = path.stat()
        self.assertEqual((after.st_size, after.st_mtime_ns), (stamp.st_size, stamp.st_mtime_ns))
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.75, places=6)           # what the evaluator kept
        doc["nodes"]["r"]["params"]["reload"] = 1                                   # the Reload button counts one more
        self.assertAlmostEqual(red_of(doc, evaluator)[0], 0.25, places=6)

    def test_the_panel_button_counts_a_reload(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        window = make_window(self)
        self.addCleanup(close_window, self)
        window.command({"op": "create", "id": "rd", "type": "Read"})
        window.reload_node("rd")
        window.reload_node("rd")
        self.assertEqual(window.dispatcher.document["nodes"]["rd"]["params"]["reload"], 2)
        window.reload_node("writer")
        self.assertEqual(window.dispatcher.document["nodes"]["writer"]["params"]["reload"], 1)


class LocalizationTests(FolderCase):
    def tampered(self, doc_path, value):
        """Overwrite the local copy of `doc_path` with a different picture, keeping the stamp that says it is current."""
        copy_path = localize.copy_path(doc_path)
        write_exr(str(copy_path), picture(value), bits="float")

    def test_off_reads_the_source_and_makes_no_copy(self):
        path = self.exr("plate.exr", 0.25)
        doc = self.read_document(path=str(path), localizationPolicy="off")
        self.assertAlmostEqual(red_of(doc)[0], 0.25, places=6)
        self.assertFalse(self.cache.exists())

    def test_on_reads_from_the_local_copy(self):
        path = self.exr("plate.exr", 0.25)
        doc = self.read_document(path=str(path), localizationPolicy="on")
        self.assertAlmostEqual(red_of(doc)[0], 0.25, places=6)
        self.assertTrue(localize.copy_path(path).exists())
        self.tampered(path, 0.9)                                                    # the copy, not the file, is what is read
        self.assertAlmostEqual(red_of(doc, Evaluator())[0], 0.9, places=6)

    def test_an_edited_source_replaces_its_copy(self):
        path = self.exr("plate.exr", 0.25)
        doc = self.read_document(path=str(path), localizationPolicy="on")
        red_of(doc)
        write_exr(str(path), picture(0.5), bits="float")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
        self.assertAlmostEqual(red_of(doc, Evaluator())[0], 0.5, places=6)

    def test_the_auto_policy_localizes_only_files_under_the_listed_folders(self):
        inside = self.exr("fast/plate.exr", 0.25)
        outside = self.exr("slow/plate.exr", 0.25)
        with unittest.mock.patch.dict(os.environ, {localize.FROM_ENV: str(self.dir / "slow")}):
            self.assertTrue(localize.wanted(localize.DEFAULT_POLICY, outside))
            self.assertFalse(localize.wanted(localize.DEFAULT_POLICY, inside))
            doc = self.read_document(path=str(outside))
            red_of(doc)
            self.assertTrue(localize.copy_path(outside).exists())
            doc = self.read_document(path=str(inside))
            red_of(doc)
            self.assertFalse(localize.copy_path(inside).exists())
        self.assertFalse(localize.wanted(localize.DEFAULT_POLICY, outside))        # no folder listed: nothing is localized

    def test_update_localization_replaces_the_copies_of_a_whole_sequence(self):
        for n in (1, 2, 3):
            self.exr(f"seq.{n:04d}.exr", n / 10.0)
        pattern = str(self.dir / "seq.%04d.exr")
        params = {"path": pattern, "localizationPolicy": "on"}
        made = localize.update_read(params)
        self.assertEqual(len(made), 3)
        self.tampered(self.dir / "seq.0002.exr", 0.99)
        made = localize.update_read(params)                                          # the button: the copy is the source again
        self.assertEqual(len(made), 3)
        doc = self.read_document(path=pattern, localizationPolicy="on")
        self.assertAlmostEqual(red_of(doc, frame=2)[0], 0.2, places=6)
        self.assertEqual(localize.update_read({"path": pattern, "localizationPolicy": "off"}), [])

    def test_the_window_button_does_it(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        path = self.exr("plate.exr", 0.25)
        window = make_window(self)
        self.addCleanup(close_window, self)
        window.command({"op": "create", "id": "rd", "type": "Read", "params": {"path": str(path), "localizationPolicy": "on"}})
        made = window.update_localization("rd")
        self.assertEqual(made, [str(localize.copy_path(path))])


class ProxyTests(FolderCase):
    def doc(self, **extra):
        return self.read_document(path=str(self.dir / "full.exr"), proxy=str(self.dir / "half.exr"), **extra)

    def test_a_proxy_stands_in_for_the_file_at_a_proxy_tier_and_is_fitted_to_its_canvas(self):
        self.exr("full.exr", 0.2)
        self.exr("half.exr", 0.8, width=W // 2, height=H // 2)
        doc = self.doc()
        value, raster = red_of(doc, tier=2)
        self.assertAlmostEqual(value, 0.8, places=6)
        self.assertEqual(raster.pixels.shape[:2], (H // 2, W // 2))
        self.assertEqual((raster.display.width, raster.display.height), (W // 2, H // 2))

    def test_at_tier_1_the_proxy_is_ignored(self):
        self.exr("full.exr", 0.2)
        self.exr("half.exr", 0.8, width=W // 2, height=H // 2)
        value, raster = red_of(self.doc(), tier=1)
        self.assertAlmostEqual(value, 0.2, places=6)
        self.assertEqual(raster.pixels.shape[:2], (H, W))

    def test_without_a_proxy_a_tier_decimates_the_full_file_as_before(self):
        self.exr("full.exr", 0.2)
        doc = self.read_document(path=str(self.dir / "full.exr"))
        value, raster = red_of(doc, tier=2)
        self.assertAlmostEqual(value, 0.2, places=6)
        self.assertEqual(raster.pixels.shape[:2], (H // 2, W // 2))

    def test_the_main_file_is_not_opened_and_may_be_missing(self):
        self.exr("half.exr", 0.8, width=W // 2, height=H // 2)
        value, raster = red_of(self.doc(), tier=2)                                   # full.exr does not exist
        self.assertAlmostEqual(value, 0.8, places=6)
        self.assertEqual(raster.pixels.shape[:2], (H // 2, W // 2))                  # the proxy keeps its own size
        with self.assertRaises((ValueError, OSError)):
            red_of(self.doc(), tier=1)

    def test_a_proxy_of_another_size_is_resampled_to_the_canvas(self):
        self.exr("full.exr", 0.2)
        image = picture(0.0, width=W // 4, height=H // 4)
        image[:, : (W // 4) // 2, 0] = 1.0                                           # left half red 1, right half red 0
        write_exr(str(self.dir / "half.exr"), image, bits="float")
        _value, raster = red_of(self.doc(), tier=2)
        self.assertEqual(raster.pixels.shape[:2], (H // 2, W // 2))
        row = raster.pixels[0, :, 0]
        self.assertGreater(row[0], 0.9)
        self.assertLess(row[-1], 0.1)
        self.assertTrue(np.all(np.diff(row) <= 1e-6))                                # a smooth, monotonic step

    def test_a_named_proxy_format_sets_the_canvas(self):
        self.exr("full.exr", 0.2)
        self.exr("half.exr", 0.8, width=W // 2, height=H // 2)
        _value, raster = red_of(self.doc(proxy_format="HD_1080"), tier=8)
        self.assertEqual(raster.pixels.shape[:2], (1080 // 8, 1920 // 8))

    def test_fit_pixels_is_an_exact_average_for_a_half_size_proxy(self):
        rng = np.random.default_rng(3)
        small = rng.random((4, 6, 4)).astype(np.float32)
        np.testing.assert_array_equal(readproxy.fit_pixels(small, 6, 4), small)
        big = readproxy.fit_pixels(small, 12, 8)
        np.testing.assert_allclose(big[0::2, 0::2].mean(), small.mean(), atol=0.05)
        down = readproxy.fit_pixels(big, 6, 4)
        self.assertEqual(down.shape, (4, 6, 4))

    def test_the_tile_path_hands_a_proxy_to_the_evaluator_with_the_same_pixels(self):
        self.exr("full.exr", 0.2)
        self.exr("half.exr", 0.8, width=W // 2, height=H // 2)
        doc = self.doc()
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=4)
        region = executor.canvas_region(doc, "r", frame=1, tier=2)
        got = executor.compose_region(doc, "r", region, frame=1, tier=2)
        np.testing.assert_allclose(got.pixels, red_of(doc, tier=2)[1].pixels, atol=1e-6)
        self.assertFalse(got.tiled)


class ReadBackTests(FolderCase):
    def doc(self, **write):
        d = Dispatcher(empty_document())
        d.execute({"op": "create", "id": "src", "type": "Constant", "params": {"width": W, "height": H, "red": 0.9, "green": 0.9, "blue": 0.9, "alpha": 1.0}})
        d.execute({"op": "create", "id": "w", "type": "Write", "params": write})
        d.execute({"op": "connect", "id": "w", "input": "image", "source": "src"})
        d.execute({"op": "view", "id": "w"})
        return d.document

    def test_reading_shows_the_file_in_place_of_the_input(self):
        self.exr("out.0007.exr", 0.3)
        doc = self.doc(path=str(self.dir / "out.%04d.exr"), frame_expr="frame + 6")
        self.assertAlmostEqual(red_of(doc, frame=1, target="w")[0], 0.9, places=6)         # reading off: the input passes through
        doc["nodes"]["w"]["params"]["reading"] = 1
        self.assertAlmostEqual(red_of(doc, frame=1, target="w")[0], 0.3, places=6)

    def test_the_file_number_follows_the_frame_mode(self):
        for n in (1001, 1002, 1003):
            self.exr(f"out.{n:04d}.exr", (n - 1000) / 10.0)
        doc = self.doc(path=str(self.dir / "out.%04d.exr"), frame_mode="start at", frame_expr="1001", reading=1)
        doc["time"].update(first=1, last=3, current=1)
        for frame, want in ((1, 0.1), (2, 0.2), (3, 0.3)):
            self.assertAlmostEqual(red_of(doc, frame=frame, target="w")[0], want, places=6)

    def test_on_error_decides_what_a_missing_frame_shows(self):
        self.exr("out.0001.exr", 0.3)
        self.exr("out.0002.exr", 0.4)
        base = dict(path=str(self.dir / "out.%04d.exr"), reading=1)
        with self.assertRaises(ValueError):
            red_of(self.doc(**base), frame=5, target="w")
        self.assertEqual(float(red_of(self.doc(**base, on_error="black"), frame=5, target="w")[1].pixels.max()), 0.0)
        self.assertAlmostEqual(red_of(self.doc(**base, on_error="hold"), frame=5, target="w")[0], 0.4, places=6)
        checker = red_of(self.doc(**base, on_error="checkerboard"), frame=5, target="w")[1].pixels
        np.testing.assert_allclose(np.unique(checker[..., 0]), [0.2, 0.8], atol=1e-6)

    def test_reload_reads_a_replaced_file_of_the_same_size_and_time(self):
        path = self.exr("out.0001.exr", 0.75)
        stamp = path.stat()
        evaluator = Evaluator()
        doc = self.doc(path=str(self.dir / "out.%04d.exr"), reading=1)
        self.assertAlmostEqual(red_of(doc, evaluator, target="w")[0], 0.75, places=6)
        replace_keeping_size_and_time(path, 0.25)
        self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), (stamp.st_size, stamp.st_mtime_ns))
        self.assertAlmostEqual(red_of(doc, evaluator, target="w")[0], 0.75, places=6)
        doc["nodes"]["w"]["params"]["reload"] = 1
        self.assertAlmostEqual(red_of(doc, evaluator, target="w")[0], 0.25, places=6)

    def test_an_edited_file_is_picked_up_without_a_reload(self):
        path = self.exr("out.0001.exr", 0.25)
        evaluator = Evaluator()
        doc = self.doc(path=str(self.dir / "out.%04d.exr"), reading=1)
        self.assertAlmostEqual(red_of(doc, evaluator, target="w")[0], 0.25, places=6)
        write_exr(str(path), picture(0.5), bits="float")
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
        self.assertAlmostEqual(red_of(doc, evaluator, target="w")[0], 0.5, places=6)

    def test_reading_without_a_path_passes_the_input_renders_whole_and_a_disabled_write_passes_the_input(self):
        doc = self.doc(reading=1)                                                    # no file set: nothing to read, the input passes
        self.assertAlmostEqual(red_of(doc, target="w")[0], 0.9, places=6)
        with self.assertRaises(ValueError):
            writenode.read_back(doc["nodes"]["w"]["params"], 1)
        self.exr("out.0001.exr", 0.3)
        doc = self.doc(path=str(self.dir / "out.%04d.exr"), reading=1)
        executor = TileExecutor(evaluator=Evaluator(), tile_edge=8)
        self.assertFalse(executor.supports_tiled(doc, "w"))
        region = executor.canvas_region(doc, "w", frame=1, tier=1)
        np.testing.assert_allclose(executor.compose_region(doc, "w", region, frame=1, tier=1).pixels, red_of(doc, target="w")[1].pixels, atol=1e-6)
        doc["nodes"]["w"]["disabled"] = True
        self.assertEqual(bypass_slot(doc["nodes"]["w"]), "image")
        self.assertAlmostEqual(red_of(doc, target="w")[0], 0.9, places=6)

    def test_a_write_round_trips_through_its_own_file(self):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        window = make_window(self)
        self.addCleanup(close_window, self)
        window.set_time(first=1, last=3, current=2)
        window.command({"op": "set", "id": "writer", "param": "path", "value": str(self.dir / "round.%04d.exr")})
        window.render_write("writer", single=False)
        rendered = [red_of(window.dispatcher.document, frame=f, target="writer")[1].pixels for f in (1, 2, 3)]
        window.command({"op": "set", "id": "writer", "param": "reading", "value": 1})
        for frame, expected in zip((1, 2, 3), rendered):
            np.testing.assert_allclose(red_of(window.dispatcher.document, frame=frame, target="writer")[1].pixels, expected, atol=1e-3)


class RenderOrderTests(FolderCase):
    def make(self, orders):
        from tests.test_2d_parity_feedback_y3 import close_window, make_window
        window = make_window(self)
        self.addCleanup(close_window, self)
        window.set_time(first=1, last=2, current=1)
        source = next(k for k, n in window.dispatcher.document["nodes"].items() if n["type"] == "Constant")
        window.command({"op": "delete", "id": "writer"})
        for name, order in orders.items():
            window.command({"op": "create", "id": name, "type": "Write", "params": {"render_order": order, "path": str(self.dir / name / "f.%04d.exr")}})
            window.command({"op": "connect", "id": name, "input": "image", "source": source})
        return window

    def test_the_order_is_the_lowest_render_order_first_with_ties_in_creation_order(self):
        window = self.make({"c": 3, "a": 1, "b": 1, "d": -2})
        self.assertEqual(writenode.render_order(window.dispatcher.document), ["d", "a", "b", "c"])
        window.command({"op": "disable", "id": "a", "value": True})
        self.assertEqual(writenode.render_order(window.dispatcher.document), ["d", "b", "c"])

    def test_render_all_renders_every_write_in_that_order(self):
        window = self.make({"late": 5, "early": 1, "middle": 3})
        seen = []
        real = window.render_write

        def record(key, single=True):
            seen.append(key)
            return real(key, single=single)
        with unittest.mock.patch.object(window, "render_write", side_effect=record):
            window.render_all_writes()
        self.assertEqual(seen, ["early", "middle", "late"])
        for name in ("early", "middle", "late"):
            self.assertEqual(sorted(p.name for p in (self.dir / name).iterdir()), ["f.0001.exr", "f.0002.exr"])

    def test_the_render_buttons_are_in_the_write_panel(self):
        from PySide6.QtWidgets import QPushButton
        window = self.make({"w1": 1})
        window.command({"op": "view", "id": "w1"})
        window.inspect("w1")
        labels = {button.text() for button in window.findChildren(QPushButton)}
        for text in ("Render current frame", "Render frame range", "Render all Write nodes", "Reload"):
            self.assertIn(text, labels)
        for button in window.findChildren(QPushButton):
            if button.text() == "Render current frame":
                button.click()
        self.assertEqual(sorted(p.name for p in (self.dir / "w1").iterdir()), ["f.0001.exr"])


if __name__ == "__main__":
    unittest.main()
