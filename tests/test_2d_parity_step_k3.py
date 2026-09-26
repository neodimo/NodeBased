"""Lane L2 step K3: Cryptomatte. A real Cryptomatte EXR is written in the test (two rank layers, the
header manifest, antialiased edges between three named objects) and read back through Read, so the
hashing, the channel layout, the header round trip and the matte maths are all exercised end to end.
Pixel assertions are exact: a matte for one name equals that object's coverage bit for bit."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nodebased import cryptomatte as cm
from nodebased.core import CHOICES, SPECS, bypass_slot
from nodebased.imaging import Evaluator
from nodebased.media import write_exr
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor
from nodebased.tiers import REGION_RULES
from tests.test_2d_parity_group_3b import Graph, evaluator_pixels, evaluator_raster

W, H = 16, 6
SET = "crypto_object"
NAMES = ("sphere", "cube", "plane")

# Per column: the objects there with their coverage, any order. Columns 0-3 sphere, 4 sphere/cube
# (0.25 / 0.75, an antialiased edge), 5-8 cube, 9 cube/plane (0.5 / 0.5), 10-14 plane, and column 15
# holds all three (0.125, 0.25, 0.5) so the second rank layer is used; its remaining 0.125 is empty.
COLUMNS = ([[("sphere", 1.0)]] * 4 + [[("sphere", 0.25), ("cube", 0.75)]] + [[("cube", 1.0)]] * 4
           + [[("cube", 0.5), ("plane", 0.5)]] + [[("plane", 1.0)]] * 5
           + [[("sphere", 0.125), ("cube", 0.25), ("plane", 0.5)]])


def truth(name):
    """The coverage of `name` in every pixel, straight from the table above."""
    out = np.zeros((H, W), np.float32)
    for x, column in enumerate(COLUMNS):
        out[:, x] = sum((cover for who, cover in column if who == name), 0.0)
    return out


def crypto_layers(ids=None):
    """crypto_object00 / 01 as HxWx4 float32 arrays, ranks sorted by coverage, best first."""
    ids = ids or {name: cm.name_to_bits(name) for name in NAMES}
    bits, cover = np.zeros((H, W, 4), np.uint32), np.zeros((H, W, 4), np.float32)
    for x, column in enumerate(COLUMNS):
        for rank, (who, coverage) in enumerate(sorted(column, key=lambda item: -item[1])):
            bits[:, x, rank], cover[:, x, rank] = ids[who], coverage
    layers = {}
    for index in range(2):
        layer = np.zeros((H, W, 4), np.float32)
        layer[..., 0] = bits[..., 2 * index].view(np.float32)
        layer[..., 1] = cover[..., 2 * index]
        layer[..., 2] = bits[..., 2 * index + 1].view(np.float32)
        layer[..., 3] = cover[..., 2 * index + 1]
        layers[f"{SET}{index:02d}"] = layer
    return layers


def manifest_metadata(name=SET, table=None):
    table = table if table is not None else {who: "%08x" % cm.name_to_bits(who) for who in NAMES}
    key = cm.set_key(name)
    return {f"cryptomatte/{key}/name": name, f"cryptomatte/{key}/hash": "MurmurHash3_32",
            f"cryptomatte/{key}/conversion": "uint32_to_float32",
            f"cryptomatte/{key}/manifest": json.dumps(table)}


def beauty():
    frame = np.zeros((H, W, 4), np.float32)
    frame[..., :3] = np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :, None]
    frame[..., 3] = 1.0
    return frame


class Scratch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.count = 0

    def exr(self, layers=None, metadata="manifest"):
        self.count += 1
        path = Path(self._dir.name) / f"crypto{self.count}.exr"
        meta = manifest_metadata() if metadata == "manifest" else metadata
        write_exr(path, beauty(), bits="float", layers=layers if layers is not None else crypto_layers(),
                  metadata=meta)
        return str(path)

    def graph(self, params=None, mask=None, **read):
        g = Graph()
        g.add("src", "Read", {"path": self.exr(**read)})
        wires = {"image": "src"}
        if mask is not None:
            g.add("mask", "Constant", dict(mask, width=W, height=H))
            wires["mask"] = "mask"
        g.add("key", "Cryptomatte", params or {}, **wires)
        return g

    def matte(self, params=None, **read):
        return evaluator_pixels(self.graph(dict(params or {}), **read).doc, "key")[..., 3]


class HashTests(unittest.TestCase):
    # The reference decoder's own test vectors (Psyop/Cryptomatte, cryptomatte_utilities_tests.py):
    # the id of each name as the float32 the file stores.
    PUBLISHED = {"hello": 6.0705627102400005616e-17, "cube": -4.08461912519e+15,
                 "sphere": 2.79018604383e+15, "plane": 3.66557617593e-11,
                 "равнина": -1.3192631212399999468e-25, "mädchen": 6.2361298211599995797e+25}

    def test_ids_match_the_published_values(self):
        for name, value in self.PUBLISHED.items():
            with self.subTest(name=name):
                self.assertEqual(cm.name_to_float(name), float(np.float32(value)))

    def test_murmurhash3_reference_vectors(self):
        # The canonical MurmurHash3_x86_32 (seed 0) answers, covering 0 to 4+ byte tails.
        for text, expected in (("", 0), ("a", 0x3C2569B2), ("ab", 0x9BBFD75F), ("abc", 0xB3DD93FA),
                               ("test", 0xBA6BD213), ("Hello, world!", 0xC0363E43),
                               ("The quick brown fox jumps over the lazy dog", 0x2E4FF723)):
            with self.subTest(text=text):
                self.assertEqual(cm.murmur3_32(text), expected)
        self.assertEqual(cm.murmur3_32("test", seed=0x9747B28C), 0x704B81DC)

    def test_a_zero_or_all_ones_exponent_is_nudged_off(self):
        seen = set()
        for n in range(6000):
            name = f"object_{n}"
            raw = cm.murmur3_32(name)
            exponent = (raw >> 23) & 255
            if exponent in (0, 255):
                seen.add(exponent)
                bits = cm.name_to_bits(name)
                self.assertEqual(bits, raw ^ (1 << 23))
                self.assertTrue(np.isfinite(cm.bits_to_float(bits)))
                self.assertNotEqual(cm.bits_to_float(bits), 0.0)
        self.assertEqual(seen, {0, 255})

    def test_set_key_is_seven_hex_digits_of_the_set_id(self):
        self.assertEqual(cm.set_key(SET), ("%08x" % cm.name_to_bits(SET))[:7])
        self.assertEqual(len(cm.set_key("crypto_material")), 7)

    def test_matte_list_syntax(self):
        sphere = cm.name_to_bits("sphere")
        self.assertEqual(cm.split_matte_list(r"a\,b, c ,, <1.5>"), ["a,b", "c", "<1.5>"])
        self.assertEqual(cm.parse_matte_list("sphere"), {sphere})
        self.assertEqual(cm.parse_matte_list(cm.id_token(sphere)), {sphere})
        self.assertEqual(cm.parse_matte_list("<0x%08x>" % sphere), {sphere})
        with self.assertRaisesRegex(ValueError, "raw id"):
            cm.parse_matte_list("<wat>")


class MatteTests(Scratch):
    def test_the_file_round_trips_layers_and_manifest(self):
        g = self.graph()
        raster = evaluator_raster(g.doc, "src")
        self.assertEqual(sorted(raster.layers), [f"{SET}00", f"{SET}01"])
        self.assertEqual(cm.layer_sets(raster), {SET: [f"{SET}00", f"{SET}01"]})
        self.assertEqual(cm.manifest_for(raster, SET), {who: cm.name_to_bits(who) for who in NAMES})
        # The ids survive the file bit for bit.
        stored = raster.layers[f"{SET}00"].pixels[..., 0].view(np.uint32)
        self.assertEqual(int(stored[0, 0]), cm.name_to_bits("sphere"))

    def test_one_name_returns_its_coverage_exactly(self):
        for name in NAMES:
            with self.subTest(name=name):
                np.testing.assert_array_equal(self.matte(dict(matte_list=name)), truth(name))
        self.assertEqual(float(self.matte(dict(matte_list="sphere"))[0, 4]), 0.25)
        self.assertEqual(float(self.matte(dict(matte_list="cube"))[0, 4]), 0.75)

    def test_the_second_rank_layer_counts(self):
        # Column 15 stores its third object in crypto_object01.
        self.assertEqual(float(self.matte(dict(matte_list="sphere"))[2, 15]), 0.125)
        self.assertEqual(float(self.matte(dict(matte_list="plane"))[2, 15]), 0.5)

    def test_two_names_sum(self):
        got = self.matte(dict(matte_list="sphere, cube"))
        np.testing.assert_array_equal(got, truth("sphere") + truth("cube"))
        self.assertEqual(float(got[0, 4]), 1.0)          # the antialiased edge closes up
        np.testing.assert_array_equal(self.matte(dict(matte_list="sphere,cube,plane"))[:, :15],
                                      np.ones((H, 15), np.float32))
        self.assertEqual(float(self.matte(dict(matte_list="sphere,cube,plane"))[0, 15]), 0.875)

    def test_raw_ids_select_without_a_manifest(self):
        token = cm.id_token(cm.name_to_bits("cube"))
        for metadata in ("manifest", None):
            with self.subTest(manifest=metadata):
                np.testing.assert_array_equal(self.matte(dict(matte_list=token), metadata=metadata), truth("cube"))

    def test_names_still_resolve_by_hash_without_a_manifest(self):
        np.testing.assert_array_equal(self.matte(dict(matte_list="plane, sphere"), metadata=None),
                                      truth("plane") + truth("sphere"))
        raster = evaluator_raster(self.graph(metadata=None).doc, "src")
        self.assertIsNone(raster.meta)
        self.assertEqual(cm.manifest_for(raster, SET), {})
        # With no header the set is still found by its crypto name.
        self.assertEqual(list(cm.layer_sets(raster)), [SET])

    def test_the_manifest_wins_over_the_hash_of_a_name(self):
        ids = {"sphere": cm.name_to_bits("marble"), "cube": cm.name_to_bits("cube"),
               "plane": cm.name_to_bits("plane")}
        table = {"marble": "%08x" % ids["sphere"], "cube": "%08x" % ids["cube"], "plane": "%08x" % ids["plane"]}
        got = self.matte(dict(matte_list="marble"), layers=crypto_layers(ids),
                         metadata=manifest_metadata(table=table))
        np.testing.assert_array_equal(got, truth("sphere"))

    def test_an_empty_or_unknown_list_is_an_empty_matte(self):
        self.assertFalse(self.matte(dict(matte_list="")).any())
        self.assertFalse(self.matte(dict(matte_list="nobody")).any())

    def test_views(self):
        g = self.graph(dict(matte_list="cube", crypto_view="final"))
        final = evaluator_pixels(g.doc, "key")
        np.testing.assert_array_equal(final[..., :3], beauty()[..., :3])       # colour untouched
        np.testing.assert_array_equal(final[..., 3], truth("cube"))            # matte in alpha
        g.d.execute(dict(op="set", id="key", param="crypto_view", value="matte"))
        matte_view = evaluator_pixels(g.doc, "key")
        for channel in range(3):
            np.testing.assert_array_equal(matte_view[..., channel], truth("cube"))
        np.testing.assert_array_equal(matte_view[..., 3], truth("cube"))
        g.d.execute(dict(op="set", id="key", param="crypto_view", value="colors"))
        colors = evaluator_pixels(g.doc, "key")
        np.testing.assert_array_equal(colors[..., 3], np.ones((H, W), np.float32))
        want = {who: cm.id_colors(np.array([cm.name_to_bits(who)], np.uint32))[0] for who in NAMES}
        for x, who in ((1, "sphere"), (6, "cube"), (12, "plane")):
            np.testing.assert_allclose(colors[3, x, :3], want[who], atol=1e-6)
        self.assertEqual(len({tuple(np.round(v, 4)) for v in want.values()}), 3)
        # An edge pixel is the coverage-weighted blend of its two colours.
        np.testing.assert_allclose(colors[3, 4, :3], 0.25 * want["sphere"] + 0.75 * want["cube"], atol=1e-6)

    def test_mix_and_mask_gate_the_matte(self):
        base = beauty()
        g = self.graph(dict(matte_list="sphere", mix=0.0))
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "key"), base)
        half = self.graph(dict(matte_list="sphere", mix=0.5))
        np.testing.assert_allclose(evaluator_pixels(half.doc, "key")[..., 3], 0.5 * truth("sphere") + 0.5, atol=1e-6)
        black = self.graph(dict(matte_list="sphere"), mask=dict(red=0, green=0, blue=0, alpha=0))
        np.testing.assert_array_equal(evaluator_pixels(black.doc, "key"), base)

    def test_bypass_passes_the_input(self):
        g = self.graph(dict(matte_list="sphere"))
        self.assertEqual(bypass_slot(g.doc["nodes"]["key"]), "image")
        g.bypass("key")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "key"), beauty())
        # The bypassed node hands the whole raster on, layers included.
        self.assertEqual(sorted(evaluator_raster(g.doc, "key").layers), [f"{SET}00", f"{SET}01"])


class LayerChoiceTests(Scratch):
    def two_sets(self):
        """crypto_object as usual plus crypto_material, whose sphere is called "marble"."""
        ids = {"sphere": cm.name_to_bits("marble"), "cube": cm.name_to_bits("cube"),
               "plane": cm.name_to_bits("plane")}
        material = {name.replace("object", "material"): layer for name, layer in crypto_layers(ids).items()}
        table = {"marble": "%08x" % ids["sphere"], "cube": "%08x" % ids["cube"], "plane": "%08x" % ids["plane"]}
        return ({**crypto_layers(), **material},
                {**manifest_metadata(), **manifest_metadata("crypto_material", table)})

    def test_several_sets_choose_by_name(self):
        layers, metadata = self.two_sets()
        # With no layer chosen the first set by name is used: crypto_material, where the sphere is "marble".
        np.testing.assert_array_equal(self.matte(dict(matte_list="marble"), layers=layers, metadata=metadata),
                                      truth("sphere"))
        self.assertFalse(self.matte(dict(matte_list="sphere"), layers=layers, metadata=metadata).any())
        named = self.matte(dict(matte_list="sphere", crypto_layer="crypto_object"), layers=layers,
                           metadata=metadata)
        np.testing.assert_array_equal(named, truth("sphere"))
        g = self.graph(dict(crypto_layer="crypto_nope"), layers=layers, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "no layer set 'crypto_nope'.*crypto_material, crypto_object"):
            evaluator_pixels(g.doc, "key")

    def test_no_cryptomatte_layers_is_an_error_that_says_so(self):
        g = self.graph(layers={"normals": np.ones((H, W, 4), np.float32)}, metadata=None)
        with self.assertRaisesRegex(ValueError, "no Cryptomatte layers"):
            evaluator_pixels(g.doc, "key")
        plain = Graph()
        plain.add("c", "Checker")
        plain.add("key", "Cryptomatte", {}, image="c")
        with self.assertRaisesRegex(ValueError, "no Cryptomatte layers"):
            evaluator_pixels(plain.doc, "key")

    def test_lower_case_channel_names_read_as_rgba(self):
        import OpenImageIO as oiio
        layers = crypto_layers()
        names = ["R", "G", "B", "A"] + [f"{name}.{c}" for name in layers for c in "rgba"]
        planes = [beauty()] + list(layers.values())
        path = str(Path(self._dir.name) / "lower.exr")
        spec = oiio.ImageSpec(W, H, len(names), oiio.TypeDesc("float"))
        spec.channelnames = names
        out = oiio.ImageOutput.create(path)
        self.assertTrue(out.open(path, spec))
        self.assertTrue(out.write_image(np.ascontiguousarray(np.concatenate(planes, axis=2))))
        out.close()
        g = Graph()
        g.add("src", "Read", {"path": path})
        g.add("key", "Cryptomatte", dict(matte_list="cube"), image="src")
        np.testing.assert_array_equal(evaluator_pixels(g.doc, "key")[..., 3], truth("cube"))


class LayersOnlyFileTests(Scratch):
    def write_raw(self, names, planes, attributes=None):
        import OpenImageIO as oiio
        path = str(Path(self._dir.name) / "raw.exr")
        spec = oiio.ImageSpec(W, H, len(names), oiio.TypeDesc("float"))
        spec.channelnames = names
        for attribute, value in (attributes or {}).items():
            spec.attribute(attribute, value)
        out = oiio.ImageOutput.create(path)
        self.assertTrue(out.open(path, spec))
        self.assertTrue(out.write_image(np.ascontiguousarray(np.concatenate(planes, axis=2))))
        out.close()
        return path

    def test_a_bare_export_with_no_beauty_reads_and_keys(self):
        """Renderers write the Cryptomatte alone (no R,G,B,A): the Read gives transparent black plus the
        layers, and a set whose name does not start with "crypto" is found through its header entry,
        with the `exr/` prefix and `red`/`green` channel spellings such files use."""
        layers = {f"uObjects{i:02d}": layer for i, layer in enumerate(crypto_layers().values())}
        names = [f"{name}.{c}" for name in layers for c in ("red", "green", "blue", "alpha")]
        meta = {key.replace("cryptomatte/", "exr/cryptomatte/", 1): value
                for key, value in manifest_metadata("uObjects").items()}
        path = self.write_raw(names, list(layers.values()), meta)
        g = Graph()
        g.add("src", "Read", {"path": path})
        g.add("key", "Cryptomatte", dict(matte_list="cube"), image="src")
        out = evaluator_pixels(g.doc, "key")
        np.testing.assert_array_equal(out[..., 3], truth("cube"))
        np.testing.assert_array_equal(out[..., :3], np.zeros((H, W, 3), np.float32))
        raster = evaluator_raster(g.doc, "src")
        self.assertEqual(list(cm.layer_sets(raster)), ["uObjects"])
        self.assertEqual(cm.manifest_for(raster, "uObjects")["cube"], cm.name_to_bits("cube"))

    def test_a_file_with_neither_beauty_nor_layers_is_still_an_error(self):
        path = self.write_raw(["depth", "mask"], [np.zeros((H, W, 2), np.float32)])
        g = Graph()
        g.add("src", "Read", {"path": path})
        with self.assertRaisesRegex(ValueError, "No RGB channels"):
            evaluator_raster(g.doc, "src")


class PickTests(Scratch):
    def test_pick_returns_the_name_under_the_cursor(self):
        raster = evaluator_raster(self.graph().doc, "src")
        members = cm.layer_sets(raster)[SET]
        self.assertEqual(cm.token_for(raster, SET, cm.pick(raster, members, 1, 2)), "sphere")
        self.assertEqual(cm.token_for(raster, SET, cm.pick(raster, members, 6, 0)), "cube")
        # On an antialiased edge the object with more coverage wins.
        self.assertEqual(cm.token_for(raster, SET, cm.pick(raster, members, 4, 0)), "cube")
        self.assertEqual(cm.token_for(raster, SET, cm.pick(raster, members, 15, 0)), "plane")

    def test_pick_without_a_manifest_gives_a_raw_id_that_selects_the_same_thing(self):
        raster = evaluator_raster(self.graph(metadata=None).doc, "src")
        members = cm.layer_sets(raster)[SET]
        token = cm.token_for(raster, SET, cm.pick(raster, members, 12, 3))
        self.assertTrue(token.startswith("<"))
        np.testing.assert_array_equal(self.matte(dict(matte_list=token), metadata=None), truth("plane"))

    def test_names_with_commas_are_escaped(self):
        self.assertEqual(cm.escape_name("a,b"), "a\\,b")
        self.assertEqual(cm.split_matte_list(cm.escape_name("a,b") + ", c"), ["a,b", "c"])

    def test_nothing_under_the_cursor_picks_nothing(self):
        raster = evaluator_raster(self.graph().doc, "src")
        members = cm.layer_sets(raster)[SET]
        empty = {name: layer for name, layer in raster.layers.items()}
        for layer in empty.values():
            layer.pixels[:] = 0
        self.assertIsNone(cm.pick(raster, members, 0, 0))


class RegistrationTests(unittest.TestCase):
    def test_registered_as_a_whole_image_node(self):
        self.assertIn("Cryptomatte", SPECS)
        self.assertEqual(SPECS["Cryptomatte"]["inputs"], ["image"])
        self.assertEqual(SPECS["Cryptomatte"]["optional_inputs"], ["mask"])
        self.assertEqual(CHOICES["crypto_view"], ["final", "matte", "colors"])
        self.assertIn("Cryptomatte", REGION_RULES)
        self.assertNotIn("Cryptomatte", SUPPORTED_TILED_KINDS)

    def test_a_graph_with_it_falls_back_to_the_evaluator(self):
        g = Graph()
        g.add("c", "Checker")
        g.add("key", "Cryptomatte", {}, image="c")
        executor = TileExecutor(evaluator=Evaluator())
        self.assertFalse(executor.supports_tiled(dict(g.doc, view="key"), "key"))


if __name__ == "__main__":
    unittest.main()
