"""Nuke's Link menu on channel, layer and format knobs (lane 10, step T6 finish, 2026-11-17; 17.0 User Guide, Linking
Expressions): Set link, Edit link, Remove link, and Ctrl/Cmd+drag of the menu onto another such knob."""
import tests.isolation  # noqa: F401  (keeps the window's layout out of the real Qt settings)

import copy
import sys
import unittest
from pathlib import Path

from tests.test_desktop import APP  # noqa: F401  (the shared QApplication)

from nodebased import channelpicker, expressions
from nodebased.animation import resolve_document
from nodebased.core import Dispatcher, validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import nuke_panel_interactions as interactions  # noqa: E402


def _doc():
    d = Dispatcher()
    d.execute({"op": "create", "id": "g", "type": "Grade", "params": {"channels": "rgba"}})
    d.execute({"op": "create", "id": "h", "type": "Grade"})
    d.execute({"op": "create", "id": "r", "type": "Reformat", "params": {"format": "HD_720"}})
    d.execute({"op": "create", "id": "s", "type": "Reformat"})
    return d


class WordLinkDocumentTests(unittest.TestCase):
    def test_families_match_the_channel_pickers(self):
        self.assertEqual(expressions.LINK_FAMILIES["channels"], channelpicker.CHANNEL_SET_PARAMS)
        self.assertEqual(expressions.LINK_FAMILIES["channel"], channelpicker.SINGLE_CHANNEL_PARAMS)

    def test_a_linked_channel_knob_follows_its_target(self):
        d = _doc()
        d.execute({"op": "set_expression", "id": "h", "param": "channels", "expression": "g.channels"})
        self.assertEqual(resolve_document(d.document, 1)["nodes"]["h"]["params"]["channels"], "rgba")
        d.execute({"op": "set", "id": "g", "param": "channels", "value": "alpha"})
        self.assertEqual(resolve_document(d.document, 1)["nodes"]["h"]["params"]["channels"], "alpha")
        self.assertEqual(d.document["nodes"]["h"]["params"]["channels"], "rgb")   # the typed value comes back on Remove link

    def test_a_chain_of_links_follows_the_first(self):
        d = _doc()
        d.execute({"op": "create", "id": "k", "type": "Grade"})
        d.execute({"op": "set_expression", "id": "k", "param": "channels", "expression": "h.channels"})
        d.execute({"op": "set_expression", "id": "h", "param": "channels", "expression": "g.channels"})
        self.assertEqual(resolve_document(d.document, 1)["nodes"]["k"]["params"]["channels"], "rgba")

    def test_a_linked_format_brings_its_size(self):
        d = _doc()
        d.execute({"op": "set_expression", "id": "s", "param": "format", "expression": "r.format"})
        params = resolve_document(d.document, 1)["nodes"]["s"]["params"]
        self.assertEqual((params["format"], params["width"], params["height"]), ("HD_720", 1280, 720))

    def test_wrong_links_are_refused(self):
        d = _doc()
        for param, text in (("channels", "g.maskChannelInput"), ("channels", "r.format"), ("channels", "g.channels + 1"),
                            ("gamma", "g.channels"), ("format", "g.channels")):
            target = "s" if param == "format" else "h"
            with self.subTest(param=param, text=text), self.assertRaises(ValueError):
                d.execute({"op": "set_expression", "id": target, "param": param, "expression": text})

    def test_a_cycle_of_links_is_refused(self):
        d = _doc()
        d.execute({"op": "set_expression", "id": "h", "param": "channels", "expression": "g.channels"})
        with self.assertRaises(ValueError):
            d.execute({"op": "set_expression", "id": "g", "param": "channels", "expression": "h.channels"})

    def test_older_documents_still_validate(self):
        d = _doc()
        validate(copy.deepcopy(d.document))


class LinkMenuInteractionTests(unittest.TestCase):
    def test_link_menu_works_like_nukes(self):
        ref = interactions.reference()
        mine = [e for e in ref["behaviours"] if e["id"] == "link.link_menu"]
        original = interactions.reference
        interactions.reference = lambda: dict(ref, behaviours=mine)
        try:
            results = interactions.probe_all()
        finally:
            interactions.reference = original
        self.assertEqual(results["link.link_menu"]["result"], "pass", results["link.link_menu"]["detail"])


if __name__ == "__main__":
    unittest.main()
