import contextlib
import io
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from nodebased.artifacts import ArtifactStore, main

class ArtifactStoreTests(unittest.TestCase):
    def test_content_addressing_gc_references_and_missing_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); store = ArtifactStore(root / "cache")
            doc = root / "shot.node"; doc.write_text("open")
            a = store.put(b"same", "scene_state", {"producer":"WriteGeo3D"})
            b = store.put(b"same", "scene_state", {"producer":"WriteGeo3D"})
            self.assertEqual(a, b); self.assertEqual(len(list((root / "cache/objects").iterdir())), 1)
            store.reference(doc, [a])
            old = store.put(b"old", "control_bundle", {})
            time.sleep(.002)
            middle = store.put(b"middle", "provider_description", {})
            time.sleep(.002)
            c = store.put(b"other", "control_bundle", {"inputs":[{"id":a}]})
            removed = store.gc(0)
            self.assertNotIn(a, removed); self.assertEqual(removed, [old, middle, c])
            with self.assertRaisesRegex(ValueError, "Unknown artifact id"):
                store.get("0" * 64)

    def test_loaded_session_document_keeps_artifacts_pinned(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(folder)
            aid = store.put(b"in use", "generated_sequence", {})
            store.reference("session:test-window", [aid])
            self.assertEqual(store.gc(0), [])

    def test_provenance_cli_is_oldest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(folder)
            a = store.put(b"scene", "scene_state", {"producer":"SceneState", "version":1, "inputs":[]})
            b = store.put(b"bundle", "control_bundle", {"producer":"ControlBundle", "version":1, "inputs":[{"id":a}]})
            out = io.StringIO()
            with patch.dict("os.environ", {"NODEBASED_CACHE": str(Path(folder) / "..")}), contextlib.redirect_stdout(out):
                # The CLI follows the same default-root convention as the running application.
                from nodebased.artifacts import default_root
                store = ArtifactStore(default_root())
                aa = store.put(b"scene", "scene_state", {"producer":"SceneState", "version":1, "inputs":[]})
                bb = store.put(b"bundle", "control_bundle", {"producer":"ControlBundle", "version":1, "inputs":[{"id":aa}]})
                main(["provenance", bb])
            self.assertLess(out.getvalue().index(aa), out.getvalue().index(bb))
            self.assertIn(aa, out.getvalue()); self.assertIn(bb, out.getvalue())
