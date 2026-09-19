import json
from pathlib import Path
import unittest

import nodebased
from nodebased import knowledge
from nodebased.core import CHOICES, LIMITS, SPECS, Dispatcher


class KnowledgeTests(unittest.TestCase):
    def test_topics_are_non_empty_and_unknown_topics_fail(self):
        for name in knowledge.TOPICS:
            with self.subTest(topic=name):
                if name == "live_state":
                    continue
                self.assertTrue(knowledge.topic(name).strip())
        with self.assertRaises(ValueError):
            knowledge.topic("not-a-topic")

    def test_version_contains_current_version(self):
        self.assertIn(nodebased.__version__, knowledge.topic("version"))

    def test_bundled_docs_match_the_repository_docs(self):
        # The packaged copies are what an agent reads at runtime; a release that updates
        # docs/ and forgets nodebased/data/docs leaves the agent describing the old build.
        bundled = Path(nodebased.__file__).resolve().parent / "data" / "docs"
        source = bundled.parent.parent.parent / "docs"
        for path in sorted(bundled.glob("*.md")):
            counterpart = source / path.name
            if counterpart.exists():
                with self.subTest(document=path.name):
                    self.assertEqual(path.read_text(encoding="utf-8"),
                                     counterpart.read_text(encoding="utf-8"))

    def test_nodes_describe_a_real_node_and_its_parameters(self):
        node_type, spec = next(iter(SPECS.items()))
        output = knowledge.topic("nodes")
        self.assertIn(node_type, output)
        for parameter in spec["params"]:
            self.assertIn(parameter, output)

    def test_limits_reflect_core_data(self):
        name, value = next(iter(LIMITS.items()))
        self.assertIn(name, knowledge.topic("limits"))
        self.assertIn(str(value[0]), knowledge.topic("limits"))
        choice_name, choices = next(iter(CHOICES.items()))
        self.assertIn(choice_name, knowledge.topic("limits"))
        self.assertIn(str(choices[0]), knowledge.topic("limits"))

    def test_bundled_docs_match_sources(self):
        root = Path(__file__).resolve().parents[1]
        names = ("VISION.md", "ARCHITECTURE.md", "AGENT_PROTOCOL.md", "COLOR_MANAGEMENT.md",
                 "PLAYBACK.md", "ANIMATION.md", "ROTO_TRACKING.md", "TIME_MODEL.md",
                 "RELEASE_NOTES.md", "3D_FOUNDATION.md", "3D_ROADMAP.md")
        for name in names:
            with self.subTest(document=name):
                self.assertEqual((root / "docs" / name).read_bytes(),
                                 (root / "nodebased" / "data" / "docs" / name).read_bytes())

    def test_live_state_requires_dispatcher_and_describes_document(self):
        with self.assertRaises(ValueError):
            knowledge.topic("live_state")
        output = knowledge.topic("live_state", Dispatcher())
        state = json.loads(output)
        self.assertIn("nodes", state)
        self.assertIn("render_errors", state)
        self.assertIsInstance(state["nodes"], dict)


if __name__ == "__main__":
    unittest.main()
