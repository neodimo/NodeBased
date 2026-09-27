import unittest

from nodebased.core import SPECS
from nodebased.nodecatalog import NODE_CATEGORIES, node_category, node_description


class NodeCatalogTests(unittest.TestCase):
    """Every SPECS type is filed in exactly one category with a description, so a lane that ships
    a node without filing it here fails this suite instead of leaving it hidden from the NODES dock."""

    def test_every_spec_is_in_exactly_one_category_with_a_description(self):
        seen = {}
        for category, kinds in NODE_CATEGORIES.items():
            for kind, description in kinds.items():
                self.assertNotIn(kind, seen,
                                 f"{kind} is filed in both {seen.get(kind)} and {category}")
                seen[kind] = category
                self.assertIsInstance(description, str)
                self.assertTrue(description.strip(), f"{kind} has no description")
        missing = set(SPECS) - set(seen)
        self.assertEqual(missing, set(), f"SPECS types missing from NODE_CATEGORIES: {sorted(missing)}")
        extra = set(seen) - set(SPECS)
        self.assertEqual(extra, set(), f"NODE_CATEGORIES types not in SPECS: {sorted(extra)}")

    def test_node_category_and_description_agree_with_the_catalog(self):
        for category, kinds in NODE_CATEGORIES.items():
            for kind, description in kinds.items():
                self.assertEqual(node_category(kind), category)
                self.assertEqual(node_description(kind), description)

    def test_unknown_kind_falls_back_to_other_and_empty_description(self):
        self.assertEqual(node_category("NotARealNode"), "Other")
        self.assertEqual(node_description("NotARealNode"), "")


if __name__ == "__main__":
    unittest.main()
