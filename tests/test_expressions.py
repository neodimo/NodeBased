"""Expression parsing, dependency resolution, and the Dispatcher expression boundary."""
import math
import unittest

from nodebased.core import Dispatcher, validate
from nodebased.expressions import ExpressionError, evaluate, parse


def expression_graph():
    dispatcher = Dispatcher()
    dispatcher.execute({"op": "batch", "commands": [
        {"op": "create", "id": "source", "type": "Constant",
         "params": {"width": 2, "height": 2, "red": 0.2, "green": 0.3,
                     "blue": 0.4, "alpha": 1.0}},
        {"op": "create", "id": "grade", "type": "Grade",
         "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
        {"op": "connect", "id": "grade", "input": "image", "source": "source"},
    ]})
    return dispatcher


class ExpressionParserTests(unittest.TestCase):
    def test_arithmetic_functions_and_frame_are_evaluated_without_python_eval(self):
        formula = parse("clamp(frame * 2 + sqrt(9), 0, 20)")
        self.assertEqual(evaluate(formula, 4, lambda ref: 0), 11.0)
        self.assertTrue(formula.uses_frame)
        self.assertEqual(formula.references, ())

    def test_parameter_references_are_collected_in_stable_order(self):
        formula = parse('knob("source", "red") + source.alpha + source.red')
        self.assertEqual(formula.references,
                         (("source", "red"), ("source", "alpha")))

    def test_unsafe_or_invalid_syntax_is_rejected(self):
        for formula in ("__import__('os')", "source.params.foo", "[1, 2][0]", "open('x')"):
            with self.subTest(formula=formula):
                with self.assertRaises(ExpressionError):
                    parse(formula)

    def test_runtime_math_error_is_reported_as_expression_error(self):
        with self.assertRaisesRegex(ExpressionError, "Division by zero"):
            evaluate(parse("1 / 0"), 1, lambda ref: 0)
        with self.assertRaisesRegex(ExpressionError, "math range"):
            evaluate(parse("exp(10000)"), 1, lambda ref: 0)


class ExpressionDocumentTests(unittest.TestCase):
    def test_expression_resolves_at_frame_and_preserves_stored_base(self):
        dispatcher = expression_graph()
        dispatcher.execute({"op": "set_expression", "id": "grade", "param": "exposure",
                            "expression": "frame * 0.5"})
        document = dispatcher.document
        self.assertEqual(document["nodes"]["grade"]["params"]["exposure"], 0.0)
        from nodebased.animation import resolve_document
        self.assertEqual(resolve_document(document, 6)["nodes"]["grade"]["params"]["exposure"], 3.0)
        self.assertEqual(resolve_document(document, 10)["nodes"]["grade"]["params"]["exposure"], 5.0)
        validate(document)

    def test_chained_reference_reads_the_other_expression_value(self):
        dispatcher = expression_graph()
        dispatcher.execute({"op": "set_expression", "id": "grade", "param": "exposure",
                            "expression": "source.red * 10"})
        dispatcher.execute({"op": "set_expression", "id": "grade", "param": "multiply",
                            "expression": "grade.exposure / 2"})
        from nodebased.animation import resolve_document
        resolved = resolve_document(dispatcher.document, 1)["nodes"]["grade"]["params"]
        self.assertEqual(resolved["exposure"], 2.0)
        self.assertEqual(resolved["multiply"], 1.0)

    def test_bad_reference_and_cycle_are_rejected_atomically(self):
        dispatcher = expression_graph()
        before = dispatcher.document
        with self.assertRaisesRegex(ValueError, "does not exist"):
            dispatcher.execute({"op": "set_expression", "id": "grade", "param": "exposure",
                                "expression": 'knob("missing", "red")'})
        self.assertEqual(dispatcher.document, before)
        dispatcher.execute({"op": "set_expression", "id": "grade", "param": "exposure",
                            "expression": "grade.multiply + 1"})
        with self.assertRaisesRegex(ValueError, "Circular expression"):
            dispatcher.execute({"op": "set_expression", "id": "grade", "param": "multiply",
                                "expression": "grade.exposure + 1"})
        self.assertNotIn("multiply", dispatcher.document["expressions"].get("grade", {}))

    def test_expression_and_curve_are_one_driver_and_set_clear_are_undoable(self):
        dispatcher = expression_graph()
        dispatcher.execute({"op": "set_expression", "id": "grade", "param": "exposure",
                            "expression": "frame"})
        with self.assertRaisesRegex(ValueError, "driven by an expression"):
            dispatcher.execute({"op": "set_key", "id": "grade", "param": "exposure",
                                "frame": 1, "value": 1.0})
        dispatcher.execute({"op": "clear_expression", "id": "grade", "param": "exposure"})
        self.assertEqual(dispatcher.document["expressions"], {})
        dispatcher.execute({"op": "undo"})
        self.assertEqual(dispatcher.document["expressions"]["grade"]["exposure"], "frame")
        dispatcher.execute({"op": "redo"})
        self.assertEqual(dispatcher.document["expressions"], {})

    def test_integer_parameter_expression_is_coerced_to_integer_and_limits_apply(self):
        dispatcher = expression_graph()
        dispatcher.execute({"op": "create", "id": "switch", "type": "Switch"})
        dispatcher.execute({"op": "set_expression", "id": "switch", "param": "which",
                            "expression": "2"})
        from nodebased.animation import resolve_document
        resolved = resolve_document(dispatcher.document, 1)["nodes"]["switch"]["params"]
        self.assertEqual(resolved["which"], 1)


if __name__ == "__main__":
    unittest.main()
