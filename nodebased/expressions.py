"""Knob expressions: parameters driven by a formula over other parameters (schema v9).

An expression is the third and last way a parameter can get its value, after the stored base
number and a schema v6 animation curve. It is stored as a string in `document["expressions"]`,
keyed by node id then parameter name, and it resolves at the same boundary curves resolve at --
`nodebased.animation.resolve_document` -- so the rest of the pipeline only ever sees a static
document. The reason is the one written into `resolve_document`'s own docstring: the tile executor
reads `node["params"]` in a dozen places, and threading a lookup through each of them is how one
of them gets missed.

Deliberately free of NumPy and Qt, like `tiers.py` and `shapes.py`, because `core.validate` has to
be able to reject `"blur = 1/0"` without importing a renderer.

## The language

Arithmetic over floats, plus:

* `frame` -- the frame being resolved. This is what makes an expression an animation source in its
  own right: `frame * 2` is a ramp with no keys in it.
* `pi`, `e`.
* A fixed function set (`FUNCTIONS`): abs, min, max, floor, ceil, round, sqrt, pow, exp, log,
  sin, cos, tan, atan2, hypot, degrees, radians, clamp, lerp, sign.
* Comparisons and `a if c else b`, because "hold at 0 until frame 10" is a thing artists write
  constantly and the alternative is a keyed curve that means less.
* References to other parameters, in two spellings of one thing:

      knob("a1b2c3d4e5f6", "translate_x")     # canonical: any node id
      grade1.exposure                          # sugar: node ids that are valid identifiers

  Both resolve to the same `(node_id, param)` reference. The sugar exists because ids created
  through the agent CLI are usually readable (`grade1`), while ids created through the GUI are
  12 hex characters that may start with a digit and so cannot be written as an attribute at all.

References are by **node id, not node name**, so renaming a node never breaks a link. Nuke links
by name and rewrites expressions on rename; that rewrite is a lossy string edit and it is how
expressions silently detach. The UI is free to *display* names.

## What is not here, on purpose

There is no `eval`. The text is parsed with `ast.parse` and then walked against a node-type
whitelist, and anything outside it is rejected at validation time rather than at render time. No
attribute access except the reference sugar, no subscripting, no comprehensions, no lambdas, no
names beyond the three above. An expression cannot import, call a method, read a file, or see any
Python object. This matters because a `.nbp` project file is something a user opens from someone
else, which makes an expression untrusted input by default.

Integer parameters are still integers: the float result goes through
`animation.coerce_value_for_param`, which rounds and clamps to `LIMITS` exactly as a curve's
result does. An expression cannot push a parameter out of its declared range.

## One driver per parameter

A parameter may have a curve or an expression, never both. `core.validate` rejects the overlap
rather than picking a winner. A silent precedence rule is the failure mode this project keeps
designing away from: the artist keys a knob, sees nothing move, and has no way to find out why.
"""
from __future__ import annotations

import ast
import math

# Long enough for real formulas, short enough that a pathological string cannot be smuggled into a
# project file and spend meaningful time in the parser.
MAX_EXPRESSION_LENGTH = 1024

# Depth of the parsed tree. Bounded because evaluation recurses, and a 5000-deep expression is a
# stack overflow rather than an error message.
MAX_EXPRESSION_DEPTH = 64

# `**` with a large exponent is the one arithmetic operation that can burn real time on small
# input, so the exponent is bounded. The result is a float either way.
MAX_EXPONENT = 1024.0


class ExpressionError(ValueError):
    """A malformed, unsafe, unresolvable or circular expression."""


def _clamp(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _sign(value):
    return 0.0 if value == 0 else math.copysign(1.0, value)


def _log(value, base=math.e):
    return math.log(value, base)


# The complete callable surface. Every entry takes and returns plain floats.
FUNCTIONS = {
    "abs": abs, "min": min, "max": max,
    "floor": math.floor, "ceil": math.ceil, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": _log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "hypot": math.hypot, "degrees": math.degrees, "radians": math.radians,
    "fmod": math.fmod, "clamp": _clamp, "lerp": _lerp, "sign": _sign,
}

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# The name that carries the frame being resolved.
FRAME_NAME = "frame"

# The canonical reference function. `knob(id, param)` rather than a bare string so that a
# reference is syntactically distinct from an arbitrary string constant.
REFERENCE_FUNCTION = "knob"

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_COMPARE_OPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


class Expression:
    """A parsed, whitelisted expression and the parameter references it makes.

    `references` is an ordered tuple of `(node_id, param)` rather than a set so that error
    messages and the dependency graph are deterministic across runs.
    """

    __slots__ = ("text", "tree", "references", "uses_frame")

    def __init__(self, text, tree, references, uses_frame):
        self.text = text
        self.tree = tree
        self.references = references
        self.uses_frame = uses_frame

    def __repr__(self):
        return f"Expression({self.text!r})"


def _reference_from_call(node):
    """`knob("id", "param")` -> ("id", "param"). Both arguments must be literal strings."""
    if len(node.args) != 2 or node.keywords:
        raise ExpressionError(f'{REFERENCE_FUNCTION}() takes exactly two arguments: '
                              f'{REFERENCE_FUNCTION}("node id", "param")')
    parts = []
    for argument in node.args:
        if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
            raise ExpressionError(f"{REFERENCE_FUNCTION}() arguments must be literal strings; a "
                                  f"computed reference cannot be checked for cycles before it runs")
        if not argument.value:
            raise ExpressionError(f"{REFERENCE_FUNCTION}() arguments must not be empty")
        parts.append(argument.value)
    return (parts[0], parts[1])


def _check(node, depth, references):
    """Walk the tree, rejecting anything outside the whitelist and collecting references.

    Returns whether the subtree reads `frame`. Raising here -- at parse time -- rather than at
    evaluation time is the whole point: a project that fails to open is better than a project that
    renders wrong on frame 200.
    """
    if depth > MAX_EXPRESSION_DEPTH:
        raise ExpressionError(f"Expression nests deeper than {MAX_EXPRESSION_DEPTH} levels")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("Only numeric literals are allowed in an expression")
        if not math.isfinite(node.value):
            raise ExpressionError("Expression literals must be finite")
        return False

    if isinstance(node, ast.Name):
        if node.id == FRAME_NAME:
            return True
        if node.id in CONSTANTS:
            return False
        if node.id in FUNCTIONS or node.id == REFERENCE_FUNCTION:
            raise ExpressionError(f"{node.id!r} is a function and must be called")
        raise ExpressionError(f"Unknown name {node.id!r}; expressions may use "
                              f"{FRAME_NAME}, {', '.join(sorted(CONSTANTS))}, "
                              f"{REFERENCE_FUNCTION}(), a node id, and {len(FUNCTIONS)} functions")

    if isinstance(node, ast.Attribute):
        # The `grade1.exposure` sugar. Exactly one level deep: `a.b.c` is not a reference to
        # anything, and silently reading it as `knob("a.b", "c")` would invent a node id.
        if not isinstance(node.value, ast.Name):
            raise ExpressionError("A parameter reference is 'node_id.param' or "
                                  f'{REFERENCE_FUNCTION}("node id", "param")')
        references.append((node.value.id, node.attr))
        return False

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("Only named functions may be called in an expression")
        name = node.func.id
        if name == REFERENCE_FUNCTION:
            references.append(_reference_from_call(node))
            return False
        if name not in FUNCTIONS:
            raise ExpressionError(f"Unknown function {name!r}; available: "
                                  f"{', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise ExpressionError(f"{name}() does not take keyword arguments")
        uses_frame = False
        for argument in node.args:
            uses_frame |= _check(argument, depth + 1, references)
        return uses_frame

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            pass
        elif type(node.op) not in _BIN_OPS:
            raise ExpressionError(f"Unsupported operator {type(node.op).__name__}")
        return _check(node.left, depth + 1, references) | _check(node.right, depth + 1, references)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            raise ExpressionError(f"Unsupported unary operator {type(node.op).__name__}")
        return _check(node.operand, depth + 1, references)

    if isinstance(node, ast.Compare):
        if any(type(op) not in _COMPARE_OPS for op in node.ops):
            raise ExpressionError("Only ==, !=, <, <=, > and >= may be compared")
        uses_frame = _check(node.left, depth + 1, references)
        for comparator in node.comparators:
            uses_frame |= _check(comparator, depth + 1, references)
        return uses_frame

    if isinstance(node, ast.BoolOp):
        uses_frame = False
        for value in node.values:
            uses_frame |= _check(value, depth + 1, references)
        return uses_frame

    if isinstance(node, ast.IfExp):
        return (_check(node.test, depth + 1, references)
                | _check(node.body, depth + 1, references)
                | _check(node.orelse, depth + 1, references))

    raise ExpressionError(f"{type(node).__name__} is not allowed in an expression")


def parse(text):
    """Parse and whitelist `text`, returning an `Expression`.

    Raises `ExpressionError` for anything that is not a safe arithmetic formula. This is the only
    entry point that turns a string into something evaluable, so there is one place to audit.
    """
    if not isinstance(text, str):
        raise ExpressionError("An expression must be a string")
    stripped = text.strip()
    if not stripped:
        raise ExpressionError("An expression must not be empty; clear it instead")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(stripped, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise ExpressionError(f"Could not parse expression: {error}") from None
    references = []
    uses_frame = _check(tree.body, 0, references)
    # De-duplicate while keeping first-seen order.
    seen, ordered = set(), []
    for reference in references:
        if reference not in seen:
            seen.add(reference)
            ordered.append(reference)
    return Expression(stripped, tree, tuple(ordered), uses_frame)


def _evaluate(node, frame, lookup):
    if isinstance(node, ast.Constant):
        return float(node.value)

    if isinstance(node, ast.Name):
        return float(frame) if node.id == FRAME_NAME else CONSTANTS[node.id]

    if isinstance(node, ast.Attribute):
        return lookup((node.value.id, node.attr))

    if isinstance(node, ast.Call):
        if node.func.id == REFERENCE_FUNCTION:
            return lookup(_reference_from_call(node))
        arguments = [_evaluate(argument, frame, lookup) for argument in node.args]
        try:
            return float(FUNCTIONS[node.func.id](*arguments))
        except (TypeError, ValueError, OverflowError) as error:
            raise ExpressionError(f"{node.func.id}(): {error}") from None

    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, frame, lookup)
        right = _evaluate(node.right, frame, lookup)
        try:
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise ExpressionError(f"Exponent magnitude exceeds {MAX_EXPONENT}")
                result = left ** right
            else:
                result = _BIN_OPS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ExpressionError("Division by zero") from None
        except (OverflowError, ValueError) as error:
            raise ExpressionError(str(error)) from None
        if isinstance(result, complex):
            raise ExpressionError("Expression produced a complex number")
        return float(result)

    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, frame, lookup)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        return 0.0 if value else 1.0

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, frame, lookup)
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, frame, lookup)
            if not _COMPARE_OPS[type(op)](left, right):
                return 0.0
            left = right
        return 1.0

    if isinstance(node, ast.BoolOp):
        values = node.values
        if isinstance(node.op, ast.And):
            for value in values:
                result = _evaluate(value, frame, lookup)
                if not result:
                    return 0.0
            return 1.0
        for value in values:
            if _evaluate(value, frame, lookup):
                return 1.0
        return 0.0

    if isinstance(node, ast.IfExp):
        chosen = node.body if _evaluate(node.test, frame, lookup) else node.orelse
        return _evaluate(chosen, frame, lookup)

    raise ExpressionError(f"{type(node).__name__} is not allowed in an expression")


def evaluate(expression, frame, lookup):
    """Evaluate a parsed `Expression` at `frame`.

    `lookup` maps a `(node_id, param)` reference to a float and is responsible for any further
    resolution -- it is what lets a reference to another expression-driven parameter work, given
    that the caller resolves in dependency order.
    """
    if isinstance(expression, str):
        expression = parse(expression)
    # `tree` is the `ast.Expression` wrapper produced by mode="eval"; the whitelist walk and the
    # evaluator both work on the single expression node underneath it.
    result = _evaluate(expression.tree.body, frame, lookup)
    if not math.isfinite(result):
        raise ExpressionError(f"Expression {expression.text!r} produced {result}")
    return result


# --- Document-level wiring ----------------------------------------------------------------------


def iter_expressions(section):
    """Yield `(node_id, param, text)` over an `expressions` section, in a deterministic order."""
    for node_id in sorted(section):
        for param in sorted(section[node_id]):
            yield node_id, param, section[node_id][param]


def resolve_order(parsed):
    """Topologically order expression-driven parameters, innermost dependency first.

    `parsed` maps `(node_id, param)` to an `Expression`. References to parameters that are *not*
    expression-driven are leaves -- their value comes from the base params or a curve -- so they
    do not participate in the ordering.

    Raises `ExpressionError` naming the cycle. A cycle is not a render-time hang to be discovered
    on a slow frame; it is a document that must not validate.

    The walk is iterative for the same reason `core.validate`'s topological sort is: a thousand
    chained expressions is a legal document, and a recursive walk would meet Python's stack limit
    and raise `RecursionError` instead of an error an artist can act on.
    """
    order, done, on_trail = [], set(), []
    for root in sorted(parsed):
        if root in done:
            continue
        # Each stack frame is (target, iterator over its expression-driven references).
        stack = [(root, iter(parsed[root].references))]
        on_trail.append(root)
        trail_set = {root}
        while stack:
            target, references = stack[-1]
            advanced = False
            for reference in references:
                if reference not in parsed or reference in done:
                    continue
                if reference in trail_set:
                    start = on_trail.index(reference)
                    loop = " -> ".join(f"{n}.{p}" for n, p in on_trail[start:] + [reference])
                    raise ExpressionError(f"Circular expression: {loop}")
                stack.append((reference, iter(parsed[reference].references)))
                on_trail.append(reference)
                trail_set.add(reference)
                advanced = True
                break
            if advanced:
                continue
            stack.pop()
            finished = on_trail.pop()
            trail_set.discard(finished)
            done.add(finished)
            order.append(finished)
    return order


def validate_expressions(section, nodes, spec_params_for):
    """Validate `document["expressions"]` against the document's nodes.

    Returns the parsed expressions keyed by `(node_id, param)` so a caller that is about to
    resolve does not have to parse twice.

    `spec_params_for(kind)` returns that node type's `SPECS[kind]["params"]`, passed in rather
    than imported so this module stays free of a `core` import cycle.
    """
    if not isinstance(section, dict):
        raise ValueError("expressions must be an object keyed by node id")
    parsed = {}
    for node_id, params in sorted(section.items()):
        if node_id not in nodes:
            raise ValueError(f"expressions[{node_id!r}] does not name a node in this document")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"expressions[{node_id!r}] must be a non-empty object of "
                             f"parameter expressions; drop the entry instead of emptying it")
        spec = spec_params_for(nodes[node_id]["type"])
        for param, text in sorted(params.items()):
            where = f"expressions[{node_id!r}][{param!r}]"
            default = spec.get(param)
            if default is None:
                raise ValueError(f"{where}: {nodes[node_id]['type']} has no parameter {param!r}")
            if isinstance(default, str) or isinstance(default, bool):
                raise ValueError(f"{where}: only numeric parameters can be expression-driven")
            try:
                parsed[(node_id, param)] = parse(text)
            except ExpressionError as error:
                raise ValueError(f"{where}: {error}") from None

    for (node_id, param), expression in parsed.items():
        where = f"expressions[{node_id!r}][{param!r}]"
        for reference in expression.references:
            target, target_param = reference
            if target not in nodes:
                raise ValueError(f"{where} references node {target!r}, which does not exist")
            target_spec = spec_params_for(nodes[target]["type"])
            default = target_spec.get(target_param)
            if default is None:
                raise ValueError(f"{where} references {target}.{target_param}, which "
                                 f"{nodes[target]['type']} does not have")
            if isinstance(default, str) or isinstance(default, bool):
                raise ValueError(f"{where} references {target}.{target_param}, which is not "
                                 f"numeric and has no value an expression could read")

    try:
        resolve_order(parsed)
    except ExpressionError as error:
        raise ValueError(str(error)) from None
    return parsed


def resolve_nodes(nodes, section, frame, spec_params_for, coerce):
    """Return `nodes` with every expression-driven parameter baked to its value at `frame`.

    `nodes` is expected to have had its curves baked already, so an expression reading an animated
    parameter reads the animated value -- expressions layer on top of curves rather than beside
    them. `coerce(value, node_id, param)` applies the parameter's int/float type and `LIMITS`
    clamp, exactly as a curve's result is coerced.

    Returns the input unchanged when there is nothing to do, so untouched graphs keep their exact
    cache keys.
    """
    if not section:
        return nodes
    parsed = {}
    for node_id, param, text in iter_expressions(section):
        node = nodes.get(node_id)
        if node is None or param not in node["params"]:
            continue
        # Same eligibility rule curves use: numeric spec defaults only, bool excluded.
        spec_default = spec_params_for(node["type"]).get(param)
        if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
            continue
        parsed[(node_id, param)] = parse(text)
    if not parsed:
        return nodes

    resolved = dict(nodes)
    values = {}

    def lookup(reference):
        if reference in values:
            return values[reference]
        target, param = reference
        node = resolved.get(target)
        if node is None:
            raise ExpressionError(f"Reference to {target}.{param}: node does not exist")
        try:
            return float(node["params"][param])
        except (KeyError, TypeError, ValueError):
            raise ExpressionError(f"Reference to {target}.{param}: not a numeric parameter") from None

    for target in resolve_order(parsed):
        node_id, param = target
        value = evaluate(parsed[target], frame, lookup)
        value = coerce(value, node_id, param)
        values[target] = float(value)
        node = resolved[node_id]
        resolved[node_id] = {**node, "params": {**node["params"], param: value}}
    return resolved
