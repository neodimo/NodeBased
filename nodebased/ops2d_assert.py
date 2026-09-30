"""Restricted, scalar condition evaluation for the Assert QA node.

Mirrors `ops2d_expression.py`'s AST-whitelist safety shape (itself built on `expressions.py`'s
scalar `FUNCTIONS`/`CONSTANTS`), but the result is one Python bool over per-image pixel
statistics rather than a per-pixel NumPy array: Assert checks something true of the *whole*
frame ("did the average stay in range"), never paints one.
"""
import ast

import numpy as np

from .expressions import CONSTANTS, FUNCTIONS, ExpressionError, MAX_EXPRESSION_DEPTH, MAX_EXPRESSION_LENGTH, MAX_EXPONENT

_STAT_VARS = {f"{c}_{stat}" for c in "rgba" for stat in ("avg", "min", "max")} | {"avg", "min", "max"}
VARS = _STAT_VARS | {"frame", "width", "height"}

_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b}
_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a, ast.Not: lambda a: not a}
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}


def _pixel_stats(pixels):
    env = {}
    for i, c in enumerate("rgba"):
        channel = pixels[..., i]
        env[f"{c}_avg"] = float(np.mean(channel))
        env[f"{c}_min"] = float(np.min(channel))
        env[f"{c}_max"] = float(np.max(channel))
    rgb = pixels[..., :3]
    env["avg"] = float(np.mean(rgb))
    env["min"] = float(np.min(rgb))
    env["max"] = float(np.max(rgb))
    return env


def _eval(node, env, depth=0):
    if depth > MAX_EXPRESSION_DEPTH:
        raise ExpressionError("Condition is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExpressionError(f"Unknown Assert variable {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, env, depth + 1), _eval(node.right, env, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(b) > MAX_EXPONENT:
            raise ExpressionError("Exponent magnitude exceeds limit")
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, env, depth + 1))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        args = [_eval(a, env, depth + 1) for a in node.args]
        return FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Compare):
        left, result = _eval(node.left, env, depth + 1), True
        for op, item in zip(node.ops, node.comparators):
            if type(op) not in _CMP:
                raise ExpressionError("Unsupported comparison")
            right = _eval(item, env, depth + 1)
            result = result and _CMP[type(op)](left, right)
            left = right
        return result
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env, depth + 1) if _eval(node.test, env, depth + 1) else _eval(node.orelse, env, depth + 1)
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env, depth + 1) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    raise ExpressionError(f"Unsupported condition construct: {type(node).__name__}")


def evaluate_condition(text, pixels, frame):
    """True/False for `text` (a restricted boolean expression) over `pixels`' own statistics."""
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Condition exceeds {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise ExpressionError(f"Condition: {error}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp, ast.DictComp, ast.Import)):
            raise ExpressionError(f"Condition: unsupported construct {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in VARS | set(CONSTANTS) | set(FUNCTIONS):
            raise ExpressionError(f"Condition: unknown variable {node.id!r}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS):
            raise ExpressionError("Condition: unsupported function")
    env = _pixel_stats(pixels)
    h, w = pixels.shape[:2]
    env.update(frame=float(frame or 0), width=float(w), height=float(h))
    try:
        return bool(_eval(tree.body, env))
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as error:
        raise ExpressionError(f"Condition: {error}") from None
