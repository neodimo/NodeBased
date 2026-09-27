"""Vectorised, restricted expression evaluation for the Expression image node."""
import ast
import operator
import numpy as np

from .expressions import ExpressionError, FUNCTIONS, CONSTANTS, MAX_EXPRESSION_LENGTH, MAX_EXPRESSION_DEPTH, MAX_EXPONENT

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: np.logical_not}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
VFUN = {"abs": np.abs, "min": np.minimum, "max": np.maximum, "floor": np.floor,
        "ceil": np.ceil, "round": np.round, "sqrt": np.sqrt, "pow": np.power,
        "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "atan2": np.arctan2, "hypot": np.hypot, "degrees": np.degrees,
        "radians": np.radians, "clamp": lambda x, a, b: np.minimum(np.maximum(x, a), b),
        "lerp": lambda a, b, t: a * (1-t) + b*t, "sign": np.sign}
VARS = set("rgba") | set("xy") | {"width", "height", "frame", "second_r", "second_g", "second_b", "second_a", "r2", "g2", "b2", "a2"}

def _eval(n, env, depth=0):
    if depth > MAX_EXPRESSION_DEPTH: raise ExpressionError("Expression is too deeply nested")
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return float(n.value)
    if isinstance(n, ast.Name):
        if n.id in env: return env[n.id]
        if n.id in CONSTANTS: return CONSTANTS[n.id]
        raise ExpressionError(f"Unknown image-expression variable {n.id!r}")
    if isinstance(n, ast.BinOp) and type(n.op) in BIN:
        a, b = _eval(n.left, env, depth+1), _eval(n.right, env, depth+1)
        if isinstance(n.op, ast.Pow) and np.any(np.abs(b) > MAX_EXPONENT): raise ExpressionError("Exponent magnitude exceeds limit")
        with np.errstate(all="ignore"): return BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in UNARY: return UNARY[type(n.op)](_eval(n.operand, env, depth+1))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in VFUN and not n.keywords:
        args = [_eval(a, env, depth+1) for a in n.args]
        with np.errstate(all="ignore"): return VFUN[n.func.id](*args)
    if isinstance(n, ast.Compare):
        left = _eval(n.left, env, depth+1); result = True
        for op, item in zip(n.ops, n.comparators):
            if type(op) not in CMP: raise ExpressionError("Unsupported comparison")
            right = _eval(item, env, depth+1); result = np.logical_and(result, CMP[type(op)](left, right)); left = right
        return result.astype(np.float32) if hasattr(result, "astype") else float(result)
    if isinstance(n, ast.IfExp): return np.where(_eval(n.test, env, depth+1), _eval(n.body, env, depth+1), _eval(n.orelse, env, depth+1))
    if isinstance(n, ast.BoolOp):
        vals = [_eval(v, env, depth+1) for v in n.values]
        op = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
        out = vals[0]
        for val in vals[1:]: out = op(out, val)
        return out
    raise ExpressionError(f"Unsupported expression construct: {type(n).__name__}")

def evaluate_channels(image, second, params, frame, origin=(0, 0), canvas_size=None):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    env = {c: image[..., i] for i, c in enumerate("rgba")}
    env.update({f"second_{c}": (second[..., i] if second is not None else np.zeros((h, w), np.float32)) for i, c in enumerate("rgba")})
    env.update({f"{c}2": env[f"second_{c}"] for c in "rgba"})
    cw, ch = canvas_size or (w, h)
    env.update(x=xx + origin[0], y=yy + origin[1], width=cw, height=ch, frame=float(frame or 0))
    out = image.copy()
    for i, c in enumerate("rgba"):
        text = str(params[f"expr_{c}"])
        if len(text) > MAX_EXPRESSION_LENGTH: raise ExpressionError(f"{c}: expression exceeds {MAX_EXPRESSION_LENGTH} characters")
        try: tree = ast.parse(text, mode="eval")
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc: raise ExpressionError(f"{c}: {exc}") from None
        # Walk every node before execution; the evaluator below only accepts the same safe subset.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp, ast.DictComp, ast.Import)):
                raise ExpressionError(f"{c}: unsupported expression construct {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in VARS | set(CONSTANTS) | set(VFUN):
                raise ExpressionError(f"{c}: unknown variable {node.id!r}")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in VFUN):
                raise ExpressionError(f"{c}: unsupported function")
        try: value = np.asarray(_eval(tree.body, env), dtype=np.float32)
        except (TypeError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as exc: raise ExpressionError(f"{c}: {exc}") from None
        if value.ndim == 0: value = np.full((h, w), value, np.float32)
        if value.shape != (h, w) or not np.all(np.isfinite(value)):
            raise ExpressionError(f"{c}: expression produced invalid pixels")
        out[..., i] = value
    return out
