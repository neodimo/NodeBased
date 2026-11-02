"""Nuke's expression language for the Expression node (2D behaviour lane 2, step B).

C-like syntax as the Reference Guide describes it: the operators `+ - * / %`, comparisons, `&&`, `||`, `!` and the conditional `a ? b : c`,
the guide's function table, the channel names `r g b a` (and `rgba.red` and the other layer.channel forms of the rgba layer), the pixel
coordinates `x` and `y`, the centred ones `cx` and `cy`, `width`, `height`, `frame`, and the temporary variables the node's name rows define.
All the arithmetic is in 32-bit float. A blank expression is zero. Comparisons and logic give 1 or 0.

A recursive-descent parser builds a tree once and numpy evaluates it over the whole image; nothing is run as Python source.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .expressions import ExpressionError, MAX_EXPRESSION_DEPTH, MAX_EXPRESSION_LENGTH

_TOKEN = re.compile(r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
                    r"|(?P<op>&&|\|\||==|!=|<=|>=|[-+*/%()<>!?:,]))")

CHANNEL_NAMES = {"r": 0, "g": 1, "b": 2, "a": 3, "red": 0, "green": 1, "blue": 2, "alpha": 3}
_F32 = np.float32


def tokenize(text):
    out, pos = [], 0
    text = str(text)
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ExpressionError(f"unexpected character {text[pos:].strip()[:1]!r}")
        pos = match.end()
        kind = match.lastgroup
        out.append((kind, match.group(kind)))
    return out


class _Parser:
    """expression := ternary; precedence from the loosest: ?: , || , && , == != , < > <= >= , + - , * / % , unary - + ! , call/primary."""

    def __init__(self, tokens):
        self.tokens, self.at = tokens, 0

    def peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else (None, None)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None or (value is not None and token[1] != value):
            raise ExpressionError(f"expected {value!r}" if value else "unexpected end of the expression")
        self.at += 1
        return token

    def accept(self, *values):
        kind, value = self.peek()
        if kind == "op" and value in values:
            self.at += 1
            return value
        return None

    def parse(self):
        tree = self.ternary(0)
        if self.at != len(self.tokens):
            raise ExpressionError(f"unexpected {self.peek()[1]!r}")
        return tree

    def ternary(self, depth):
        if depth > MAX_EXPRESSION_DEPTH:
            raise ExpressionError("the expression is too deeply nested")
        cond = self.binary(0, depth + 1)
        if self.accept("?"):
            yes = self.ternary(depth + 1)
            self.take(":")
            no = self.ternary(depth + 1)
            return ("if", cond, yes, no)
        return cond

    _LEVELS = (("||",), ("&&",), ("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%"))

    def binary(self, level, depth):
        if level == len(self._LEVELS):
            return self.unary(depth)
        left = self.binary(level + 1, depth)
        while True:
            op = self.accept(*self._LEVELS[level])
            if op is None:
                return left
            left = ("bin", op, left, self.binary(level + 1, depth))

    def unary(self, depth):
        if depth > MAX_EXPRESSION_DEPTH:
            raise ExpressionError("the expression is too deeply nested")
        op = self.accept("-", "+", "!")
        if op:
            return ("un", op, self.unary(depth + 1))
        return self.primary(depth)

    def primary(self, depth):
        kind, value = self.peek()
        if kind == "num":
            self.at += 1
            return ("num", float(value))
        if kind == "name":
            self.at += 1
            if self.accept("("):
                args = []
                if not self.accept(")"):
                    while True:
                        args.append(self.ternary(depth + 1))
                        if self.accept(")"):
                            break
                        self.take(",")
                return ("call", value, args)
            return ("name", value)
        if self.accept("("):
            inner = self.ternary(depth + 1)
            self.take(")")
            return inner
        raise ExpressionError("unexpected end of the expression" if kind is None else f"unexpected {value!r}")


_CACHE = {}


def parse(text):
    """The tree of one expression; a blank one is zero."""
    text = str(text)
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"the expression exceeds {MAX_EXPRESSION_LENGTH} characters")
    if text.strip() == "":
        return ("num", 0.0)
    if text not in _CACHE:
        if len(_CACHE) > 512:
            _CACHE.clear()
        _CACHE[text] = _Parser(tokenize(text)).parse()
    return _CACHE[text]


def _smoothstep(a, b, x):
    t = np.clip((x - a) / np.where(b == a, 1.0, b - a), 0.0, 1.0)
    return np.where(x < a, 0.0, np.where(x >= b, 1.0, t * t * (3.0 - 2.0 * t)))


def _mantissa(x):
    return np.frexp(x)[0]


def _exponent(x):
    return np.frexp(x)[1].astype(_F32)


_UNARY = {"abs": np.abs, "fabs": np.abs, "acos": np.arccos, "asin": np.arcsin, "atan": np.arctan, "ceil": np.ceil, "floor": np.floor, "cos": np.cos,
          "cosh": np.cosh, "sin": np.sin, "sinh": np.sinh, "tan": np.tan, "tanh": np.tanh, "degrees": np.degrees, "radians": np.radians, "exp": np.exp,
          "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "int": np.trunc, "trunc": np.trunc, "rint": np.rint, "pow2": np.square,
          "exponent": _exponent, "logb": _exponent, "mantissa": _mantissa}
_BINARY = {"atan2": np.arctan2, "fmod": np.fmod, "hypot": np.hypot, "pow": np.power, "ldexp": lambda x, e: np.ldexp(x, e.astype(np.int32)), "step": lambda a, x: np.asarray(x >= a, _F32)}
_TERNARY = {"lerp": lambda a, b, x: a + (b - a) * x, "mix": lambda a, b, x: a + (b - a) * x, "smoothstep": _smoothstep}
_UNSUPPORTED = ("random", "noise", "fBm", "turbulence", "from_byte", "from_rec709", "from_sRGB", "to_byte", "to_rec709", "to_sRGB")
FUNCTION_NAMES = frozenset(_UNARY) | frozenset(_BINARY) | frozenset(_TERNARY) | {"clamp", "max", "min", "pi", "true", "false", "frame", "x", "y"}


def _truth(v):
    return np.asarray(v != 0, _F32)


def evaluate(tree, env):
    """The tree over the numpy arrays (or scalars) in `env`; the result is a float32 array or scalar."""
    kind = tree[0]
    if kind == "num":
        return _F32(tree[1])
    if kind == "name":
        name = tree[1]
        if name in env:
            return env[name]
        if name in FUNCTION_NAMES and name in ("pi", "true", "false", "frame", "x", "y"):
            return evaluate(("call", name, []), env)
        raise ExpressionError(f"unknown name {name!r}")
    if kind == "un":
        value = evaluate(tree[2], env)
        return -value if tree[1] == "-" else (value if tree[1] == "+" else np.asarray(value == 0, _F32))
    if kind == "if":
        return np.where(evaluate(tree[1], env) != 0, evaluate(tree[2], env), evaluate(tree[3], env)).astype(_F32)
    if kind == "bin":
        op = tree[1]
        a, b = evaluate(tree[2], env), evaluate(tree[3], env)
        with np.errstate(all="ignore"):
            if op == "+": return a + b
            if op == "-": return a - b
            if op == "*": return a * b
            if op == "/": return a / b
            if op == "%": return np.fmod(a, b)
            if op == "<": return np.asarray(a < b, _F32)
            if op == ">": return np.asarray(a > b, _F32)
            if op == "<=": return np.asarray(a <= b, _F32)
            if op == ">=": return np.asarray(a >= b, _F32)
            if op == "==": return np.asarray(a == b, _F32)
            if op == "!=": return np.asarray(a != b, _F32)
            if op == "&&": return np.asarray((a != 0) & (b != 0), _F32)
            return np.asarray((a != 0) | (b != 0), _F32)
    name, args = tree[1], [evaluate(a, env) for a in tree[2]]
    with np.errstate(all="ignore"):
        if name in _UNSUPPORTED:
            raise ExpressionError(f"{name}() is not supported: Nuke's {name} values cannot be reproduced here")
        if name == "pi": return _F32(math.pi)
        if name == "true": return _F32(1.0)
        if name == "false": return _F32(0.0)
        if name in ("frame", "x", "y"):
            return env[name]
        if name in ("max", "min") and args:
            fn = np.maximum if name == "max" else np.minimum
            out = args[0]
            for item in args[1:]:
                out = fn(out, item)
            return out
        if name == "clamp" and len(args) in (1, 3):
            low, high = (_F32(0.0), _F32(1.0)) if len(args) == 1 else (args[1], args[2])
            return np.minimum(np.maximum(args[0], low), high)
        if name in _UNARY and len(args) == 1: return _UNARY[name](args[0])
        if name in _BINARY and len(args) == 2: return _BINARY[name](*args)
        if name in _TERNARY and len(args) == 3: return _TERNARY[name](*args)
    if name in FUNCTION_NAMES or name in _UNARY or name in _BINARY or name in _TERNARY:
        raise ExpressionError(f"{name}() was given {len(args)} argument(s)")
    raise ExpressionError(f"unknown function {name!r}")


def names_used(tree, out=None):
    out = set() if out is None else out
    if tree[0] == "name":
        out.add(tree[1])
    elif tree[0] in ("un",):
        names_used(tree[2], out)
    elif tree[0] == "bin":
        names_used(tree[2], out), names_used(tree[3], out)
    elif tree[0] == "if":
        for part in tree[1:]:
            names_used(part, out)
    elif tree[0] == "call":
        for part in tree[2]:
            names_used(part, out)
    return out


def channel_set(text, default):
    """The rgba indices a row's channel set names: red, green, blue, alpha, rgb, rgba, all, none, rgba.red and so on, separated by spaces or commas."""
    words = [w for w in re.split(r"[\s,]+", str(text if text not in (None, "") else default).strip().lower()) if w]
    picked = set()
    for word in words:
        word = word[5:] if word.startswith("rgba.") else word
        if word in ("all", "rgba"):
            picked |= {0, 1, 2, 3}
        elif word == "rgb":
            picked |= {0, 1, 2}
        elif word == "none":
            continue
        elif word in CHANNEL_NAMES:
            picked.add(CHANNEL_NAMES[word])
        else:
            raise ExpressionError(f"channel {word!r} is not in the rgba layer")
    return sorted(picked)


def evaluate_rows(image, second, params, frame, origin=(0, 0), canvas_size=None):
    """The Expression node: temp variables first (in order), then each of the four channel rows in order, a later row overriding an earlier one."""
    h, w = image.shape[:2]
    cw, ch = canvas_size or (w, h)
    cols = (np.arange(w, dtype=np.float64) + origin[0] + 0.5).astype(_F32)
    rows = (np.arange(h, dtype=np.float64) + origin[1] + 0.5).astype(_F32)
    xx = np.broadcast_to(cols[None, :], (h, w))
    yy = np.broadcast_to((_F32(ch) - rows)[:, None], (h, w))              # Nuke's y counts up from the bottom row
    half_w = _F32(cw) / _F32(2.0)
    env = {"r": image[..., 0], "g": image[..., 1], "b": image[..., 2], "a": image[..., 3],
           "x": xx, "y": yy, "cx": (xx - half_w) / half_w, "cy": (yy - _F32(ch) / _F32(2.0)) / half_w,
           "width": _F32(cw), "height": _F32(ch), "frame": _F32(float(frame or 0))}
    for index, name in enumerate(("red", "green", "blue", "alpha")):
        env[f"rgba.{name}"] = image[..., index]
    sec = second if second is not None else np.zeros_like(image)
    for index, name in enumerate("rgba"):
        env[f"second_{name}"] = env[f"{name}2"] = sec[..., index]
    taken = set(env) | FUNCTION_NAMES
    for slot in range(4):
        name = str(params.get(f"temp_name{slot}", "") or "").strip()
        if not name:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in taken:
            raise ExpressionError(f"temporary variable name {name!r} is not usable")
        try:
            env[name] = np.asarray(evaluate(parse(params.get(f"temp_expr{slot}", "")), env), _F32)
        except ExpressionError as error:
            raise ExpressionError(f"temporary variable {name}: {error}") from None
    out = image.copy()
    for letter, default in zip("rgba", ("red", "green", "blue", "alpha")):
        text = params.get(f"expr_{letter}", "")
        try:
            value = np.asarray(evaluate(parse(text), env), _F32)
            picked = channel_set(params.get(f"expr_channels_{letter}"), default)
        except ExpressionError as error:
            raise ExpressionError(f"{letter}: {error}") from None
        value = np.broadcast_to(value, (h, w))
        for index in picked:
            out[..., index] = value
    return out
