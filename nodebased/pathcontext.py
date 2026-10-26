"""Nuke's context variables on Read and Write (fidelity step F): four key / value pairs that fill `%{key}` in the file path.

A Read whose path is `/shots/%{SHOT}/plate.%04d.exr` with `key1` = SHOT and `value1` = sh010 reads `/shots/sh010/plate.%04d.exr`; the
same pairs serve a Write. A pair with an empty key does nothing; a `%{name}` no pair names stays as typed, so the missing file
is named in the error with its variable still showing. The substitution runs where a node's parameters are resolved for a frame
(`animation.resolve_params`), so the evaluator, the tile executor and the digests all see the substituted path; the stored path
keeps the variables. The `%{...}` spelling is NodeBased's reading of the guide, which gives the four key / value knobs but not
the syntax that uses them.
"""
from __future__ import annotations

import re

COUNT = 4
PARAMS = {name: "" for index in range(1, COUNT + 1) for name in (f"key{index}", f"value{index}")}
KINDS = ("Read", "Write")
_VARIABLE = re.compile(r"%\{([^{}%]+)\}")


def pairs(params):
    """The (key, value) pairs in use: those whose key is not empty."""
    out = []
    for index in range(1, COUNT + 1):
        key = str(params.get(f"key{index}", "") or "").strip()
        if key:
            out.append((key, str(params.get(f"value{index}", "") or "")))
    return out


def uses(params):
    return bool(pairs(params))


def substitute(text, params):
    """`text` with every `%{key}` a pair names replaced by its value (the first pair wins when a key repeats)."""
    table = {}
    for key, value in pairs(params):
        table.setdefault(key, value)
    if not table or "%{" not in str(text):
        return text
    return _VARIABLE.sub(lambda match: table.get(match.group(1).strip(), match.group(0)), str(text))


def apply(kind, params):
    """`params` with the path filled in for a Read or Write that has context pairs; the same dict otherwise."""
    if kind in KINDS and uses(params) and isinstance(params.get("path"), str):
        substituted = substitute(params["path"], params)
        if substituted != params["path"]:
            return {**params, "path": substituted}
    return params
