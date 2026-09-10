# Animation curves — design contract

This document describes the shape and semantics of the animation system as it ships today
(curve-only, two interpolations, numeric parameters) and how the same shape admits Bezier
tangents, expressions, a curve editor, clip time mappings, and non-numeric values later.
Reading this is optional for callers; the implementation is the source of truth, and this file
documents *why* the shape looks the way it does.

## Storage shape (schema v6)

```json
{
  "version": 6,
  "nodes":  { "<id>": { "type": "Grade", "params": { "exposure": 0.0, ... }, ... } },
  "view":   "<id>",
  "time":   { "first": 1, "last": 24, "current": 1, "fps": 24.0 },
  "animation": {
    "curves": {
      "<node_id>": {
        "<param_name>": {
          "interpolation": "constant | linear",
          "keys": [
            { "frame": <int>, "value": <number> },
            ...
          ]
        }
      }
    }
  }
}
```

A few hard rules, enforced by ``nodebased.animation.validate_curve`` and ``nodebased.core.validate``:

* ``animation.curves`` is the only place curve data lives. Node ``params`` are never
  mutated by an animation edit.
* Curve values must be finite numbers. Frames must be integers, strictly increasing, and
  inside ``FRAME_LIMITS``.
* Only numeric ``SPECS`` params are animatable. ``bool``, ``str``, and choice enums are
  not (the latter would need a separate interpolation contract).
* Each ``(node_id, param)`` slot holds *one* curve at a time. Editing replaces it
  (``set_key`` may add or replace a key, ``delete_key`` removes one key, ``clear_curve``
  removes the whole slot).

## Resolution semantics (per-frame, per-node)

``nodebased.animation.resolve_params`` is the single source of truth. Given a node, its
``animation_section``, the current frame, the node's spec params, and ``LIMITS``, it returns
a new params dict with curve overrides applied.

* **Before the first key / after the last key** — the curve is a no-op; the node's stored
  ``params`` value applies. This is what makes "no animation" the cheap default and
  keeps the static graph rendering byte-identically to a v5 document.
* **At an exact key frame** — that key's value, after type coercion.
* **Between keys (linear)** — ``v0 + (v1 - v0) * (frame - f0) / (f1 - f0)``.
* **Between keys (constant)** — the previous key's value (step-and-hold).
* **Type coercion at resolution** — if the param's spec default is an integer, the resolved
  value rounds to ``int(round(value))`` and then clamps to ``LIMITS``. Floats stay float
  and clamp to ``LIMITS``. This means a user can store ``1.6`` on a curve for an int
  parameter and the evaluator rounds it consistently at every frame.
* **No cache invalidation on shape** — the evaluator hashes the resolved params, not
  the stored params. Static nodes (no curves, or curves whose range doesn't cover the
  current frame) hash to the same digest they had pre-v6, so the existing cache stays
  warm.

## Determinism

Animation does not add any nondeterministic surface to the serialized document. Two
properties matter:

1. **Editing a curve does not change ``node["params"]``.** Two documents with the same
   ``nodes`` and ``animation`` hash to the same resolved graph at the same frame.
2. **Order-independent evaluation.** ``resolve_params`` reads only the curve for one
   ``(node, param)`` slot, never multiple curves at once. Multi-param animations are
   independent curves composited at evaluation time.

The evaluator's cache key includes the frame and the *resolved* params (post-curve),
not the stored params. This means: same frame + same stored params + same curves ⇒ same
cache entry. The cache is also frame-keyed implicitly via the resolved params digest,
so a node whose resolved value differs across frames re-keys naturally without separate
frame bookkeeping in the cache.

## Dispatcher operations (machine-discoverable)

Three new atomic, undoable ops. Each takes one undo slot and either succeeds or rolls
back the document to its prior state.

| Op           | Required fields                                              | Effect                                                                  |
| ------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| ``set_key``  | ``id``, ``param``, ``frame``, ``value``                      | Insert or replace the key at ``frame``; sort; validate; set interpolation if new. |
| ``delete_key`` | ``id``, ``param``, ``frame``                                | Remove the key at ``frame``; if the curve is empty, drop the slot.    |
| ``clear_curve`` | ``id``, ``param``                                          | Remove the whole ``(node, param)`` curve.                              |

``describe`` advertises each op's field names and types so an agent doesn't have to read
the source.

Errors are explicit and machine-greppable. Examples (not exhaustive):

* ``animation set_key: value 100 outside [-20, 20] for 'exposure'``
* ``animation set_key: 'frame' must be an integer``
* ``animation delete_key: no curve on 'g'.'exposure'``
* ``animation.curves[...].<param>: only numeric parameters can be animated``

## How the design admits later features

### Bezier tangents

The curve envelope becomes

```json
{ "interpolation": "bezier",
  "keys": [
    { "frame": 0, "value": 0.0, "in_tangent": [-2, 1.0], "out_tangent": [3, 1.0] },
    ...
  ] }
```

``evaluate_curve`` gains a ``bezier`` branch; ``merge_key`` accepts the new shape;
``validate_curve`` enforces tangent structure. ``CURVE_INTERPOLATIONS`` adds
``"bezier"``; ``describe`` advertises it. The same ``resolve_params`` entry point does
not change.

### Expressions

A curve can grow an optional ``"expression"`` field that takes precedence over
``"keys"`` at resolution time:

```json
{ "interpolation": "linear",
  "expression": "math.sin(frame * 0.1) * 0.5",
  "keys": [] }
```

``evaluate_curve`` tries ``expression`` first, then falls back to key-based interpolation.
``validate_curve`` does a syntactic parse of the expression (full sandboxing is a
follow-on).

### Clip time mappings

Clips are document-level (one document can host several compositions). Each clip gets
its own ``(first, last)`` and a per-node ``clip_frame_offset``. ``resolve_params`` would
accept an extra ``frame`` argument that's already mapped through the clip's range before
the curve is evaluated. No change to the curve shape itself; only the resolved frame
that gets fed to ``evaluate_curve``.

### Nonnumeric values (string paths, choices, colors)

Currently only numeric ``SPECS`` params are animatable because ``resolve_params`` calls
``coerce_value_for_param`` which only handles numbers. The path to nonnumeric animation
is to introduce per-type interpolators (string concat, choice-tween, color-rgb-lerp) and
register them on the curve's ``interpolation`` field, e.g.
``"interpolation": "color_rgba"``. Each adds a branch in ``evaluate_curve`` and a
spec-validation rule; nothing else moves.

### Curve editor / dopesheet / playback

These are UI concerns. The data model already gives them everything they need:

* per-node, per-param ``(interpolation, keys)``
* a project-wide time range in ``time.first / time.last``
* an agent command surface that round-trips every edit

A dopesheet renders the ``curves`` flat against the timeline; a curve editor edits a
single ``(node, param)`` slot's keys; playback scrubs ``time.current``. None of those
need new schema.

## Out of scope (deliberately)

* **Auto-tangents / curve fitting / smoothing.** These are user-driven authoring choices
  and belong in the curve editor.
* **Pre-roll / post-roll hold.** The current spec uses "out-of-range = base value". A
  user who wants hold-at-end adds a key at the desired end frame. If hold-at-end becomes
  a real requirement later, an ``extrapolation`` field on the curve admits it without a
  schema bump.
* **Per-parameter time remapping.** A frame offset per param (e.g., stagger identical
  effects across frames) is handled at the node level by reading a global time mapping;
  not needed at the curve level today.
* **Animation of inputs (wiring animation).** Inputs are graph topology; a future
  ``animated_wires`` slot is a separate concern, not part of this layer.
