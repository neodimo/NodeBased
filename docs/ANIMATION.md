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

* **Endpoint hold (Nuke-style extrapolation).** Before the first key the curve returns the
  first key's value; after the last key it returns the last key's value. Both ``constant``
  and ``linear`` interpolations use this rule. A present curve always overrides the
  node's stored ``params`` for every frame — the curve is not a patch on top of the base
  for out-of-range frames.
* **At an exact key frame** — that key's value, after type coercion.
* **Between keys (linear)** — ``v0 + (v1 - v0) * (frame - f0) / (f1 - f0)``.
* **Between keys (constant)** — the previous key's value (step-and-hold).
* **No curve at all** — the node renders from its stored ``params`` byte-identically to a
  v5 document. That is the path that keeps pre-v6 graphs stable: the upgrade only adds
  the ``animation`` field; an empty curves section is a no-op for the evaluator.
* **Type coercion at resolution** — if the param's spec default is an integer, the resolved
  value rounds to ``int(round(value))`` and then clamps to ``LIMITS``. Floats stay float
  and clamp to ``LIMITS``. This means a user can store ``1.6`` on a curve for an int
  parameter and the evaluator rounds it consistently at every frame.
* **No cache invalidation on shape** — the evaluator hashes the resolved params, not
  the stored params. Two frames where ``resolve_params`` produces equal dicts share a
  cache entry; an animated node re-keys naturally when its resolved value differs across
  frames.

## Crossing into the tile executor

``resolve_params`` covers the reference evaluator, which resolves curves node-by-node as it
walks the chain. The tile executor cannot use that shape: it reads ``node["params"]`` from a
dozen independent sites — content digest, canvas size, Read bounds, source generation,
Switch branch selection, kernel dispatch — and any site that skipped a curve lookup would
produce a parameter frozen at its stored base value. That failure is silent. The graph
animates through the reference evaluator, freezes through tiles, and tiles are the viewer's
default path since v0.9.

``nodebased.animation.resolve_document`` closes it by baking curves once, at the
``TileExecutor`` API boundary (``compose_region``, ``canvas_size``, ``canvas_region``).
Everything downstream sees a static document and cannot miss a site.

* A document with no curves is returned **unchanged by identity**, as is an animated
  document on a frame where every curve resolves to the stored value. Unanimated graphs
  therefore keep their exact cache keys; resolution costs them a dict lookup.
* The returned copy carries an empty curve set, so a second resolve is a no-op rather than
  a re-evaluation of the same frame.
* **Ordering:** curve resolution runs *before* ``tiers.scale_params`` on both paths. A
  pixel-unit parameter such as a Blur radius must be scaled from the value the frame
  actually uses; scaling the stored base value would proxy an animated blur at the wrong
  size.

Pinned by ``AnimationThroughTileExecutorTests`` in ``tests/test_animation.py``: tile output
matches the reference at frames 1/5/10 across tiers 1/2/4, and the frames are asserted to
actually differ, so a frozen parameter cannot pass by matching an equally frozen reference.

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

The current validator (``validate_curve`` + ``validate``) strictly requires the exact
shape above: ``{interpolation, keys: [{frame, value}]}``. Adding a Bezier ``in_tangent``
field, an ``expression`` string, a clip-time offset, or a non-numeric value type is
**not** "additive" without further work — it is a schema and protocol evolution that
needs at least:

1. A new curve or key shape recognised by ``validate_curve`` (and rejected otherwise).
2. A bumped document schema (``SCHEMA_VERSION`` -> N+1) plus an ``upgrade_document``
   step that fills the new shape with backwards-compatible defaults for old docs.
3. A bumped ``describe`` response that advertises the new interpolation / field names so
   agents and the inspector can discover them without reading the source.
4. New tests covering the new shape, the upgrade path, and the discoverability response.

The paragraphs below describe the *intent* for each follow-on, not a claim that the
current code accepts them. Each path is additive to ``evaluate_curve`` and
``resolve_params`` (the resolver entry point does not change), but it is not additive to
the validator or the agent protocol — those need explicit shape and version bumps.

### Bezier tangents

The curve envelope would become

```json
{ "interpolation": "bezier",
  "keys": [
    { "frame": 0, "value": 0.0, "in_tangent": [-2, 1.0], "out_tangent": [3, 1.0] },
    ...
  ] }
```

``evaluate_curve`` gains a ``bezier`` branch; ``merge_key`` accepts the new shape;
``validate_curve`` enforces tangent structure. ``CURVE_INTERPOLATIONS`` adds
``"bezier"``; ``describe`` advertises it. Requires a schema bump.

### Expressions

A curve would grow an optional ``"expression"`` field that takes precedence over
``"keys"`` at resolution time:

```json
{ "interpolation": "linear",
  "expression": "math.sin(frame * 0.1) * 0.5",
  "keys": [] }
```

``evaluate_curve`` would try ``expression`` first, then fall back to key-based
interpolation. ``validate_curve`` would do a syntactic parse of the expression (full
sandboxing is a follow-on). Requires a schema bump and a sandbox policy.

### Clip time mappings

Clips would live at the document level (one document hosts several compositions). Each
clip gets its own ``(first, last)`` and a per-node ``clip_frame_offset``. ``resolve_params``
would accept an extra ``frame`` argument that's already mapped through the clip's range
before the curve is evaluated. The curve shape itself doesn't change, but the document
does (new ``clips`` slot), and the agent protocol needs a new ``clip`` op set.

### Nonnumeric values (string paths, choices, colors)

Currently only numeric ``SPECS`` params are animatable because ``resolve_params`` calls
``coerce_value_for_param`` which only handles numbers. The path to nonnumeric animation
is per-type interpolators (string concat, choice-tween, color-rgb-lerp) registered on
the curve's ``interpolation`` field (e.g. ``"interpolation": "color_rgba"``), plus a
validator pass that admits them only on the matching param kinds. Each adds a branch in
``evaluate_curve`` and a spec-validation rule; the curve shape still needs a schema bump
if any new field is required (e.g. per-channel tangents for color).

### Curve editor / dopesheet / playback

These are UI concerns. The data model already gives them everything they need once the
``animation`` section exists:

* per-node, per-param ``(interpolation, keys)``
* a project-wide time range in ``time.first / time.last``
* an agent command surface that round-trips every edit

A dopesheet renders the ``curves`` flat against the timeline; a curve editor edits a
single ``(node, param)`` slot's keys; playback scrubs ``time.current``. None of those
need new schema — they are pure UI layers over the existing shape. The playback loop
already exists (see ``nodebased/playback.py`` in the main worktree); the animation
data is consumed by the same evaluator and integrates with the existing time range
without further protocol work.

## Out of scope (deliberately)

* **Auto-tangents / curve fitting / smoothing.** These are user-driven authoring choices
  and belong in the curve editor.
* **Per-parameter extrapolation policy.** The current spec hard-codes endpoint hold
  (Nuke default). A future ``extrapolation`` field (``constant``, ``linear``,
  ``cycle``, ``pingpong``) would admit other policies, but it would require a schema
  bump — the curve shape gains an ``extrapolation`` key that must be advertised in
  ``describe`` and accepted by ``validate_curve``.
* **Per-parameter time remapping.** A frame offset per param (e.g., stagger identical
  effects across frames) is handled at the node level by reading a global time mapping;
  not needed at the curve level today.
* **Animation of inputs (wiring animation).** Inputs are graph topology; a future
  ``animated_wires`` slot is a separate concern, not part of this layer.
