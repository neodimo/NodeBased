# Roto, tracking and channel routing — design contract

Status: contract written 2026-09-10 before implementation; **implemented on
`main` 2026-09-12** and covered by `tests/test_roto.py`. Judged the way
`docs/PLAYBACK.md` and `docs/EVALUATION_TIERS.md` were judged — the contract is
fixed first, then the tests prove each clause.

Scope: three node types (`Roto`, `Tracker`, `ChannelShuffle`), one schema step
(**v8** `node_data`), and the ROI/proxy declarations they owe
`docs/EVALUATION_TIERS.md`.

This was designed against a tree where schema v6 (animation curves) was the tip,
so the contract calls the new section v7. By the time it landed, `main` had
already spent v7 on project settings (OCIO/ACES and viewer background), so
`node_data` is **v8** and the chain runs v6 → v7 (settings) → v8 (`node_data`).
A v7 document that already carries a `settings` section upgrades without losing
it. Read "v7" below as "v8" wherever it names this section.

## Why these three together

Roto and tracking are the two M1 features that cannot be expressed as a pure
function of `(params, inputs, frame)`. A grade is four numbers; a roto shape is
a variable-length list of points, each of which is independently animatable, and
a tracker is a variable-length list of per-frame sampled positions. Both need
somewhere in the document to live that is not `node["params"]`, and both need
that storage to interpolate over time.

That is the same problem parameter animation solves, so the rule for this pass
is: **do not invent a second animation representation.** Every time-varying
number in a roto shape or a track is expressed with the *same* curve envelope
schema v6 defines for parameters, and resolves with the same semantics.

`ChannelShuffle` is here because a matte is only useful if it can be routed into
another image's alpha, and the existing `Shuffle` node cannot do that — it has
one input. See "Relationship to the existing Shuffle node" below.

## Storage shape (schema v8)

v8 assumes v6 (`animation.curves`) and v7 (`settings`) already exist and adds
one top-level section:

```json
{
  "version": 8,
  "nodes":  { "<id>": { "type": "Roto", "params": {...}, ... } },
  "view":   "<id>",
  "time":   { "first": 1, "last": 24, "current": 1, "fps": 24.0 },
  "animation": { "curves": { ... } },
  "settings":  { "color": { ... }, "viewer": { ... } },
  "node_data": {
    "<roto_node_id>":    { "shapes": [ <shape>, ... ] },
    "<tracker_node_id>": { "tracks": [ <track>, ... ] }
  }
}
```

`node_data` is a generic per-node side-data section, keyed by node id exactly
the way `animation.curves` is. It exists as one section rather than a
`roto:`/`tracking:` pair so that the next node type with structured payload —
a paint stroke list, a generative operator's conditioning set — adds a payload
schema rather than a new top-level document key.

Hard rules, enforced by `nodebased.core.validate`:

* A `node_data` key must name a node that exists. Orphan entries are an error,
  not garbage collected silently.
* The payload shape is decided by the node's `type`. `Roto` takes exactly
  `{"shapes": [...]}`; `Tracker` takes exactly `{"tracks": [...]}`. Any other
  node type appearing in `node_data` is an error.
* An absent entry means "empty", the same way an absent curve means "use the
  stored parameter". This keeps a freshly created Roto out of the document
  until it has geometry.
* `delete` drops the node's `node_data` entry in the same atomic edit.

### Animatable scalars

Every number inside a shape or a track is an *animatable scalar*, which is
either a plain number or a base value plus a v6 curve:

```json
12.5
{ "value": 12.5, "curve": { "interpolation": "linear",
                            "keys": [ {"frame": 1, "value": 0.0},
                                      {"frame": 24, "value": 30.0} ] } }
```

The `curve` object is byte-for-byte the envelope `docs/ANIMATION.md` defines,
and resolution is `nodebased.animation.evaluate_curve` itself:

* at an exact key frame — that key's value;
* between keys — `linear` interpolates, `constant` holds the previous key;
* before the first key or after the last key — **endpoint hold**: the first or
  last key's value;
* no `curve` at all — the base `value`.

**Corrected against the pre-merge draft.** This document originally specified
that a frame outside the key range falls back to the base `value`, because the
spike carried its own `resolve_scalar` and that is what it did. v6 extrapolates
by holding the endpoints, matching Nuke, and the substitution the draft asked
for has now happened: `nodebased.shapes` imports `validate_curve` and
`evaluate_curve` from `nodebased.animation` rather than reimplementing them. One
engine, one set of rules — a shape point and a node knob now extrapolate
identically, and a key list a knob would reject cannot survive inside a shape.
The base `value` is what applies when a scalar carries no curve; it is no longer
an out-of-range fallback.

`nodebased.shapes.validate_curve` adds only the payload location to the message,
so an artist gets `points[3].x: ...` instead of a rule violation with no address.

### Shape

```json
{
  "name": "shape1",
  "mode": "union",
  "opacity": 1.0,
  "feather": 0.0,
  "points": [
    { "x": 100.0, "y": 100.0,
      "in_x": 0.0, "in_y": 0.0, "out_x": 0.0, "out_y": 0.0 },
    ...
  ]
}
```

* `mode` is `union` or `subtract`, applied in list order against the matte
  accumulated so far. Union is `a + b - a*b`; subtract is `a * (1 - b)`. Both
  are chosen because they stay in `[0, 1]` for inputs in `[0, 1]` and neither
  double-brightens an overlap.
* `in_*` / `out_*` are cubic tangent handles stored **relative to their point**,
  the Nuke/After Effects convention. All-zero tangents on a span make that span
  a straight line, so a polygon is a bezier with no handles rather than a
  separate shape type.
* `opacity` in `[0, 1]` and `feather` in pixels are per shape, both animatable.
* At least three points. A shape is always closed for the purpose of generating
  a matte; open/stroked splines are out of scope for this pass and are not
  silently closed — they are rejected, because there is no way to store one.

### Track

```json
{ "name": "track1", "enabled": 1, "x": 480.0, "y": 270.0 }
```

`x` and `y` are animatable scalars — a solved track is a curve with a key on
every analysed frame, which is why `constant`/`linear` interpolation is enough
for it. `enabled` is also an animatable scalar, resolved and then thresholded at
`>= 0.5`, so a track can be switched off across an occlusion without deleting
its keys.

## Coordinates

Shape points and track positions are in **pixel coordinates of the node's own
output space, origin at the top-left, y increasing downward, pixel centres at
`i + 0.5`.** This is the convention `Evaluator._transform` already uses, and
matching it is what lets a tracked point be handed to a Transform without a
conversion nobody remembers to apply.

## Roto

```
Roto  inputs: []   params: width, height, invert
```

A generator, not a filter. It has no image input, so it cannot inherit a format
from upstream and then quietly disagree with it; the artist states the format.
Combining the matte with a plate of a different size hits the existing
shape-mismatch rule and raises rather than resampling.

Output is a **matte artifact**: float32, scene-linear, premultiplied, with
`rgb == a == coverage`. That is a premultiplied white matte, so it composites
correctly through every existing kernel with no special case, and reads as a
sensible greyscale image in the viewer. `invert` produces `1 - coverage` in all
four channels.

`docs/EVALUATION_TIERS.md` clause C6 requires cache entries to carry a declared
artifact type. The type registry (`nodebased.core.ARTIFACT_TYPES`) is declared
here — `Roto` is `matte`, everything else is `image` — but the cache does not
yet *store* the type; typed cache entries are v0.9.0 work. This pass owes the
declaration, not the storage, and does not claim otherwise.

### Rasterisation

Deterministic, CPU, NumPy:

1. Resolve every shape's scalars at the requested frame.
2. Flatten each cubic span to a polyline. A span whose two handles are both zero
   emits one line segment exactly; otherwise the segment count is derived from
   the control polygon's length and clamped to `[4, 64]`, so the same geometry
   always flattens the same way and the cache digest stays meaningful.
3. Scanline fill, non-zero winding, with `SUBSAMPLES` sub-scanlines per pixel
   row and *exact* fractional coverage horizontally. Vertical antialiasing is
   therefore quantised to `1/SUBSAMPLES` while horizontal is continuous. This
   asymmetry is a known limitation of the reference implementation, recorded
   rather than hidden.
4. Feather is a separable box blur of that shape's coverage by the feather
   radius. This is an approximation of a true offset-curve feather: it softens
   symmetrically about the edge instead of growing the shape outward, and a
   feather wider than a thin shape's waist will thin it. Nuke's roto does not
   behave this way. Recorded as a deliberate reference-implementation choice.
5. Combine into the accumulated matte per `mode`, scaled by `opacity`.

## Tracker

```
Tracker  inputs: [image]  optional: [mask]
         params: reference_frame, mode, apply_translate, apply_rotate,
                 apply_scale, filter, mix
```

The Tracker is a Transform whose transform is *solved from data* instead of
typed in. Given the tracks resolved at `reference_frame` and at the requested
frame, it fits a 2D similarity (translate, rotate, uniform scale) and applies it
through the same `Evaluator._transform` the Transform node uses. It honours the
established optional-`mask` + `mix` contract unchanged.

* One usable track — translation only; rotation and scale are not observable
  from a single point and are reported as identity rather than guessed.
* Two or more — closed-form least-squares similarity fit (Umeyama, reflection
  excluded).
* `apply_rotate` / `apply_scale` **constrain the fit**, they do not zero the
  result afterwards: a translation-only fit against rotating tracks is the
  best translation, not the translation component of a rotation.
  `apply_translate` does zero the translation after the fit, because that is
  what "match rotation but not position" means.
* `mode: match_move` applies the solved motion; `mode: stabilise` applies its
  exact algebraic inverse, expressed in the same `(center, rotate, scale,
  translate)` parameterisation so it round-trips through one kernel.
* A track only participates if it is enabled at **both** the reference frame and
  the requested frame. Zero participating tracks resolves to identity. This is a
  declared behaviour, not a fallback: erroring on an occlusion gap mid-scrub
  would make the node unusable, and the alternative — holding the last solve —
  would make the result depend on scrub order, which breaks cache determinism.

### Analysis

**Not implemented. `nodebased.tracker.analyse` does not exist.** The draft
described it as library surface awaiting a UI; what shipped has no analysis at
any layer. Nothing in this build looks at pixels to produce a track position —
track positions are authored through the `set_tracks` op and nothing else, which
means a Tracker is useful to an agent or a script and not yet to an artist with
a plate.

The design still stands as the intended approach when someone builds it:
zero-mean normalised cross-correlation of a pattern window against a search
window, integer peak, then parabolic sub-pixel refinement on the correlation
surface in x and y independently, walking frames in order and seeding each
frame's search from the previous frame's result. It would be a pure function of
the images it is given, so deterministic and testable.

The solve on top of it is real and tested: `nodebased.tracker.solve` is the
closed-form least-squares 2D similarity (Umeyama without reflection). Tracks are
matched **by index** rather than by name, because a track list is an ordered
payload and index is what `set_tracks` preserves. A track disabled at either the
reference frame or the current frame leaves the solve at that frame instead of
contributing a stale position. One usable track measures translation only; none
resolves to identity rather than to an undefined region.

## ChannelShuffle

```
ChannelShuffle  inputs: [A]  optional: [B]
                params: out_red, out_green, out_blue, out_alpha
```

Each output channel names its source explicitly as one of `A.r A.g A.b A.a
B.r B.g B.b B.a 0 1`. Two failures are errors rather than conveniences:

* naming a `B.*` source while `B` is unwired — an unwired input is not silently
  black, because "my alpha went black" is precisely the bug that costs an
  afternoon;
* `A` and `B` disagreeing on shape — no silent resampling, consistent with the
  mask/mix rule.

### Relationship to the existing Shuffle node

`Shuffle` already routes `R G B A 0 1` from a single input, and it is not
deprecated or changed by this pass. `ChannelShuffle` is the two-input form,
which is what a roto workflow actually needs: take RGB from the plate and alpha
from the Roto in one node, instead of Premult/Merge gymnastics. The two nodes
deliberately use different parameter names (`red_from` vs `out_red`) because
`nodebased.core.CHOICES` is keyed by parameter name globally, so a shared name
would force both nodes to share one option list.

## Evaluation tier declarations

Required by `docs/EVALUATION_TIERS.md` clause C2. Declared in
`nodebased/tiers.py`; the existing coverage test fails if a kind is missing.

| Node | ROI rule | Meaning |
| --- | --- | --- |
| `Roto` | generator | Reads nothing. Produces exactly the requested region. |
| `Tracker` | solved inverse transform | Inverse-maps the requested rectangle's corners through the solved similarity, takes the bounding box, pads by the filter support. The `mask` slot takes the requested region. |
| `ChannelShuffle` | identity | Both `A` and `B` are read at the requested region. |

`Tracker`'s rule cannot be computed from `params` alone — the transform lives in
`node_data` and depends on the frame. `nodebased.tiers.input_regions` therefore
accepts an optional pre-solved transform for that kind and **raises without
one**. Falling back to full frame would be exactly the silent erasure clause C2
forbids.

### Proxy (clause C3)

`Roto`'s `width`/`height` join `PIXEL_UNIT_PARAMS`. The harder half is that
pixel units also live in `node_data`: shape point positions, tangent handles and
feather radii, and track positions are all in pixels and all wrong at tier 2 if
left alone. `nodebased.tiers.scale_node_data` scales them by `1/tier` alongside
`scale_params`. A roto shape that did not scale would key a different part of
the frame at proxy resolution, which is worse than a slow viewer.

## Cache digest

Both new node kinds have output that depends on `node_data`, so the digest folds
it in. It folds in the **resolved** payload at the requested frame, not the
stored payload, exactly as `docs/ANIMATION.md` specifies for parameters:

* a static shape resolves identically at every frame and keeps its cache entry
  while the artist scrubs;
* an animated shape re-keys naturally per frame with no separate frame
  bookkeeping in the cache;
* editing a point invalidates, because the resolved payload changed.

### Tiled execution

All three kinds are deliberately **absent** from `tiles.SUPPORTED_TILED_KINDS`,
so a graph containing one reports `supports_tiled() == False` and falls back to
the reference `Evaluator` through the existing explicit telemetry. Two reasons:
roto feather is a box blur that needs halo handling a tile boundary does not yet
provide, and `Tracker`'s region is data-dependent, so the tile scheduler would
have to solve before it could plan.

The tile digest carries no `node_data` term. That is safe only while the two
sets stay disjoint, so `tests/test_roto.py` asserts the disjointness directly —
the day someone adds `Roto` to the tiled set, that test fails and forces the
digest change first. The alternative, folding an always-`None` payload term into
every tile digest now, would churn every existing tile cache key for no present
benefit.

## What this pass deliberately does not do

* No GUI shape drawing, point dragging, on-viewer transform handles or track
  markers. Shapes and tracks are created through the validated command boundary
  (`set_shapes` / `set_tracks`), which is the same boundary an agent uses. There
  is no hidden GUI-only edit path, and there is also no ergonomic one.
* No `roto_set_key` convenience op. v6 is merged now, so the stated blocker is
  gone; the op is simply not written. When it is, it reuses
  `nodebased.animation.merge_key` rather than growing a second key-insertion
  code path. Until then, keying a point means sending the whole shape list
  through `set_shapes`.
* No track analysis. See "Analysis" above — the function the draft described was
  never written.
* No open/stroked splines, no per-point feather, no motion blur, no shape
  linking to a Tracker, no planar tracking, no ROI-limited or tiered execution —
  the evaluator still runs full frame; this pass only declares its rules.
