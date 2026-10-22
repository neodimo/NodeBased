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
Roto  inputs: []   optional: bg   params: width, height, invert, premultiply_mode
```

A generator, not a filter. With nothing wired it has no image input, so it cannot
inherit a format from upstream and then quietly disagree with it; the artist states
the format. Combining the matte with a plate of a different size hits the existing
shape-mismatch rule and raises rather than resampling.

The optional `bg` input is Nuke's Roto input. Wired, the node passes the plate's
RGB through and the shape's coverage becomes its alpha, at the plate's own format
(`width` and `height` are then ignored), so the shape is drawn over the plate it is
traced against. An empty Roto passes the plate through unchanged so it remains visible
while drawing begins. Nuke documents its Premultiply control with a default of `none`;
`rgb` and `rgba` multiply the corresponding input channels by shape coverage. `O` with a node selected wires the selection to `bg`. Documents
saved before the slot existed gain it unwired on load and render the matte as before.

Each shape has `closed` (default `true`) and animatable `stroke_width` (default 1 pixel).
Closed shapes keep the historical fill and feather raster exactly. An open shape is a
round-capped, round-joined stroke centred on its Bezier curve; its feather expands outward
from both sides. Width and feather use image pixels and scale with proxy tiers. Enter ends
an open spline; clicking its first point still closes it. Existing documents omit `closed`
and therefore continue to load as closed shapes without a schema migration.

With no `bg`, output remains a **matte artifact**: float32, scene-linear, premultiplied, with
`rgb == a == coverage`. That is a premultiplied white matte, so it composites
correctly through every existing kernel with no special case, and reads as a
sensible greyscale image in the viewer. With no `bg`, `invert` produces `1 -
coverage` in all four channels; with a plate it inverts the shape alpha while
leaving the pass-through RGB unchanged.

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
4. New shapes use the edge-distance feather: the interior remains solid to the
   outline, then coverage falls to zero at the feather distance along the
   outline normal. Negative values move the feather inward. Linear, smooth and
   Gaussian choices shape that ramp. Per-point X/Y offsets are interpolated
   along each flattened span and alter its feather distance locally. Older
   documents retain their original separable box-blur model through the
   schema v22 `feather_model` migration field; the panel offers a one-click switch. The
   selected shape draws a dashed feather outline with per-point handles; drag
   a handle or Ctrl-drag a point, and use **Reset feather on selected points**
   to clear offsets. These edits are one undo step and key animated offsets at
   the current frame. Shape transforms and full Tracker follow transform the
   edge outline together with its control points.
5. Combine into the accumulated matte per `mode`, scaled by `opacity`.

## Tracker

```
Tracker  inputs: [image]  optional: [mask]
         params: reference_frame, mode, apply_translate, apply_rotate,
                 apply_scale, smoothing, pattern_radius, search_radius,
                 adaptive_update, tracking_channels, filter, mix
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

`nodebased.tracker.analyse` accepts a frame mapping or `frame -> Raster` provider and returns an
ordered `{frame: (x, y)}` mapping plus per-frame confidence for the interactive viewer. The reference pattern is sampled from
float32 scene-linear premultiplied RGB using luminance by default; callers may
select red, green, blue, alpha, or a set of those channels. Matching is zero-mean
normalised cross-correlation over explicit square pattern and search radii. The
integer maximum is refined independently in x and y with a clamped three-sample
parabola. Forward and backward frames search around the previous result, so the
result is deterministic and independent of scrub order.

Coordinates use pixel centres (`i + 0.5`) and the Raster data-window origin,
including negative origins. Invalid radii, insufficient texture, incomplete
pattern windows, searches with no complete candidate window, and weak matches
below the declared NCC confidence floor raise `AnalysisError` with the failing
condition. A weak match is reported as a possible occlusion instead of silently
writing an arbitrary peak. Cancellation raises
`concurrent.futures.CancelledError` before returning partial results.

The desktop Tracker inspector provides **Add track point at reference…**,
**Analyze forward**, and **Analyze backward**, with up to 16 track points. Pattern and search squares are draggable in the viewed image; dragging a point writes an undoable key at the current frame. First/Through controls bound an analysis or range clear. Track paths are coloured by the per-frame match error. Export buttons bake linked-source Transform and CornerPin nodes from the solved samples.
Picking is transient; successful analysis appends the
track and writes constant-interpolation x/y keys for every analyzed frame in one
validated, atomic, undoable `set_tracks` command. Existing names and enabled
values are copied unchanged. Failure or cancellation leaves the Dispatcher
document untouched. The path uses the reference evaluator and stays off the
tiled execution path; it does not alter `TileKey` or claim tiled provenance.

The solve on top of it is real and tested: `nodebased.tracker.solve` is the
closed-form least-squares 2D similarity (Umeyama without reflection). Tracks are
matched **by index** rather than by name, because a track list is an ordered
payload and index is what `set_tracks` preserves. A track disabled at either the
reference frame or the current frame leaves the solve at that frame instead of
contributing a stale position. One usable track measures translation only; none
resolves to identity rather than to an undefined region.

## PlanarTracker

`PlanarTracker` tracks a four-corner plane selected in the Viewer. Set its reference frame, connect an image, choose **Pick four region corners…**, then click the quadrilateral clockwise. **Track plane forward and backward** evaluates the connected image over the requested range, samples a feature grid, follows Lucas–Kanade flow in both directions and robustly fits a homography at each frame. The four corners are stored as v8 scalar x/y tracks, with per-frame fit error, so they remain editable and serializable like Tracker points.

The Viewer draws the tracked quad and colours it by its fit error. **Export CornerPin** creates a CornerPin with keyed source and destination corners. In Roto's Track link menu, choose the PlanarTracker and set Follow to **Planar** to transform the shape by the reference-to-current homography. Region picking uses four clicks; drawing or selecting an existing curved Roto shape as the tracking region is not implemented. The analysis currently runs synchronously on CPU and uses the full-frame evaluator; PlanarTracker is an analysis tap whose image output passes through unchanged.

Synthetic tests recover a known 24-frame projective move within one pixel, repeat the solve deterministically, retain a one-pixel translation under a quarter-region occlusion, assert the exported CornerPin corner keys, and verify Roto follows the plane homography. These tests do not establish performance or robustness on production plates.

## Stabilize

`Stabilize` is a standalone image node with its own tracked-point payload and analysis controls. It uses the same similarity fit as Tracker and always applies the inverse motion; smoothing blends nearby track positions around the reference frame. It supports the optional mask + mix contract and shares the viewer point, box, path and export workflow.

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
| `Roto` | generator | Reads nothing. TileExecutor rasterises each requested tile at the selected tier; conservative shape bounds skip shapes that cannot touch that tile. |
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

Roto and RotoPaint render requested tiles at the selected proxy resolution.
Roto's conservative shape bounds include feather and stroke expansion; shapes
outside a tile are skipped before rasterisation. RotoPaint fetches the plate and
clone source regions needed by the tile, including the neighborhood required by
blur and smear strokes. Output is byte-identical to the full-frame reference at
every tested proxy tier. Existing Roto and RotoPaint cache digests are preserved.

Tracker and Stabilize analysis requests only the reference pattern and each
candidate search region through TileExecutor's ROI path. PlanarTracker requests
the selected quadrilateral's bounding box plus a search margin. Tracker analysis
cache digests are unchanged. PlanarTracker still uses a rectangular padded ROI;
curved Roto masks are not an analysis-region option. Tracker and PlanarTracker
analysis run on the preview worker, report frame progress in the status bar,
and expose cancellation in the properties panel. PlanarTracker checks cancellation
between frame renders and adjacent-frame flow solves; an in-progress flow solve
finishes before cancellation is observed.

#### Lane 8 Q1 baseline profile (before optimization)

`cProfile` on the synthetic 40-shape 1920 x 1080 Roto, `TileExecutor(tile_edge=256)`, one
compose per tier. These are cumulative seconds under the profiler, so they identify where work
goes rather than serving as the timing comparison below.

Tier 1 top entries: `compose` 3.347; `compose_region` 3.347; `roto.rasterise` 41 calls / 3.328;
`_compute_node_digests` 2.620; digest callback 2.619; `_solve_roto` 2.619;
`Evaluator.evaluate_raster` 2.618; `_windowed_kernel` 2.610; Roto `_kernel` 2.610; `feather`
131 calls / 1.308. The 41 raster calls show repeated per-tile shape rasterisation. The full matte
kernel is also invoked while establishing the legacy-compatible Roto digest; the tile scheduler
and assembly copies do not appear among the top ten.

Tier 8 top entries: `compose` 0.125; `compose_region` 0.125; `roto.rasterise` 2 calls / 0.118;
`coverage` 80 calls / 0.097; `_render_tile` 0.063; `_roto_tile` 0.063; `_compute_node_digests`
0.062; digest callback 0.062; `_solve_roto` 0.062; `Evaluator.evaluate_raster` 0.062. At this
small tier only one tile is needed, so the 42 ms gap in the earlier unprofiled comparison cannot
be assigned to many-tile scheduling; most observed work is the full-frame digest raster plus
tile rasterisation. The large-tier overhead is principally repeated shape rasterisation.

On this machine, the synthetic 40-shape 1920 x 1080 Roto now uses the reference evaluator
for full-frame requests; the five-run medians were 2,299 ms at tier 1 and 44.0 ms at tier 8,
within 5% of the earlier full-frame baseline (2,244 ms and 44 ms). The earlier tiled path took
2,769 ms and 86 ms. The baseline profile above locates the 525 ms tier-1 penalty in 40
additional per-tile matte rasterisations; at tier 8, the extra 42 ms came from a full matte
render for digest calculation plus the actual tile render. Scheduling and assembly copies were
outside the profiled top ten. Roto tile digests now hash resolved shape data without rendering.

For interactive regions, each entry compares five-run median full-frame evaluator time with
direct tiled-region time on that same synthetic scene. Tier 1 (1920 x 1080): a 960 x 540
quarter-frame region took 2,301 ms before and 166 ms after; a 512 x 512 crop took 2,304 ms
before and 76 ms after. Tier 8 (240 x 135): a 120 x 67 quarter-frame region took 44.0 ms
before and 13.7 ms after; a 64 x 64 crop (the tier-8 equivalent of 512 x 512) took 43.9 ms
before and 8.3 ms after. Full-frame and tile-reference pixels remain byte-identical at tiers 1
and 8. These measurements cover Roto; dense RotoPaint interactive regions remain unmeasured.

## What this pass deliberately does not do

* The Viewer paints Tracker and Stabilize point markers, pattern/search boxes and error-coloured paths. The desktop viewer offers
  an artist-facing Roto overlay when the selected node is the viewed Roto: **Draw
  shape…** collects a closed Bezier (click for cusp points, drag for mirrored
  smooth tangents, click the first point or press Enter to close, Esc to cancel).
  Selected points expose tangent handles; dragging bends the span, Ctrl-drag
  breaks the pair, Z mirrors the handles and Shift+Z leaves a cusp. Ctrl+Alt-click
  on an outline splits its cubic span, and Delete removes selected points while
  preserving the three-point minimum. Gestures replace the whole payload through
  the validated `set_shapes` Dispatcher command (`set_paint_items` for RotoPaint),
  so they share validation, undo, save and agent semantics. The overlay is painted in the
  foreground rather than inserted into the scene; it therefore cannot alter
  `itemsBoundingRect()`, the format/display window, tile coordinates or EXR data
  window. Proxy tiers map screen coordinates back to the node's pixel space.
  Dragging an animated point updates/adds the current-frame key in its v8 scalar
  envelope instead of removing the curve. Shape mode, opacity and feather
  remain command-editable payload fields. The shape list supports a Nuke-style per-shape
  transform with animatable translate X/Y, rotate, scale X/Y and centre X/Y. The default
  identity transform preserves existing documents. The shape's centre defaults to the
  bounds centre; transforming scales and rotates tangent vectors along with their points.
  Tracker-linked shapes default to translation-only follow, preserving the prior behavior;
  choosing **Full Tracker solve** applies the Tracker's enabled translate/rotate/scale
  solution about the shape centre. Render and overlay call the same shape resolver.
* No `roto_set_key` convenience op. v6 is merged now, so the stated blocker is
  gone; the op is simply not written. When it is, it reuses
  `nodebased.animation.merge_key` rather than growing a second key-insertion
  code path. Until then, keying a point means sending the whole shape list
  through `set_shapes`.
* Planar tracking is available through PlanarTracker and Roto's planar follow link. Picking a curved
  Roto shape as the analysis region remains unsupported; the planar region is selected with four
  Viewer clicks.
* The 40-shape Roto profile and ROI timings above are synthetic measurements, not a recorded
  artist session. Dense RotoPaint interactive-region timing remains unmeasured.

## Transform viewer handle

With multiple points selected, the overlay bounds the set and dragging inside it moves
all selected points in one undoable edit. Ctrl+A selects every point in the shape chosen
in the Roto list. Corner handles scale both axes; edge handles scale one axis, and Shift
keeps a corner scale uniform. The rotate handle turns the selection around its centre.
Dragging the centre moves the pivot while compensating translation so the shape stays put.
Alt-dragging a whole-shape box edits its animated transform; ordinary box gestures edit
selected points, including their tangent vectors. Each drag is one undo step.

## RotoPaint and DustBust (schema v13)

`RotoPaint` takes a plate and an optional second image. Its `node_data` payload is
`{"items": [...]}`: shapes and pressure-sampled strokes occupy one ordered layer
list. Every item stores blend mode, opacity and visibility. A stroke stores a
polyline of `{x, y, pressure}` points, brush `size`, `hardness`, `opacity` and
`spacing`, a tool (`paint`, `eraser`, `clone`, `reveal`, `blur`, `sharpen`,
`smear`, `dodge` or `burn`), and a frame lifetime. Existing v12 documents
upgrade without paint payload; Roto shape records and rendering remain unchanged.

The deterministic CPU renderer has both full-frame reference and direct tile paths. Paint is premultiplied
colour, eraser attenuates the plate, clone samples the plate using its source
offset and requested source frame, and reveal uses input2. The DustBust viewer
preset records a single-frame clone dab from the previous frame. One viewer
gesture becomes one `set_paint_items` command and one undo step. Qt tablet
pressure is recorded when provided; ordinary mouse gestures use pressure 1.0.
Proxy tiers scale stroke coordinates, brush size and clone source offsets while
leaving pressure and spacing unchanged.

Known bounds: the four supported layer blend modes are deliberately simple, and
clone samples outside the frame edge-clamped. DustBust does detect specks, on request,
through its "Detect specks…" button.

### Paint strokes (plan 19, step W1)

The paint tools are `RotoPaint`'s; `Roto` stays the shapes-only generator Nuke's own
Roto is. A stroke carries a tool and these per-stroke settings.

- **Brush**: colour, `opacity`, `size` (diameter in pixels), `spacing` and
  per-point pressure, which scales both the dab's radius and its strength.
  `hardness` (0-1) is the fraction of the radius that is solid; the dab falls off
  smoothly from there to the rim, so 1.0 is a hard edge with a one-pixel
  antialiased rim and 0.0 a falloff across the whole radius. Schema v19 sets
  hardness 1.0 on every stroke saved earlier, because before v19 the number only
  nudged the rim.
- **Eraser** restores the plate; **Reveal** shows `input2`; **Clone** copies the
  plate from `source_offset` pixels away, from the previous frame (`relative`) or a
  fixed source frame; **Blur**, **Sharpen** and **Smear** retouch under the dab with a
  radius or reach driven by the brush size; **Dodge** and **Burn** lift or drop
  toward white or black by the brush `strength`.
- **Lifetime**: `all`, `single` (one frame), `range` (first to last, inclusive) or
  `from_current`. A stroke outside its lifetime does not render.
- Items composite in list order, shapes and strokes together. The Layers list shows
  each with its lifetime, and hides, reorders and deletes it, each a single undo step.
- **Viewer**: with a RotoPaint open in the viewer, B, E and C pick the brush, eraser
  and clone tool (B stays the blue channel for every other node); dragging paints at
  the chosen size and the whole drag is one stroke and one undo step.
- **Mask and mix**: the optional `mask` input limits where the paint layer shows
  and `mix` fades it; outside the mask the plate passes through.
- **Tiles**: strokes render only the requested tile plus required blur/smear neighborhood,
  and clone reads the matching source region. Tiled output equals the full frame at every seam.
  The tile digest folds in the Evaluator's digest, which carries the strokes and the
  plate at a clone's source frames.

Not built: Nuke's per-stroke colour correction and its onion-skin.

When a Transform is selected in the properties panel, its on-screen handle is
shown while the Viewer displays that Transform or anything downstream of it in
the same format. The box translates the image, the dashed outer ring rotates
it, and corner and edge handles scale it uniformly about the centre. The round
centre marker is the pivot.

Ctrl-drag the centre marker to move the pivot. This compensates translation so
the rendered image does not move; it makes changing the point about which a
later rotation or scale happens safe and predictable. Handle state is painted
as a Viewer foreground overlay and a drag is committed as one undoable batch
only on release.

Roto and tracker gestures have priority over this handle. In particular, a
Transform handle never consumes a Roto point edit or tracker point-picking
click.

## Lane 8 step notes

**2026-10-20, Lane 8 step T2 finish 1 (partial).** The merged interaction and PlanarTracker worker changes remain as described above. Real-display QA on the 1920 × 1080 comp used the RTX 3080 Ti and llvmpipe. On the RTX, first picture after a Roto move arrived 1.142 s from the first move, after release; full-tier settle took 3.888 s. An uncached scrub presented tier 8 at 0.595 s, tier 4 at 0.978 s and tier 1 at 4.914 s. At the 24 fps target, measured presentation was 0.125 fps at tier 1, 2.624 fps at tier 4 and 12.84 fps with automatic adaptation; the tier readout stepped through 1, 2 and 4 and finished at 1/4. The RTX run held the exclusive card lock. On llvmpipe, with Lane 4's integrated-adapter progression benchmark running concurrently, first picture after a Roto move arrived 1.414 s from the first move, after release; full-tier settle took 4.416 s. The uncached scrub first picture arrived at 0.808 s and settled at 9.446 s. Playback measured 0.125 fps at tier 1, 1.618 fps at tier 4 and 0.125 fps in automatic mode (tier readout 1 and 2 during the sample). The llvmpipe rates are contention-affected. Both adapters miss the 0.3-second first-picture target; the RTX automatic mode also misses the set 24 fps rate. The first provisional playback pass counted preview generations; its rates are excluded. The cube-face pick remains a bad pick because it is flat and featureless; the checker-floor failures at grids 8, 16 and 24 still point to a separate robustness limitation. The under-0.3-second first-picture and set-rate playback gates remain unmet.

### Lane 8 T1: ROI correctness

The tile executor keeps the picture visible when a node lacks a tile-native kernel: it renders that node full-frame, crops to the requested region and names the fallback in the ROI badge. Tracker and Stabilize now use the solved Transform inverse map to request their input patch, while retaining the untouched output patch for mask/mix. Their badges disappear on that native path. Requests crossing the target data window clip to its edge. ROI handle presses are resolved before Roto drawing, so a selected Roto no longer turns a box-handle press into a new shape.

The bundled `tests/fixtures/qa.nbcomp` is compared against the full evaluator inside a 960 × 540 box, both with its Tracker upstream and with the Roto background wired directly to RotoPaint; both comparisons are byte-identical. The test uses the QA plate sequence from `scratch/nb-qa/1015-roi` when available.

At tier 1 on llvmpipe, five cold-cache ROI-off runs before the backdrop reveal left the outside dimmed until the full frame arrived in 3.92–3.98 s. Afterward, the first paint (8.4–9.3 ms) showed the current ROI patch over the saved full frame at normal strength; refinement completed in 4.02–4.58 s.

On the 12 GB RTX 3080 Ti at tier 1, five ROI-off runs before the reveal painted in 8.2–9.4 ms but kept the backdrop dim until the full frame arrived in 3.78–3.90 s. After the change, first paint remained 8.3–9.2 ms and showed the saved full frame at normal strength; refinement took 3.95–4.48 s. The change fixes what is visible while refinement runs; it does not claim to accelerate that full-frame render. Raw runs are in `scratch/nb-lanes/run/bench-L8-roi-off-rtx3080ti-{before,after}.{json,log}`.

### Lane 8 step A: live-picture profile (partial)

The committed bench records monotonic mouse, edit, request, worker, graph, display-cache, conversion, card-handoff and paint events. Both baselines use the 1920 × 1080 QA comp. The RTX 3080 Ti measured first Roto-drag picture at 1.039 s after release, 3.768 s to settle, 0.566 s to first uncached-scrub picture, 5.098 s to settle, and 2.622 distinct tier-4 playback frames per second against a 24 fps target. llvmpipe measured 1.026 s after release, 3.728 s to settle, 0.569 s to first scrub picture, 5.509 s to settle, and 2.493 playback frames per second. Final playback status read “display cache hit · 4 ms” on the RTX and “display cache hit · 5 ms” on llvmpipe; these describe individual cache-hit requests, while the misses trigger the expensive graph work below. Playback samples were 8 seconds on the RTX and 2 seconds on llvmpipe.

Largest inclusive graph-root evaluation costs, in milliseconds:

| Series | RTX 3080 Ti, tier: ms | llvmpipe, tier: ms | Other measured evidence |
|---|---|---|---|
| Roto drag | 1: 2,498; 4: 353; 8: 101, 92, 89 | 1: 2,495; 4: 365; 8: 122, 95 (four calls) | Six preview requests and six worker starts overlap the drag; zero new frames arrive before release. |
| Uncached scrub | 1: 3,738; 4: 351; 8: 204; Read at 8: 0.7 | 1: 4,118; 4: 357; 8: 201; Read at 8: 0.6 | Four graph evaluations on each adapter; one of five display-cache lookups hits. |
| Tier-4 playback | 4: 384, 371, 370, 364, 362 | 4: 378, 375, 372, 370, 366 | RTX: 45 lookups, 23 hits, 22 graph evaluations in 8 s. llvmpipe: 13 lookups, 7 hits, 6 evaluations in 2 s. |

The drag timeline records requests arriving after successive mouse moves while graph work is still running, followed by release and a later delivered frame. This supports request supersession during the move as the reason no picture arrives in-drag. The baseline trace did not attribute the “tile fallback” status to a node; the after-run trace now identifies Roto and records the condition: requests covering at least 80% of the canvas use the full-frame evaluator. Qt-thread and worker cProfile captures are in each `benchmarks/live-picture/profile-1020-<series>.txt`; event timelines are in matching JSON files. Two optimizations landed: an active live-drag render can finish and appear while newer snapshots queue, and a selected-tier cache hit skips the lower progressive renders. The targeted offscreen preview and playback-queue tests pass.


Latest three-adapter comparison (each finish3 playback sample ran for 8 seconds; finish3 uses the final per-contour cache code):

| Series | RTX 3080 Ti baseline → finish2 → finish3 | llvmpipe baseline → finish2 → finish3 | Integrated laptop stand-in finish2 → finish3 |
|---|---|---|---|
| Roto drag first picture | 1.039 s after release → 0.5145 s in-drag → 0.4739 s in-drag | 1.026 s after release → 0.5109 s in-drag → 0.5171 s in-drag | No picture during drag; 45.356 → 45.358 s to settle |
| Uncached scrub first picture | 0.566 s → 0.6316 s → 0.4871 s | 0.569 s → 0.4731 s → 0.4885 s | 0.6588 s → 0.7658 s |
| Tier-4 playback | 2.622 fps → 19.104 fps → 19.363 fps | 2.493 fps → 20.444 fps → 18.834 fps | 4.607 fps → 8.100 fps |

The largest Roto evaluation costs fell materially after local-bounds rasterization, static-matte reuse and per-contour reuse. On RTX, the tier-1 drag evaluation fell from 2,498 ms to 659 ms and the uncached-scrub tier-1 evaluation from 3,738 ms to 1,455 ms; on llvmpipe they fell from 2,495 to 486 ms and 4,118 to 1,289 ms. Median tier-4 Roto evaluation fell from 370 to 166 ms on RTX and 372 to 167 ms on llvmpipe. The finish3 RTX run now clears the 0.5-second first-picture gate (0.474 s in-drag); llvmpipe narrowly misses at 0.517 s. Both delivered frames during the drag. Playback remains below 24 fps: 19.363 RTX, 18.834 llvmpipe, and 8.100 on the integrated stand-in. The laptop was pinned to CPUs 0–3 with 2× UI scaling. Its drag again delivered no picture during movement and took 45.358 s to settle. Scrub first picture is under 0.5 s on both discrete/virtual adapters, but not the laptop (0.766 s).

The follow-up worker profile (full session under cProfile; its wall timings include profiling overhead) found 28 `read_media_raster` calls totaling 2.821 s, including 28 named-layer reads at 2.663 s; it also found 28 RotoPaint rasterizations at 82 ms each. `benchmarks/live-picture/profile-1022-followup.txt` records those calls. The common solid-color paint-over path now avoids copying its read-only input and allocating a full-frame solid layer. Multichannel EXR beauty and named layers now share one channel decode when within the existing layer-cache size limit. A pixel-exact paint test checks input immutability; a multichannel EXR test checks that one full channel buffer serves beauty and named layers.

The paint-only fast-path samples, captured before the shared-EXR-decode change, are `after-rtx3080ti-paint-fastpath.json`, `after-llvmpipe-paint-fastpath.json` and `after-laptop-standin-paint-fastpath.json`. They measured playback at 20.242, 20.345 and 7.867 fps respectively; first drag picture was 0.510 s on RTX and 0.498 s on llvmpipe, while the integrated stand-in delivered no in-drag picture.

The final-code post-shared-decode runs are `benchmarks/live-picture/after-rtx3080ti-shareddecode.json`, `after-llvmpipe-shareddecode.json` and `after-laptop-standin-shareddecode.json`. RTX: 0.516 s to an in-drag picture, 22.089 fps; llvmpipe: 0.497 s in-drag, 21.215 fps; integrated stand-in: no picture during drag, 45.361 s to settle, 13.611 fps. The uncached scrub first picture was 0.450 s on RTX, 0.458 s on llvmpipe and 0.656 s on the laptop. The cached-range offscreen playback test passes, while these real playback runs remain below 24 fps on all adapters. The 0.5-second drag-picture gate passes on llvmpipe and misses on RTX; the laptop still produces no in-drag picture. The lane remains partial against these measured gates.

The October 22 corrected-coordinate follow-up supersedes the integrated-drag and playback rows above. Root cause: the benchmark generated Qt logical pointer coordinates but sent them to `xdotool`, which expects physical X11 pixels; at 2× scaling the 1920 × 1080 logical window also extended to 3840 × 2160 physical pixels, leaving the drag target outside the Xvfb display. The committed harness now scales pointer positions by the device-pixel ratio and sizes the window to the screen's logical geometry. It also paces drag moves so Qt receives held-button move events. This was a benchmark input/window geometry failure, not an integrated-adapter render failure.

Fresh corrected 8-second runs show a picture during the held drag on every adapter. Measured from the first mouse-move event actually received by Qt to the first new picture: RTX 3080 Ti 0.198 s (11 frames during drag), llvmpipe 0.193 s (11), and integrated Radeon stand-in 0.177 s (10). The older start-to-picture number includes the deliberate press/setup interval and is 0.540, 0.525 and 0.516 s respectively; use `first_move_to_picture_s` for the 0.5-second first-move gate. Uncached-scrub first pictures were 0.456, 0.452 and 0.501 s.

Cold tier-4 playback measured 21.330 fps on RTX, 20.979 on llvmpipe and 15.463 on the 4-core integrated stand-in, all below 24 fps. A separate range-prewarmed run produced 192/192, 192/192 and 193/193 display-cache hits during playback with zero graph evaluations, yet measured 23.842, 23.845 and 23.841 fps. That isolates the remaining near-target shortfall to playback/display cadence rather than graph rendering; the exact 24 fps gate remains unmet. The corrected cached-range verifier checks cache residency by content identity and frame while allowing region-keyed entries. Focused coordinate tests and the benchmark self-check pass. The lane remains partial: the 24 fps playback gate is still open on every adapter.
