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

### Shape and node settings from the Nuke parity pass (fidelity step F)

Every setting below is optional. A shape saved without it draws exactly as before, and its resolved form (and so its cache digest)
carries nothing new. The scorer pairs each with the Nuke knob of the same job (`docs/PARITY_2D_FIDELITY.md`); where this document
gives a reading of the guide's one-line description, that reading is NodeBased's and has not been compared with a Nuke render.

Per shape, in the payload:

* `feather_on` (default true). False draws the hard outline: the shape's feather width and every point's feather offset are ignored.
* `falloff_power` (0 to 16, default 1). Nuke's numeric *feather falloff*. The feathered coverage is raised to this power, so 1 is
  the unchanged ramp, below 1 the feather holds opaque longer (0 is a hard edge out to the feather's reach) and above 1 it drops away
  faster. It acts after the linear, smooth or Gaussian choice.
* `transform.skew_x`, `transform.skew_y` (degrees, animatable) and `transform.skew_order` (`XY` or `YX`). About the centre, the
  skew comes first, then the scale, then the rotation, then the translation. Order `XY` shears X then Y.
* `extra_matrix`: 16 numbers, row by row (Nuke's 4 x 4 `transform_matrix`). Applied last, to image coordinates: `x' = m0 x + m1 y + m3`,
  `y' = m4 x + m5 y + m7`. A stroke's width follows the matrix's scale; at a proxy tier the translation column is divided by the tier.
* Motion blur. `motion_blur` (the switch) with `motion_blur_samples` (1 to 64), `motion_blur_shutter` (frames, 0 to 8),
  `motion_blur_offset_type` (`start`, `centred`, `end`, `custom`) and `motion_blur_offset` (frames, for `custom`). The shape,
  transform included, is drawn at `samples + 1` evenly spaced times across the open shutter, ends included, and the pictures are
  averaged. The shutter opens at the frame (`start`), straddles it (`centred`), closes at it (`end`) or opens `offset` frames from it
  (`custom`). A shape saved with only `motion_blur: true` keeps its original blur across the two neighbouring frames.
* `lifetime`: `{"type": "all" | "single" | "start_to" | "to_end" | "range", "start": frame, "end": frame}`. The shape exists only
  on the frames the type selects (`single` at `start`, `start_to` up to `end`, `to_end` from `start`, `range` from `start` to `end`).
* `cap_start`, `cap_end` (`rounded`, `square`, `butt`; open splines only). Rounded is the original end. Butt stops at the end point.
  Square runs on half the stroke width past it with straight sides.

Per node, in `params`: `outputMask` (a layer name, default `none`) also writes the matte, after the node's opacity, to that layer in
all four channels; a node with it set renders whole, not in tiles. `shape_motionblur_on` (default on) lets the shapes' own motion
blur settings count. `global_motionblur_on` (default off) with `global_motionblur`, `global_motionblur_shutter`,
`global_motionblur_shutter_offset_type` and `global_motionblur_shutter_offset` applies one set of motion blur settings to every shape
that has none of its own.

Not honoured, with the reason in `docs/parity/nodebased_to_nuke.json`: the clone and reveal source knobs (`shape_source`,
`source_*`; RotoPaint holds the source stage), the stereo `view` and `stereo_offset` (no views to choose), the spline `key` spinner
(keys are set with the key button beside each field), and the tracking tab (`pretrack_filter` to `reference_frame`; a shape follows
a Tracker or PlanarTracker through `track_link`).

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

Auto-tracking's `warp` can refine that initial translation as Translate/Rotate,
Translate/Scale, Translate/Rotate/Scale, or Affine with a bounded Gauss-Newton
fit over the pattern. `max_iter` caps the fit and `epsilon` ends it when the
parameter update is small. Translate remains the default path. With warp fitting,
the analysis `info` result carries rotation and scale per frame; the Tracker
inspector stores these as animated per-track values so a single track can
contribute its pattern's rotation and scale to the solved Transform.
`auto_regrab_pattern` updates the reference pattern after each tracked frame
and accumulates those per-frame warp values.

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

### RotoPaint settings from the Nuke parity pass (fidelity step H)

Every setting below is optional and an older document draws as it did. Per item in the payload, on shapes and strokes alike:

* `locked` (boolean). The viewer refuses to edit the item. No pixel changes, as in Nuke.
* `source`: `color` (default), `foreground` or `background`. What the item paints. A shape paints white and a stroke its colour under
  `color`; `foreground` paints the `input2` image where the item draws (black when nothing is wired); `background` paints the node's
  plate, which is how a stroke puts the plate back over earlier paint. Strokes use it with the paint tool.
* `transform` (translate, rotate, scale, skew, centre, skew order, each animatable) and `extra_matrix`: the Transform tab, with the
  arithmetic of Roto's shapes (the section above). A stroke's points move with it and its brush size follows the transform's
  scale (the square root of its area).
* `motion_blur`, `motion_blur_samples`, `motion_blur_shutter`, `motion_blur_offset_type`, `motion_blur_offset`: the Motion Blur tab,
  also as Roto has it. A stroke is drawn at each of the `samples + 1` times across the shutter and the pictures are averaged; a stroke
  that does not move is sharp. The node's `shape_motionblur_on` and `global_motionblur*` params work as Roto's do.

Shapes also take Roto's own shape settings: `feather_on`, `falloff_power`, `feather_falloff`, `feather_model`, `invert`, `closed`,
`stroke_width`, `cap_start`, `cap_end` and `lifetime`. A shape outside its lifetime is hidden.

Strokes:

* `effect` (0 to 4096, default 0). The Stroke tab's effect parameter: for blur and sharpen the filter radius in pixels, for smear how
  far the colour is dragged, for dodge and burn the strength. 0 keeps the brush's own settings.
* `dynamic_hardness` (boolean). Pressure scales the hardness along the stroke.
* `buildup` (boolean). Overlapping dabs add up, `1 - (1 - a)(1 - a_dab)`, instead of the strongest dab alone setting the opacity.
* `write_on`: `{"start": s, "end": e}`, fractions of the path's length. Only that stretch is drawn; the cut ends are interpolated.
* The Clone tab: `source_transform` (`rotate` in degrees, `scale_x`, `scale_y`, `skew_x`, `skew_y`, `skew_order`, `center_x`,
  `center_y`) turns, scales and skews the source about its centre before the source offset moves it; `source_round` rounds the offset
  to whole pixels; `source_black_outside` makes pixels the source does not cover black and transparent instead of repeating its edge;
  `source_filter` (`impulse`, `cubic`, `keys`, `simon`, `rifman`, `mitchell`, `parzen`, `notch`) is the resampling filter, `impulse`
  (the nearest pixel, the original behaviour) when none is named. The guide names the eight filters and gives no formula: `cubic` and
  `keys` are the Keys cubic convolution (a = -0.5), `simon` and `rifman` the same family with a = -0.75 and -1, `mitchell` the
  Mitchell-Netravali filter (B = C = 1/3), `parzen` the cubic B-spline and `notch` the two-pixel tent. These readings are
  NodeBased's and have not been compared with a Nuke render.

Per node, in `params`: `premultiply_mode` (`none`, `rgb`, `rgba`; Nuke's `premultiply`) multiplies the colour (and with `rgba` the
alpha) by the alpha of the painted result before the mask and mix. `outputMask` (a layer name, default `none`) also writes the
coverage of everything the node drew, the alpha it leaves on an empty canvas, scaled by `opacity`, to that layer in all four channels;
a node with it set renders whole, not in tiles. The node has Nuke's mask strip: `channels` (default `all`), `maskChannelInput`,
`invert_mask`, `inject` and `fringe`, around the paint as they are around every filter, and the global motion blur knobs of Roto.
A stroke whose clone source is rotated, scaled, skewed or filtered renders whole, not in tiles.

Not honoured, with the reason in `docs/parity/nodebased_to_nuke.json`: the stereo `view`, `stereo_offset` and `source_view`, the
spline `key` spinner, the `format` knob (the node keeps the wired image's format), and the tracking tab (`pretrack_filter` to
`reference_frame`).

## Edge snap and fit to edges (past Nuke)

Nuke's Roto has neither feature. Both are opt-in: with **Snap to edges** off (the default) every gesture behaves exactly as before, and a
saved shot stores no setting for them, loads and renders bit-identical. They read the image at the Roto node's input (`bg`, or `image` on
RotoPaint) at the current frame through the evaluator and never change it.

**The edge field** (`nodebased/rotoedges.py`). Edge strength is the gradient magnitude of the maximum over the RGB channels (luminance can
cancel an edge between a saturated colour and a grey of the same luma; for a grey plate the two are equal), after a Gaussian of sigma 1 px,
with the Scharr kernel normalised so a unit step gives 1 per pixel. Only a region is computed: a window around the cursor for a snap, a
band around the shape for a fit, rounded out to a 32 px grid, and cached by (node, frame, input digest, region). An edit to the input
changes the digest and misses. Below an absolute gradient floor, or 20% of the region's gradient range, nothing is an edge, so a point
with no edge nearby stays where it is.

**Snap to edges.** A "Snap to edges" checkbox and a "Snap radius" field (px, default 8) sit in the strip of the Select tools, Add Points and
the shape tools; they are shared by every tool of the node (stored next to Autokey under the node's `__global__` tool settings, and left out
when at their defaults). When on, a point placed by a shape tool (a Rectangle or Ellipse snaps its two drag corners), a point dragged with a
Select tool and a point added on a spline move to the strongest edge within the radius on release, to a sub-pixel position. The preview while
dragging shows the snapped position. Feather drags, tangent drags and multi-point moves do not snap. **Hold Alt+Shift** for a gesture without
snapping: Ctrl already drags a feather and starts a freehand sketch, Alt alone pans the picture and Shift alone ends an open curve, so only the
pair is free (Shift ends an open curve as before unless Alt is held and Snap is on). One undo step per gesture, as before.

**Fit to edges.** A "Fit band" field (px, default 12) and a "Fit to edges" button in the Select tools' strip, and "Fit to edges" in the shape
list's right-click menu, refine every point of the selected shapes, closed or open, to the edges within the band: an active contour in which each
point moves along its normal toward the strongest edge, a second-difference smoothness term keeps curvature close to the original, spacing is
kept, tangents turn and scale with their neighbours, a cusped corner goes to the corner where two edges meet, and feather offsets stay
relative to their point. With Autokey on it writes a key at the current frame (the same commit path as a point drag), with Autokey off it
changes the static shape. One undo step covers all selected shapes; locked shapes and shapes in locked layers are skipped.

**Ripple through Fit to edges.** With the Select tools' Ripple edit on, Fit to edges on the current frame is treated like a hand point edit
(`rotoripple.ripple_shapes`): each point's fitted offset, the difference between its fitted and its previous coordinate at the current frame,
is added to that point's other keys inside the ripple range (all, from start, to end or range), keys outside the range stay put, and a point
with no keys simply takes the fitted value. The fit and the ripple are one command, so one undo restores every key. Track points and Fill
in-betweens already write keys of their own over a range of frames, so ripple edit does not apply to them: when it is on and one of them runs,
the status bar adds "ripple edit does not apply: ... writes its own keys".

Measured on synthetic images (tests/test_roto_edges_i1.py): a snap from 5 px outside a disc lands within 0.09 px of its edge; a 16-point circle
perturbed by up to 4 px fits to 0.3 px RMS; a perturbed square reaches its corners within 0.25 px; fitting a 24-point shape on a 1920 x 1080
plate computes a 672 x 672 px region in about 15 ms and fits in about 15 ms. The plate of the playback fixture is a smooth gradient with no
edges, so Fit to edges leaves a shape on it unchanged, by design.

## Point tracking (past Nuke)

"Track shape" planar-tracks the shape's bounding quad, so an outline follows a rigid or perspective move but not a subject that bends or
stretches; in Nuke a deforming subject needs a SmartVector or tracker-linking setup. **Track points** tracks every control point with its
own small patch of the image and uses the planar track only to steer and steady it. It is a separate command: "Track shape" behaves as
before, and a saved shot stores nothing new until the command is run, so saved shots render bit-identical.

**Where.** A "Track points" button beside "Track selected shape or stroke" in the Roto and RotoPaint panels, with a direction (Forward and
backward, Forward, Backward) and the first and last frame (they start at the timeline's), and "Track points" in the shape list's right-click
menu (forward and backward over the timeline). The reference frame is the current frame, the one whose outline is on screen: tracking runs
away from it, so keys before it (when tracking forward) and after it (when tracking backward) are left alone. Select the shapes in the viewer
or the list first.

**How a frame is tracked** (`nodebased/rotopointtrack.py`, pure numpy; the planar track of the shape's bounding quad is
`planar.track_region`, the one "Track shape" uses). For each point:

1. The search window (12 px, patch 7 px either side of its centre) is centred on where the planar track puts the point plus the offset the
   point's own match last had from the plane.
2. A normalised cross-correlation finds the best whole-pixel position of the patch, and Lucas-Kanade steps refine it to a fraction of a pixel.
   The refinement lets the patch stretch and shear a little (a six-parameter warp held toward identity by a small ridge) so a point on a
   subject that is deforming is not dragged toward the side that deforms least; the confidence is the correlation of the warped patch. The
   patch is grabbed again from the frame only when its correlation falls under 0.92, so small errors do not add up. Samples use a cubic
   filter because a bilinear one blurs a half-pixel sample more than a whole-pixel one and the two then disagree by half a pixel.
3. The point is the planar position pulled toward its own match by the confidence: none of it under 0.7, all of it above 0.8.
4. Each confident point's offset from the plane is then moved a quarter of the way toward the confidence-weighted average of its two
   neighbours' offsets along the outline, so a noisy match does not tear the shape. Points under the threshold are left exactly on the
   planar prediction and take no part in their neighbours' averages.
5. Tangent and feather offsets turn and scale with the planar map at the point (its local 2x2 linear part), so a rotating or scaling subject
   carries them round; they do not take on the local deformation of the points around them.

**What is written.** One undo step for the whole run, through the same commit path as a point drag (`set_shapes` or `set_paint_items`). Every
tracked frame gets a key on each point's anchor, and on a tangent or feather offset only where it moved; the reference frame gets a key too
when a point was static, so it stays where it was. Autokey rules apply: a shape already keyed has its keys written (and overwritten) at each
tracked frame whatever the Autokey switch says; a static shape is keyed only with Autokey on, and with it off the command changes nothing and
says so in the status bar. Locked shapes and shapes in locked layers are skipped. Shape transforms and tracker links are inverted frame by
frame, so keys land in the shape's own space as for any point edit (a shape with a rotating transform whose centre is left at its default
follows the points' bounding box, which moves with the keys: set the centre if that matters).

**Flags.** A point whose confidence is under 0.7 on a frame is drawn larger in red at that frame in the viewer, and the status bar names
the points and their frame ranges ("disc point 4 (frames 6–8)"), so the artist can fix them. The flags are viewer state only: the document
holds no trace of them, and they disappear from a shape whose point count changed. A flagged point is not lost: its patch is kept from the
last good frame and its search stays centred near the plane, so it picks itself up again when the cause (an occluder, a blink of motion
blur) passes.

**Limits.** Matching uses the channels, pre-track filter and clamp of the node's tracking settings, like "Track shape". A region too plain
for the planar tracker (fewer than four visible features) falls back to searching around each point's own position, which needs small motion.
Under a deformation faster than about 7% of the patch per frame, the patch is grabbed again every frame and drift can show: key the shape
at intervals. A flagged point falls back to the planar position, which on a deforming subject can sit several pixels from where the point
was; its neighbours' good matches do not pull it back.

Measured on synthetic frames (tests/test_roto_point_track_i2.py): a textured disc translating 1.3 and 0.7 px a frame is followed by all 24
points of its outline to within 0.38 px over 10 frames (the planar track alone: 0.19 px); the same disc with its right half stretching 7%
of its reference width every frame is followed by the right-side points to within 0.5 px where the planar track alone misses by up to
3.6 px; an occluded patch drops under the threshold on the frames it is hidden, nothing else is flagged, and the point rejoins its patch
when the occluder goes. A 24-point shape on 1920 x 1080 frames takes about 130 ms a frame for the matching (frames already decoded, quiet machine); the
planar track that steers it costs about 0.6 s a frame on a 640 x 640 px region around the shape (the same cost as "Track shape"), and the frame
decode comes on top. The run is in the background with progress in the status bar.

## In-betweens (past Nuke)

Between two keyed frames a Roto shape is interpolated blindly, so the artist redraws every in-between the interpolation misses. **Fill
in-betweens** gives them a draft to correct. It is opt-in: nothing changes until the artist runs the command, existing interpolation is
untouched, a saved shot stores none of its settings at their defaults, and one undo step reverts a whole run.

**What it does** (`nodebased/rotoinbetween.py`, built on the edge field above). For each frame strictly between the two keys, it takes the
shape as the renderer resolves it today (interpolated, with its transform and tracker link), runs `rotoedges.fit` on *that frame's* input
image inside a band, and proposes a key on every point the fit moved by more than a threshold. Points that did not move keep the
interpolation, and a frame whose points all stayed put gets no key, so a shape the interpolation already fits gains nothing. A key sets all
six coordinates of the moved point (position and both tangents). A coordinate that was static is first pinned at both ends of the range, so
no other frame changes. Frames the artist already keyed are skipped, never overwritten, and the end keys are never touched.

**In the app.** "Fill in-betweens" is a button in the Select tools' strip and an entry in the shape list's right-click menu. It acts on the
selected shapes, between the nearest keys around the current frame (on a key it runs to the next key), or between the strip's **Fill from**
and **Fill to** frames when they are set (both 0 means the nearest keys). Other strip fields: **Fill band** (px, default 8), **Fill threshold**
(px, default 0.75) and **Fill every Nth frame** (default 1, so only every Nth frame between the keys is fitted and keyed). Locked shapes and
shapes in locked layers are skipped. The keys go through the same commit path as a point drag, one undo step covers every shape, and the status
bar reports "added N keys on M frames" (a key is one point at one frame), or says why nothing was added (no key on both sides of the frame, no
image on the Roto input, or the interpolation already fits).

Measured on synthetic images (tests/test_roto_inbetween_i3.py): a disc that grows with ease-in between frames 1 and 9, keyed only at those two,
is missed by plain interpolation by 2.2 to 5.0 px RMS on frames 2 to 8; after the fill every in-between is within 0.2 px RMS of the true edge
(7 frames and 112 point keys for a 16-point shape). A disc that grows linearly gets no key. The fit on a 24-point shape at 1920 x 1080 takes
about 30 ms per in-between frame once the plate is rendered; the plate's own evaluation per frame comes on top.

## GPU mattes (past Nuke)

Roto shapes can be drawn on the graphics card instead of the processor. It is a switch, **off by default**: nothing changes for anyone
who has not turned it on, and the processor drawer stays the reference. **Settings → Interface → "Draw Roto mattes on the graphics card
(experimental)"** turns it on at once (the cached mattes are dropped so the picture shows the new path); `NODEBASED_ROTO_GPU=1` or `0`
overrides the saved choice and Settings says so under the row. The code is `nodebased/rotogpu.py`; `roto.rasterise` offers it every
full-frame request first and draws on the processor whenever it answers "no".

**What runs on the card.** Closed Bezier and B-Spline shapes with the box feather, per shape, in the processor drawer's own order:
the fill (non-zero winding, eight sub-scanlines per pixel row, exact horizontal coverage), the box blur of the feather (the Blur node's
window, edge pixels repeated at the patch edge), clamp, the falloff curve (linear, smooth or gaussian) and falloff power, the shape's
own invert, opacity and colour alpha, and the shape's blend: over, plus, multiply, minus, from, difference, exclusion, max, min,
screen, and the subtract mode. The processor flattens each outline and works out which sub-scanlines every edge crosses, in double
precision, by the drawer's own rule, so no sub-scanline is gained or lost; the card does the rest in 32-bit floats. All shapes of a
matte go through four compute passes in all (fill every row of every shape, blur across, blur down, then one finishing pass that
composites each pixel through the shapes that cover it in order), and only the shapes' patches come back, because copying the whole
frame back costs about 11 ms at 1080p (3 GB/s from the card) whatever is drawn. A compute pass was chosen over a render pass: the fill
needs a sort of each sub-scanline's crossings and a sum into a row one thread owns, and a rasteriser's multisample patterns cannot
give the drawer's exact horizontal coverage.

**What stays on the processor** (the whole matte, so a mixed Roto is never half-drawn on the card): open strokes and RotoPaint
brushes, the signed-distance ("edge") feather of a Bezier shape, motion blur samples, picture sources (clone, reveal, the
background and foreground sources), the Merge-style blend modes (under, atop, overlay and the rest), any request for a region of the
frame (the tile path), a feather wider than 256 pixels, a scanline crossed more than 128 times, patches larger than the card's buffer
limit, and the read-ahead matte process (a child process never touches the card; with the switch on, the read-ahead takes the thread
path instead, so the card draws those mattes too). A graphics error of any kind (no adapter, lost device, a shader that will not
compile) falls back to the processor matte for that call and is counted (`rotogpu.counters`); three errors in a row turn the path
off for the session until Settings turns it on again.

**How close.** Not bit-identical: the card fills in 32-bit floats where the drawer fills in 64-bit, so edge positions differ by a few
millionths of a pixel and a coverage by up to about 1e-5. The largest difference over all 32 test cases is 1.2e-5 (an 8-bit step is
3.9e-3, so the worst difference is about 1/300 of one level and only a pixel sitting on a rounding boundary can land a level apart) and the mean difference is 3e-8 or less. The same cases give the same figures on all three
adapters on this machine (RTX 3080 Ti, Radeon 8060S, llvmpipe). Per case, the largest and mean absolute difference over the whole RGBA
frame against the processor drawer (`tests/test_roto_gpu_i4.py`; `benchmarks/2d-playback/roto-gpu-1110/parity-*.json` has all 32; `playback-summary.json` and `matte-gpu-*.json` hold the timings):

| Case | Largest difference | Mean difference |
|---|---|---|
| convex quad | 3.6e-07 | 5.6e-10 |
| concave curve | 8.1e-06 | 9.9e-09 |
| concave comb | 6.9e-06 | 2.1e-09 |
| self-crossing bowtie | 5.1e-06 | 1.6e-08 |
| self-crossing star | 4.6e-06 | 2.4e-08 |
| tiny triangle | 3.0e-08 | 1.5e-11 |
| sub-pixel sliver | 8.9e-08 | 7.5e-11 |
| huge feather 90 | 2.7e-07 | 2.1e-08 |
| feather 12 linear | 3.0e-07 | 8.9e-09 |
| feather 12 smooth | 3.0e-07 | 8.2e-09 |
| feather 12 gaussian | 4.2e-07 | 1.2e-08 |
| falloff power 0.5 | 1.5e-06 | 9.9e-09 |
| rotated, scaled, moved | 3.0e-07 | 5.0e-09 |
| b-spline | 2.4e-07 | 4.6e-09 |
| half off the frame | 2.4e-07 | 6.2e-09 |
| subtract mode | 4.2e-07 | 1.0e-08 |
| plus blend | 6.3e-06 | 5.8e-09 |
| multiply blend | 2.5e-06 | 4.4e-09 |
| difference blend | 8.3e-07 | 8.1e-09 |
| 30 shapes at 1080p | 1.2e-05 | 4.6e-09 |

Tolerance of the tests: 5e-5. Because the result is not bit-identical the switch stays off until DiMo approves the tolerance.

**Speed** (`tools/benchmark_roto_gpu.py`, `roto.rasterise` on resolved shapes, no evaluator and no window; median of seven or nine
drawings per pass, range over three alternating passes; the machine was idle, 1080p; `matte-gpu-*.json`).

| 1080p | Processor ms | Card ms (RTX 3080 Ti) | Faster by | Card ms (Radeon 8060S) | Card ms (llvmpipe) |
|---|---|---|---|---|---|
| 30 shapes, 12-point curves, feather 12 | 35.7 to 36.8 | 15.7 to 17.3 | 2.2x | 10.0 to 11.3 (3.3x) | 28.8 to 29.6 (1.2x) |
| 30 shapes, 50-point curves, feather 18, larger | 65.7 to 70.2 | 26.2 to 26.9 | 2.5x | 14.9 to 15.1 (4.4x) | 57.2 to 65.3 (1.2x) |
| 10 shapes, 12-point curves, feather 12 | 13.1 to 14.7 | 8.1 to 10.2 | 1.6x | | |
| 1 shape, 50-point curve, feather 18 | 10.7 to 11.5 | 9.8 to 12.8 | 1.2x | 7.0 to 7.2 (1.5x) | 12.7 to 13.1 (0.8x) |

At 4K the 30-shape cases take 122 to 129 ms and 242 to 244 ms on the processor against 48 to 50 ms and 87 to 89 ms on the RTX
3080 Ti (2.5x and 2.7x). The card's time is mostly the copy back to the processor (about 3 GB/s on the discrete card, so a patch the
size of the frame costs 11 ms to read) and, when the card has been idle, its clocks falling: with 0.25 s between drawings the same 30
shapes took 41 ms instead of 17. A Radeon sharing memory with the processor pays almost nothing for the copy, hence its larger gain.
The processor's own figure is 36 ms because that drawer is the fast one of the Roto drawer setting.

Through the evaluator, the Roto node of the 30-shape comp takes 42 ms per frame on the processor and 25 ms on the card (frames 2 to
24, uncached, same pictures to 6e-7).

**Playback: no gain yet.** `benchmark_2d_playback.py` on the real display (`--display :0`, no sampler, uncached, 1080p, 48 frames),
with the playback comp's Roto replaced by 30 animated shapes (`tools/benchmark_roto_gpu.py --write-comp`), three alternating passes
each: the processor path (matte read ahead in its own process) gives 6.9 frames per second on the first lap and an interval p95 of
154 to 155 ms; the processor on the thread path (`NODEBASED_ROTO_PROCESS=0`) 6.9 to 7.0 and 156 to 158 ms; the card 6.4 to 6.6 and 161
to 172 ms. The matte is not what limits this comp: it is already drawn ahead of the playhead on a spare core, so a faster matte buys
nothing, and the card's work, its idle clocks and the copy back cost a few milliseconds on the frames that pace the run. Where the
matte is on the critical path (a Roto node evaluated while the artist scrubs or edits) the card is the faster draw. Follow-ups that
could change the playback result: keep the matte on the card inside the 2D island instead of reading it back, and raise the card's
clocks for a play with `cardclock` as the 3D path does; neither is part of this switch.

## Outline from a click (past Nuke)

In Nuke the artist clicks every point of the first outline by hand. Here the Bezier and B-Spline tools can outline a subject from one click.
It is a switch, **off by default**: with **Outline from click** off (a toggle in the strip of the Bezier and B-Spline tools, shared by the
node and left out of a saved shot while off) nothing changes, and the tools draw point by point as before. With it on:

* **Click inside the subject.** The outline appears as an ordinary new shape (a closed Bezier, or a B-Spline for that tool), selected, in
  **one undo step**. Everything that works on a shape works on it: move points, edge fit, in-betweens, point tracking.
* **Shift-drag draws a box first** for a harder subject (a subject the same colour as part of its background, or touching a twin). The box
  stays on the picture; the next click inside it outlines the subject inside the box and nothing outside it. A click outside the box ignores it.
* **Esc** cancels a pending box. A click where nothing can be found, or on a Roto node with no picture connected to its background input,
  says so in the status bar and changes nothing.

**How it finds the outline** (`nodebased/rotooutline.py`; pure numpy, no new dependency):

1. *Segmentation* by a continuous min-cut. GrabCut's graph cut needs a max-flow solver and numpy has none, so the cut is solved in its convex
   form, `min sum g |grad u| + sum f u` for `0 <= u <= 1`, with the Chambolle-Pock primal-dual iteration (a few array operations per step).
   `f` is the log-likelihood ratio of each pixel's colour under a foreground and a background colour mixture (three full-covariance Gaussians
   fitted by EM); the foreground model starts from a small disc around the click and the background model from the frame border, or from
   everything outside the box. `g` is small across strong colour edges so the cut follows them. As in GrabCut the mixtures are refitted from the
   result and the cut is solved again (three rounds). It runs on a copy of the picture reduced to 256 pixels on the long side; without a box a
   second, finer pass runs on the subject's bounding box.
2. *Contour*: the region connected to the click, holes filled, traced with marching squares (sub-pixel). The outline is the silhouette, so a
   hole through the subject is not cut out.
3. *Edge refinement*: the contour is resampled evenly and refined to the full-resolution edges with the active contour of the edge-snap
   feature (above).
4. *Bezier fit*: corners are found from the turning angle and Schneider's least-squares cubic fit runs between them. The error tolerance grows
   until there are at most 40 points, so straight runs get few points and curves more; a straight span gets zero tangents; a result with fewer than
   8 points is subdivided to 8. A last fit-to-edges pass is kept only if it raises the edge response along the outline without moving a point
   more than 1.5 px (on corners it rounds them off, and then it is not used). For the B-Spline tool the Bezier outline is turned into B-spline
   control points that pass through it; a B-spline has no corners, so sharp tips round off there.

**Measured** (`tests/test_roto_outline_i5.py`, 320 x 280 synthetic subjects at 4x oversampling; distance from every vertex of the outline to
the true boundary):

| Subject | Points | Mean | Max |
| --- | --- | --- | --- |
| Disc, radius 70 | 8 | 0.19 px | 0.45 px |
| Rounded rectangle | 14 | 0.26 px | 0.71 px |
| Five-point star, concave points found within 2.5 px | 16 | 0.20 px | 0.67 px |
| Disc on a noisy background | 8 | 0.25 px | 0.56 px |
| Low contrast (a 0.08 step with sensor noise) | 8 | 0.19 px | 0.49 px |

The tests hold a mean of 1.0 px and a maximum of 3.0 px (1.5 and 4.0 for low contrast) and 8 to 40 points. A click off the subject's centre gives the
same outline, and a box separates a subject from an overlapping twin that is otherwise taken as one blob. **Timing**, quiet machine, one
thread: 0.47 to 0.67 s for a 320 x 280 frame and 0.9 s for a 1920 x 1080 frame (a 220 px subject on a patterned plate; the radius comes out within
0.7 px). The viewer runs it on the click, with a wait cursor.

**Limits.** A subject the same colour as its background everywhere has no outline to find: draw a box, or place points by hand. The background
model comes from the frame border, so a subject that fills the frame needs a box. Fur, hair and motion blur have a soft edge that this does not
model; the outline follows the strongest edge.

## Lane 2 step notes

### Lane 2 I6: outlines with holes

Outline from click (`nodebased/rotooutline.py`) now cuts the holes out of a subject. `outline_with_holes(image, seed, box)` returns the
outline of the whole subject (what `outline_from_click` returns: the silhouette, holes filled) and one closed outline per interior hole.
The hole finder compares the segmented subject with its silhouette: whatever the silhouette holds that the subject does not is a hole.
A hole counts only when it covers **at least 1% of the silhouette's area and at least 16 cells of the 256 px working image**
(`MIN_HOLE_FRACTION`, `MIN_HOLE_CELLS`); a smaller region is a speck and stays part of the subject. Each hole goes through the same
contour, edge refinement and Bezier fit as the outline, so it follows the image's edges to the same accuracy. In the viewer the click adds
the outline and then each hole as a shape set to **subtract**, directly after the outline in the shape list (biggest hole first), all in one
command: one undo removes the outline and every hole, and the outline is the selected shape. With the B-Spline tool the holes are
B-Spline shapes too. Holes take the settings of the outline's tool except the blending mode, which is subtract.

Measured on synthetic subjects with 4 x 4 supersampled edges (tests/test_roto_outline_holes_i6.py): a ring (disc with a hole), a letter-like
block with two holes and a disc with one hole and a 3 px speck each give the expected number of holes (1, 2 and 1), and in the rendered matte
every pixel that disagrees with the true coverage lies within 0.93 px of the true edge. A hole that touches the subject's edge is part of the
background, not a hole; an island of a different colour inside a hole is cut out with the hole.

## Strokes that stick to the plate (past Nuke)

In Nuke a RotoPaint stroke painted on one frame stays where it was painted. To make it follow a moving plate the artist links it to a
tracker, or builds a SmartVector and a VectorDistort. Here a stroke (paint, eraser, clone, reveal, blur, sharpen, smear, dodge, burn) has a
**Stick to plate** command that carries it with the picture. It is **off unless run**: a stroke with no keys renders exactly as before, and a
saved shot that never used the command is unchanged.

* **Where.** Select a stroke, go to the frame it was painted on, and use **Stick to plate** in the Tracking section of the RotoPaint properties
  (next to Track points, with the same direction and frame range), in the right-click menu of the shape list, or in the strip of the Select
  tools. One undo step covers every key it writes. A locked stroke, or one with a transform of its own, is left alone and the status bar says so.
* **What it writes.** The stroke's points get keys on every frame in the range (and keep the painted frame exactly as painted), the same
  kind of key Track points gives a shape point, so lifetime, ripple, undo and the curve tools treat it as ordinary animation. The stroke
  also gets a `stick` entry with the local scale and turn of the motion, keyed the same way: the **brush size follows the scale** and the **clone
  offset follows the scale and turn**, so a clone keeps its source in the same place relative to the stroke as the picture turns and stretches under it.
* **Before the painted frame and after the range** the stroke holds the position of its first or last key.

**How it follows the picture** (`nodebased/rotostick.py`; pure numpy, reuses `opticalflow.flow_pair`):

1. A dense flow from the previous frame to the next is composed into a field that tells where every pixel of the painted frame has gone. A chain
   of small steps is accurate, but its errors add up.
2. Every third frame the picture is warped back by the field and a second flow, against the painted frame, adds the small leftover motion to
   it, so drift does not build up. This is the direct check.
3. Each point of the stroke moves by the field averaged over a 7 x 7 pixel neighbourhood. The mean 2 x 2 map of the field over the stroke gives
   the scale and the turn.
4. **When the plate is covered.** The warped picture is compared with the painted frame at up to 48 anchors along the stroke (normalised
   correlation of a 13 x 13 patch). A stroke whose anchors no longer match (an object passes over it, or the flow failed) falls back to the planar
   motion of its region: a homography fitted through the samples around the stroke that still match, and the planar tracker (`planar.track_region`)
   when too few do. The status bar names the frames ("stroke1: planar motion on frames 4\u20139"), and the field is reset to the planar motion so
   the chain picks the plate up again when it shows. A few low anchors on an otherwise good frame put only those points on the planar motion.

**Measured** (`tests/test_roto_paint_stick_i7.py`; a 320 x 240 smoothed random texture rendered through an analytic map, a 24-point curved
stroke painted on frame 1; distance from every point to where the plate really carried it, on frame 10):

| Plate motion | Mean | Max |
| --- | --- | --- |
| Translating (1.2 px right, 0.7 px down a frame) | 0.23 px | 0.64 px |
| Rotating (0.012 rad a frame, 6.2 deg in all, with a drift) | 0.22 px | 0.67 px |
| Bending (a non-rigid sine warp that moves points by different amounts) | 0.21 px | 0.72 px |

The tests hold 0.5 px mean and 1.5 px maximum, on every frame on the way. The rotation reported on frame 10 of the rotating plate is 0.108 rad
(true 0.108), and the scale on a zooming plate 1.135 (true 1.135). A clone offset of 40 x -25 px carried to frame 10 of the rotating plate puts
the source within 0.2 px of where the plate carried it. With a patch of a different texture sweeping over the stroke on frames 3 to 9, the
stroke falls back on frames 4 to 9, stays within 0.4 px mean and 0.8 px at worst on those frames, and is back on its texture when the
patch has gone. In the app the stroke ends frame 11 within 0.22 px mean and 0.56 px maximum.
**Timing**, quiet machine, one thread: 0.68 s a frame (16.4 s for 24 frames) for a 32-point stroke about 520 px wide on a 1920 x 1080 plate, working on the 779 x 459 pixel region around it. The time grows with the size of that region and the number of frames, and a short stroke is faster. The planar tracker's own pass over the range is added once, and only when too few samples match to fit a plane directly.

**Limits.** The stroke follows the picture, not the object it is painted on: a stroke on a surface that the plate hides for good, or on pixels
that are not in the plate at all, has nothing to follow. A plate with no texture under the stroke (a flat wall) has no motion to measure; the
planar fallback then reports it. The stroke's own transform (Roto transform knobs) is not combined with it. The clone offset keeps the stretch
and turn of the picture, not shear.

### Lane 2 I8: shapes from a matte (past Nuke)

A keyer, a generated matte or a painted alpha often gets a matte 90 % of the way, and the last 10 % is roto. Nuke cannot turn a matte into
shapes. In the shape list menu of a Roto node, **Shapes from matte** reads a matte from the Roto's Background or Mask input, traces it, and
adds ordinary editable shapes. The dialog asks for the source input, the channel (alpha, red, green, blue or luminance), the threshold
(default 50 %), the minimum area (default 64 px), key every Nth frame (default 4) and the first and last frame (default the timeline
range). Nothing in a document changes unless the command is run: it adds no node parameter and no schema field, and a saved shot
renders bit-identical.

**One frame (`rotooutline.matte_outlines`).** Each 4-connected region of at least the minimum area at or above the threshold becomes a
closed Bezier shape; each hole of at least the minimum area inside it becomes another shape set to **subtract**, and a smaller hole stays
filled. An island inside a hole is a shape that adds again, so nesting depth decides the mode (even adds, odd subtracts) and the shapes are
ordered by depth. The outline lies on the threshold level of the matte at sub-pixel accuracy (marching squares on the float matte; a
binary matte is smoothed by 0.8 px first), then goes through the contour resampling, corner finding and Bezier fitting that Outline from
click uses, with 8 to 40 points per outline. There is no edge refinement pass: the contour is already on the matte's own level.

**Feather.** The edge width of a region is measured as the area of its 10 % to 90 % band divided by the length of its outline, and
divided by 0.8 to give the full rise. The shape gets a **box feather** whose window (2 x feather + 1 px) rises over that width, so the
rendered edge rises like the matte's and the 50 % level stays on the outline. A matte whose edge is under about 2 px wide gets feather 0
(hard edge). Box feather values are whole pixels, so the rendered edge width is within one pixel of the matte's.

**A frame range (`rotomatte`).** The matte is traced on every Nth frame and the last frame. Each outline is matched to the previous key's
outline by overlap (the pair must be each other's best overlap and share at least 10 % of their union). The matched outline is resampled
from its matched start point: the previous points are laid on the new outline at the same shares of the way round, starting where they fit
best after the shift of the centres, and each corner moves to a nearby corner of the new outline, with the points between following
proportionally. The handles are then fitted to the new outline, so point k is on the same side of the shape on every key. Points are
keyed with linear curves, and the shape's lifetime is the range of frames it exists on.

A region that splits, merges, appears or vanishes starts or ends a shape. When a gap between two keys holds such an event, or the matte
moves further than its own width so nothing overlaps, that gap is traced frame by frame so the event lands on its real frame, and the
extra frames become keys. Each event is named in the status bar, for example "matte2 split off matte1 at frame 7". A merge ends the shape
with the smaller overlap ("matte2 merged into matte1"). Fast motion over a thin region and regions that cross each other are matched by
overlap alone, so they may swap.

**In-betweens.** The frames between keys are interpolated and then filled with the in-betweens fit against each frame's own matte (band 8 px,
threshold 0.75 px). The fit snaps to the strongest gradient, which on a hard matte can tie between two pixel rows, so a frame keeps the
fitted keys only when they bring the outline closer to the matte's level contour than the interpolation was (measured by the softened matte
along the outline). The fill can therefore only improve a frame.

Measured (tests/test_roto_matte_shapes_i8.py, 4 x 4 supersampled mattes; the tests print the figures; no timing is asserted):

| Case | Result |
| --- | --- |
| disc, radius 70 | one shape, 8 points, rendered edge within 0.09 px of the true edge |
| ring | outline and one subtracting hole, radii 80.0 and 35.0 within 0.4 px, ring full and hole empty |
| two discs; island in a hole | two shapes; union, subtract, union |
| soft edges, ramps of 6, 10 and 16 px | feather 2, 4 and 7; rendered edge width within 20 % of the matte's; largest alpha difference 0.14, 0.10 and 0.08 |
| moving ellipse, 12 frames, key every 4th | worst disagreeing pixel 0.94 px from the true edge on keyed and in-between frames; point 0 stays within 15 degrees round the shape |
| accelerating disc, keys 1 and 5 | interpolation alone misses by 13.2 px; with the fill 0.92 px |
| two discs that part (frame 7) | one split event on frame 7; shapes live 1 to 12 and 7 to 12; every frame within 1 px |
| a region that vanishes and one that appears | vanished on frame 7, appeared on frame 9; lifetimes 1 to 6, 1 to 12 and 9 to 12 |

Trace time on one frame: 0.07 s for two soft-edged regions on a 1920 x 1080 matte, 0.37 s for 12 frames of a 320 x 280 matte including the fill
(measured while other lanes were running). One undo step removes every shape the command added.

## Lane 8 step notes

**2026-10-20, Lane 8 step T2 finish 1 (partial).** The merged interaction and PlanarTracker worker changes remain as described above. Real-display QA on the 1920 × 1080 comp used the RTX 3080 Ti and llvmpipe. On the RTX, first picture after a Roto move arrived 1.142 s from the first move, after release; full-tier settle took 3.888 s. An uncached scrub presented tier 8 at 0.595 s, tier 4 at 0.978 s and tier 1 at 4.914 s. At the 24 fps target, measured presentation was 0.125 fps at tier 1, 2.624 fps at tier 4 and 12.84 fps with automatic adaptation; the tier readout stepped through 1, 2 and 4 and finished at 1/4. The RTX run held the exclusive card lock. On llvmpipe, with Lane 4's integrated-adapter progression benchmark running concurrently, first picture after a Roto move arrived 1.414 s from the first move, after release; full-tier settle took 4.416 s. The uncached scrub first picture arrived at 0.808 s and settled at 9.446 s. Playback measured 0.125 fps at tier 1, 1.618 fps at tier 4 and 0.125 fps in automatic mode (tier readout 1 and 2 during the sample). The llvmpipe rates are contention-affected. Both adapters miss the 0.3-second first-picture target; the RTX automatic mode also misses the set 24 fps rate. The first provisional playback pass counted preview generations; its rates are excluded. The cube-face pick remains a bad pick because it is flat and featureless; the checker-floor failures at grids 8, 16 and 24 still point to a separate robustness limitation. The under-0.3-second first-picture and set-rate playback gates remain unmet.

### Lane 8 T1: ROI correctness

The tile executor keeps the picture visible when a node lacks a tile-native kernel: it renders that node full-frame, crops to the requested region and names the fallback in the ROI badge. Tracker and Stabilize now use the solved Transform inverse map to request their input patch, while retaining the untouched output patch for mask/mix. Their badges disappear on that native path. Requests crossing the target data window clip to its edge. ROI handle presses are resolved before Roto drawing, so a selected Roto no longer turns a box-handle press into a new shape.

The bundled `tests/fixtures/qa.nbcomp` is compared against the full evaluator inside a 960 × 540 box, both with its Tracker upstream and with the Roto background wired directly to RotoPaint; both comparisons are byte-identical. The test uses the QA plate sequence from `scratch/nb-qa/1015-roi` when available.

At tier 1 on llvmpipe, five cold-cache ROI-off runs before the backdrop reveal left the outside dimmed until the full frame arrived in 3.92–3.98 s. Afterward, the first paint (8.4–9.3 ms) showed the current ROI patch over the saved full frame at normal strength; refinement completed in 4.02–4.58 s.

On the 12 GB RTX 3080 Ti at tier 1, five ROI-off runs before the reveal painted in 8.2–9.4 ms but kept the backdrop dim until the full frame arrived in 3.78–3.90 s. After the change, first paint remained 8.3–9.2 ms and showed the saved full frame at normal strength; refinement took 3.95–4.48 s. The change fixes what is visible while refinement runs; it does not claim to accelerate that full-frame render. Raw runs are in `scratch/nb-lanes/run/bench-L8-roi-off-rtx3080ti-{before,after}.{json,log}`.

### Lane 8 step B: live-picture gates (finish 1 complete)

The remaining gaps are closed on the RTX 3080 Ti / Vulkan with the 1920 × 1080 QA comp. Solid-over RotoPaint strokes now rasterize and composite only their clipped stroke bounds; the expanded regression compares output bytes against full-tile rasterization. The field benchmark now keeps Properties-slider movement horizontal and records the largest gap between new pictures during each drag. A first run on a 1920 × 1080 virtual desktop put the Properties panel below the drawable screen (its slider was at y=1293); the valid rerun used a 1920 × 1800 desktop so the controls were visible.

| Series | Before (RTX 3080 Ti) | After (RTX 3080 Ti) | Gate |
|---|---|---|---|
| Drag | Roto first picture 1.039 s after movement, no in-drag picture, 3.768 s settle; other subjects not measured separately | Roto 0.189 s first / 0.182 s max gap / 0.930 s settle; RotoPaint 0.034 / 0.202 / 0.849; Tracker 0.022 / 0.278 / 0.763; PlanarTracker 0.031 / 0.082 / 0.477; Properties mix slider 0.004 / 0.195 / 0.702 s, with 8 pictures and knob value 1.000 → 0.771 | All five first pictures under 0.3 s, max gaps under 0.3 s and settles under 2 s |
| Uncached scrub | Tier 8 0.566 s; earlier T2 tier 4 0.978 s; settle 5.098 s | Tier 8 0.241 s; tier 4 0.050 s; newest of 24 scrubbed frames received tier 8 | Both first-picture targets pass; newest frame shown at tier 8 |
| Playback | Cold tier 4 2.622 fps; automatic 12.84 fps | Warm automatic tier 1 23.731 fps over 8 s; warm cached tier 4 23.72 fps over 8.01 s, all 24 frames cached and in order | Both meet the ≥23 fps gate; three dropped frames in the tier-4 sample |

Maximum in-drag picture gaps were 0.1821 s for Roto, 0.2023 s for RotoPaint, 0.2784 s for Tracker, 0.0824 s for PlanarTracker, and 0.1947 s for Properties. These come from newly shown frame events between the first movement and release. All requested RTX drag, scrub and playback gates now pass. Existing llvmpipe and 2×/four-core integrated stand-in observations from the prior pass remain recorded above; these finish-1 changes were not rerun on those adapters. No A6000 or off-site laptop numbers are claimed.

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

Cold tier-4 playback measured 21.330 fps on RTX, 20.979 on llvmpipe and 15.463 on the 4-core integrated stand-in, all below 24 fps. A separate range-prewarmed run produced 192/192, 192/192 and 193/193 display-cache hits during playback with zero graph evaluations, yet measured 23.842, 23.845 and 23.841 fps. That isolates the remaining near-target shortfall to playback/display cadence rather than graph rendering; the exact 24 fps gate remains unmet. Those hit counts come from the archived event timelines. The old direct residency field queried `region=None` and undercounted region-keyed entries; the harness now checks residency by content identity and frame, but the corrected verifier was not rerun end-to-end because other lanes held the exclusive GPU queue. Focused coordinate tests and the benchmark self-check pass. The lane remains partial: the 24 fps playback gate is still open on every adapter.

Cold first-play profile (October 22): `benchmarks/live-picture/profile-1022-cold-rtx.json` records 22 foreground source decodes during the 8-second tier-4 playback interval, all on the preview worker (median 85.3 ms, p95 98.2 ms, 1.946 s cumulative). It also records 22 RotoPaint rasterizations (median 47.2 ms, 1.038 s cumulative), so source decode was the largest serial per-frame cost. The profile revealed why the existing read-ahead had no effect on this composition: broad Roto requests use the full-frame Evaluator, which did not consult the DecodeAheadPool, and prefetch traversal omitted optional wired inputs such as `Roto.bg`. The implementation now prefetches those optional ancestors and lets both the tile path and full-frame Evaluator consume the same bounded, immutable `Raster` decode, including named layers. Initial four-worker cold adapter reruns are recorded in `after-*-cold-decodepool.json`; the measured contention and follow-up tuning are summarized below.

The first four-worker rerun exposed a regression despite the cache working: it supplied 21/22 foreground source reads on each adapter, while playback fell to 15.564 fps RTX, 15.571 fps llvmpipe and 7.617 fps on the 4-core integrated stand-in (which dropped 24 frames). Timeline evidence showed four concurrent EXR decoders contending with the preview/UI path. With one worker, RTX measured 21.356 fps and llvmpipe 20.953 fps, with 21/22 read-ahead hits on both. A clean one-worker integrated sample in `after-laptop-standin-cold-decodepool-w1-clean.json` measured 13.741 fps; because even one background decoder starved the 4-core stand-in, the final app policy disables read-ahead when the process has four or fewer available CPUs. The final no-prefetch integrated run measured 16.473 fps with zero dropped frames. Thus the decode path is now affinity-aware and no longer regresses low-core playback, but the strict cold 24 fps target remains unmet on all adapters. Current final comparisons are `after-rtx3080ti-cold-decodepool-w1.json`, `after-llvmpipe-cold-decodepool-w1.json` and `after-laptop-standin-cold-adaptive-decodepool.json` (the integrated name denotes the affinity-selected disabled-pool path). The earlier `after-laptop-standin-cold-decodepool-w1.json` overlapped a separate CPU test and is excluded from comparison. Focused tile, decode-pool, media, bounding-box, live-picture benchmark and RotoPaint suites pass (114 tests). The lane remains PARTIAL; further work must reduce graph/evaluation time or display cadence enough to reach 24 fps without starving the low-core preview.
