# 2D node parity with Nuke

Status: audit, written 2026-09-21 against `main` at `8b49dfa`. This is the first deliverable of
lane L2 (`context/lanes.md`) and the document every later L2 commit flips a row in — a node moves
from **missing** or **partial** to **supported** only once it has a kernel on both evaluation
paths (`nodebased/imaging.py`'s `Evaluator` and `nodebased/tileexec.py`'s `TileExecutor`), mask and
mix where Nuke has them, bypass behaviour, `LIMITS`/`CHOICES`, knobs, a docs row flipped here, and
tests with pixel assertions — see `tests/test_bypass.py` for the shape that proof takes.

## Method

Node classes and one-line descriptions are read from the Foundry Nuke 17.0/17.1 Reference Guide
(`learn.foundry.com/nuke/17.0/content/reference_guide/`), grouped exactly as Nuke's own Toolbar
groups them (`getting_started/using_interface/using_toolbar.html`): Image, Draw, Time, Channel,
Color, Filter, Keyer, Merge, Transform, Metadata, Other. Nuke's 3D and Particles/Deep/Views
toolbar groups are out of this lane's scope (L3, L5, L6). Within each group, rows are ordered by
judged daily-use rank for a working compositor — highest first — not alphabetically as Nuke's own
docs list them.

Status is judged against `main` at `8b49dfa` (`nodebased/core.py` `SPECS`):

- **supported** — a NodeBased node exists with materially the same behaviour.
- **partial** — some of the behaviour exists (as a knob on another node, or a degraded
  approximation), but not as Nuke's dedicated node or not to Nuke's full extent.
- **missing** — nothing in NodeBased does this today.

Today's full 2D node set on `main`: Read, Constant, Checker, Grade, ColorCorrect, Blur, Transform,
Crop, Shuffle, ChannelShuffle, Roto, Tracker, Merge, Premult, Unpremult, Dot, Switch, Viewer,
Write. Nineteen nodes against roughly 140 in Nuke's 2D toolbar groups.

## Image

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Read | supported | `Read`, with padded-sequence patterns, colourspace/alpha-mode and per-node missing-frame policy (`docs/TIME_MODEL.md`). |
| 2 | Write | supported | `Write`, EXR/PNG only (`WRITE_FILE_TYPES`); Nuke writes far more formats but the two here are real. |
| 3 | Viewer | supported | `Viewer`; no A/B wipe or channel-isolation UI beyond `to_qimage`'s single-channel view yet. |
| 4 | Constant | supported | `Constant`. |
| 5 | UDIM Import | missing | No UDIM texture-patch import; not needed until a texturing workflow exists. |
| 6 | CurveTool | missing | No per-frame pixel analysis-to-curve node. |
| 7 | Profile | missing | No in-graph performance-probe node; `TASKLOG.md` timings substitute today. |

## Draw

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Roto | partial | `Roto` rasterises Bezier/B-spline shapes with feather and mode (`docs/ROTO_TRACKING.md`), but has none of RotoPaint's paint tools. |
| 2 | Text | missing | No text-rendering node at all — used constantly for slates, burn-ins and labeling; not in this lane's ranked list but high daily use. |
| 3 | Rectangle | missing | On the lane's ranked list (group e). No solid-rectangle generator over an input. |
| 4 | Ramp | missing | On the lane's ranked list (group e). No linear gradient generator. |
| 5 | Radial | missing | On the lane's ranked list (group e). No radial gradient generator. |
| 6 | Noise | missing | On the lane's ranked list (group e). No procedural noise generator. |
| 7 | RotoPaint | missing | Roto's paint tools (clone, blur, dodge strokes) do not exist; a materially bigger lift than plain Roto. |
| 8 | LightWrap | missing | Common comp-finishing node; not on the lane's ranked list. |
| 9 | Grain / ScannedGrain | missing | No film-grain synthesis or scan-grain matching. |
| 10 | DustBust / MarkerRemoval | missing | Roto-driven paint-out tools; depend on RotoPaint. |
| 11 | Flare / Glint / Sparkles | missing | Stylised lens-artifact generators; low daily use outside specific shots. |
| 12 | Dither | missing | Quantization-noise node; low priority without 8-bit delivery paths yet. |
| 13 | Grid | missing | Debug/reference overlay; rarely used in a finished comp. |

## Time

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| — | *(none supported)* | | NodeBased has no time-manipulation node today; `Read`'s `frame_offset` is the only time control anywhere in the graph. |
| 1 | TimeOffset | missing | On the lane's ranked list (group g). Shifts a clip forwards/backwards in time. |
| 2 | FrameHold | missing | On the lane's ranked list (group g). Freezes on one frame for every output frame. |
| 3 | Retime | missing | On the lane's ranked list (group g). Speed change with an interpolation choice. |
| 4 | TimeClip | missing | Offsets/reverses playback range; overlaps with `TimeOffset` and `Read`'s range. |
| 5 | FrameRange | missing | Restricts the frame range a branch presents downstream. |
| 6 | AppendClip | missing | Splices clips head-to-tail; needs a multi-clip timeline concept `docs/TIME_MODEL.md` does not have yet. |
| 7 | TimeBlur / TimeWarp / TimeEcho | missing | Motion-blur-adjacent retiming; needs `Retime`/`Kronos`-grade groundwork first. |
| 8 | Kronos / OFlow / SmartVector / VectorToMotion | missing | Optical-flow retiming; a research-grade lift, far below `Retime` in priority. |
| 9 | NoTimeBlur | missing | A cache-shaping hint node with no analogue in the current evaluator. |

## Channel

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Shuffle | supported | `Shuffle`, single-input channel routing with constants. |
| 2 | ShuffleCopy | partial | `ChannelShuffle` covers the same two-input explicit-routing need but as a distinct node rather than Nuke's unified Shuffle/ShuffleCopy pair. |
| 3 | Copy | missing | On the lane's ranked list (group b). Replaces named channels in B with channels from A — distinct from a full Merge. |
| 4 | ChannelMerge | missing | On the lane's ranked list (group b). Per-channel arithmetic merge into a chosen output channel. |
| 5 | Remove | missing | Deletes channels/layers from a stream; NodeBased's four-channel RGBA model has no extra layers to remove yet. |

## Color

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Grade | supported | `Grade` (exposure/multiply/offset), plus mask + mix. |
| 2 | ColorCorrect | supported | `ColorCorrect` (lift/gamma/gain/saturation), plus mask + mix. |
| 3 | Invert | missing | On the lane's ranked list (group b), first row. No dedicated invert node; only achievable today by hand-tuning Grade/ColorCorrect into an approximation. |
| 4 | Saturation | missing | On the lane's ranked list (group b). `ColorCorrect.saturation` exists as a knob, not as Nuke's standalone scrub-friendly node. |
| 5 | Multiply | missing | On the lane's ranked list (group b) as its own node, separate from Grade's `multiply` knob. |
| 6 | Add | missing | On the lane's ranked list (group b), separate from Grade's `offset` knob. |
| 7 | Gamma | missing | On the lane's ranked list (group b), separate from ColorCorrect's `gamma` knob. |
| 8 | Clamp | missing | On the lane's ranked list (group b). No range-constraining node; HDR values pass through every filter unclamped today. |
| 9 | HueCorrect | missing | Per-hue-range saturation/luma adjustment; common grading tool, not on the ranked list. |
| 10 | Exposure | partial | `Grade.exposure` covers the stop-based math; no standalone node with Nuke's black-point-preserving formula. |
| 11 | HSVTool | missing | Combined hue/saturation/value adjuster; overlaps HueCorrect in use case. |
| 12 | ColorMatrix | missing | Arbitrary 3x3 RGB matrix; niche outside specific grain/colour-space fixes. |
| 13 | Colorspace / OCIOColorspace / OCIODisplay / OCIOLookTransform / OCIOFileTransform / OCIOLogConvert | partial | The document's colour pipeline is a single fixed ACES config (`docs/COLOR_MANAGEMENT.md`; `validate_settings` rejects any other config/working-space/display); `Read.colorspace` offers a fixed enum, but there is no general per-node colourspace-conversion node. |
| 14 | Log2Lin / PLogLin | missing | Log/lin conversion nodes; redundant while the working space is fixed ACEScg with no log-encoded intermediate format. |
| 15 | MatchGrade / ColorTransfer | missing | Automatic grade-matching between two clips; a research-grade addition. |
| 16 | Expression | missing | Per-channel Tcl-like formula node; NodeBased's expression system (`docs/EXPRESSIONS` via `expressions.py`) exists at the parameter level, not as an image-processing node. |
| 17 | CrossTalk | missing | Channel-bleed simulation; low daily use. |
| 18 | Posterize / SoftClip / Toe | missing | Specialised tonal-response shapers; occasional use. |
| 19 | HistEQ / Histogram / MinColor / Sampler | missing | Analysis/diagnostic tools rather than image transforms; lowest priority in this group. |
| 20 | GenerateLUT / Vectorfield | missing | LUT-authoring and LUT-application nodes; no LUT I/O anywhere in NodeBased yet. |
| 21 | Truelight | n/a | Superseded by Baselight for Nuke per Foundry's own docs; not a parity target. |

## Filter

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Blur | supported | `Blur`, separable box filter, plus mask + mix. |
| 2 | Sharpen | missing | On the lane's ranked list (group c). Common finishing filter; no unsharp-mask kernel exists. |
| 3 | Erode (filter) | missing | On the lane's ranked list (group c) as `FilterErode`. Matte grow/shrink; used on every key. |
| 4 | Median | missing | On the lane's ranked list (group c). Despeckle/cleanup filter. |
| 5 | Glow | missing | On the lane's ranked list (group c). Bloom/glow finishing effect. |
| 6 | Soften | missing | On the lane's ranked list (group c). Gaussian-leaning soften, distinct from `Blur`'s box filter. |
| 7 | DropShadow | missing | Very common compositing finishing node; not on the ranked list. |
| 8 | Defocus / ZDefocus | missing | Disc-based defocus, the second most common blur after box blur; `ZDefocus` needs a depth channel NodeBased has no concept of yet. |
| 9 | DirBlur | missing | Directional/zoom blur; common for speed/impact effects. |
| 10 | EdgeBlur / EdgeExtend | missing | Matte-edge treatment tools; depend on having Erode/Blur first. |
| 11 | Erode (fast) | missing | The GPU-cheap sibling of Erode (filter); same priority tier once one lands. |
| 12 | Denoise / DegrainSimple | missing | Grain/noise removal; a research-grade filter, not a box/erode-class kernel. |
| 13 | MotionBlur / MotionBlur2D / MotionBlur3D / VectorBlur | missing | All depend on a motion-vector pipeline NodeBased does not have. |
| 14 | Bilateral | missing | Edge-preserving smooth; specialised, occasional use. |
| 15 | Convolve / Matrix | missing | User-supplied kernel filters; power-user tools, low daily use. |
| 16 | EdgeDetect / Emboss / BumpBoss / Laplacian | missing | Stylised/edge-analysis filters; occasional use. |
| 17 | Inpaint | missing | Content-aware fill; a research-grade addition. |
| 18 | GodRays / VolumeRays / LevelSet | missing | Specialised lighting/level-set filters; low daily use. |
| 19 | ZSlice | missing | Depth-channel slicing; needs a depth channel concept. |
| 20 | Bokeh | missing | Depth-driven bokeh blur; needs a depth channel concept. |

## Keyer

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Keyer | missing | On the lane's ranked list (group d) as luminance keying. Foundry's built-in collection of simple keyers; none exist. |
| 2 | ChromaKeyer | missing | On the lane's ranked list (group d). The everyday green/bluescreen keyer; highest-use keyer in a working pipeline. |
| 3 | Difference | missing | On the lane's ranked list (group d) as "Difference key". Simplest keyer, also usable as a comparison tool. |
| 4 | HueKeyer | missing | On the lane's ranked list (group d). Hue-based matte extraction. |
| 5 | Keylight | missing | Industry-standard colour-difference keyer; very high daily use where licensed, but the algorithm is proprietary — out of reach without a from-scratch equivalent. |
| 6 | Primatte / Ultimatte | missing | Commercial keying algorithms with the same licensing barrier as Keylight. |
| 7 | IBKColor / IBKGizmo | missing | Image-based-keying pair; depends on a clean-plate workflow, lower priority than the single-input keyers above. |
| 8 | Cryptomatte / Encryptomatte | missing | ID-matte extraction from renderer metadata; NodeBased's `Render3D` has no Cryptomatte-style ID pass to key from yet. |

## Merge

See the dedicated "Merge operations" audit below for the `operation` dropdown itself. This table
covers Nuke's separate single-purpose Merge-toolbar nodes.

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Merge | partial | `Merge` exists with mask + mix and 16 of Nuke's 30 documented operations (see below); the node itself is supported, its operation coverage is not. |
| 2 | Premult | supported | `Premult`. |
| 3 | Unpremult | supported | `Unpremult`. |
| 4 | Switch | supported | `Switch` (two inputs; Nuke's goes to any number). |
| 5 | Dissolve | missing | On the lane's ranked list (group b). Weighted cross-fade between two inputs; very common transition tool. |
| 6 | KeyMix | missing | On the lane's ranked list (group b). Copies A into B wherever a mask is non-black; distinct from `Merge`'s per-pixel mask gate because it is a hard copy, not a blend. |
| 7 | AddMix | missing | `over` plus a premultiply step; a thin wrapper over existing `Merge`+`Premult`, low priority once both exist. |
| 8 | Blend | missing | Weighted average across any number of inputs; `Merge`'s `average`/`plus` operations (once added) cover the two-input case. |
| 9 | CopyRectangle / CopyBBox | missing | Rectangular patch/bbox-copy tools; specialised cleanup use. |
| 10 | ContactSheet | missing | Debug/review grid of inputs; not a compositing operation. |
| 11 | TimeDissolve | missing | A `Dissolve` driven by a time curve instead of a static mix; depends on `Dissolve` landing first. |
| 12 | ZMerge | missing | Depth-sorted merge; needs a depth channel concept. |
| 13 | Absminus / In / Matte / Max / Min / Multiply / Out / Plus / Screen | n/a | Each is a single-operation convenience wrapper around one `Merge` operation; covered by the operations audit below rather than as separate nodes. |
| 14 | VariableSwitch | n/a | A Nuke-scripting Variables/scope construct, not an image operation; not a parity target. |

### Merge operations

`MERGE_OPERATIONS` in `nodebased/core.py` today (16, alphabetised for this table): `atop`,
`difference`, `divide`, `in`, `mask`, `max`, `min`, `minus`, `multiply`, `out`, `over`, `plus`,
`screen`, `stencil`, `under`, `xor`. Verified against Foundry's documented algorithms
(`learn.foundry.com/.../merge_operations.html`): the `_merge_op` formulas in `imaging.py` for
`over`, `under`, `atop`, `xor`, `in`, `out`, `mask`, `stencil`, `plus`, `minus`, `multiply`,
`screen`, `max`, `min`, `difference` match Nuke's documented algorithms exactly, so the existing
16 are correctness-checked, not just present.

Nuke ships 30 named operations in the `operation` dropdown. Missing, ranked:

| Rank | Operation | Algorithm | Reason it matters |
|---|---|---|---|
| 1 | `average` | `(A+B)/2` | On the lane's ranked list (group a); common for blending two mattes or plates. |
| 2 | `from` | `B-A` | On the lane's ranked list (group a); the inverse of `minus`, used constantly for paint-fix deltas. |
| 3 | `hypot` | `sqrt(A*A+B*B)` | On the lane's ranked list (group a); the standard alternative to `screen` for reflections/glints. |
| 4 | `matte` | `Aa+B(1-a)` | Premultiplied-over on already-unpremultiplied inputs; common when working with straight-alpha layers. |
| 5 | `disjoint-over` | `A+B(1-a)/b, A+B if a+b<1` | Fixes the dark-seam artifact plain `over` produces on abutting held-out CG passes; a real production need once multi-pass CG compositing is common. |
| 6 | `conjoint-over` | `A+B(1-a/b), A if a>b` | Companion to `disjoint-over` for overlapping (rather than abutting) held-out passes. |
| 7 | `copy` | `A` | Trivial once mask/mix exist on `Merge`; mostly a convenience over wiring A straight through. |
| 8 | `exclusion` | `A+B-2AB` | A softer `difference`; occasional use, e.g. gentler comparison mattes. |
| 9 | `geometric` | `2AB/(A+B)` | Another averaging mode; rare in practice. |
| 10 | `overlay` | `multiply if B<0.5, screen if B>0.5` | Common in paint/2D-art workflows, less so in photographic compositing. |
| 11 | `hard-light` | `multiply if A<0.5, screen if A>0.5` | `overlay` with the roles of A and B swapped; same priority tier. |
| 12 | `soft-light` | `B(2A+(B(1-AB))) if AB<1, 2AB otherwise` | Gentler `hard-light`; lowest-use of the photographic blend modes here. |
| 13 | `color-dodge` | brighten B towards A | Photo-editing-style blend mode; rare in VFX compositing specifically. |
| 14 | `color-burn` | darken B towards A | Same tier as `color-dodge`. |

## Transform

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Transform | supported | `Transform` (translate/rotate/scale/center/filter), plus mask + mix, inverse-mapped with sub-pixel filtering. |
| 2 | Crop | supported | `Crop`, plus mask + mix, shrinks the data window. |
| 3 | Tracker | partial | `Tracker` applies a solved match-move/stabilise transform (`docs/ROTO_TRACKING.md`), but there is no standalone `Stabilize` node and no point-tracking UI beyond what `Tracker`'s node_data already carries. |
| 4 | Reformat | missing | On the lane's ranked list, its own numbered deliverable ("a real format model") precisely because it changes every downstream node's notion of format; needs a written design (`docs/` doc + integrator sign-off) before code, per the lane brief. |
| 5 | CornerPin2D | missing | On the lane's ranked list (group f) as `CornerPin`. Four-point perspective pin, a very common screen-replacement tool. |
| 6 | Mirror | missing | On the lane's ranked list (group f). Flip/flop around the format centre. |
| 7 | Stabilize | partial | `Tracker.mode = "stabilise"` covers the pixel math; Nuke exposes it as its own node with its own knob set. |
| 8 | Position | missing | Integer-pixel move, a restricted subset of `Transform`; low priority once `Transform` exists. |
| 9 | AdjustBBox | missing | Expands/crops the bounding box without moving pixels; a bounding-box utility (`docs/BOUNDING_BOX.md`) NodeBased's box model could grow into. |
| 10 | BlackOutside | missing | Fills outside the bounding box with black; a thin utility over the same bounding-box model. |
| 11 | STMap | missing | Absolute-position pixel remapping from a UV pass; needs a UV/vector-channel convention. |
| 12 | IDistort | missing | UV-channel-driven warp; same dependency as `STMap`. |
| 13 | GridWarp / GridWarpTracker / SplineWarp | missing | Bezier/spline-grid warping tools; a significant standalone feature, low priority against the ranked list. |
| 14 | Tile | missing | Scaled-down tiled copies; niche, mostly for texture/pattern work. |
| 15 | VectorCornerPin / VectorDistort | missing | SmartVector-driven paint propagation; depends on the Time group's `SmartVector` landing first. |
| 16 | PointsTo3D / Reconcile3D | missing | 2D/3D point-correspondence tools; sit closer to L3's scope than L2's. |
| 17 | TVIScale | missing | Legacy power-of-two scaler; effectively superseded by `Transform`/`Reformat`. |

## Metadata

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| — | *(none supported)* | | NodeBased carries no per-image metadata concept anywhere in the graph or the document schema. |
| 1 | ViewMetaData | missing | Read-only metadata inspector; the cheapest of the five to add once any metadata exists to inspect. |
| 2 | ModifyMetaData | missing | Add/edit/remove metadata key-value pairs. |
| 3 | CopyMetaData | missing | Copies metadata from one stream to another. |
| 4 | AddTimeCode | missing | Timecode injection; depends on the Time group existing first. |
| 5 | CompareMetaData | missing | Diagnostic diff tool; lowest daily use of the five. |

Not on the lane's ranked node list; genuinely low priority until a Read/Write round-trip carries
metadata worth inspecting.

## Other

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Dot | supported | `Dot`, a neutral graph reroute. |
| 2 | NoOp | missing | Passthrough label node distinct from `Dot`; low priority, `Dot` already covers the graph-tidying use case. |
| 3 | Backdrop | missing | Node-graph organisational box; a UI feature (out of L2's evaluator/kernel scope) rather than a pixel node. |
| 4 | PostageStamp | partial | `node_thumbnail`/`DEFAULT_THUMBNAIL_TYPES` already give Read/Constant/Checker an inline postage-stamp-style preview in the graph; there is no standalone node that re-displays another node's output elsewhere in the graph the way Nuke's `PostageStamp` does. |
| 5 | Group / Input / Output | missing | Nesting/sub-graph nodes; a real architectural feature, well beyond a single L2 deliverable. |
| 6 | BlinkScript | missing | User GPU-kernel DSL; out of scope without a Blink-equivalent runtime. |
| 7 | BurnIn | missing | Text/metadata burn-in for review; depends on the `Text` node (Draw group) landing first. |
| 8 | LiveGroup / LiveInput / VariableGroup / Link | missing | Multi-artist collaboration and scripting-variable features; far out of L2's kernel-parity scope. |
| 9 | DiskCache | n/a | NodeBased's disk tier is a transparent evaluator feature (`docs/EVALUATION_TIERS.md` C4), not an explicit graph node; no parity gap. |
| 10 | Precomp / Root / Annotations / Assert / AudioRead | missing | Script-management, project-settings and QA nodes; lowest priority in this group. |

## Summary

Of the roughly 140 Nuke 2D-toolbar node classes surveyed, 8 are **supported** (Read, Constant,
Viewer, Write, Shuffle, Premult, Unpremult, Switch, Dot — nine, including Dot from Other), 7 are
**partial** (Roto, ShuffleCopy, Exposure, Colorspace/OCIO family, Merge, Tracker, Stabilize,
PostageStamp — eight), and the remainder are **missing**. The lane's ranked build order (groups
a–g in `context/lanes.md`) targets the highest-daily-use missing rows first: Merge operations,
Invert/Clamp/Saturation/Multiply/Add/Gamma, Erode/Median/Sharpen/Glow/Soften, the four keyers,
Ramp/Radial/Rectangle/Noise, Mirror/CornerPin, then the Time group, with Reformat last and gated
on a written design.
