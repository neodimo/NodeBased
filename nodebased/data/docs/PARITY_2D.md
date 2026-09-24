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
| 2 | Text | supported | `Text`, rendered through Qt's own text rasteriser offscreen (no new dependency); font availability is a machine property, not something NodeBased embeds or controls. |
| 3 | Rectangle | supported | `Rectangle`, an area box with fractional softness, colour and an optional image input the box is composited over. |
| 4 | Ramp | supported | `Ramp`, a linear gradient between two points and two colours, Nuke's own knob names. |
| 5 | Radial | supported | `Radial`, a soft-edged disc inscribed in an area box. |
| 6 | Noise | supported | `Noise`, Nuke's own fractal-noise knobs (size/z_slice/octaves/lacunarity/gain/gamma) plus a seed so two runs can be compared deterministically. |
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
| 1 | TimeOffset | supported | `TimeOffset` (`time_offset`, `reverse`), evaluates its input at `frame - time_offset` (or `+` when reversed) via a nested evaluate call — see `docs/TIME_MODEL.md`. |
| 2 | FrameHold | supported | `FrameHold` (`first_frame`, `increment`), freezes on `first_frame` at increment 0 (Nuke's own default), or steps forward every `increment` frames. |
| 3 | Retime | supported | `Retime`, simplified: nearest-frame sampling only, no frame blending. `input_range`/`output_range` plus `speed` map a frame linearly (`input_start + (frame - output_start) * speed`); the range end knobs are carried for Nuke-parity naming and are not consulted by the simplified mapping. |
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
| 3 | Copy | supported | `Copy`, plus mask + mix. Four `copy_*` knobs (one per output channel) each pick a source channel from A or `"none"` to leave that channel as B's own — a real reduction of Nuke's per-channel `from`/`to` pairs onto this app's fixed four-channel model. |
| 4 | ChannelMerge | supported | `ChannelMerge`, plus mask + mix. `a_channel`/`b_channel` pick one scalar channel from each input, `operation` reuses NodeBased's own 30-operation `MERGE_OPERATIONS` vocabulary (rather than Nuke's separate ChannelMerge-specific dropdown) treating each side's own value as its own alpha — Nuke's documented convention for compositing a single channel — and `out_channel` picks the destination; every other output channel is copied unchanged from B. |
| 5 | Remove | missing | Deletes channels/layers from a stream; NodeBased's four-channel RGBA model has no extra layers to remove yet. |

## Color

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Grade | supported | `Grade` (exposure/multiply/offset), plus mask + mix. |
| 2 | ColorCorrect | supported | `ColorCorrect` (lift/gamma/gain/saturation), plus mask + mix. |
| 3 | Invert | supported | `Invert`, plus mask + mix. A `channels` knob (rgb/rgba/alpha) selects what gets inverted; `rgb` (the default) leaves alpha untouched, matching Nuke. |
| 4 | Saturation | supported | `Saturation`, plus mask + mix. Same luma-weighted math as `ColorCorrect.saturation`, split onto its own scrub-friendly node with no channels knob (inherently RGB), matching Nuke. |
| 5 | Multiply | supported | `Multiply`, plus mask + mix and a `channels` knob. Reuses Grade's own `multiply` param name and its existing `LIMITS` — the node is literally that one knob, alone. |
| 6 | Add | supported | `Add`, plus mask + mix and a `channels` knob. Reuses Grade's own `offset` param name and `LIMITS` for the same reason. |
| 7 | Gamma | supported | `Gamma`, plus mask + mix and a `channels` knob. Reuses ColorCorrect's own `gamma` param name and `LIMITS`; `sign(x)*|x|^(1/gamma)` per selected channel, same formula ColorCorrect already uses. |
| 8 | Clamp | supported | `Clamp`, plus mask + mix. `minimum`/`maximum` bounds with independent `clamp_min`/`clamp_max` enable toggles and a `channels` knob, matching Nuke's own control set. |
| 9 | HueCorrect | missing | Per-hue-range saturation/luma adjustment; common grading tool, not on the ranked list. |
| 10 | Exposure | supported | `Exposure`, plus mask + mix and `channels`. Knobs follow the reference guide: `exposure_mode` (Nuke's `mode`, renamed because `mode` already belongs to Tracker in the shared `CHOICES`; `stops` or `densities`), `blackpoint`, `gang`, `red`/`green`/`blue`. Formula `(in - blackpoint) * gain` per channel, so a pixel at the black point becomes 0; `stops` gain is `2 ** exposure`, `densities` gain is `10 ** (density / 0.6)`. With `gang` on, `red` drives all three channels; alpha (under `channels = rgba`) takes red's gain. Not covered: Nuke's `Lights` and `Cineon` adjust-in modes, the `colorspace` Cineon offset variant, and the (un)premult-by channel. |
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
| 2 | Sharpen | supported | `Sharpen`, unsharp mask (`amount`/`size` knobs, Nuke's own names), plus mask + mix and a `channels` knob. |
| 3 | Erode (filter) | missing | Nuke's analytic (non-fast) erode, a distance-falloff kernel distinct from the box min/max filter `Erode`/`Dilate` (row 11) now cover; lower priority now that the common fast case exists. |
| 4 | Median | supported | `Median`, square despeckle window (`size`), plus mask + mix and a `channels` knob. |
| 5 | Glow | supported | `Glow`, threshold + size + brightness + tint, added back over the input, plus mask + mix and a `channels` knob. |
| 6 | Soften | supported | `Soften`, plus mask + mix and `channels`. `soften_size` is the kernel's pixel reach: a separable Gaussian with sigma = size / 3, truncated at `ceil(size)` pixels (three sigma) and renormalised so weights sum to exactly 1; below 0.5 it is the identity. Its padded region rule in `tiers.py` is Blur's (`ceil(size)` pixels of halo), so tiles are seamless, asserted against the full-frame evaluator. No per-axis size: `Blur` has none either. |
| 7 | DropShadow | missing | Very common compositing finishing node; not on the ranked list. |
| 8 | Defocus / ZDefocus | missing | Disc-based defocus, the second most common blur after box blur; `ZDefocus` needs a depth channel NodeBased has no concept of yet. |
| 9 | DirBlur | missing | Directional/zoom blur; common for speed/impact effects. |
| 10 | EdgeBlur / EdgeExtend | missing | Matte-edge treatment tools; depend on having Erode/Blur first. |
| 11 | Erode (fast) | supported | `Erode` and `Dilate`, a shared box morphological min/max kernel: `Erode`'s own signed `erode_size` matches Nuke's Erode (fast) (negative dilates); `Dilate` is its positive twin, kept as a separate node as Nuke's toolbar does. Plus mask + mix and a `channels` knob on both. |
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
| 1 | Keyer | supported | `Keyer`, plus mask + mix. `keyer_operation` (luminance/red/green/blue/saturation/min/max) picks the per-pixel quantity a four-point range (`range_a`..`range_d`) ramps to alpha — 0 at or below A, ramping up between A and B, 1 through C, ramping down between C and D, 0 at or above D — and `invert` flips the result. |
| 2 | ChromaKeyer | missing | On the lane's ranked list (group d). The everyday green/bluescreen keyer; highest-use keyer in a working pipeline. |
| 3 | Difference | supported | `Difference`, the two-input colour-difference keyer (`MERGE_LIKE_KINDS`: bypass passes B), plus mask + mix. Alpha is `clamp((max channel |A-B| - offset) * gain, 0, 1)`; output is B's colour with the new alpha. |
| 4 | HueKeyer | supported | `HueKeyer`, plus mask + mix. Nuke's own hue-range/softness knobs are simplified to numeric fields (`hue_center`, `hue_width`, `hue_softness`, all degrees) plus a hard saturation range (`sat_min`, `sat_max`); `invert` flips the result. |
| 5 | Keylight | missing | Industry-standard colour-difference keyer; very high daily use where licensed, but the algorithm is proprietary — out of reach without a from-scratch equivalent. |
| 6 | Primatte / Ultimatte | missing | Commercial keying algorithms with the same licensing barrier as Keylight. |
| 7 | IBKColor / IBKGizmo | missing | Image-based-keying pair; depends on a clean-plate workflow, lower priority than the single-input keyers above. |
| 8 | Cryptomatte / Encryptomatte | missing | ID-matte extraction from renderer metadata; NodeBased's `Render3D` has no Cryptomatte-style ID pass to key from yet. |

## Merge

See the dedicated "Merge operations" audit below for the `operation` dropdown itself. This table
covers Nuke's separate single-purpose Merge-toolbar nodes.

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Merge | supported | `Merge` with mask + mix and all 30 of Nuke's documented operations (see below); the last eleven landed in step 3a. |
| 2 | Premult | supported | `Premult`. |
| 3 | Unpremult | supported | `Unpremult`. |
| 4 | Switch | supported | `Switch` (two inputs; Nuke's goes to any number). |
| 5 | Dissolve | supported | `Dissolve`, plus mask + mix. `which` (0..1) cross-fades linearly between A and B (0 is A, 1 is B); the node's own `mask`/`mix` blend that dissolved result against B on top, the same outer contract every Merge-family node here shares. |
| 6 | KeyMix | supported | `Keymix`, plus mask + mix and an `invert_mask` toggle. Copies A into B wherever the wired mask's alpha is non-zero (or non-zero after inverting); unwired, `mix` alone gates a full copy of A over B, matching Nuke's "no mask = full effect" default. |
| 7 | AddMix | missing | `over` plus a premultiply step; a thin wrapper over existing `Merge`+`Premult`, low priority once both exist. |
| 8 | Blend | missing | Weighted average across any number of inputs; `Merge`'s `average`/`plus` operations cover the two-input case. |
| 9 | CopyRectangle / CopyBBox | missing | Rectangular patch/bbox-copy tools; specialised cleanup use. |
| 10 | ContactSheet | missing | Debug/review grid of inputs; not a compositing operation. |
| 11 | TimeDissolve | missing | A `Dissolve` driven by a time curve instead of a static mix; depends on `Dissolve` landing first. |
| 12 | ZMerge | missing | Depth-sorted merge; needs a depth channel concept. |
| 13 | Absminus / In / Matte / Max / Min / Multiply / Out / Plus / Screen | n/a | Each is a single-operation convenience wrapper around one `Merge` operation; covered by the operations audit below rather than as separate nodes. |
| 14 | VariableSwitch | n/a | A Nuke-scripting Variables/scope construct, not an image operation; not a parity target. |

### Merge operations

`MERGE_OPERATIONS` in `nodebased/core.py` now holds all 30 of Nuke's operations: `atop`,
`average`, `color-burn`, `color-dodge`, `conjoint-over`, `copy`, `difference`, `disjoint-over`,
`divide`, `exclusion`, `from`, `geometric`, `hard-light`, `hypot`, `in`, `mask`, `matte`, `max`,
`min`, `minus`, `multiply`, `out`, `over`, `overlay`, `plus`, `screen`, `soft-light`, `stencil`,
`under`, `xor`. Verified against Foundry's documented algorithms
(`learn.foundry.com/.../merge_operations.html`): the `_merge_op` formulas in `imaging.py` match
Nuke's documented algorithms, so the operations are correctness-checked, not just present.
`average`, `from` and `hypot` landed 2026-09-21 (group (a)); the remaining eleven landed in step 3a
(2026-09-24). Tests: `tests/test_phase_a.py::MergeOperationTests` and
`tests/test_2d_parity_group_3a_merge.py` (hand-computed pixels for every new operation, including
the branch edges), the mix=0/mix=1/HDR/negative-input tests that iterate `CHOICES["operation"]`,
and `tests/test_tileexec.py::MergeMaskTests::test_tiled_masked_merge_matches_the_reference_for_every_operation`
(tile path parity, which picks up new operations automatically).

The eleven added in step 3a, with the formulas as implemented (A foreground, B background, `a` and
`b` the alphas, all per channel including alpha, premultiplied inputs):

| Operation | Formula |
|---|---|
| `matte` | `A*a + B*(1-a)` |
| `disjoint-over` | `A + B` where `a + b < 1`, else `A + B*(1-a)/b` |
| `conjoint-over` | `A` where `a > b`, else `A + B*(1 - a/b)` |
| `copy` | `A` |
| `exclusion` | `A + B - 2AB` |
| `geometric` | `2AB/(A+B)` |
| `overlay` | `2AB` where `B < 0.5`, else `1 - 2(1-A)(1-B)` (multiply and screen, rescaled so the halves meet at B = 0.5) |
| `hard-light` | `overlay` with A and B swapped |
| `soft-light` | `B(2A + B(1-AB))` where `AB < 1`, else `2AB` (Foundry's documented test, on the raw product) |
| `color-dodge` | `B/(1-A)` |
| `color-burn` | `1 - (1-B)/A` |

Division convention (HDR and zero inputs always give finite results): any divisor with magnitude
at most 1e-6 counts as unusable and the term takes a fixed limit. `disjoint-over`: the
`B*(1-a)/b` term is 0 where `b` is unusable. `conjoint-over`: the result is `A` where `b` is
unusable. `geometric`: 0 where `A+B` is unusable. `color-dodge`: 1 where `1-A` is unusable and
`B > 0`, otherwise 0 (so A at or above 1 saturates rather than flipping sign). `color-burn`: 0
where `A` is unusable, except 1 when `B >= 1`. `ChannelMerge` shares the same vocabulary and the
same helper, with each channel value standing in for its own alpha.

## Transform

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Transform | supported | `Transform` (translate/rotate/scale/center/filter), plus mask + mix, inverse-mapped with sub-pixel filtering. |
| 2 | Crop | supported | `Crop`, plus mask + mix, shrinks the data window. |
| 3 | Tracker | partial | `Tracker` applies a solved match-move/stabilise transform (`docs/ROTO_TRACKING.md`), but there is no standalone `Stabilize` node and no point-tracking UI beyond what `Tracker`'s node_data already carries. |
| 4 | Reformat | supported | `Reformat`: type (to format / scale / to box), a node-local named-format preset list (`HD_1080`, `HD_720`, `UHD_4K`, `2K_DCP`, `Square_1K`, or Custom) that resolves into the node's own width/height/pixel_aspect the moment it is chosen, resize type (none/width/height/fit/fill/distort), center/flip/flop/turn, filter, preserve bounding box, plus mask + mix. Unlike every other node here it changes the *display* window itself, not just the data window. The document-level named-format registry the audit originally sketched ("stored in the document so a later node can refer to a format by name") would touch files other lanes own, so this ships node-local instead — see the design note below the summary. |
| 5 | CornerPin2D | supported | `CornerPin` (four "to" points, four "from" points, forward or inverse direction, filter), plus mask + mix. A projective (four-point) warp sharing `Transform`'s inverse-map-then-resample shape; the output data window follows the destination quad's bounds, the format itself is unchanged. |
| 6 | Mirror | supported | `Mirror` (`flip_x`/`flip_y`), plus mask + mix. Flips about the format centre, so it is excluded from the tile path (like `Transform`/`Crop`: flipping is canvas-origin-dependent) and falls back to the full-frame evaluator. |
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

**2026-09-22, group (b).** Ten more rows flip from missing to supported: Invert, Clamp, Multiply,
Add, Gamma, Saturation (Color), Dissolve, Keymix, Copy, ChannelMerge (Merge/Channel). Every one
ships on both evaluation paths (identity ROI rule, zero halo), with mask + mix, `LIMITS`/`CHOICES`
entries, Nuke-matched knobs and pixel-asserted tests (`tests/test_2d_parity_group_b.py`). The
supported count against the 8b49dfa baseline above is now 19 (9 + these 10); groups (c) onward —
Erode/Median/Sharpen/Glow/Soften, the keyers, generators, Mirror/CornerPin, Time, Reformat — are
still open.

**2026-09-23, step 2c1 (six filter nodes).** Six more rows flip from missing to supported: Sharpen,
Median, Glow, Erode (fast, as `Erode` and `Dilate`), and Mirror (Transform menu). Erode/Dilate/
Median/Sharpen/Glow share Blur's padded-filter shape (a kernel radius/size grows the requested
input region in `tiers.py` so tiles have their neighbours, asserted seamless against the
full-frame evaluator); Mirror is canvas-origin-dependent like Transform/Crop and is excluded from
the tile path by the same precedent. Every node ships mask + mix, a `channels` selector where
Nuke has one, `LIMITS`/`CHOICES` entries, Nuke-matched knobs and pixel-asserted tests
(`tests/test_2d_parity_group_c1.py`). Nuke's separate, non-fast `Erode (filter)` (an analytic
falloff kernel, not a box min filter) and `Soften` remain missing. The supported count is now 25
(19 + these 6); Soften, the four keyers, generators, CornerPin, Time and Reformat are still open.

**2026-09-23, step 2c2 (five draw nodes).** Five more rows flip from missing to supported: Ramp,
Radial, Rectangle, Noise and Text (Draw menu). Each states its own format like Roto, but unlike
Roto also takes an optional "image" input the shape is composited over (Nuke's own Draw-node
convention: over the input where the shape's alpha is set) and an optional mask, so bypassing one
now passes that optional image through, or a transparent frame at its own format when nothing is
wired — `core.bypass_slot`/`core.DRAW_KINDS`, fixing the same disabled-generator gap Constant/
Checker/Roto had latently carried since bypassing any of the three previously crashed. Ramp/
Radial/Rectangle/Noise are pure numpy; Text renders through Qt's own text rasteriser offscreen
(QPainter/QFont on a QImage), so it needs no new dependency — which fonts are actually available
is a machine property, stated plainly rather than hidden. Every node ships on both evaluation
paths from one shared pure function (`Evaluator._draw_shape`), so a tile's render and the
full-frame reference are pixel-identical by construction rather than by comparison; mask + mix;
`LIMITS`/`CHOICES` entries; Nuke-matched knobs; theme colours; and pixel-asserted tests
(`tests/test_2d_parity_group_c2.py`). The supported count is now 30 (25 + these 5); Soften, the
four keyers, CornerPin, Time and Reformat are still open.

**2026-09-23, step 2c3 (three keyer nodes).** Three more rows flip from missing to supported:
Keyer, HueKeyer and Difference (Keyer menu). Keyer keys a chosen per-pixel quantity (luminance,
red, green, blue, saturation, min or max) through a four-point range ramp into alpha, RGB
untouched, with an invert toggle; HueKeyer simplifies Nuke's own hue-range/softness knobs to
numeric fields (hue center, width and softness, all in degrees) plus a hard saturation range;
Difference is the two-input colour-difference keyer, sharing `MERGE_LIKE_KINDS`' own bypass
(passes B) and windowing convention, with its alpha the largest per-channel difference between A
and B shaped by `offset` and `gain`. Every node ships mask + mix, `LIMITS`/`CHOICES` entries,
Nuke-matched knobs, a theme colour and pixel-asserted tests (`tests/test_2d_parity_group_c3.py`).
The supported count is now 33 (30 + these 3); of the four keyers only ChromaKeyer remains missing.
Soften, CornerPin, Time and Reformat are still open.

**2026-09-23, step 2c4 (three time nodes).** Three more rows flip from missing to supported:
TimeOffset, FrameHold and Retime (Time menu). None takes a mask or mix, matching Nuke's own
Time-menu nodes; each is a producer with its own time mapping rather than a pixel kernel — its
required input is evaluated at a remapped frame through a nested `Evaluator.evaluate_raster` call
(the "clip" shape `docs/TIME_MODEL.md` sketches), and its cache digest folds in that nested call's
own content digest rather than this walk's stale, wrong-frame `hashes[source]`, so a still stays
one cache entry across every frame it is asked for while a curve edited on a keyframe elsewhere is
still picked up. TimeOffset shifts by a signed frame count (`reverse` flips which direction the
offset applies); FrameHold holds on `first_frame` (increment 0, Nuke's own default) or steps
forward every `increment` frames; Retime is nearest-frame only, no frame blending, with the range
end knobs carried for Nuke-parity naming but not consulted by the simplified linear mapping. All
three are excluded from the tile path, like `Transform`/`Crop`/`Mirror` before them, because the
tile executor has no per-tile notion of "a different frame" — a graph containing one falls back to
the full-frame evaluator, asserted equal to it. Every node ships bypass (passthrough at the
current, un-remapped frame), `LIMITS` entries, Nuke-matched knobs, a theme colour and tests against
an animated upstream, including a cache-behaviour proof (`tests/test_2d_parity_group_c4.py`). The
supported count is now 36 (33 + these 3); of the Time group, TimeClip, FrameRange, AppendClip and
the motion-blur/optical-flow-grade retimers remain missing. Soften, CornerPin and Reformat are
still open.

**2026-09-23, step 2c5 (Reformat, CornerPin — the last of group c).** Two more rows flip from
missing to supported: Reformat and CornerPin2D (Transform menu). **Design note on Reformat's
format model:** the audit above originally gated Reformat on "a written design (`docs/` doc +
integrator sign-off)" because a document-level named-format registry — every node able to refer to
a shared format by name — would touch files other lanes own (`core.py`'s document schema). This
build ships the node-local escape hatch the lane brief allows instead: `format` is a small built-in
preset list (`HD_1080` 1920x1080, `HD_720` 1280x720, `UHD_4K` 3840x2160, `2K_DCP` 2048x1080,
`Square_1K` 1024x1024, or `Custom`) that resolves into the node's own `width`/`height`/
`pixel_aspect` params the instant it is chosen (`core._resolve_reformat_format`, called from both
"create" and "set"), so those three params stay the only pixel-unit state the kernel and the
proxy-tier scaler (`tiers.PIXEL_UNIT_PARAMS`) ever read — a named preset stays correct at every
playback tier for exactly that reason. `type` (`to_format`/`scale`/`to_box`) picks how the target
size is derived: the format list or `Custom`'s own fields, an upstream-relative percentage
(`scale`), or the same width/height/pixel_aspect fields read directly (`to_box` deliberately reuses
them rather than a parallel set of box knobs). `resize_type` (none/width/height/fit/fill/distort)
then places the source inside that target box, with `center`/`flip`/`flop`/`turn` and the existing
`filter` choice; `turn` solves the resize against the *swapped* working format the way Nuke's own
turn knob does. Reformat is the one node in this file whose own output *display* window is not its
source's: every other bounding-box-aware node here (`Transform`, `Crop`, `Mirror`, the group c4
time nodes) only ever moves the *data* window, which is what let `docs/EVALUATION_TIERS.md`'s tile
executor keep treating "the target's canvas size" as "walk back to the nearest generator and use
its size" — `nodebased/tileexec.py`'s `_first_generator`/`_canvas_size_for_chain` now stop at a
Reformat node instead of walking through it, for `to_format`/`to_box`; a `scale`-type Reformat's
canvas size is upstream-relative and is not resolved by that walk, a known gap noted where the code
special-cases it (Reformat still renders correctly through `Evaluator.evaluate` either way, since
it is excluded from the tile path entirely, below). `pixel_aspect` is carried on the node for
format-parity but, like Retime's unused range-end knobs, is not consulted by the resample math,
which works in square pixels throughout this build. CornerPin is the simpler of the two: a
projective four-point warp (`from1..4`, `to1..4`, `direction` forward/inverse) sharing `Transform`'s
own inverse-map-then-resample shape and `_filter_window`/`_filtered_pixels` dispatch — like
`Transform` it only ever moves the data window, the format itself is unchanged. Both nodes are
excluded from the tile path, like `Transform`/`Crop`/`Mirror`/the group c4 time nodes before them
— Reformat because it is both canvas-origin-dependent and changes the canvas size outright, which
the tile executor's single fixed-canvas model has no notion of; CornerPin because a projective warp
is coordinate-dependent exactly like Transform's affine one — a graph containing either falls back
to the full-frame evaluator, asserted equal to it. Both nodes ship mask + mix, `LIMITS`/`CHOICES`
entries, Nuke-matched knobs, a theme colour and pixel/window-asserted tests, including the brief's
own worked examples (`tests/test_2d_parity_group_2c5.py`): Reformat to `HD_720` from a 1920x1080
input gives a 1280x720 display window with centred content staying centred, and `resize_type =
"none"` keeps the pixels and only changes the window; CornerPin with `to` points equal to `from`
points is the identity, and pinning the top edge inward by 25% moves a known pixel to the computed
position. The supported count is now 38 (36 + these 2); of the Transform group, Stabilize (as its
own node), Position, AdjustBBox, BlackOutside, STMap, IDistort, GridWarp family, Tile,
VectorCornerPin/VectorDistort, PointsTo3D/Reconcile3D and TVIScale remain missing, and Soften
(Filter group) is still open.

**2026-09-24, step 3a part 1 (the eleven remaining Merge operations).** `Merge` and `ChannelMerge`
now carry all 30 of Nuke's operations: `matte`, `disjoint-over`, `conjoint-over`, `copy`,
`exclusion`, `geometric`, `overlay`, `hard-light`, `soft-light`, `color-dodge` and `color-burn`
join the existing 19. Every division is guarded and the limit convention is stated in the "Merge
operations" section above; the tile path shares the evaluator's formulas, asserted by the existing
every-operation tile parity test. The Merge row flips from partial to supported, so the supported
count is now 39 (38 + Merge).

**2026-09-24, step 3a parts 2 and 3 (Soften, Exposure).** Two more rows flip to supported, so the
supported count is now 41 (39 + these 2). `Soften` is
the Filter-menu Gaussian-leaning blur: a separable Gaussian, sigma one third of `soften_size`,
truncated at three sigma, so a single bright pixel becomes a symmetric kernel summing to the
original energy and size 0 is the identity. `Exposure` is the standalone Color-menu node with the
black-point-preserving formula `(in - blackpoint) * gain` in `stops` (`2 ** exposure`) or
`densities` (`10 ** (density / 0.6)`, the reference guide's 0.6-gamma negative stock) mode; the
knob names come from the reference guide (`blackpoint`, `gang`, `red`, `green`, `blue`), with
Nuke's `mode` stored as `exposure_mode`. Both ship mask + mix, `LIMITS`/`CHOICES` entries, knobs,
a theme colour and both evaluation paths (the tile path calls the evaluator's own kernels and is
asserted equal, across several tiles for Soften) in `tests/test_2d_parity_group_3a_filters.py`.
Soften joins Blur's family of padded filters; Exposure is pointwise. Still open in these groups:
Exposure's `Lights` and `Cineon` modes, and the rest of the Filter group's missing rows.
