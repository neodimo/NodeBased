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
| 2 | Write | supported | `Write`, EXR/PNG only (`WRITE_FILE_TYPES`); Nuke writes far more formats but the two here are real. An EXR from a multichannel input writes every layer in one part (docs/3D_FOUNDATION.md "Multichannel output"). |
| 3 | Viewer | supported | `Viewer` with nine inputs, A/B buffers, wipe, over, under, minus and difference compare modes (`docs/PLAYBACK.md`), and R/G/B/A channel solo. |
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
| 8 | LightWrap | supported | `LightWrap`, Nuke's two-input finishing node: inputs `fg` and `bg`, plus mask + mix; a bypass passes `fg`. Knobs `intensity`, `wrap_diffuse` (Nuke's `diffuse`, renamed because `diffuse` is a 0 to 1 `LIMITS` key on `Relight`), `fgblur`, `bgblur`, `wrap_threshold`, `highlight_merge` (`plus`, `screen`, `max`, `over`), `use_constant_highlight` and the highlight `red`/`green`/`blue`. The foreground alpha is softened by `fgblur`, its inverse blurred by `wrap_diffuse` and doubled (a straight edge reads 0.5, so strength is full at the edge), then multiplied by the foreground alpha, so light appears only inside the matte's edge band and never far inside or outside it. The wrapped light is the background blurred by `bgblur` minus `wrap_threshold` (or the constant colour), scaled by `intensity` and folded into the premultiplied foreground by `highlight_merge`; alpha is the foreground's. Both paths: the padded region rule in `tiers.py` asks fg and bg for `max(ceil(fgblur) + ceil(wrap_diffuse), ceil(bgblur))` pixels of padding, tiles asserted against the full-frame evaluator with and without a mask. Not covered: Nuke's `luminance`, `saturation` and separate highlight colour on non-constant wraps, and a background whose display window differs from the foreground's (raises, like Merge). |
| 9 | Grain / ScannedGrain | partial | `Grain`, plus mask + mix: synthetic grain with Nuke's per-channel `red_size`/`green_size`/`blue_size` and intensity (`red_intensity` ..., Nuke's `red_m`), a `seed`, and `luminance_weighted` with `black` as the floor of the weight. The noise is zero-mean lattice noise (spacing = `size` pixels; 1 or less is one independent value per pixel), so the mean of a flat area is preserved (asserted: mean within 0.002, standard deviation equal to the intensity at size 1). It is a pure function of the absolute pixel position, the channel and `seed + frame`, so the tile path matches the full frame exactly, the grain moves every frame, and `seed` = minus the frame freezes it (Nuke's `-frame` recipe). Not covered: `ScannedGrain`, the film-stock presets, `irregularity`, `minimum` and Nuke's "apply only through alpha"; intensity is an additive amplitude, so its defaults (0.05) are not Nuke's (0.416, 0.46, 0.85), and larger sizes interpolate between lattice values, which lowers the variance a little. Zero halo, on both paths. |
| 10 | DustBust / MarkerRemoval | missing | Roto-driven paint-out tools; depend on RotoPaint. |
| 11 | Flare / Glint / Sparkles | missing | Stylised lens-artifact generators; low daily use outside specific shots. |
| 12 | Dither | supported | `Dither`, plus mask + mix and `channels` (default `rgb`, so alpha is left alone). Quantises to `bits` (1 to 16) per channel after adding triangular (TPDF) noise of +/- `dither_amount` least-significant bits, so a smooth ramp becomes fine grain whose average is the original value instead of contour bands (asserted: a flat 0.3 at 4 bits averages 0.3 within 0.004, where a plain quantise is off by more than 0.01). The noise is a hash of the absolute pixel position, the channel and `seed`, never a random stream, so a seed repeats exactly on every run and the tile path, which hands the kernel each tile's canvas origin, matches the full frame pixel for pixel. Values are clamped to 0 to 1 by the quantiser. Proxy tiers hash their own (decimated) pixel positions, so the grain is not the full-resolution grain. Zero halo, on both paths. |
| 13 | Grid | supported | `Grid`, Nuke's Draw Grid: vertical and horizontal lines every `spacing_x`/`spacing_y` pixels (or `number_x`/`number_y` lines across the format when above zero), `grid_offset_*`, `line_width`, colour, plus the optional image the lines are composited over and mask + mix. Pure numpy through the shared draw function, so the evaluator and the tile path draw the same pixels. |

## Time

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | TimeOffset | supported | `TimeOffset` (`time_offset`, `reverse`), evaluates its input at `frame - time_offset` (or `+` when reversed) via a nested evaluate call — see `docs/TIME_MODEL.md`. |
| 2 | FrameHold | supported | `FrameHold` (`first_frame`, `increment`), freezes on `first_frame` at increment 0 (Nuke's own default), or steps forward every `increment` frames. |
| 3 | Retime | supported | `Retime`, simplified: nearest-frame sampling only, no frame blending. `input_range`/`output_range` plus `speed` map a frame linearly (`input_start + (frame - output_start) * speed`); the range end knobs are carried for Nuke-parity naming and are not consulted by the simplified mapping. |
| 4 | TimeClip | supported | `TimeClip` (`first`, `last`, `frame_range_type` custom/all, `before`/`after` hold, loop, bounce or black, `time_offset`). The input is evaluated at `frame - time_offset`; outside `[first, last]` the policy decides (black is a transparent frame of the nearest in-range frame's size). Nuke's `reverse` and expression modes are not carried; the offset knob is `time_offset` (the global `offset` key is Grade's float). |
| 5 | FrameRange | supported | `FrameRange` (`first_frame`, `last_frame`, `before`/`after`). The document has no per-branch frame range, so it presents the range by clamping, looping, bouncing or blanking frames outside it; `AppendClip` reads it (and a custom-range `TimeClip`) as a clip's length. |
| 6 | AppendClip | supported | `AppendClip`, eight optional `clip0`..`clip7` slots played head to tail from `first_frame`. A clip's length is the range of a directly upstream FrameRange/TimeClip (through bypassed nodes and Dots), else its `length<i>` knob (0 skips it), sampled from that range's first frame or from frame 1. `dissolve` frames cross-fade consecutive clips (clips must share a format); before the first clip and after the last the end frames hold. |
| 7 | TimeBlur / TimeWarp / TimeEcho | missing | Motion-blur-adjacent retiming; needs `Retime`/`Kronos`-grade groundwork first. |
| 8 | Kronos / OFlow / SmartVector / VectorToMotion | missing | Optical-flow retiming; a research-grade lift, far below `Retime` in priority. |
| 9 | NoTimeBlur | missing | A cache-shaping hint node with no analogue in the current evaluator. |

## Channel

| Rank | Nuke node | Status | Reason |
|---|---|---|---|
| 1 | Shuffle | supported | `Shuffle`, single-input channel routing with constants, and a `layer` knob (step 5c) naming one of the input's named layers (`Raster.layers`: what a multichannel EXR `Read` or `Render3D` carries). Empty or `rgba` shuffles the input's own channels, exactly as before; a layer name routes that layer's R, G, B, A (a vector layer's X, Y, Z fill R, G, B, a single-channel layer such as `depth` fills all three, alpha is 1) through the same four `*_from` choices, so any layer can be sent to RGBA. The output data window is the layer's. An unknown name is an error that lists the layers the input has (or says it has none). The knob is a text field, not a drop-down, because the layer list belongs to the evaluated input. Named layers exist only on the whole-image path (a tile carries one RGBA array), so `TileExecutor.supports_tiled` sends a graph containing a layered `Shuffle` to the full-frame evaluator; an unlayered one stays on the tile path. |
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
| 9 | HueCorrect | partial | `HueCorrect`, plus mask + mix. Nuke edits free-form per-hue curves; this is the smallest honest model of them: a fixed set of **six** hue anchors (red, yellow, green, cyan, blue, magenta, 60 degrees apart, the hue axis Nuke's curve editor uses; the brief's eight bands could not be sourced, so six is what is claimed), each with a saturation multiplier (`sat_red` ... `sat_magenta`) and a luminance multiplier (`lum_red` ... `lum_magenta`), all defaulting to 1 (identity). Between neighbouring anchors the multiplier follows a smoothstep (`3t^2 - 2t^3`): continuous, flat at every anchor, wrapping from magenta to red. Saturation scales chroma about Rec.709 luma; the luminance multiplier is faded in by HSV saturation so neutrals are untouched. `hue_shift` (degrees) rotates chroma about the neutral axis, keeping the channel average. Alpha untouched. Not covered: a curve editor, Nuke's `r_sup`/`g_sup`/`b_sup` suppression curves, per-channel `red`/`green`/`blue` curves and `sat_thrsh`. |
| 10 | Exposure | supported | `Exposure`, plus mask + mix and `channels`. Knobs follow the reference guide: `exposure_mode` (Nuke's `mode`, renamed because `mode` already belongs to Tracker in the shared `CHOICES`; `stops` or `densities`), `blackpoint`, `gang`, `red`/`green`/`blue`. Formula `(in - blackpoint) * gain` per channel, so a pixel at the black point becomes 0; `stops` gain is `2 ** exposure`, `densities` gain is `10 ** (density / 0.6)`. With `gang` on, `red` drives all three channels; alpha (under `channels = rgba`) takes red's gain. Not covered: Nuke's `Lights` and `Cineon` adjust-in modes, the `colorspace` Cineon offset variant, and the (un)premult-by channel. |
| 11 | HSVTool | partial | `HSVTool`, plus mask + mix. Hue (degrees), saturation and value are adjusted together, limited by a hue range (`hue_range_min`/`hue_range_max`, Nuke's `huesrcs`; min above max wraps through 360, a span of 360 takes every hue), a saturation range and a brightness range (`saturation_range_*`, `brightness_range_*`), each with a linear rolloff (`hue_rolloff` in degrees, the others in value units); the adjustment is weighted by the product of the three range weights. `hue_rotation` rotates hue (120 degrees turns pure red into pure green, asserted); `sat_adjust` and `brt_adjust` (Nuke's `saturation`/`brightness` adjustments, renamed because those names belong to other nodes) scale saturation and value by `1 + adjust`, or with `set_saturation`/`set_brightness` move them to the adjust value. The reference guide does not define the scaling; scaling is inferred and unverified against Nuke. A brightness or saturation range whose upper end is 1 or more is open above, so HDR values are inside a default range. `output_alpha` writes the combined range weight into alpha. Not covered: `srccolor`/`dstcolor` colour replacement, the per-range mask channels and the alpha conversion choice. Zero halo, on both paths. |
| 12 | ColorMatrix | supported | `ColorMatrix`, plus mask + mix. Nine knobs `matrix_00` ... `matrix_22` (row, then column; identity by default), `out.r = matrix_00 * r + matrix_01 * g + matrix_02 * b`, alpha untouched. `invert` applies the inverse matrix; a singular matrix (`abs(det) < 1e-9`) has no inverse, so the node then passes its input through unchanged instead of producing NaNs. A permutation matrix swaps channels bit-exactly. Not covered: Nuke's `channels` selector and (un)premult-by. |
| 13 | Colorspace / OCIOColorspace / OCIODisplay / OCIOLookTransform / OCIOFileTransform / OCIOLogConvert | partial | The document's colour pipeline is a single fixed ACES config (`docs/COLOR_MANAGEMENT.md`; `validate_settings` rejects any other config/working-space/display); `Read.colorspace` offers a fixed enum, but there is no general per-node colourspace-conversion node. |
| 14 | Log2Lin / PLogLin | missing | Log/lin conversion nodes; redundant while the working space is fixed ACEScg with no log-encoded intermediate format. |
| 15 | MatchGrade / ColorTransfer | missing | Automatic grade-matching between two clips; a research-grade addition. |
| 16 | Expression | missing | Per-channel Tcl-like formula node; NodeBased's expression system (`docs/EXPRESSIONS` via `expressions.py`) exists at the parameter level, not as an image-processing node. |
| 17 | CrossTalk | missing | Channel-bleed simulation; low daily use. |
| 18 | Posterize / SoftClip / Toe | partial | `Posterize` and `SoftClip`, each plus mask + mix. `Posterize` snaps each selected channel (`channels`, default `rgb`) to `colors` evenly spaced levels between 0 and 1 (`colors` = 2 gives exactly the levels 0 and 1, asserted), clamping HDR values to the top level. `SoftClip` has Nuke's four `conversion` modes (`none`, `preserve hue and brightness`, `preserve hue and saturation`, `logarithmic compress`) and `softclip_min` (0.8) / `softclip_max` (1). The logarithmic mode maps min..max onto min..1 with a curve of slope 1 at min, so values at or below min are untouched and the join is smooth, and values above max keep climbing past 1; a max of 1 or less leaves nothing to compress (asserted). The preserve modes act only on pixels with a component above `softclip_max`: desaturation toward the luma keeps brightness, scaling the whole pixel down keeps hue and saturation. The reference guide does not give the exact curve, so the curve shape is this repository's choice. Not covered: `Toe`. Zero halo, on both paths. |
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
| 7 | DropShadow | supported | `DropShadow`, plus mask + mix. Knobs `angle` (degrees, counter-clockwise from +x with y up, so -45 points down and right), `distance`, `shadow_size`, `opacity` and the shadow `red`/`green`/`blue` (default black). `shadow_size` is Nuke's `size`, renamed because `size` is a global `LIMITS` key with a minimum of 1 and a different meaning on `Checker` and `Noise`. The input's alpha is moved by (`distance`, `angle`) rounded to whole pixels, blurred with the `Soften` kernel (sigma = `shadow_size` / 3, reach `ceil(shadow_size)`), scaled by `opacity`, tinted, and composited UNDER the input (`out = input + shadow * (1 - input alpha)` on premultiplied RGBA), so opaque input pixels are untouched, opacity 0 and a transparent input are the identity, and a semi-transparent input shows the shadow through it. Outside the frame counts as transparent (zero fill). Like `Blur` the output stays inside the input's data window, so a shadow that falls off it is clipped rather than growing the box. Padded region rule in `tiers.py`: `ceil(distance) + ceil(shadow_size)`, tiles asserted against the full-frame evaluator. Not covered: Nuke's shadow-only output and multi-shadow layering. |
| 8 | Defocus / ZDefocus | partial | `Defocus` is supported: Nuke's disc blur without depth, plus mask + mix and `channels`. `defocus` is the disc radius in pixels and `aspect` its width / height (semi-axes `defocus` by `defocus / aspect`). Every output pixel is the plain average of the input pixels whose centres fall inside that ellipse, so a bright point becomes a flat-topped disc (unlike `Blur`'s box and `Soften`'s Gaussian) and the weights sum to exactly 1; below a half pixel it is the identity. Its padded region rule in `tiers.py` is the larger semi-axis rounded up, so tiles are seamless, asserted against the full-frame evaluator. `ZDefocus` stays missing: it needs a depth channel NodeBased has no concept of yet. |
| 9 | DirBlur | supported | `DirBlur`, plus mask + mix and `channels`; all three of Nuke's types landed, chosen by `blur_type` (`linear`, `radial`, `zoom`). Each output pixel averages bilinear samples along a path through it, weights summing to 1, so a single pixel keeps its energy. `linear`: a box of `length` pixels along `angle` (degrees, counter-clockwise from +x with y up, as in Nuke), whole-pixel taps with the two end taps weighted by coverage, so `length` 8 is exactly eight pixels and `length` 1 or less is the identity. `zoom`: samples on the ray through `center_x`/`center_y`, scale 1 ± `length` / 200 (`length` is a percentage). `radial`: samples on the circle about the centre over a sweep of `angle` degrees (`length` unused). The centre is in canvas pixels and defaults to the middle of the default 960x540 format. Only `linear` is on the tile path, with a padded region rule of `ceil(length / 2) + 1` pixels, asserted against the full-frame evaluator across several tiles; `zoom` and `radial` read from anywhere in the frame, so `TileExecutor.supports_tiled` sends a graph containing one to the full-frame evaluator, the precedent Transform and Mirror set. Not covered: Nuke's per-type extras (`fade`, `samples` control, motion-vector input). |
| 10 | EdgeBlur / EdgeExtend | supported | Two nodes, each with mask + mix, on both paths. `EdgeBlur` blurs only along the matte's edge: `edgeblur_size` (Nuke's `size`, renamed for the same `LIMITS` reason as `shadow_size`) is the Gaussian's reach (the `Soften` kernel) and `edge_mult` scales the band the blur is confined to, `edge_mult * edgeblur_size` pixels either side of the edge. The band is the alpha's own box dilate minus box erode, so it is 1 near a hard edge, fractional across a soft one and 0 in flat areas; `out = image + band * (blurred - image)`, so detail farther than the band from any edge is untouched bit for bit (asserted: a fine checker inside a square keeps every pixel beyond 3 pixels of the edge, band width measured at exactly 3 + 3 rows per edge, while a plain `Blur` smears it). `channels` picks what is blurred. Padded rule: the wider of `ceil(size)` and `ceil(band)`. `EdgeExtend` pushes edge colour outward to kill dark fringes: pixels with alpha at least `extend_threshold` are trusted, every other pixel takes the mean straight colour of its known 8 neighbours, one ring per step for `ceil(extend_size)` steps, alpha unchanged. The output colour is UNPREMULTIPLIED (follow with `Premult`, or feed a filter that would otherwise bleed black), so mask + mix blend an unpremultiplied result with the premultiplied input. Asserted on a constructed edge: a dark half-transparent fringe pixel and the first transparent pixel both take the interior's pure red, and colour stops `ceil(extend_size)` pixels out. Padded rule: `ceil(extend_size)`. Not covered: Nuke's `EdgeBlur` tint and the alpha-only or luminance-driven edge sources, and an `EdgeExtend` that outputs premultiplied colour with negative-alpha extension. |
| 11 | Erode (fast) | supported | `Erode` and `Dilate`, a shared box morphological min/max kernel: `Erode`'s own signed `erode_size` matches Nuke's Erode (fast) (negative dilates); `Dilate` is its positive twin, kept as a separate node as Nuke's toolbar does. Plus mask + mix and a `channels` knob on both. |
| 12 | Denoise / DegrainSimple | missing | Grain/noise removal; a research-grade filter, not a box/erode-class kernel. |
| 13 | MotionBlur / MotionBlur2D / MotionBlur3D / VectorBlur | partial | `VectorBlur` is supported (step 5c), plus mask + mix; `MotionBlur`, `MotionBlur2D` and `MotionBlur3D` stay missing. The vector field is a motion-vector image in pixels per frame (x to the right, y down): `uv_layer` names a layer of the wired `uv` input (or of the image input itself when `uv` is not wired, so a multichannel EXR drives its own blur), an empty `uv_layer` takes the `uv` input's channels, and `u_channel`/`v_channel` pick which two channels are x and y. Each pixel averages bilinear samples of the image along its own vector: `vector_scale` (Nuke's `scale`) multiplies the vector, its length is capped at `max_length` pixels (0 = no cap), and the samples sit one pixel apart from `t = vector_offset` to `vector_offset + 1` (`vector_offset`, default 0, moves the shutter). `vector_method` `forward` samples at `p - t * v`, so a point moving by `v` leaves a streak that runs with its motion (a dot moved 6 pixels right per frame smears over the 7 pixels from itself to its right, each at 1/7); `backward` samples at `p + t * v`, the same streak on the other side. Weights sum to 1, so a uniform field gives every pixel the same streak and zero vectors are the identity (both asserted). `vector_alpha` `weighted` averages the straight colour with the samples' alpha as weight and keeps the pixel's own alpha (colour smears, the matte does not); `none` is the plain premultiplied average, which fades the matte along the streak. The output keeps the source's data window like `Blur` (a streak past it is clipped). The `uv` slot is required in the sense that the node refuses to run with neither a wired `uv` nor a `uv_layer`. Excluded from the tile path (named layers do not reach it), its region rule in `tiers.py` grows the image request by `max_length + 1` (whole image when uncapped). At a proxy tier `vector_scale` and `max_length` are divided by the tier, because the vectors are measured in full-resolution pixels. Not covered: Nuke's `MotionBlur` family (transform- and camera-driven blur), per-pixel sample counts other than one per pixel of length, and a `vector_method` that reads the vector at the destination pixel instead of the source. |
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
| 7 | AddMix | supported | `AddMix`, plus mask + mix, inputs `A` and `B` (a bypass passes `B`). A is premultiplied, then merged `over` B with the same gate as `Merge`; asserted byte-equal to `Merge` over of a premultiplied A. Zero halo, on both paths. |
| 8 | Blend | partial | `Blend`, plus mask + mix and `channels` (default `rgba`): the weighted average of up to **eight** inputs (`in0` ... `in7`; the first two required, gaps are skipped) with a `weight0` ... `weight7` each; `normalize` (on) divides by the weight sum, off returns the weighted sum. Equal weights give the mean of the wired inputs (asserted for three and for eight). A bypass passes the first wired input. Both paths (all inputs requested at the output region, tiles asserted equal with and without a mask). Not covered: more than eight inputs, and Nuke's `fringe`, `inject` and per-channel mask choice. |
| 9 | CopyRectangle / CopyBBox | partial | `CopyRectangle`, plus mask + mix, `channels` (default `rgba`), inputs `A` and `B` (a bypass passes `B`). Copies A's channels over B inside the `area` box: `area_x`, `area_y` (left, top) and `area_r`, `area_t` (right, bottom edge, canvas pixels, rows counted from the top like `Crop`; Nuke's `xyrt` counts rows from the bottom). A pixel is inside when its centre is, so integer edges copy exact whole pixels (asserted); `softness` fades the copy over that fraction of half the shorter side, inward from every edge. The tile path hands the kernel each tile's canvas origin, so seams match the full frame. Not covered: `CopyBBox`. Zero halo, on both paths. |
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
| 4 | Reformat | supported | `Reformat`: type (to format / scale / to box), a named `format` that resolves against the document-wide format registry first (`settings.formats`, seeded with `HD_1080`, `HD_720`, `UHD_4K`, `2K_DCP`, `Square_1K`) and the built-in list second, or Custom, into the node's own width/height/pixel_aspect, resize type (none/width/height/fit/fill/distort), center/flip/flop/turn, filter, preserve bounding box, plus mask + mix. Unlike every other node here it changes the *display* window itself, not just the data window. Step 4c added the document-level registry the audit sketched, so two Reformats naming one format share one meaning — see the design note below the summary. |
| 5 | CornerPin2D | supported | `CornerPin` (four "to" points, four "from" points, forward or inverse direction, filter), plus mask + mix. A projective (four-point) warp sharing `Transform`'s inverse-map-then-resample shape; the output data window follows the destination quad's bounds, the format itself is unchanged. |
| 6 | Mirror | supported | `Mirror` (`flip_x`/`flip_y`), plus mask + mix. Flips about the format centre, so it is excluded from the tile path (like `Transform`/`Crop`: flipping is canvas-origin-dependent) and falls back to the full-frame evaluator. |
| 7 | Stabilize | partial | `Tracker.mode = "stabilise"` covers the pixel math; Nuke exposes it as its own node with its own knob set. |
| 8 | Position | supported | `Position`, `translate_x` / `translate_y` as integers (Nuke's `translate`). Pixels and data window move together by whole pixels, nothing is resampled (the moved pixels are byte-identical) and the display window stays put; a pixel at (10, 10) shifted by (3, -2) lands at (13, 8). No mask or mix, as in Nuke. Excluded from the tile path (window-moving, like `Transform`), so a graph containing it uses the full-frame evaluator; its own read is declared in `tiers.py` as the output region shifted back. At a proxy tier the integer shift is scaled and rounded to a whole pixel of that tier. |
| 9 | AdjustBBox | supported | `AdjustBBox`: `numpixels` grows the data window by that many pixels on every side (zero fill) or, negative, shrinks it (pixels outside are dropped); pixels that remain are byte-identical and do not move. `clip_to_format` then intersects the window with the display window. Shrinking past nothing leaves an empty window. No mask or mix, as in Nuke; excluded from the tile path like `Position`. Not covered: Nuke's separate left/right/top/bottom `numpixels` and its `extra` overscan handling. |
| 10 | BlackOutside | supported | `BlackOutside`, no knobs. Everything outside the data window is already "no pixel here" (0) in NodeBased's window model; the node grows the data window by one pixel on every side so that ring is a real black border a later filter reads instead of extending the edge pixel, which is Nuke's purpose for it. Interior pixels are unchanged, the display window is untouched, and stacking two grows the window by two each side. Excluded from the tile path like `Position`. |
| 11 | STMap | supported | `STMap`, plus mask + mix (step 5c). Absolute remap: each output pixel takes the image sampled at (`u * width`, `(1 - v) * height`), u and v normalised to the image's display window with v running bottom to top (Nuke's convention) and pixel centres at half integers, so a map holding every pixel's own centre is the identity (asserted for nearest and bilinear) and a constant shift of `k / width` in u moves the picture `k` pixels left. The map is `uv_layer` of the wired `uv` input, or of the image input itself when `uv` is unwired (the layer a `Render3D` `uv` pass or a multichannel EXR carries), or the `uv` input's own channels when `uv_layer` is empty; `u_channel`/`v_channel` pick the channels. `filter` is nearest, bilinear or cubic, the `Transform` resampler; `uv_outside` `black` (coordinates outside the image sample transparent black) or `clamp` (they stick to the nearest edge pixel). NaN in the map counts as 0. The output covers the map's data window, as in Nuke, and the map's absence outside it is transparent black. With no `uv` and no `uv_layer` the node raises an error rather than passing the image on looking plausible. Excluded from the tile path (it reads the image wherever the map points, and named layers do not reach tiles); its region rule requests the whole image and the map and mask pointwise. Not covered: Nuke's `blur_scale` and `uv` alpha handling. |
| 12 | IDistort | supported | `IDistort`, plus mask + mix (step 5c). Relative offsets in pixels: with `d = (uv + uv_offset) * uv_scale` (`uv_scale_x/y`, `uv_offset_x/y`), `out(p) = image(p - d)`, so a positive u moves the picture right and a positive v moves it down, the way a forward motion vector carries a pixel (a map of constant (5, -2) moves a dot from (10, 8) to (15, 6), asserted; a zero map is the identity; scale and offset apply before sampling; fractional offsets interpolate). The map source, `filter` and window rules are `STMap`'s except that the output keeps the image's data window (what is pushed out of it is clipped). Excluded from the tile path; whole-image region rule. At a proxy tier `uv_scale_x/y` are divided by the tier, because the map holds full-resolution pixels. |
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
| 2 | NoOp | supported | `NoOp`, a passthrough that (unlike `Dot`) has a properties panel and keeps a `note` knob for the artist; bypassed or enabled it passes its input untouched, on both paths. |
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
format model (superseded in part by the shared registry of step 4c, below):** the audit above originally gated Reformat on "a written design (`docs/` doc +
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

**2026-09-24, step 3b (Defocus, DirBlur, DropShadow, Position, BlackOutside, AdjustBBox).** Five rows
flip to supported and one goes from missing to partial, so the supported count is now 46 (41 + these 5).
`Defocus` is the disc blur without depth (`defocus` radius, `aspect`, `channels`, mask + mix): a flat-topped
disc of equal weights summing to 1, built row by row from running sums, with a padded region rule of
the larger disc semi-axis; `ZDefocus` stays missing because there is no depth channel, so the row is
partial. `DirBlur` has `blur_type` `linear`, `radial` and `zoom`, all three landed; only `linear` is on the
tile path (padded by `ceil(length / 2) + 1`), and a graph containing `radial` or `zoom` falls back to the
full-frame evaluator, the precedent `Transform` and `Mirror` set. `DropShadow` puts the input's alpha,
offset, blurred by `shadow_size` (Nuke's `size`, renamed for the global `LIMITS` key), tinted and scaled
by `opacity`, under the input, padded by `ceil(distance) + ceil(shadow_size)`. `Position`, `BlackOutside`
and `AdjustBBox` change only the data window: they are handled once in the evaluator, carry no mask or
mix (Nuke has none), and are excluded from the tile path. Every node ships `LIMITS`/`CHOICES` entries,
Nuke-matched knobs, a theme colour, bypass through `core.bypass_slot`, and pixel- and window-asserted
tests with tile-versus-evaluator parity across several tiles (`tests/test_2d_parity_group_3b.py`).
Of the Transform group, Stabilize (as its own node), STMap, IDistort, GridWarp family, Tile,
VectorCornerPin/VectorDistort, PointsTo3D/Reconcile3D and TVIScale remain missing.

**2026-09-24, step 4a (HueCorrect, ColorMatrix).** One row flips to supported and one goes from missing
to partial, so the supported count is now 47 (46 + ColorMatrix). `ColorMatrix` is a 3x3 RGB matrix as
nine knobs (`matrix_00` ... `matrix_22`) with an `invert` toggle; a singular matrix passes the input
through when inverted. `HueCorrect` is the reduced model described in its row: six hue anchors, each
with a saturation and a luminance multiplier, smoothstep-interpolated around the hue circle, plus a
`hue_shift`; it stays partial because there is no curve editor and no suppression curves. Both ship
mask + mix, `LIMITS` entries, knobs, a theme colour, bypass through `core.bypass_slot`, and both
evaluation paths (pointwise, identity region rule; the tile path calls the evaluator's own kernels and
is asserted equal across several tiles) in `tests/test_2d_parity_group_4a.py`. HSVTool remains missing.

**2026-09-24, step 4b (TimeClip, FrameRange, AppendClip).** Three more Time rows flip from missing to
supported, so the supported count is now 50 (47 + these 3). All three are producers with their own time
mapping like TimeOffset, FrameHold and Retime: the input is evaluated through the same nested
`Evaluator.evaluate_raster` call at the mapped frame and the cache digest folds in the nested digests,
so a still stays one cache entry per clip. TimeClip and FrameRange share one range mapping (hold clamps,
loop repeats the range, bounce ping-pongs without repeating the end frames, black is transparent).
AppendClip evaluates one clip, or two inside a `dissolve` overlap with hand-checkable weights
`(k + 1) / (dissolve + 1)`. None takes a mask or mix, matching Nuke's Time menu. All three are excluded
from the tile path and fall back to the full-frame evaluator, asserted equal to it; the tile executor's
canvas sizing follows the clip active at the frame. The same commit fixes the step 2c4 digest for a
bypassed time node, which dropped the passed-through input's own digest. Limits: FrameRange presents its
range by frame mapping because the document has no per-branch frame range (a request for the document
model is in the lane report), and an AppendClip clip with no FrameRange/TimeClip directly upstream needs
its `length<i>` knob. Tests: `tests/test_2d_parity_step_4b.py`. Of the Time group, TimeBlur, TimeWarp,
TimeEcho and the optical-flow retimers remain missing.

**2026-09-24, step 4c (shared format registry, Grid, NoOp).** Two rows flip from missing to supported (Grid in
Draw, NoOp in Other), so the supported count is now 52 (50 + these 2); the Reformat row gains the registry.
**Update to the Reformat design note (step 2c5):** the document-level registry the audit sketched now
exists as `settings.formats`, a name -> `{width, height, pixel_aspect}` map seeded with the five built-in
formats. It arrives as an additive upgrade (no version bump, like the other additive options in
`upgrade_document`): an old document gets the built-in list on load and every Reformat renders identically,
and a hand-built document without the section still validates and falls back to the built-ins
(`core.document_formats`). A Reformat's `format` resolves against the registry first and the node-local
`REFORMAT_FORMATS` second, still into the node's own `width`/`height`/`pixel_aspect`, which stay the only
pixel-unit state the kernel and the proxy-tier scaler read. What changed is who writes them: the new `format`
document op edits the registry and touches the Reformats in the same undoable command: `set` (add or update
`name`, `width`, `height`, `pixel_aspect`) re-resolves every Reformat naming the entry, `rename` (`name` to
`new_name`) rewrites their `format`, and `delete` turns them into `Custom` at the window they last had. A
Reformat may name any registry entry, so `validate` accepts registry names for that knob and `CHOICES["format"]`
stays the built-in list for discovery only. `Grid` draws a line at every column where
`(x - offset) mod step < line_width` (likewise rows), with fractional widths giving fractional coverage at the
line's trailing edge; `number_*` above zero sets the step to the format size divided by the count. Both nodes
are on the tile path (Grid through the shared draw function, so seams match, NoOp as a zero-halo passthrough).
Limits: the format editor UI is a request to lane 1 (in the lane report), and Grid draws axis-aligned
hard-edged lines only (no angle, no antialiased sub-pixel positioning). Tests: `tests/test_2d_parity_group_4c.py`.

**2026-09-25, step 5a (EdgeBlur, EdgeExtend, LightWrap, Dither).** Four rows flip from missing to supported
(EdgeBlur and EdgeExtend share the Filter row 10, LightWrap and Dither are Draw rows 8 and 12), the first half of
the fourth parity pass. All four are on the tile path with mask + mix, `LIMITS`/`CHOICES` entries, knobs and
theme colours. `EdgeBlur`, `EdgeExtend` and `LightWrap` are padded filters (region rules in `tiers.py`, halos in
`tiles.resolve_halo`); `LightWrap` is the first padded two-input node, so its rule pads fg and bg equally and
its bypass passes `fg`. `Dither`'s noise is hashed from the absolute pixel position so a tile, given its canvas
origin, matches the full frame exactly. Renamed knobs (`edgeblur_size`, `wrap_diffuse`) avoid clashing with the
global `LIMITS` keys `size` and `diffuse`. Limits: Nuke's EdgeBlur tint and LightWrap luminance/saturation knobs
are not modelled, `EdgeExtend` emits unpremultiplied colour, and Dither's grain differs between proxy tiers.
Tests: `tests/test_2d_parity_step_5a.py`.

**2026-09-26, step 5b (Grain, Posterize, SoftClip, HSVTool, AddMix, Blend, CopyRectangle).** The second half of the
fourth parity pass. By table rows the supported count is now 60 (59 + AddMix) and the partial count 13 (8 + 5):
Grain, Posterize/SoftClip, HSVTool, Blend and CopyRectangle are partial because each row also names a node or knob
set that is still missing (ScannedGrain, Toe, HSVTool colour replacement, Blend beyond eight inputs, CopyBBox).
All seven have mask + mix, `LIMITS`/`CHOICES` entries, knobs with Nuke's names (renamed where a name is already
another node's knob: `red_intensity` for `red_m`, `sat_adjust`/`brt_adjust`), theme colours, bypass through
`core.bypass_slot`, and both evaluation paths. AddMix and CopyRectangle join `MERGE_LIKE_KINDS` (bypass passes `B`);
Blend has no `B`, so its bypass passes the first wired numbered input. `Evaluator._kernel` gained an `origin`
argument (a canvas position, default 0, 0) and `_filtered_pixels` a `frame`, so Grain's noise and CopyRectangle's box
are functions of absolute pixel position and, for Grain, of the frame on both paths. Unverified: Grain's amplitude
scale and HSVTool's saturation and brightness scaling are this repository's reading of Nuke's knobs, not measured
against Nuke. Tests: `tests/test_2d_parity_step_5b.py`.
