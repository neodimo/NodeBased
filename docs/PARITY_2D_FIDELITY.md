# 2D node fidelity against Nuke

**Overall fidelity: 87.0 / 100** (167 nodes scored, rank-weighted; 89.6 counting only the nodes NodeBased has).

This is the score measured by `tools/audit_nuke_fidelity.py` (re-run it from a checkout; a test fails when this file drifts); the first measurement, step A on 2026-10-25, was 54.5. [PARITY_2D.md](PARITY_2D.md) answers *is there a node*; this file answers *how much of the real Nuke node is in it*. Every number is computed from the Nuke 17.0 Reference Guide's own tables, an explicit knob-by-knob mapping and formula tests on NodeBased's evaluator. Nothing here has been compared with a real Nuke render yet: the pixel-truth column is `unmeasured` until a folder of renders exists (`tools/make_nuke_reference_pack.py` writes the scripts that produce them).

## How a node is scored

Per node, five components, each a fraction from 0 to 1:

| Component | Weight | What it counts |
|---|---|---|
| knobs | 40 | The Nuke node's knobs from the guide's tables, minus the shared chrome below, enable checkboxes, Python callbacks, GPU options and (for Read, Write, Viewer) per-format option tabs. Each knob NodeBased pairs with one counts; knobs on the node's main tab weigh 2, the rest 1. |
| defaults | 10 | Of the paired knobs whose Nuke default the guide states and whose NodeBased default is comparable, the fraction that agree. |
| sockets | 10 | The Nuke node's input sockets (mask included, A/B and numbered inputs as listed) that a NodeBased input is paired with. |
| chrome | 15 | The shared node chrome the Nuke node has: channels selector, mask channel, mask invert, mask inject, mask fringe, (un)premult by, its invert, mix, bbox handling. |
| math | 25 | Pixel tests of NodeBased's kernel against the behaviour the guide states (the 30 Merge operations, KeyMix, Gamma, Invert, Multiply, Add, Clamp, Premult, Unpremult, ColorMatrix) or Nuke's long-standing form where the page names the knobs but states no formula (Grade, Exposure; marked *convention*). Only nodes with such cases have this component. |
| pixels | 30 | Pixel truth against real Nuke renders. **Not measured yet**: it joins the formula once `--reference-dir` has renders; until then it is not in any number here. |

`fidelity = 100 * sum(weight * fraction) / sum(weight)` over the components the node has. A component with nothing to count (the guide lists no inputs, no formula cases, no renders) is left out and the rest renormalised. A Nuke node with no NodeBased counterpart scores 0. A node whose guide page lists no inputs or knobs cannot be measured and is left out (listed at the end).

**Overall** is the mean of node fidelity weighted by daily-use rank: a node in row *r* of its group weighs `1 / r`, shared equally by the Nuke nodes of a combined row (`Grain / ScannedGrain` gives each half of `1 / 9`). Rows marked n/a in PARITY_2D.md are not counted.

## By group

| Group | Rank-weighted | Plain mean | Nodes |
|---|---|---|---|
| Image | 79.4 | 80.7 | 6 |
| Draw | 79.6 | 58.4 | 17 |
| Time | 92.2 | 70.5 | 13 |
| Channel | 95.5 | 95.8 | 5 |
| Color | 90.6 | 72.9 | 34 |
| Filter | 81.1 | 60.5 | 32 |
| Keyer | 87.3 | 64.6 | 11 |
| Merge | 94.1 | 88.1 | 12 |
| Transform | 83.9 | 61.6 | 20 |
| Metadata | 99.4 | 98.9 | 5 |
| Other | 76.3 | 32.4 | 12 |

## Per node

Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages (`knobs` with the count of Nuke knobs covered, `defaults`, `sockets`, `chrome`, `math` with passed/total cases) and the pixel-truth column.

### Image

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Read | `Read` | supported | 80.9 | 92 (28/32) | 35 | - | - | - | no fixture |
| 2 | Write | `Write` | supported | 63.8 | 56 (25/56) | 41 | - | 100 | - | no fixture |
| 3 | Viewer | `Viewer` | supported | 89.7 | 83 (20/24) | - | 100 | 100 | - | no fixture |
| 4 | Constant | `Constant` | supported | 96.2 | 100 (4/4) | 75 | - | 100 | - | no fixture |
| 5 | UDIM Import | - | missing | n/a | - | - | - | - | - | no fixture |
| 6 | CurveTool | `CurveTool` | supported | 53.3 | 50 (7/12) | 100 | 100 | 0 | - | no fixture |
| 7 | Profile | `Profile` | supported | 100.0 | 100 (2/2) | 100 | - | 100 | - | no fixture |

### Draw

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Roto | `Roto` | supported | 81.8 | 66 (48/75) | 100 | 100 | 100 | - | no fixture |
| 2 | Text | `Text` | supported | 98.2 | 98 (40/41) | 93 | 100 | 100 | - | no fixture |
| 3 | Rectangle | `Rectangle` | supported | 88.6 | 81 (9/13) | 91 | 100 | 100 | - | no fixture |
| 4 | Ramp | `Ramp` | supported | 92.2 | 89 (9/10) | 83 | 100 | 100 | - | no fixture |
| 5 | Radial | `Radial` | supported | 84.9 | 74 (9/14) | 91 | 100 | 100 | - | no fixture |
| 6 | Noise | `Noise` | supported | 76.0 | 60 (12/26) | 80 | 100 | 100 | - | no fixture |
| 7 | RotoPaint | `RotoPaint` | supported | 91.3 | 84 (73/88) | 100 | 100 | 100 | - | no fixture |
| 8 | LightWrap | `LightWrap` | supported | 60.8 | 36 (7/27) | 62 | 100 | 100 | - | no fixture |
| 9 | Grain | `Grain` | supported | 51.7 | 56 (8/14) | 62 | 100 | 0 | - | no fixture |
| 9 | ScannedGrain | `ScannedGrain` | supported | 20.0 | 0 (0/1) | - | 100 | - | - | no fixture |
| 10 | DustBust | - | supported | 0.0 | 0 (0/5) | - | 0 | - | - | no fixture |
| 10 | MarkerRemoval | - | supported | 0.0 | 0 (0/5) | - | 0 | - | - | no fixture |
| 11 | Flare | `Flare` | supported | 27.1 | 7 (3/50) | 33 | 100 | - | - | no fixture |
| 11 | Glint | `Glint` | supported | 38.4 | 31 (4/13) | 40 | 100 | 17 | - | no fixture |
| 11 | Sparkles | `Sparkles` | supported | 22.6 | 9 (2/32) | 0 | 100 | - | - | no fixture |
| 12 | Dither | `Dither` | supported | 69.3 | 50 (2/4) | 70 | 100 | 100 | - | no fixture |
| 13 | Grid | `Grid` | supported | 89.6 | 83 (10/14) | 92 | 100 | 100 | - | no fixture |

### Time

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TimeOffset | `TimeOffset` | supported | 100.0 | 100 (6/6) | 100 | 100 | - | - | no fixture |
| 2 | FrameHold | `FrameHold` | supported | 100.0 | 100 (7/7) | 100 | 100 | - | - | no fixture |
| 3 | Retime | `Retime` | supported | 97.7 | 97 (14/15) | 100 | 100 | - | - | no fixture |
| 4 | TimeClip | `TimeClip` | supported | 100.0 | 100 (10/10) | 100 | 100 | - | - | no fixture |
| 5 | FrameRange | `FrameRange` | supported | 100.0 | 100 (2/2) | 100 | 100 | - | - | no fixture |
| 6 | AppendClip | `AppendClip` | supported | 100.0 | 100 (6/6) | 100 | 100 | - | - | no fixture |
| 7 | TimeBlur | `TimeBlur` | supported | 91.7 | 100 (4/4) | 50 | 100 | - | - | no fixture |
| 7 | TimeEcho | `TimeEcho` | supported | 61.1 | 67 (2/3) | 0 | 100 | - | - | no fixture |
| 7 | TimeWarp | `TimeWarp` | supported | 69.4 | 67 (2/3) | 50 | 100 | - | - | no fixture |
| 8 | Kronos | `Kronos` | supported | 24.2 | 18 (4/26) | 50 | 25 | - | - | no fixture |
| 8 | OFlow | `OFlow` | supported | 56.1 | 48 (9/21) | 43 | 100 | - | - | no fixture |
| 8 | SmartVector | `SmartVector` | supported | 16.2 | 12 (1/11) | 0 | 50 | - | - | no fixture |
| 8 | VectorToMotion | `VectorToMotion` | supported | n/a | - | - | - | - | - | no fixture |
| 9 | NoTimeBlur | - | missing | 0.0 | 0 (0/2) | - | 0 | - | - | no fixture |

### Channel

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Shuffle | `Shuffle` | supported | 93.3 | 90 (5/6) | 100 | 100 | - | - | no fixture |
| 2 | ShuffleCopy | `ShuffleCopy` | supported | 100.0 | 100 (4/4) | - | 100 | - | - | no fixture |
| 3 | Copy | `Copy` | supported | 97.8 | 100 (10/10) | 100 | 100 | 89 | - | no fixture |
| 4 | ChannelMerge | `ChannelMerge` | supported | 90.7 | 100 (4/4) | 56 | 100 | 83 | - | no fixture |
| 5 | Remove | `Remove` | supported | 97.3 | 100 (4/4) | 80 | 100 | 100 | - | no fixture |

### Color

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Grade | `Grade` | supported | 100.0 | 100 (10/10) | 100 | 100 | 100 | 100 (6/6) | unmeasured |
| 2 | ColorCorrect | `ColorCorrect` | supported | 95.4 | 91 (21/23) | 100 | 100 | 100 | - | no fixture |
| 3 | Invert | `Invert` | supported | 98.9 | 100 (1/1) | 89 | 100 | 100 | 100 (1/1) | unmeasured |
| 4 | Saturation | `Saturation` | supported | 100.0 | 100 (1/1) | 100 | 100 | 100 | - | no fixture |
| 5 | Multiply | `Multiply` | supported | 98.8 | 100 (1/1) | 88 | 100 | 100 | 100 (1/1) | unmeasured |
| 6 | Add | `Add` | supported | 98.9 | 100 (1/1) | 89 | 100 | 100 | 100 (1/1) | unmeasured |
| 7 | Gamma | `Gamma` | supported | 98.9 | 100 (1/1) | 89 | 100 | 100 | 100 (2/2) | unmeasured |
| 8 | Clamp | `Clamp` | supported | 70.8 | 50 (4/8) | 92 | 100 | 100 | 67 (2/3) | unmeasured |
| 9 | HueCorrect | `HueCorrect` | supported | 73.3 | 50 (1/2) | 100 | 100 | 100 | - | no fixture |
| 10 | ColorLookup | `ColorLookup` | supported | 53.3 | 12 (1/8) | 100 | 100 | 100 | - | no fixture |
| 11 | Exposure | `Exposure` | supported | 93.6 | 86 (6/7) | 93 | 100 | 100 | 100 (2/2) | unmeasured |
| 12 | HSVTool | `HSVTool` | partial | 72.8 | 59 (14/25) | 100 | 100 | - | - | no fixture |
| 13 | ColorMatrix | `ColorMatrix` | supported | 100.0 | 100 (2/2) | 100 | 100 | 100 | 100 (1/1) | unmeasured |
| 14 | Colorspace | `Colorspace` | supported | 82.2 | 67 (6/9) | 100 | 100 | 100 | - | no fixture |
| 14 | OCIOColorspace | `OCIOColorspace` | supported | 61.9 | 29 (2/11) | 100 | 100 | 100 | - | no fixture |
| 14 | OCIODisplay | `OCIODisplay` | supported | 37.6 | 23 (3/17) | 33 | 100 | - | - | no fixture |
| 14 | OCIOFileTransform | `OCIOFileTransform` | supported | 88.1 | 80 (3/4) | 91 | 100 | 100 | - | no fixture |
| 14 | OCIOLogConvert | `OCIOLogConvert` | supported | 100.0 | 100 (1/1) | 100 | 100 | 100 | - | no fixture |
| 14 | OCIOLookTransform | `OCIOLookTransform` | supported | 68.0 | 40 (4/14) | 100 | 100 | 100 | - | no fixture |
| 15 | Log2Lin | `Log2Lin` | supported | 88.2 | 80 (4/5) | 92 | 100 | 100 | - | no fixture |
| 15 | PLogLin | `PLogLin` | supported | 83.8 | 80 (4/5) | 92 | 67 | 100 | - | no fixture |
| 16 | ColorTransfer | - | supported | 0.0 | 0 (0/1) | - | 0 | - | - | no fixture |
| 16 | MatchGrade | `MatchGrade` | supported | 26.0 | 8 (3/33) | - | 100 | - | - | no fixture |
| 17 | Expression | `Expression` | supported | 60.0 | 25 (4/16) | 100 | 100 | 100 | - | no fixture |
| 18 | CrossTalk | `CrossTalk` | supported | 100.0 | 100 (1/1) | 100 | 100 | 100 | - | no fixture |
| 19 | Posterize | `Posterize` | supported | 98.5 | 100 (1/1) | 89 | 100 | 100 | - | no fixture |
| 19 | SoftClip | `SoftClip` | supported | 100.0 | 100 (3/3) | 100 | 100 | - | - | no fixture |
| 19 | Toe | `Toe` | supported | 71.7 | 50 (1/2) | 88 | 100 | 100 | - | no fixture |
| 20 | HistEQ | `HistEQ` | supported | 76.9 | 100 (1/1) | - | 100 | 0 | - | no fixture |
| 20 | Histogram | `Histogram` | supported | 65.1 | 67 (2/3) | 100 | 100 | 14 | - | no fixture |
| 20 | MinColor | `MinColor` | supported | 15.4 | 0 (0/4) | - | 100 | 0 | - | no fixture |
| 20 | Sampler | `Sampler` | supported | 30.8 | 25 (1/4) | - | 100 | 0 | - | no fixture |
| 21 | GenerateLUT | `GenerateLUT` | supported | 28.8 | 18 (1/9) | 0 | 100 | - | - | no fixture |
| 21 | Vectorfield | `Vectorfield` | supported | 42.3 | 38 (4/10) | 0 | 100 | - | - | no fixture |

### Filter

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Blur | `Blur` | supported | 98.7 | 100 (4/4) | 90 | 100 | 100 | - | no fixture |
| 2 | Sharpen | `Sharpen` | supported | 97.9 | 100 (7/7) | 85 | 100 | 100 | - | no fixture |
| 3 | Erode (filter) | `ErodeFilter` | supported | 100.0 | 100 (2/2) | 100 | 100 | 100 | - | no fixture |
| 4 | Median | `Median` | supported | 71.4 | 50 (1/2) | 86 | 100 | 100 | - | no fixture |
| 5 | Glow | `Glow` | supported | 63.4 | 36 (4/11) | 80 | 100 | 100 | - | no fixture |
| 6 | Soften | `Soften` | supported | 97.9 | 100 (7/7) | 85 | 100 | 100 | - | no fixture |
| 7 | DropShadow | `DropShadow` | supported | 48.7 | 56 (5/9) | 20 | 50 | - | - | no fixture |
| 8 | Defocus | `Defocus` | supported | 66.3 | 40 (2/5) | 88 | 100 | 100 | - | no fixture |
| 8 | ZDefocus | `ZDefocus` | supported | 56.5 | 27 (8/36) | 64 | 100 | 100 | - | no fixture |
| 9 | DirBlur | `DirBlur` | supported | 60.0 | 33 (4/12) | 67 | 100 | 100 | - | no fixture |
| 10 | EdgeBlur | `EdgeBlur` | supported | 55.2 | 22 (2/9) | 75 | 100 | 100 | - | no fixture |
| 10 | EdgeExtend | `EdgeExtend` | supported | 40.0 | 0 (0/6) | 100 | 50 | 100 | - | no fixture |
| 11 | Erode (fast) | `Erode` | supported | 98.1 | 100 (1/1) | 86 | 100 | 100 | - | no fixture |
| 12 | DegrainSimple | `DegrainSimple` | supported | 95.6 | 100 (3/3) | 67 | 100 | 100 | - | no fixture |
| 12 | Denoise | `Denoise` | supported | 18.1 | 6 (2/37) | 50 | 33 | - | - | no fixture |
| 13 | MotionBlur | `MotionBlur` | supported | 26.4 | 33 (2/6) | 0 | 25 | - | - | no fixture |
| 13 | MotionBlur2D | `MotionBlur2D` | supported | 63.9 | 75 (3/4) | 33 | 50 | - | - | no fixture |
| 13 | MotionBlur3D | `MotionBlur3D` | supported | 55.6 | 50 (3/6) | 33 | 100 | - | - | no fixture |
| 13 | VectorBlur | `VectorBlur` | supported | 57.9 | 29 (4/15) | 67 | 100 | 100 | - | no fixture |
| 14 | Bilateral | `Bilateral` | supported | 61.9 | 40 (2/5) | 88 | 67 | 100 | - | no fixture |
| 15 | Convolve | `Convolve` | supported | 46.7 | 0 (0/4) | 100 | 100 | 100 | - | no fixture |
| 15 | Matrix | `Matrix` | supported | 100.0 | 100 (2/2) | 100 | 100 | 100 | - | no fixture |
| 16 | BumpBoss | `BumpBoss` | supported | 42.2 | 0 (0/6) | 100 | 67 | 100 | - | no fixture |
| 16 | EdgeDetect | `EdgeDetect` | supported | 66.1 | 36 (2/7) | 100 | 100 | 100 | - | no fixture |
| 16 | Emboss | `Emboss` | supported | 54.5 | 18 (2/13) | 86 | 100 | 100 | - | no fixture |
| 16 | Laplacian | `Laplacian` | supported | 44.4 | 0 (0/4) | 83 | 100 | 100 | - | no fixture |
| 17 | Inpaint | `Inpaint` | supported | 10.3 | 0 (0/11) | - | 67 | 0 | - | no fixture |
| 18 | GodRays | `GodRays` | supported | 37.1 | 30 (3/10) | 33 | 100 | 17 | - | no fixture |
| 18 | LevelSet | `LevelSet` | supported | 100.0 | 100 (8/8) | 100 | 100 | - | - | no fixture |
| 18 | VolumeRays | `VolumeRays` | supported | 23.2 | 4 (1/24) | - | 100 | - | - | no fixture |
| 19 | ZSlice | `ZSlice` | supported | 76.7 | 100 (3/3) | 50 | 100 | 17 | - | no fixture |
| 20 | Bokeh | - | supported | 0.0 | 0 (0/42) | - | 0 | - | - | no fixture |

### Keyer

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Keyer | `Keyer` | supported | 96.3 | 100 (10/10) | 78 | 100 | - | - | no fixture |
| 2 | ChromaKeyer | `ChromaKeyer` | supported | 97.8 | 100 (16/16) | 87 | 100 | - | - | no fixture |
| 3 | Difference | `Difference` | supported | 100.0 | 100 (4/4) | 100 | 100 | - | - | no fixture |
| 4 | HueKeyer | `HueKeyer` | supported | 96.7 | 100 (4/4) | 75 | 100 | 100 | - | no fixture |
| 5 | Keylight | `ScreenKeyer` | partial | 98.0 | 98 (34/35) | 97 | 100 | - | - | no fixture |
| 6 | Primatte | - | missing | 0.0 | 0 (0/26) | - | 0 | 0 | - | no fixture |
| 6 | Ultimatte | - | missing | 0.0 | 0 (0/60) | - | 0 | - | - | no fixture |
| 7 | IBKColor | `IBKColor` | supported | 79.2 | 100 (6/6) | 25 | 50 | - | - | no fixture |
| 7 | IBKGizmo | `IBKGizmo` | supported | 59.7 | 47 (7/15) | 71 | 100 | - | - | no fixture |
| 8 | Cryptomatte | `Cryptomatte` | partial | 40.0 | 25 (1/4) | - | 100 | - | - | no fixture |
| 8 | Encryptomatte | `Encryptomatte` | partial | 42.9 | 29 (1/4) | - | 100 | - | - | no fixture |

### Merge

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Merge | `Merge` | supported | 100.0 | 100 (9/9) | 100 | 100 | 100 | 100 (29/29) | unmeasured |
| 2 | Premult | `Premult` | supported | 100.0 | 100 (2/2) | 100 | 100 | 100 | 100 (1/1) | unmeasured |
| 3 | Unpremult | `Unpremult` | supported | 100.0 | 100 (2/2) | 100 | 100 | 100 | 100 (1/1) | unmeasured |
| 4 | Switch | `Switch` | supported | 100.0 | 100 (1/1) | 100 | 100 | - | - | no fixture |
| 5 | Dissolve | `Dissolve` | supported | 100.0 | 100 (1/1) | 100 | 100 | 100 | - | no fixture |
| 6 | KeyMix | `Keymix` | supported | 70.0 | 50 (1/2) | 100 | 100 | 33 | 100 (1/1) | unmeasured |
| 7 | AddMix | `AddMix` | supported | 46.7 | 0 (0/6) | 100 | 100 | 100 | - | no fixture |
| 8 | Blend | `Blend` | supported | 93.8 | 100 (2/2) | 83 | 100 | 80 | - | no fixture |
| 9 | CopyBBox | `CopyBBox` | supported | n/a | - | - | - | - | - | no fixture |
| 9 | CopyRectangle | `CopyRectangle` | supported | 100.0 | 100 (5/5) | 100 | 100 | 100 | - | no fixture |
| 10 | ContactSheet | `ContactSheet` | supported | 90.9 | 100 (11/11) | 45 | 100 | - | - | no fixture |
| 11 | TimeDissolve | `TimeDissolve` | supported | 100.0 | 100 (3/3) | 100 | 100 | - | - | no fixture |
| 12 | ZMerge | `ZMerge` | supported | 55.6 | 33 (1/3) | 100 | 100 | - | - | no fixture |

### Transform

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Transform | `Transform` | supported | 96.2 | 100 (15/15) | 77 | 100 | - | - | no fixture |
| 2 | Crop | `Crop` | supported | 95.8 | 100 (5/5) | 75 | 100 | - | - | no fixture |
| 3 | Tracker | `Tracker` | supported | 85.3 | 78 (41/54) | 79 | 100 | 100 | - | no fixture |
| 4 | PlanarTracker | `PlanarTracker` | supported | n/a | - | - | - | - | - | no fixture |
| 5 | Reformat | `Reformat` | supported | 83.0 | 81 (13/16) | 73 | 100 | - | - | no fixture |
| 6 | CornerPin2D | `CornerPin` | supported | 100.0 | 100 (24/24) | 100 | 100 | - | - | no fixture |
| 7 | Mirror | `Mirror` | supported | 100.0 | 100 (2/2) | 100 | 100 | - | - | no fixture |
| 8 | Stabilize | `Stabilize` | supported | n/a | - | - | - | - | - | no fixture |
| 9 | Position | `Position` | supported | 100.0 | 100 (1/1) | 100 | 100 | - | - | no fixture |
| 10 | AdjustBBox | `AdjustBBox` | supported | 83.3 | 100 (1/1) | 0 | 100 | - | - | no fixture |
| 11 | BlackOutside | `BlackOutside` | supported | 100.0 | - | - | 100 | - | - | no fixture |
| 12 | STMap | `STMap` | supported | 31.1 | 33 (2/6) | 0 | 100 | 0 | - | no fixture |
| 13 | IDistort | `IDistort` | supported | 40.0 | 50 (4/8) | 0 | 100 | 0 | - | no fixture |
| 14 | GridWarp | `GridWarp` | supported | 13.1 | 6 (2/34) | 0 | 50 | 17 | - | no fixture |
| 14 | SplineWarp | `SplineWarp` | supported | 28.6 | 18 (4/23) | 0 | 67 | 50 | - | no fixture |
| 15 | Tile | `Tile` | supported | 78.3 | 80 (4/5) | 50 | 100 | - | - | no fixture |
| 16 | VectorCornerPin | `VectorCornerPin` | supported | 20.0 | 0 (0/17) | - | 100 | - | - | no fixture |
| 16 | VectorDistort | `VectorDistort` | supported | 44.4 | 33 (2/6) | 100 | 100 | 0 | - | no fixture |
| 17 | PointsTo3D | `PointsTo3D` | supported | 48.3 | 60 (6/9) | 0 | 50 | - | - | no fixture |
| 18 | Reconcile3D | `Reconcile3D` | supported | 36.7 | 13 (1/14) | 100 | 67 | - | - | no fixture |
| 19 | TVIScale | `TVIScale` | supported | 30.0 | 0 (0/3) | 100 | 100 | 17 | - | no fixture |
| 20 | LensDistortion | `Undistort` | supported | 17.8 | 2 (1/58) | 0 | 100 | - | - | no fixture |

### Metadata

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ViewMetaData | `ViewMetaData` | supported | 100.0 | 100 (3/3) | - | 100 | - | - | no fixture |
| 2 | ModifyMetaData | `ModifyMetaData` | supported | 100.0 | 100 (1/1) | - | 100 | - | - | no fixture |
| 3 | CopyMetaData | `CopyMetaData` | supported | 100.0 | 100 (5/5) | 100 | 100 | - | - | no fixture |
| 4 | AddTimeCode | `AddTimeCode` | supported | 94.4 | 100 (5/5) | 67 | 100 | - | - | no fixture |
| 5 | CompareMetaData | `CompareMetaData` | supported | 100.0 | 100 (1/1) | - | 100 | - | - | no fixture |

### Other

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Dot | `Dot` | supported | 100.0 | - | - | 100 | - | - | no fixture |
| 2 | NoOp | `NoOp` | supported | 100.0 | - | - | 100 | - | - | no fixture |
| 3 | Backdrop | `Backdrop` | supported | n/a | - | - | - | - | - | no fixture |
| 4 | PostageStamp | `PostageStamp` | supported | n/a | - | - | - | - | - | no fixture |
| 5 | Group | `Group` | supported | 0.0 | 0 (0/1) | - | 0 | - | - | no fixture |
| 5 | Input | `Input` | supported | n/a | - | - | - | - | - | no fixture |
| 5 | Output | `Output` | supported | n/a | - | - | - | - | - | no fixture |
| 6 | BlinkScript | - | missing | 0.0 | 0 (0/19) | - | 0 | - | - | no fixture |
| 7 | BurnIn | `BurnIn` | supported | 88.3 | 100 (18/18) | 30 | 100 | - | - | no fixture |
| 8 | Link | - | missing | 0.0 | - | - | 0 | - | - | no fixture |
| 8 | LiveGroup | - | missing | 0.0 | 0 (0/11) | - | 0 | - | - | no fixture |
| 8 | LiveInput | - | missing | 0.0 | 0 (0/4) | - | - | - | - | no fixture |
| 8 | VariableGroup | - | missing | 0.0 | 0 (0/1) | - | 0 | - | - | no fixture |
| 10 | Precomp | `Precomp` | supported | 100.0 | 100 (9/9) | 100 | - | - | - | no fixture |
| 11 | Root | - | missing | 0.0 | 0 (0/43) | - | - | - | - | no fixture |
| 13 | Assert | `Assert` | supported | n/a | - | - | - | - | - | no fixture |
| 14 | AudioRead | - | missing | 0.0 | 0 (0/10) | - | - | - | - | no fixture |

## The 30 largest gaps

Ranked by `rank weight * (100 - fidelity)`, the share of the overall shortfall each node accounts for. This is the backlog for the following steps.

| # | Nuke node | NodeBased | Weighted gap | Fidelity | What is missing |
|---|---|---|---|---|---|
| 1 | Read (Image #1) | `Read` | 19.10 | 80.9 | 4 secondary knobs missing: `importsequence`, `sequence`, `shownmetadata`, `metadatafilter` |
| 2 | Roto (Draw #1) | `Roto` | 18.23 | 81.8 | 27 secondary knobs missing |
| 3 | Write (Image #2) | `Write` | 18.08 | 63.8 | primary knobs missing: `proxy`, `read_all_lines`; 29 secondary knobs missing |
| 4 | BlinkScript (Other #6) | - | 16.67 | 0.0 | no node: needs a Blink-equivalent runtime (PARITY_2D.md Other #6) |
| 5 | NoTimeBlur (Time #9) | - | 11.11 | 0.0 | no node (PARITY_2D.md Time #9) |
| 6 | Root (Other #11) | - | 9.09 | 0.0 | no node: per-script project settings (PARITY_2D.md Other #11) |
| 7 | Primatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 8 | Ultimatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 9 | CurveTool (Image #6) | `CurveTool` | 7.78 | 53.3 | primary knobs missing: `operation`, `go`; 3 secondary knobs missing: `color`, `ccrange`, `avgframes`; node chrome missing: channels selector |
| 10 | AddMix (Merge #7) | `AddMix` | 7.62 | 46.7 | primary knobs missing: `A`, `B`, `output`, `premultiplied`, `nonlinear`, `lut` |
| 11 | DropShadow (Filter #7) | `DropShadow` | 7.33 | 48.7 | primary knobs missing: `enable_dropshadow_effect`, `inherit_input_color`, `size`, `operation`; sockets missing: `bg` |
| 12 | Glow (Filter #5) | `Glow` | 7.32 | 63.4 | primary knobs missing: `saturation`, `W`, `nonlinear`, `effect_only`, `filter`, `quality`, `crop` |
| 13 | Median (Filter #4) | `Median` | 7.14 | 71.4 | primary knobs missing: `ignore_top_line` |
| 14 | AudioRead (Other #14) | - | 7.14 | 0.0 | no node: no audio pipeline (PARITY_2D.md Other #14) |
| 15 | Shuffle (Channel #1) | `Shuffle` | 6.67 | 93.3 | 1 secondary knob missing: `out2` |
| 16 | Group (Other #5) | `Group` | 6.67 | 0.0 | primary knobs missing: `lock_connections`; sockets missing: `unnamed` |
| 17 | STMap (Transform #12) | `STMap` | 5.74 | 31.1 | primary knobs missing: `blur`, `blur_scale`, `maskChannel`, `premultiplied`; node chrome missing: channels selector, mask invert |
| 18 | Inpaint (Filter #17) | `Inpaint` | 5.28 | 10.3 | primary knobs missing: `fillRegion`, `smoothness`, `LABColourSpace`; 8 secondary knobs missing: `stretchAmount`, `stretchDirection`, `detailSource`, `detailAmount`, `translate` ...; sockets missing: `Detail`; node chrome missing: channels selector |
| 19 | KeyMix (Merge #6) | `Keymix` | 5.00 | 70.0 | primary knobs missing: `maskChannel`; node chrome missing: channels selector, bbox handling |
| 20 | DustBust (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 21 | MarkerRemoval (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 22 | Bokeh (Filter #20) | - | 5.00 | 0.0 | provided as ZDefocus bokeh kernels, not a node (PARITY_2D.md Filter #20) |
| 23 | Tracker (Transform #3) | `Tracker` | 4.90 | 85.3 | 13 secondary knobs missing |
| 24 | LightWrap (Draw #8) | `LightWrap` | 4.90 | 60.8 | primary knobs missing: `onlywrap`, `disableluma`, `enableglow`, `saturation`, `LumaTolerance`; 15 secondary knobs missing |
| 25 | ColorLookup (Color #10) | `ColorLookup` | 4.67 | 53.3 | primary knobs missing: `use_precomputed`, `range`, `source`, `target`, `setRGB`, `setRGBA`, `setA` |
| 26 | IDistort (Transform #13) | `IDistort` | 4.62 | 40.0 | primary knobs missing: `blur`, `blur_scale`, `maskChannel`, `premultiplied`; node chrome missing: channels selector, mask invert |
| 27 | DirBlur (Filter #9) | `DirBlur` | 4.44 | 60.0 | primary knobs missing: `BlurLayer`, `holdout`, `target`, `UseTarget`, `Samples`, `PixelAspect`, `Quality`, `pixeloffset` |
| 28 | ScannedGrain (Draw #9) | `ScannedGrain` | 4.44 | 20.0 | primary knobs missing: `fullGrain` |
| 29 | LensDistortion (Transform #20) | `Undistort` | 4.11 | 17.8 | primary knobs missing: `distortionModelPreset`, `lens`; 55 secondary knobs missing |
| 30 | Noise (Draw #6) | `Noise` | 4.00 | 76.0 | primary knobs missing: `type`, `nyquist`; 12 secondary knobs missing: `translate`, `rotate`, `scaling`, `skew`, `center` ... |

## Formula cases

Each case runs one node on the 16 x 12 float fixture (HDR values from -0.5 to 12, alpha from 0 to 1) through `Evaluator._kernel` and compares every pixel with the stated behaviour (absolute and relative tolerance 1e-05; a failing case shows its largest error rounded to four decimals and the first pixel in raster order that has it). `guide` means the guide's text states the behaviour; `convention` means it does not (see the ambiguities below). `inexpressible` means NodeBased has no knob that can set what Nuke's knob sets, so it cannot even attempt the case.

| Case | Source | Result | Max abs error | Worst pixel (y, x, channel) |
|---|---|---|---|---|
| `merge_atop` | guide | pass | within tolerance | - |
| `merge_average` | guide | pass | within tolerance | - |
| `merge_color-burn` | guide | pass | within tolerance | - |
| `merge_color-dodge` | guide | pass | within tolerance | - |
| `merge_conjoint-over` | guide | pass | within tolerance | - |
| `merge_copy` | guide | pass | within tolerance | - |
| `merge_difference` | guide | pass | within tolerance | - |
| `merge_disjoint-over` | guide | pass | within tolerance | - |
| `merge_divide` | guide | pass | within tolerance | - |
| `merge_exclusion` | guide | pass | within tolerance | - |
| `merge_from` | guide | pass | within tolerance | - |
| `merge_geometric` | guide | pass | within tolerance | - |
| `merge_hard-light` | guide | pass | within tolerance | - |
| `merge_hypot` | guide | pass | within tolerance | - |
| `merge_in` | guide | pass | within tolerance | - |
| `merge_mask` | guide | pass | within tolerance | - |
| `merge_matte` | guide | not scored: the guide's algorithm text for matte ('screen(A,a,B,b) if A==a, B if a<=0, ...') cannot be implemented without guessing | - | - |
| `merge_max` | guide | pass | within tolerance | - |
| `merge_min` | guide | pass | within tolerance | - |
| `merge_minus` | guide | pass | within tolerance | - |
| `merge_multiply` | guide | pass | within tolerance | - |
| `merge_out` | guide | pass | within tolerance | - |
| `merge_over` | guide | pass | within tolerance | - |
| `merge_overlay` | guide | pass | within tolerance | - |
| `merge_plus` | guide | pass | within tolerance | - |
| `merge_screen` | guide | pass | within tolerance | - |
| `merge_soft-light` | guide | pass | within tolerance | - |
| `merge_stencil` | guide | pass | within tolerance | - |
| `merge_under` | guide | pass | within tolerance | - |
| `merge_xor` | guide | pass | within tolerance | - |
| `keymix_default` | guide | pass | within tolerance | - |
| `gamma_2.2` | guide | pass | within tolerance | - |
| `gamma_0.5` | guide | pass | within tolerance | - |
| `invert_rgb` | guide | pass | within tolerance | - |
| `multiply_1.5` | guide | pass | within tolerance | - |
| `add_0.25` | guide | pass | within tolerance | - |
| `clamp_default` | guide | pass | within tolerance | - |
| `clamp_0.2_0.8` | guide | pass | within tolerance | - |
| `clamp_to_values` | guide | inexpressible | - | - |
| `premult_default` | guide | pass | within tolerance | - |
| `unpremult_default` | guide | pass | within tolerance | - |
| `colormatrix_rows` | guide | pass | within tolerance | - |
| `grade_affine` | convention | pass | within tolerance | - |
| `grade_default_clamps` | convention | pass | within tolerance | - |
| `grade_levels` | convention | pass | within tolerance | - |
| `grade_gamma` | convention | pass | within tolerance | - |
| `grade_white_clamp` | convention | pass | within tolerance | - |
| `grade_reverse` | convention | pass | within tolerance | - |
| `exposure_stops` | convention | pass | within tolerance | - |
| `exposure_stops_blackpoint` | convention | pass | within tolerance | - |

## What a reader should know about the reference data

- The guide has no type column. A knob's `type` in `docs/parity/nuke_nodes.json` is inferred from its default (`enabled`/`disabled` is a bool, a number is a number, several numbers a list) and from the choices the text lists.
- Where a default reads 'Dependent on ...', 'dependent on file' or is prose, the data records `null` with a note. These knobs count as covered or missing like any other but never enter the defaults component.
- The guide does not say how a colour or position knob is stored. NodeBased splits them into scalar params (`red`, `green`, `blue`, `alpha`; `p0_x`, `p0_y`); the mapping file lists them as `components` of one Nuke knob.
- Multiply's `value` is listed with default 0, which would blank the image; Nuke's own default is 1. The score skips that default (`skip_default`).
- Exposure's `mode` default is listed as `Densities`; the long-standing Nuke default is `Stops`. NodeBased defaults to stops, so the defaults component counts it as a mismatch.
- The guide spells Sharpen's, Soften's and Laplacian's quality knob `quailty`; the data keeps the guide's spelling.
- Clamp's `MinClampTo_ enable` has a stray space in the guide's table; the data removes whitespace from knob names (`MinClampTo_enable`).
- The guide repeats scripting names inside one node (`invert`, `output`, `color`, `area`): the data suffixes the repeats in table order (`area#2`, `area#3`, `area#4` are the y, r and t halves of CopyRectangle's area).
- No guide page exists for PlanarTracker, and the Stabilize and Backdrop pages list no inputs or knobs; those nodes are not scored.
- Gamma is described as 'the actual function is 1/pow(value)'. That sentence is not a function of the pixel; the formula case reads it as `out = in ** (1 / value)` and tests only non-negative input.
- The guide states no formula for Grade or Exposure. Their cases use Nuke's long-standing forms and are marked *convention*; the lane brief's `A*(in-B)^gamma` shape is the Grade form.
- The Merge page's algorithm for `matte` ('screen(A,a,B,b) if A==a, B if a<=0, ...') cannot be implemented without guessing, so that case is listed and not scored. `soft-light`, `hard-light`, `screen`, `multiply` and `divide` are implemented literally from the guide's text, which is not always what Nuke does outside 0..1; a failure there is a difference from the guide as written.
- The guide's mask input is listed as a socket only on some pages whose nodes do have a mask channel knob (for example Crop has neither). The sockets component counts what each page lists.
- Features NodeBased keeps in a payload or panel rather than in a node param (Roto and RotoPaint shape and stroke fields, the Viewer's look and proxy state, the metadata viewers) are paired as `panel` entries, each citing the document that describes it; they count as covered. A Nuke knob NodeBased cannot honour is paired as `unsupported` with the reason; it counts as missing, so listing it explains a gap without closing it. Roto's CornerPin export tab is not paired.

### Guide pages without a row in PARITY_2D.md

The guide's group pages list these nodes that no row of PARITY_2D.md covers, so this score cannot see them: `Image/checkerboard`, `Image/colorbars`, `Image/colorwheel`, `Image/unrealreader`, `Time/add_32_pulldown`, `Time/frameblend`, `Time/remove_32_pulldown`, `Time/temporalmedian`, `Time/vectorgenerator`, `Channel/addchannels`, `Color/cliptest`, `Color/cmstestpattern`, `Color/hueshift`, `Color/ociocdltransform`, `Color/rolloffcontrast`, `Filter/degrainblue`, `Filter/erode_blur`, `Merge/layercontactsheet`, `Merge/mergeexpression`, `Transform/camerashake`, `Transform/card3d`, `Transform/gridwarptracker`, `Transform/sphericaltransform`, `Transform/transformmasked`, `Other/stickynote`.

### Nodes not scored

- UDIM Import (Image #5): the guide page lists no inputs and no knobs.
- VectorToMotion (Time #8): the guide page lists no inputs and no knobs.
- CopyBBox (Merge #9): the guide page lists no inputs and no knobs.
- PlanarTracker (Transform #4): the guide page lists no inputs and no knobs.
- Stabilize (Transform #8): the guide page lists no inputs and no knobs.
- Backdrop (Other #3): the guide page lists no inputs and no knobs.
- PostageStamp (Other #4): the guide page lists no inputs and no knobs.
- Input (Other #5): the guide page lists no inputs and no knobs.
- Output (Other #5): the guide page lists no inputs and no knobs.
- Assert (Other #13): the guide page lists no inputs and no knobs.
- Truelight (Color #22): n/a in PARITY_2D.md.
- Absminus (Merge #13): n/a in PARITY_2D.md.
- In (Merge #13): n/a in PARITY_2D.md.
- Matte (Merge #13): n/a in PARITY_2D.md.
- Max (Merge #13): n/a in PARITY_2D.md.
- Min (Merge #13): n/a in PARITY_2D.md.
- Multiply (Merge #13): n/a in PARITY_2D.md.
- Out (Merge #13): n/a in PARITY_2D.md.
- Plus (Merge #13): n/a in PARITY_2D.md.
- Screen (Merge #13): n/a in PARITY_2D.md.
- VariableSwitch (Merge #14): n/a in PARITY_2D.md.
- DiskCache (Other #9): n/a in PARITY_2D.md.
- Annotations (Other #12): n/a in PARITY_2D.md.

### NodeBased 2D kinds paired with no Nuke node

These kinds are in the node catalogue's Image to Other groups but are not scored, because no node in PARITY_2D.md pairs with them: `Checker`, `ReadBundle`, `ConditionedRead`, `Generate`, `ChannelShuffle`, `VectorGenerator`, `Dilate`, `Redistort`, `GridWarpTracker`.

