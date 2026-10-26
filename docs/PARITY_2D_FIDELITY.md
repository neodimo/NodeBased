# 2D node fidelity against Nuke

**Overall fidelity: 65.2 / 100** (167 nodes scored, rank-weighted; 67.1 counting only the nodes NodeBased has).

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
| Image | 48.4 | 38.2 | 6 |
| Draw | 62.5 | 51.5 | 17 |
| Time | 56.2 | 53.3 | 13 |
| Channel | 87.4 | 81.0 | 5 |
| Color | 69.3 | 68.5 | 34 |
| Filter | 59.0 | 54.3 | 32 |
| Keyer | 54.5 | 46.3 | 11 |
| Merge | 61.5 | 73.6 | 12 |
| Transform | 67.8 | 54.1 | 20 |
| Metadata | 87.1 | 80.8 | 5 |
| Other | 70.2 | 22.3 | 12 |

## Per node

Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages (`knobs` with the count of Nuke knobs covered, `defaults`, `sockets`, `chrome`, `math` with passed/total cases) and the pixel-truth column.

### Image

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Read | `Read` | supported | 63.3 | 58 (15/32) | 86 | - | - | - | no fixture |
| 2 | Write | `Write` | supported | 51.2 | 32 (12/56) | 55 | - | 100 | - | no fixture |
| 3 | Viewer | `Viewer` | supported | 30.8 | 25 (6/24) | - | 100 | 0 | - | no fixture |
| 4 | Constant | `Constant` | supported | 30.8 | 50 (2/4) | 0 | - | 0 | - | no fixture |
| 5 | UDIM Import | - | missing | n/a | - | - | - | - | - | no fixture |
| 6 | CurveTool | `CurveTool` | supported | 53.3 | 50 (7/12) | 100 | 100 | 0 | - | no fixture |
| 7 | Profile | `Profile` | supported | 0.0 | 0 (0/2) | - | - | 0 | - | no fixture |

### Draw

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Roto | `Roto` | supported | 52.1 | 18 (10/75) | 100 | 100 | 80 | - | no fixture |
| 2 | Text | `Text` | supported | 63.9 | 38 (12/41) | 79 | 100 | 100 | - | no fixture |
| 3 | Rectangle | `Rectangle` | supported | 88.6 | 81 (9/13) | 91 | 100 | 100 | - | no fixture |
| 4 | Ramp | `Ramp` | supported | 92.2 | 89 (9/10) | 83 | 100 | 100 | - | no fixture |
| 5 | Radial | `Radial` | supported | 84.9 | 74 (9/14) | 91 | 100 | 100 | - | no fixture |
| 6 | Noise | `Noise` | supported | 76.0 | 60 (12/26) | 80 | 100 | 100 | - | no fixture |
| 7 | RotoPaint | `RotoPaint` | supported | 41.2 | 27 (22/88) | 100 | 100 | 0 | - | no fixture |
| 8 | LightWrap | `LightWrap` | supported | 60.8 | 36 (7/27) | 62 | 100 | 100 | - | no fixture |
| 9 | Grain | `Grain` | supported | 51.7 | 56 (8/14) | 62 | 100 | 0 | - | no fixture |
| 9 | ScannedGrain | `ScannedGrain` | supported | 20.0 | 0 (0/1) | - | 100 | - | - | no fixture |
| 10 | DustBust | - | supported | 0.0 | 0 (0/5) | - | 0 | - | - | no fixture |
| 10 | MarkerRemoval | - | supported | 0.0 | 0 (0/5) | - | 0 | - | - | no fixture |
| 11 | Flare | `Flare` | supported | 27.1 | 7 (3/50) | 33 | 100 | - | - | no fixture |
| 11 | Glint | `Glint` | supported | 38.4 | 31 (4/13) | 40 | 100 | 17 | - | no fixture |
| 11 | Sparkles | `Sparkles` | supported | 22.6 | 9 (2/32) | 0 | 100 | - | - | no fixture |
| 12 | Dither | `Dither` | supported | 66.4 | 50 (2/4) | 67 | 100 | 88 | - | no fixture |
| 13 | Grid | `Grid` | supported | 89.6 | 83 (10/14) | 92 | 100 | 100 | - | no fixture |

### Time

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TimeOffset | `TimeOffset` | supported | 55.6 | 33 (2/6) | 100 | 100 | - | - | no fixture |
| 2 | FrameHold | `FrameHold` | supported | 57.6 | 36 (2/7) | 100 | 100 | - | - | no fixture |
| 3 | Retime | `Retime` | supported | 45.2 | 34 (5/15) | 33 | 100 | - | - | no fixture |
| 4 | TimeClip | `TimeClip` | supported | 60.0 | 40 (4/10) | 100 | 100 | - | - | no fixture |
| 5 | FrameRange | `FrameRange` | supported | 100.0 | 100 (2/2) | 100 | 100 | - | - | no fixture |
| 6 | AppendClip | `AppendClip` | supported | 55.6 | 33 (2/6) | 100 | 100 | - | - | no fixture |
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
| 3 | Copy | `Copy` | supported | 71.6 | 80 (8/10) | 100 | 100 | 11 | - | no fixture |
| 4 | ChannelMerge | `ChannelMerge` | supported | 86.7 | 100 (4/4) | 50 | 100 | 67 | - | no fixture |
| 5 | Remove | `Remove` | supported | 53.3 | 25 (1/4) | 50 | 100 | 100 | - | no fixture |

### Color

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Grade | `Grade` | supported | 41.1 | 20 (2/10) | 100 | 100 | 88 | 0 (0/4) | unmeasured |
| 2 | ColorCorrect | `ColorCorrect` | supported | 92.9 | 91 (21/23) | 100 | 100 | 88 | - | no fixture |
| 3 | Invert | `Invert` | supported | 56.7 | 0 (0/1) | 86 | 100 | 88 | 100 (1/1) | unmeasured |
| 4 | Saturation | `Saturation` | supported | 97.5 | 100 (1/1) | 100 | 100 | 88 | - | no fixture |
| 5 | Multiply | `Multiply` | supported | 96.7 | 100 (1/1) | 86 | 100 | 88 | 100 (1/1) | unmeasured |
| 6 | Add | `Add` | supported | 96.9 | 100 (1/1) | 88 | 100 | 88 | 100 (1/1) | unmeasured |
| 7 | Gamma | `Gamma` | supported | 96.9 | 100 (1/1) | 88 | 100 | 88 | 100 (2/2) | unmeasured |
| 8 | Clamp | `Clamp` | supported | 68.9 | 50 (4/8) | 91 | 100 | 88 | 67 (2/3) | unmeasured |
| 9 | HueCorrect | `HueCorrect` | supported | 70.8 | 50 (1/2) | 100 | 100 | 88 | - | no fixture |
| 10 | ColorLookup | `ColorLookup` | supported | 50.8 | 12 (1/8) | 100 | 100 | 88 | - | no fixture |
| 11 | Exposure | `Exposure` | supported | 91.6 | 86 (6/7) | 92 | 100 | 88 | 100 (2/2) | unmeasured |
| 12 | HSVTool | `HSVTool` | partial | 72.8 | 59 (14/25) | 100 | 100 | - | - | no fixture |
| 13 | ColorMatrix | `ColorMatrix` | supported | 98.1 | 100 (2/2) | 100 | 100 | 88 | 100 (1/1) | unmeasured |
| 14 | Colorspace | `Colorspace` | supported | 79.7 | 67 (6/9) | 100 | 100 | 88 | - | no fixture |
| 14 | OCIOColorspace | `OCIOColorspace` | supported | 59.4 | 29 (2/11) | 100 | 100 | 88 | - | no fixture |
| 14 | OCIODisplay | `OCIODisplay` | supported | 37.6 | 23 (3/17) | 33 | 100 | - | - | no fixture |
| 14 | OCIOFileTransform | `OCIOFileTransform` | supported | 85.5 | 80 (3/4) | 90 | 100 | 88 | - | no fixture |
| 14 | OCIOLogConvert | `OCIOLogConvert` | supported | 97.5 | 100 (1/1) | 100 | 100 | 88 | - | no fixture |
| 14 | OCIOLookTransform | `OCIOLookTransform` | supported | 65.5 | 40 (4/14) | 100 | 100 | 88 | - | no fixture |
| 15 | Log2Lin | `Log2Lin` | supported | 85.6 | 80 (4/5) | 91 | 100 | 88 | - | no fixture |
| 15 | PLogLin | `PLogLin` | supported | 81.2 | 80 (4/5) | 91 | 67 | 88 | - | no fixture |
| 16 | ColorTransfer | - | supported | 0.0 | 0 (0/1) | - | 0 | - | - | no fixture |
| 16 | MatchGrade | `MatchGrade` | supported | 26.0 | 8 (3/33) | - | 100 | - | - | no fixture |
| 17 | Expression | `Expression` | supported | 57.1 | 25 (4/16) | 100 | 100 | 86 | - | no fixture |
| 18 | CrossTalk | `CrossTalk` | supported | 100.0 | 100 (1/1) | 100 | 100 | 100 | - | no fixture |
| 19 | Posterize | `Posterize` | supported | 95.8 | 100 (1/1) | 88 | 100 | 88 | - | no fixture |
| 19 | SoftClip | `SoftClip` | supported | 100.0 | 100 (3/3) | 100 | 100 | - | - | no fixture |
| 19 | Toe | `Toe` | supported | 68.6 | 50 (1/2) | 86 | 100 | 86 | - | no fixture |
| 20 | HistEQ | `HistEQ` | supported | 76.9 | 100 (1/1) | - | 100 | 0 | - | no fixture |
| 20 | Histogram | `Histogram` | supported | 65.1 | 67 (2/3) | 100 | 100 | 14 | - | no fixture |
| 20 | MinColor | `MinColor` | supported | 15.4 | 0 (0/4) | - | 100 | 0 | - | no fixture |
| 20 | Sampler | `Sampler` | supported | 30.8 | 25 (1/4) | - | 100 | 0 | - | no fixture |
| 21 | GenerateLUT | `GenerateLUT` | supported | 28.8 | 18 (1/9) | 0 | 100 | - | - | no fixture |
| 21 | Vectorfield | `Vectorfield` | supported | 42.3 | 38 (4/10) | 0 | 100 | - | - | no fixture |

### Filter

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Blur | `Blur` | supported | 54.4 | 25 (1/4) | 83 | 100 | 83 | - | no fixture |
| 2 | Sharpen | `Sharpen` | supported | 54.8 | 29 (2/7) | 71 | 100 | 83 | - | no fixture |
| 3 | Erode (filter) | `ErodeFilter` | supported | 96.7 | 100 (2/2) | 100 | 100 | 83 | - | no fixture |
| 4 | Median | `Median` | supported | 67.8 | 50 (1/2) | 83 | 100 | 83 | - | no fixture |
| 5 | Glow | `Glow` | supported | 59.8 | 36 (4/11) | 78 | 100 | 83 | - | no fixture |
| 6 | Soften | `Soften` | supported | 46.5 | 14 (1/7) | 67 | 100 | 83 | - | no fixture |
| 7 | DropShadow | `DropShadow` | supported | 48.7 | 56 (5/9) | 20 | 50 | - | - | no fixture |
| 8 | Defocus | `Defocus` | supported | 62.8 | 40 (2/5) | 86 | 100 | 83 | - | no fixture |
| 8 | ZDefocus | `ZDefocus` | supported | 52.8 | 27 (8/36) | 62 | 100 | 83 | - | no fixture |
| 9 | DirBlur | `DirBlur` | supported | 56.1 | 33 (4/12) | 62 | 100 | 83 | - | no fixture |
| 10 | EdgeBlur | `EdgeBlur` | supported | 51.4 | 22 (2/9) | 71 | 100 | 83 | - | no fixture |
| 10 | EdgeExtend | `EdgeExtend` | supported | 40.0 | 0 (0/6) | 100 | 50 | 100 | - | no fixture |
| 11 | Erode (fast) | `Erode` | supported | 94.4 | 100 (1/1) | 83 | 100 | 83 | - | no fixture |
| 12 | DegrainSimple | `DegrainSimple` | supported | 91.7 | 100 (3/3) | 62 | 100 | 83 | - | no fixture |
| 12 | Denoise | `Denoise` | supported | 18.1 | 6 (2/37) | 50 | 33 | - | - | no fixture |
| 13 | MotionBlur | `MotionBlur` | supported | 26.4 | 33 (2/6) | 0 | 25 | - | - | no fixture |
| 13 | MotionBlur2D | `MotionBlur2D` | supported | 63.9 | 75 (3/4) | 33 | 50 | - | - | no fixture |
| 13 | MotionBlur3D | `MotionBlur3D` | supported | 55.6 | 50 (3/6) | 33 | 100 | - | - | no fixture |
| 13 | VectorBlur | `VectorBlur` | supported | 54.0 | 29 (4/15) | 62 | 100 | 83 | - | no fixture |
| 14 | Bilateral | `Bilateral` | supported | 58.3 | 40 (2/5) | 86 | 67 | 83 | - | no fixture |
| 15 | Convolve | `Convolve` | supported | 43.3 | 0 (0/4) | 100 | 100 | 83 | - | no fixture |
| 15 | Matrix | `Matrix` | supported | 96.7 | 100 (2/2) | 100 | 100 | 83 | - | no fixture |
| 16 | BumpBoss | `BumpBoss` | supported | 38.9 | 0 (0/6) | 100 | 67 | 83 | - | no fixture |
| 16 | EdgeDetect | `EdgeDetect` | supported | 66.1 | 36 (2/7) | 100 | 100 | 100 | - | no fixture |
| 16 | Emboss | `Emboss` | supported | 50.1 | 18 (2/13) | 83 | 100 | 80 | - | no fixture |
| 16 | Laplacian | `Laplacian` | supported | 40.7 | 0 (0/4) | 80 | 100 | 83 | - | no fixture |
| 17 | Inpaint | `Inpaint` | supported | 10.3 | 0 (0/11) | - | 67 | 0 | - | no fixture |
| 18 | GodRays | `GodRays` | supported | 37.1 | 30 (3/10) | 33 | 100 | 17 | - | no fixture |
| 18 | LevelSet | `LevelSet` | supported | 100.0 | 100 (8/8) | 100 | 100 | - | - | no fixture |
| 18 | VolumeRays | `VolumeRays` | supported | 23.2 | 4 (1/24) | - | 100 | - | - | no fixture |
| 19 | ZSlice | `ZSlice` | supported | 76.7 | 100 (3/3) | 50 | 100 | 17 | - | no fixture |
| 20 | Bokeh | - | supported | 0.0 | 0 (0/42) | - | 0 | - | - | no fixture |

### Keyer

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Keyer | `Keyer` | supported | 67.8 | 60 (6/10) | 67 | 100 | - | - | no fixture |
| 2 | ChromaKeyer | `ChromaKeyer` | supported | 30.0 | 20 (3/16) | 0 | 100 | - | - | no fixture |
| 3 | Difference | `Difference` | supported | 66.7 | 50 (2/4) | 100 | 100 | - | - | no fixture |
| 4 | HueKeyer | `HueKeyer` | supported | 66.7 | 50 (2/4) | 50 | 100 | 100 | - | no fixture |
| 5 | Keylight | `ScreenKeyer` | partial | 56.6 | 43 (12/35) | 92 | 75 | - | - | no fixture |
| 6 | Primatte | - | missing | 0.0 | 0 (0/26) | - | 0 | 0 | - | no fixture |
| 6 | Ultimatte | - | missing | 0.0 | 0 (0/60) | - | 0 | - | - | no fixture |
| 7 | IBKColor | `IBKColor` | supported | 79.2 | 100 (6/6) | 25 | 50 | - | - | no fixture |
| 7 | IBKGizmo | `IBKGizmo` | supported | 59.7 | 47 (7/15) | 71 | 100 | - | - | no fixture |
| 8 | Cryptomatte | `Cryptomatte` | partial | 40.0 | 25 (1/4) | - | 100 | - | - | no fixture |
| 8 | Encryptomatte | `Encryptomatte` | partial | 42.9 | 29 (1/4) | - | 100 | - | - | no fixture |

### Merge

| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Merge | `Merge` | supported | 52.5 | 11 (1/9) | 100 | 100 | 67 | 72 (21/29) | unmeasured |
| 2 | Premult | `Premult` | supported | 38.9 | 0 (0/2) | - | 100 | 0 | 100 (1/1) | unmeasured |
| 3 | Unpremult | `Unpremult` | supported | 38.9 | 0 (0/2) | - | 100 | 0 | 100 (1/1) | unmeasured |
| 4 | Switch | `Switch` | supported | 100.0 | 100 (1/1) | 100 | 100 | - | - | no fixture |
| 5 | Dissolve | `Dissolve` | supported | 96.0 | 100 (1/1) | 100 | 100 | 80 | - | no fixture |
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
| 2 | Crop | `Crop` | supported | 36.0 | 20 (1/5) | - | 100 | - | - | no fixture |
| 3 | Tracker | `Tracker` | supported | 44.6 | 9 (5/54) | 50 | 100 | 100 | - | no fixture |
| 4 | PlanarTracker | `PlanarTracker` | supported | n/a | - | - | - | - | - | no fixture |
| 5 | Reformat | `Reformat` | supported | 83.0 | 81 (13/16) | 73 | 100 | - | - | no fixture |
| 6 | CornerPin2D | `CornerPin` | supported | 50.4 | 38 (10/24) | 50 | 100 | - | - | no fixture |
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
| 3 | CopyMetaData | `CopyMetaData` | supported | 36.0 | 20 (1/5) | - | 100 | - | - | no fixture |
| 4 | AddTimeCode | `AddTimeCode` | supported | 67.8 | 60 (3/5) | 67 | 100 | - | - | no fixture |
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
| 7 | BurnIn | `BurnIn` | supported | 48.9 | 45 (8/18) | 12 | 100 | - | - | no fixture |
| 8 | Link | - | missing | 0.0 | - | - | 0 | - | - | no fixture |
| 8 | LiveGroup | - | missing | 0.0 | 0 (0/11) | - | 0 | - | - | no fixture |
| 8 | LiveInput | - | missing | 0.0 | 0 (0/4) | - | - | - | - | no fixture |
| 8 | VariableGroup | - | missing | 0.0 | 0 (0/1) | - | 0 | - | - | no fixture |
| 10 | Precomp | `Precomp` | supported | 18.8 | 19 (2/9) | - | - | - | - | no fixture |
| 11 | Root | - | missing | 0.0 | 0 (0/43) | - | - | - | - | no fixture |
| 13 | Assert | `Assert` | supported | n/a | - | - | - | - | - | no fixture |
| 14 | AudioRead | - | missing | 0.0 | 0 (0/10) | - | - | - | - | no fixture |

## The 30 largest gaps

Ranked by `rank weight * (100 - fidelity)`, the share of the overall shortfall each node accounts for. This is the backlog for the following steps.

| # | Nuke node | NodeBased | Weighted gap | Fidelity | What is missing |
|---|---|---|---|---|---|
| 1 | Grade (Color #1) | `Grade` | 58.88 | 41.1 | primary knobs missing: `blackpoint`, `whitepoint`, `black`, `white`, `gamma`, `reverse`, `black_clamp`, `white_clamp`; node chrome missing: mask fringe; formula cases failing: `grade_affine`, `grade_default_clamps`, `grade_levels`, `grade_gamma` |
| 2 | Roto (Draw #1) | `Roto` | 47.88 | 52.1 | 65 secondary knobs missing; node chrome missing: mask fringe |
| 3 | Merge (Merge #1) | `Merge` | 47.45 | 52.5 | primary knobs missing: `sRGB`, `screen_alpha`, `metainput`, `rangeinput`, `Achannels`, `Bchannels`, `output`, `also_merge`; node chrome missing: mask fringe, bbox handling; formula cases failing: `merge_color-burn`, `merge_color-dodge`, `merge_divide`, `merge_geometric`, `merge_hard-light`, `merge_multiply` and 2 more |
| 4 | Blur (Filter #1) | `Blur` | 45.56 | 54.4 | primary knobs missing: `filter`, `quality`, `crop`; node chrome missing: mask fringe |
| 5 | TimeOffset (Time #1) | `TimeOffset` | 44.44 | 55.6 | primary knobs missing: `rounding_mode`, `enable_path_mask`, `mask_patterns`, `inject_mask` |
| 6 | Read (Image #1) | `Read` | 36.70 | 63.3 | primary knobs missing: `localizationPolicy`, `updateLocalization`, `proxy`, `proxy_format`, `reload`; 12 secondary knobs missing: `key1`, `value1`, `key2`, `value2`, `key3` ... |
| 7 | ChromaKeyer (Keyer #2) | `ChromaKeyer` | 35.00 | 30.0 | primary knobs missing: `_view`, `_screenGain`, `_screenBalance`; 10 secondary knobs missing: `_chromaGain`, `_whitePoint`, `_blackPoint`, `_alphaBias`, `_insideMask` ... |
| 8 | Keyer (Keyer #1) | `Keyer` | 32.22 | 67.8 | primary knobs missing: `input`, `output`, `combine`, `range` |
| 9 | Crop (Transform #2) | `Crop` | 32.00 | 36.0 | primary knobs missing: `softness`, `reformat`, `intersect`, `crop` |
| 10 | Premult (Merge #2) | `Premult` | 30.56 | 38.9 | primary knobs missing: `alpha`, `invert`; node chrome missing: channels selector |
| 11 | Write (Image #2) | `Write` | 24.42 | 51.2 | primary knobs missing: `proxy`, `render_order`, `Render`, `reading`, `on_error`, `reload`, `read_all_lines`; 37 secondary knobs missing |
| 12 | Viewer (Image #3) | `Viewer` | 23.08 | 30.8 | primary knobs missing: `frame_range`, `frame_range_lock`, `frame_increment`, `rgb_only`, `ignore_pixel_aspect`, `zoom_lock`, `show_overscan`, `full_frame_processing` and 10 more; node chrome missing: channels selector |
| 13 | Sharpen (Filter #2) | `Sharpen` | 22.62 | 54.8 | primary knobs missing: `minimum`, `maximum`, `filter`, `quailty`, `crop`; node chrome missing: mask fringe |
| 14 | CopyMetaData (Metadata #3) | `CopyMetaData` | 21.33 | 36.0 | primary knobs missing: `mergeMode`, `metadatafilterMode`, `imageMetadatafilter`, `imageMetadatafilterMode` |
| 15 | FrameHold (Time #2) | `FrameHold` | 21.21 | 57.6 | primary knobs missing: `rounding_mode`, `setToCurrentFrame`; 3 secondary knobs missing: `enable_path_mask`, `mask_patterns`, `inject_mask` |
| 16 | Unpremult (Merge #3) | `Unpremult` | 20.37 | 38.9 | primary knobs missing: `alpha`, `invert`; node chrome missing: channels selector |
| 17 | Tracker (Transform #3) | `Tracker` | 18.47 | 44.6 | primary knobs missing: `add_track`, `del_tracks`, `select_all`, `average_tracks`; 45 secondary knobs missing |
| 18 | Retime (Time #3) | `Retime` | 18.26 | 45.2 | primary knobs missing: `input.first_lock`, `input.last_lock`, `reverse`, `output.first_lock`, `output.last_lock`, `before`, `after`, `filter` and 1 more; 1 secondary knob missing: `warp` |
| 19 | Text (Draw #2) | `Text` | 18.04 | 63.9 | primary knobs missing: `yjustify`, `index`, `global_font_scale`, `font_width`, `font_height`, `kerning`, `tracking`, `baseline_shift` and 1 more; 20 secondary knobs missing |
| 20 | Constant (Image #4) | `Constant` | 17.31 | 30.8 | primary knobs missing: `first`, `last`; node chrome missing: channels selector |
| 21 | BlinkScript (Other #6) | - | 16.67 | 0.0 | no node: needs a Blink-equivalent runtime (PARITY_2D.md Other #6) |
| 22 | Invert (Color #3) | `Invert` | 14.43 | 56.7 | primary knobs missing: `clamp`; node chrome missing: mask fringe |
| 23 | Profile (Image #7) | `Profile` | 14.29 | 0.0 | primary knobs missing: `frame_range`, `open_profile_panel`; node chrome missing: channels selector |
| 24 | NoTimeBlur (Time #9) | - | 11.11 | 0.0 | no node (PARITY_2D.md Time #9) |
| 25 | Difference (Keyer #3) | `Difference` | 11.11 | 66.7 | primary knobs missing: `output`, `output#2` |
| 26 | TimeClip (Time #4) | `TimeClip` | 10.00 | 60.0 | primary knobs missing: `fadeIn`, `fadeOut`, `frame_mode`, `origfirst`, `origlast`, `reverse` |
| 27 | Copy (Channel #3) | `Copy` | 9.48 | 71.6 | primary knobs missing: `metainput`, `rangeinput`; node chrome missing: channels selector, mask channel, mask invert, mask inject, mask fringe, (un)premult by, (un)premult invert, bbox handling |
| 28 | Remove (Channel #5) | `Remove` | 9.33 | 53.3 | primary knobs missing: `channels2`, `channels3`, `channels4` |
| 29 | Root (Other #11) | - | 9.09 | 0.0 | no node: per-script project settings (PARITY_2D.md Other #11) |
| 30 | Soften (Filter #6) | `Soften` | 8.92 | 46.5 | primary knobs missing: `minimum`, `maximum`, `amount`, `filter`, `quailty`, `crop`; node chrome missing: mask fringe |

## Formula cases

Each case runs one node on the 16 x 12 float fixture (HDR values from -0.5 to 12, alpha from 0 to 1) through `Evaluator._kernel` and compares every pixel with the stated behaviour (absolute and relative tolerance 1e-05; a failing case shows its largest error rounded to four decimals and the first pixel in raster order that has it). `guide` means the guide's text states the behaviour; `convention` means it does not (see the ambiguities below). `inexpressible` means NodeBased has no knob that can set what Nuke's knob sets, so it cannot even attempt the case.

| Case | Source | Result | Max abs error | Worst pixel (y, x, channel) |
|---|---|---|---|---|
| `merge_atop` | guide | pass | within tolerance | - |
| `merge_average` | guide | pass | within tolerance | - |
| `merge_color-burn` | guide | fail | 29 | (3, 6, 1) |
| `merge_color-dodge` | guide | fail | 47 | (2, 4, 1) |
| `merge_conjoint-over` | guide | pass | within tolerance | - |
| `merge_copy` | guide | pass | within tolerance | - |
| `merge_difference` | guide | pass | within tolerance | - |
| `merge_disjoint-over` | guide | pass | within tolerance | - |
| `merge_divide` | guide | fail | 36 | (3, 3, 1) |
| `merge_exclusion` | guide | pass | within tolerance | - |
| `merge_from` | guide | pass | within tolerance | - |
| `merge_geometric` | guide | fail | 4 | (1, 6, 0) |
| `merge_hard-light` | guide | fail | 241 | (2, 8, 1) |
| `merge_hypot` | guide | pass | within tolerance | - |
| `merge_in` | guide | pass | within tolerance | - |
| `merge_mask` | guide | pass | within tolerance | - |
| `merge_matte` | guide | not scored: the guide's algorithm text for matte ('screen(A,a,B,b) if A==a, B if a<=0, ...') cannot be implemented without guessing | - | - |
| `merge_max` | guide | pass | within tolerance | - |
| `merge_min` | guide | pass | within tolerance | - |
| `merge_minus` | guide | pass | within tolerance | - |
| `merge_multiply` | guide | fail | 0.75 | (3, 4, 1) |
| `merge_out` | guide | pass | within tolerance | - |
| `merge_over` | guide | pass | within tolerance | - |
| `merge_overlay` | guide | pass | within tolerance | - |
| `merge_plus` | guide | pass | within tolerance | - |
| `merge_screen` | guide | fail | 132 | (2, 8, 1) |
| `merge_soft-light` | guide | fail | 924 | (2, 9, 1) |
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
| `grade_affine` | convention | fail | 0.1 | (0, 0, 0) |
| `grade_default_clamps` | convention | fail | 0.75 | (0, 0, 0) |
| `grade_levels` | convention | inexpressible | - | - |
| `grade_gamma` | convention | inexpressible | - | - |
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
- Features NodeBased keeps in a payload or panel rather than in a node param (Roto and RotoPaint shape and stroke fields, the Viewer's look and proxy state, the metadata viewers) are paired as `panel` entries, each citing the document that describes it; they count as covered. Roto's transform, motion blur and CornerPin export tabs are not paired.

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

