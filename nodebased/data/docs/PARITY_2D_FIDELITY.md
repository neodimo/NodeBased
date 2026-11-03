# 2D node fidelity against Nuke

**Overall fidelity: 88.6 / 100** (167 nodes scored, rank-weighted; 90.8 counting only the nodes NodeBased has).

Under the previous formula (knob presence, defaults, sockets, chrome and formula cases only, used from 2026-10-25 to 2026-11-02) the same tree scores 92.0 (94.4 counting only the nodes NodeBased has). The difference is the requirements added on 2026-11-02: knobs that act, Nuke's layout, per-channel splits, and behaviour that counts against a node until a pixel case checks it.

This is the score measured by `tools/audit_nuke_fidelity.py` (re-run it from a checkout; a test fails when this file drifts); the first measurement, step A on 2026-10-25, was 54.5. [PARITY_2D.md](PARITY_2D.md) answers *is there a node*; this file answers *how much of the real Nuke node is in it*. Every number is computed from the Nuke 17.0 Reference Guide's own tables, an explicit knob-by-knob mapping and formula tests on NodeBased's evaluator. Nothing here has been compared with a real Nuke render yet: the pixel-truth column is `unmeasured` until a folder of renders exists (`tools/make_nuke_reference_pack.py` writes the scripts that produce them).

## How a node is scored

Per node, up to nine components, each a fraction from 0 to 1. The parts marked *(11/2)* are the requirements added on 2026-11-02 (DiMo, 11/1: "I need these nodes in a much better place" and "match known layouts and how some can hold multiple channel values"):

| Component | Weight | What it counts |
|---|---|---|
| knobs | 30 | The Nuke node's knobs from the guide's tables, minus the shared chrome below, enable checkboxes, Python callbacks, GPU options and (for Read, Write, Viewer) per-format option tabs. Each knob NodeBased pairs with one counts; knobs on the node's main tab weigh 2, the rest 1. |
| acts | 20 | *(11/2)* Of the paired knobs, the share (main tab 2, the rest 1) that change NodeBased's output pixels when moved off their default on the fixture, through `Evaluator._kernel` (`tools/nuke_fidelity_requirements.py`). A knob that moves nothing alone is tried again with each other paired knob moved first. Knobs that act only over time, on metadata or on the bounding box are left out (listed below), and so are taps, whose behaviour is their input. Only nodes whose kernel runs on the fixture have this component. |
| defaults | 10 | Of the paired knobs whose Nuke default the guide states and whose NodeBased default is comparable, the fraction that agree. |
| sockets | 10 | The Nuke node's input sockets (mask included, A/B and numbered inputs as listed) that a NodeBased input is paired with. |
| chrome | 10 | The shared node chrome the Nuke node has: channels selector, mask channel, mask invert, mask inject, mask fringe, (un)premult by, its invert, mix, bbox handling. |
| math | 25 | Pixel tests of NodeBased's kernel against the behaviour the guide states (the 30 Merge operations, KeyMix, Gamma, Invert, Multiply, Add, Clamp, Premult, Unpremult, ColorMatrix) or Nuke's long-standing form, or a published measurement of Nuke itself, where the page names the knobs but states no formula (Grade, Exposure, ColorCorrect and the other grading nodes; marked *convention*). *(11/2)* A node whose kernel runs on the fixture and that has no case at all scores 0 here (behaviour unchecked); the previous formula left it out, which is how a node with no behaviour checked could score 100. A node whose every case is listed `not scored`, with its reason, and a tap, stay out. |
| layout | 10 | *(11/2)* Nuke's knob layout: half is the share of paired knobs that sit in the guide table's order in the Properties panel (the longest run already in order), half the share of paired main-tab knobs whose label reads as the guide's label or the knob's scripting name (case and punctuation ignored; a label the guide gives a whole row of knobs, or `n/a`, is not compared). Section breaks and which knobs share a row are not measured yet. |
| splits | 10 | *(11/2)* For the Nuke knobs that split per channel (an AColor knob into r g b a, a WH knob into w h; `docs/parity/nuke_split_knobs.json`, see below), the share that are splittable NodeBased params (`core.SPLITTABLE`). A knob paired with one NodeBased param per channel and no one-slider form counts half. Only nodes the table lists have this component. |
| pixels | 30 | Pixel truth against real Nuke renders. **Not measured yet**: it joins the formula once `--reference-dir` has renders; until then it is not in any number here. |

`fidelity = 100 * sum(weight * fraction) / sum(weight)` over the components the node has. A component with nothing to count (the guide lists no inputs, no formula cases, no renders) is left out and the rest renormalised. A Nuke node with no NodeBased counterpart scores 0. A node whose guide page lists no inputs or knobs cannot be measured and is left out (listed at the end).

Root is the one Nuke node scored against something that is not a node: the project settings. Its pairing (`document_settings` in `docs/parity/nodebased_to_nuke.json`) names document fields, such as `time.first` or `settings.color.view`, where other pairings name a node's params. A Root knob counts as covered only when its document field exists and changes behaviour (`tests/test_2d_parity_notimeblur.py`, `RootPairingTests`, shows each one acting); every other Root knob is listed there as unsupported with a one-line reason and scores as missing. The defaults component compares a paired field with the value a new project starts with (`core.demo_document`). Root has no input sockets, shared chrome or formula cases, so those components are left out and only knobs and defaults are renormalised.

**Overall** is the mean of node fidelity weighted by daily-use rank: a node in row *r* of its group weighs `1 / r`, shared equally by the Nuke nodes of a combined row (`Grain / ScannedGrain` gives each half of `1 / 9`). Rows marked n/a in PARITY_2D.md are not counted.

## By group

| Group | Rank-weighted | Plain mean | Previous formula | Nodes |
|---|---|---|---|---|
| Image | 80.9 | 84.7 | 85.9 | 6 |
| Draw | 77.2 | 65.4 | 86.5 | 17 |
| Time | 96.1 | 78.9 | 96.1 | 13 |
| Channel | 95.8 | 95.4 | 98.9 | 5 |
| Color | 94.5 | 79.4 | 94.4 | 34 |
| Filter | 83.0 | 71.4 | 91.2 | 32 |
| Keyer | 88.4 | 67.5 | 88.3 | 11 |
| Merge | 91.7 | 83.6 | 99.4 | 12 |
| Transform | 89.2 | 74.7 | 91.2 | 20 |
| Metadata | 99.0 | 98.2 | 99.4 | 5 |
| Other | 80.1 | 42.6 | 80.3 | 12 |

Behaviour checked: 54 of the 167 scored nodes have at least one scored pixel case against what the node computes (the `math` column below); 26 nodes whose kernel runs on the fixture have none and score 0 on `math`.

Knobs that act: 281 of the 323 paired knobs probed change the output pixels; 42 change nothing on the fixture (listed per node below). Splits: 8 of the 14 nodes with split knobs in the table split all of them.

## Per node

Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages (`knobs` with the count of Nuke knobs covered, `acts` with acting/probed knobs, `defaults`, `sockets`, `chrome`, `math` with passed/total cases, `unchecked` when no case exists, `layout`, `splits` with fully split/listed knobs) and the pixel-truth column. `Prev.` is the node under the previous formula.

### Image

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Read | `Read` | supported | 81.7 | 87.1 | 100 (32/32) | - | 35 | - | - | - | 73 | - | no fixture |
| 2 | Write | `Write` | supported | 66.9 | 67.2 | 61 (27/56) | - | 42 | - | 100 | - | 76 | - | no fixture |
| 3 | Viewer | `Viewer` | supported | 90.0 | 89.7 | 83 (20/24) | - | - | 100 | 100 | - | - | - | no fixture |
| 4 | Constant | `Constant` | supported | 71.4 | 96.2 | 100 (4/4) | 100 (2/2) | 75 | - | 100 | 0 unchecked | 75 | - | no fixture |
| 5 | UDIM Import | - | missing | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 6 | CurveTool | `CurveTool` | supported | 98.1 | 100.0 | 100 (12/12) | - | 100 | 100 | 100 | - | 86 | - | no fixture |
| 7 | Profile | `Profile` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | - | 100 | - | - | - | no fixture |

### Draw

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Roto | `Roto` | supported | 60.9 | 81.8 | 66 (48/75) | - | 100 | 100 | 100 | 0 unchecked | 81 | - | no fixture |
| 2 | Text | `Text` | supported | 95.4 | 98.2 | 98 (40/41) | - | 93 | 100 | 100 | - | 80 | - | no fixture |
| 3 | Rectangle | `Rectangle` | supported | 98.6 | 98.7 | 100 (13/13) | - | 87 | 100 | 100 | 100 (9/9) | 100 | - | unmeasured |
| 4 | Ramp | `Ramp` | supported | 98.4 | 98.5 | 100 (10/10) | - | 85 | 100 | 100 | 100 (6/6) | 100 | - | unmeasured |
| 5 | Radial | `Radial` | supported | 98.7 | 98.8 | 100 (14/14) | - | 88 | 100 | 100 | 100 (4/4) | 100 | - | unmeasured |
| 6 | Noise | `Noise` | supported | 99.6 | 99.5 | 100 (26/26) | - | 96 | 100 | 100 | - | 100 | 100 (1/1) | unmeasured |
| 7 | RotoPaint | `RotoPaint` | supported | 66.8 | 91.3 | 84 (73/88) | - | 100 | 100 | 100 | 0 unchecked | 83 | - | no fixture |
| 8 | LightWrap | `LightWrap` | supported | 66.1 | 98.6 | 100 (27/27) | 41 (8/27) | 89 | 100 | 100 | 0 unchecked | 89 | - | no fixture |
| 9 | Grain | `Grain` | supported | 96.7 | 95.0 | 88 (13/14) | 100 (13/13) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 9 | ScannedGrain | `ScannedGrain` | supported | 15.4 | 20.0 | 0 (0/1) | - | - | 100 | - | 0 unchecked | - | - | no fixture |
| 10 | DustBust | - | supported | 0.0 | 0.0 | 0 (0/5) | - | - | 0 | - | - | - | - | no fixture |
| 10 | MarkerRemoval | - | supported | 0.0 | 0.0 | 0 (0/5) | - | - | 0 | - | - | - | - | no fixture |
| 11 | Flare | `Flare` | supported | 35.4 | 27.1 | 7 (3/50) | 75 (2/3) | 33 | 100 | - | 0 unchecked | 67 | - | no fixture |
| 11 | Glint | `Glint` | supported | 45.6 | 38.4 | 31 (4/13) | 100 (4/4) | 40 | 100 | 17 | 0 unchecked | 75 | - | no fixture |
| 11 | Sparkles | `Sparkles` | supported | 40.6 | 22.6 | 9 (2/32) | 100 (2/2) | 0 | 100 | - | 0 unchecked | 100 | - | no fixture |
| 12 | Dither | `Dither` | supported | 94.9 | 99.2 | 100 (4/4) | 75 (3/4) | 92 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 13 | Grid | `Grid` | supported | 99.3 | 99.4 | 100 (14/14) | - | 94 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |

### Time

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TimeOffset | `TimeOffset` | supported | 100.0 | 100.0 | 100 (6/6) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 2 | FrameHold | `FrameHold` | supported | 100.0 | 100.0 | 100 (7/7) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 3 | Retime | `Retime` | supported | 98.3 | 97.7 | 97 (14/15) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 4 | TimeClip | `TimeClip` | supported | 100.0 | 100.0 | 100 (10/10) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 5 | FrameRange | `FrameRange` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 6 | AppendClip | `AppendClip` | supported | 96.7 | 100.0 | 100 (6/6) | - | 100 | 100 | - | - | 80 | - | no fixture |
| 7 | TimeBlur | `TimeBlur` | supported | 89.6 | 91.7 | 100 (4/4) | - | 50 | 100 | - | - | 88 | - | no fixture |
| 7 | TimeEcho | `TimeEcho` | supported | 54.2 | 61.1 | 67 (2/3) | - | 0 | 100 | - | - | 25 | - | no fixture |
| 7 | TimeWarp | `TimeWarp` | supported | 75.0 | 69.4 | 67 (2/3) | - | 50 | 100 | - | - | 100 | - | no fixture |
| 8 | Kronos | `Kronos` | supported | 33.1 | 24.2 | 18 (4/26) | - | 50 | 25 | - | - | 71 | - | no fixture |
| 8 | OFlow | `OFlow` | supported | 61.4 | 56.1 | 48 (9/21) | - | 43 | 100 | - | - | 80 | - | no fixture |
| 8 | SmartVector | `SmartVector` | supported | 17.1 | 16.2 | 12 (1/11) | - | 0 | 50 | - | - | - | - | no fixture |
| 8 | VectorToMotion | `VectorToMotion` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 9 | NoTimeBlur | `NoTimeBlur` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | 100 | - | - | 100 | - | no fixture |

### Channel

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Shuffle | `Shuffle` | supported | 100.0 | 100.0 | 100 (6/6) | 100 (1/1) | 100 | 100 | - | 100 (7/7) | 100 | - | unmeasured |
| 2 | ShuffleCopy | `ShuffleCopy` | supported | 85.0 | 100.0 | 100 (4/4) | - | - | 100 | - | - | 25 | - | no fixture |
| 3 | Copy | `Copy` | supported | 99.0 | 98.3 | 100 (10/10) | 100 (8/8) | 100 | 100 | 89 | 100 (8/8) | 100 | - | unmeasured |
| 4 | ChannelMerge | `ChannelMerge` | supported | 95.7 | 94.2 | 100 (4/4) | 100 (4/4) | 67 | 100 | 83 | 100 (13/13) | 100 | - | unmeasured |
| 5 | Remove | `Remove` | supported | 97.1 | 97.3 | 100 (4/4) | - | 80 | 100 | 100 | - | 100 | - | no fixture |

### Color

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Grade | `Grade` | supported | 100.0 | 100.0 | 100 (10/10) | 100 (10/10) | 100 | 100 | 100 | 100 (6/6) | 100 | 100 (7/7) | unmeasured |
| 2 | ColorCorrect | `ColorCorrect` | supported | 97.9 | 96.5 | 91 (21/23) | 100 (20/20) | 100 | 100 | 100 | 100 (20/20) | 100 | 100 (20/20) | unmeasured |
| 3 | Invert | `Invert` | supported | 98.9 | 98.9 | 100 (1/1) | 100 (1/1) | 89 | 100 | 100 | 100 (1/1) | - | - | unmeasured |
| 4 | Saturation | `Saturation` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 100 (13/13) | - | - | unmeasured |
| 5 | Multiply | `Multiply` | supported | 98.9 | 98.8 | 100 (1/1) | 100 (1/1) | 88 | 100 | 100 | 100 (1/1) | - | 100 (1/1) | unmeasured |
| 6 | Add | `Add` | supported | 99.0 | 98.9 | 100 (1/1) | 100 (1/1) | 89 | 100 | 100 | 100 (1/1) | - | 100 (1/1) | unmeasured |
| 7 | Gamma | `Gamma` | supported | 99.0 | 98.9 | 100 (1/1) | 100 (1/1) | 89 | 100 | 100 | 100 (2/2) | - | 100 (1/1) | unmeasured |
| 8 | Clamp | `Clamp` | supported | 90.5 | 99.4 | 100 (8/8) | 100 (8/8) | 94 | 100 | 100 | 100 (3/3) | 88 | 0 (0/2) | unmeasured |
| 9 | HueCorrect | `HueCorrect` | supported | 82.4 | 80.0 | 50 (1/2) | - | 100 | 100 | 100 | 100 (15/15) | - | - | unmeasured |
| 10 | ColorLookup | `ColorLookup` | supported | 88.6 | 90.0 | 75 (6/8) | - | 100 | 100 | 100 | 100 (7/7) | 67 | - | unmeasured |
| 11 | Exposure | `Exposure` | supported | 94.9 | 93.6 | 86 (6/7) | 100 (6/6) | 93 | 100 | 100 | 100 (2/2) | 92 | - | unmeasured |
| 12 | HSVTool | `HSVTool` | partial | 88.4 | 80.8 | 59 (14/25) | 100 (9/9) | 100 | 100 | - | 100 (11/11) | 100 | - | unmeasured |
| 13 | ColorMatrix | `ColorMatrix` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 14 | Colorspace | `Colorspace` | supported | 87.7 | 83.6 | 78 (7/9) | 100 (7/7) | 100 | 100 | 100 | 70 (14/20) | 100 | - | unmeasured |
| 14 | OCIOColorspace | `OCIOColorspace` | supported | 45.9 | 61.9 | 29 (2/11) | - | 100 | 100 | 100 | 0 unchecked | 50 | - | no fixture |
| 14 | OCIODisplay | `OCIODisplay` | supported | 46.3 | 37.6 | 23 (3/17) | 100 (1/1) | 33 | 100 | - | 0 unchecked | 83 | - | no fixture |
| 14 | OCIOFileTransform | `OCIOFileTransform` | supported | 54.7 | 77.3 | 60 (2/4) | - | 90 | 100 | 100 | 0 unchecked | 50 | - | no fixture |
| 14 | OCIOLogConvert | `OCIOLogConvert` | supported | 76.2 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 0 unchecked | - | - | no fixture |
| 14 | OCIOLookTransform | `OCIOLookTransform` | supported | 49.5 | 68.0 | 40 (4/14) | - | 100 | 100 | 100 | 0 unchecked | 50 | - | no fixture |
| 15 | Log2Lin | `Log2Lin` | supported | 100.0 | 100.0 | 100 (5/5) | 100 (5/5) | 100 | 100 | 100 | 100 (8/8) | 100 | - | unmeasured |
| 15 | PLogLin | `PLogLin` | supported | 97.1 | 96.7 | 100 (5/5) | 100 (5/5) | 100 | 67 | 100 | 100 (6/6) | 100 | - | unmeasured |
| 16 | ColorTransfer | - | supported | 0.0 | 0.0 | 0 (0/1) | - | - | 0 | - | - | - | - | no fixture |
| 16 | MatchGrade | `MatchGrade` | supported | 37.8 | 26.0 | 8 (3/33) | - | - | 100 | - | - | 67 | - | no fixture |
| 17 | Expression | `Expression` | supported | 100.0 | 100.0 | 100 (16/16) | - | 100 | 100 | 100 | 100 (15/15) | 100 | - | unmeasured |
| 18 | CrossTalk | `CrossTalk` | supported | 100.0 | 100.0 | 100 (1/1) | - | 100 | 100 | 100 | 100 (5/5) | - | - | unmeasured |
| 19 | Posterize | `Posterize` | supported | 98.6 | 98.5 | 100 (1/1) | 100 (1/1) | 89 | 100 | 100 | - | - | - | unmeasured |
| 19 | SoftClip | `SoftClip` | supported | 100.0 | 100.0 | 100 (3/3) | 100 (2/2) | 100 | 100 | - | 100 (5/5) | 100 | - | unmeasured |
| 19 | Toe | `Toe` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (1/1) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 20 | HistEQ | `HistEQ` | supported | 83.3 | 80.0 | 100 (1/1) | - | 100 | 100 | 0 | - | - | - | unmeasured |
| 20 | Histogram | `Histogram` | supported | 73.5 | 65.1 | 67 (2/3) | - | 100 | 100 | 14 | - | 100 | - | no fixture |
| 20 | MinColor | `MinColor` | supported | 100.0 | 100.0 | 100 (4/4) | - | 100 | 100 | 100 | - | 100 | - | unmeasured |
| 20 | Sampler | `Sampler` | supported | 35.0 | 30.8 | 25 (1/4) | - | - | 100 | 0 | - | - | - | no fixture |
| 21 | GenerateLUT | `GenerateLUT` | supported | 30.9 | 28.8 | 18 (1/9) | - | 0 | 100 | - | - | - | - | no fixture |
| 21 | Vectorfield | `Vectorfield` | supported | 44.2 | 42.3 | 38 (4/10) | - | 0 | 100 | - | - | 50 | - | no fixture |

### Filter

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Blur | `Blur` | supported | 90.2 | 99.0 | 100 (4/4) | 100 (3/3) | 90 | 100 | 100 | 100 (4/4) | 88 | 0 (0/1) | unmeasured |
| 2 | Sharpen | `Sharpen` | supported | 83.7 | 97.9 | 100 (7/7) | 83 (5/6) | 85 | 100 | 100 | - | 86 | 0 (0/1) | unmeasured |
| 3 | Erode (filter) | `ErodeFilter` | supported | 90.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | - | 100 | 0 (0/1) | unmeasured |
| 4 | Median | `Median` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 5 | Glow | `Glow` | supported | 89.9 | 98.8 | 100 (11/11) | 100 (10/10) | 88 | 100 | 100 | 100 (4/4) | 85 | 0 (0/1) | unmeasured |
| 6 | Soften | `Soften` | supported | 83.7 | 97.9 | 100 (7/7) | 83 (5/6) | 85 | 100 | 100 | - | 86 | 0 (0/1) | unmeasured |
| 7 | DropShadow | `DropShadow` | supported | 66.5 | 92.6 | 100 (9/9) | 89 (8/9) | 56 | 100 | - | 0 unchecked | 65 | - | no fixture |
| 8 | Defocus | `Defocus` | supported | 88.6 | 82.9 | 60 (3/5) | 100 (3/3) | 89 | 100 | 100 | 100 (5/5) | 100 | - | unmeasured |
| 8 | ZDefocus | `ZDefocus` | supported | 74.1 | 69.2 | 31 (9/36) | - | 67 | 100 | 100 | 100 (2/2) | 93 | - | unmeasured |
| 9 | DirBlur | `DirBlur` | supported | 60.3 | 94.7 | 100 (12/12) | 30 (3/10) | 60 | 100 | 100 | 0 unchecked | 74 | - | no fixture |
| 10 | EdgeBlur | `EdgeBlur` | supported | 53.6 | 55.2 | 22 (2/9) | 100 (2/2) | 75 | 100 | 100 | 0 unchecked | 75 | - | no fixture |
| 10 | EdgeExtend | `EdgeExtend` | supported | 82.6 | 75.0 | 50 (3/6) | 100 (3/3) | 100 | 50 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 11 | Erode (fast) | `Erode` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 100 (5/5) | - | - | unmeasured |
| 12 | DegrainSimple | `DegrainSimple` | supported | 88.1 | 95.6 | 100 (3/3) | - | 67 | 100 | 100 | - | 50 | - | no fixture |
| 12 | Denoise | `Denoise` | supported | 25.4 | 18.1 | 6 (2/37) | - | 50 | 33 | - | - | 50 | - | no fixture |
| 13 | MotionBlur | `MotionBlur` | supported | 25.0 | 26.4 | 33 (2/6) | - | 0 | 25 | - | - | 25 | - | no fixture |
| 13 | MotionBlur2D | `MotionBlur2D` | supported | 68.1 | 63.9 | 75 (3/4) | - | 33 | 50 | - | - | 100 | - | no fixture |
| 13 | MotionBlur3D | `MotionBlur3D` | supported | 63.9 | 55.6 | 50 (3/6) | - | 33 | 100 | - | - | 100 | - | no fixture |
| 13 | VectorBlur | `VectorBlur` | supported | 56.1 | 57.9 | 29 (4/15) | - | 67 | 100 | 100 | - | 38 | - | no fixture |
| 14 | Bilateral | `Bilateral` | supported | 64.2 | 61.9 | 40 (2/5) | - | 88 | 67 | 100 | - | 75 | - | no fixture |
| 15 | Convolve | `Convolve` | supported | 100.0 | 100.0 | 100 (4/4) | - | 100 | 100 | 100 | - | 100 | - | unmeasured |
| 15 | Matrix | `Matrix` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 16 | BumpBoss | `BumpBoss` | supported | 31.4 | 42.2 | 0 (0/6) | - | 100 | 67 | 100 | 0 unchecked | - | - | no fixture |
| 16 | EdgeDetect | `EdgeDetect` | supported | 80.8 | 80.6 | 64 (5/7) | 86 (4/5) | 100 | 100 | 100 | - | 65 | - | unmeasured |
| 16 | Emboss | `Emboss` | supported | 71.1 | 54.5 | 18 (2/13) | 100 (2/2) | 86 | 100 | 100 | - | 100 | - | unmeasured |
| 16 | Laplacian | `Laplacian` | supported | 91.3 | 100.0 | 100 (4/4) | 50 (2/4) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 17 | Inpaint | `Inpaint` | supported | 98.7 | 98.8 | 100 (11/11) | - | 91 | 100 | 100 | - | 100 | - | no fixture |
| 18 | GodRays | `GodRays` | supported | 45.5 | 37.1 | 30 (3/10) | 100 (3/3) | 33 | 100 | 17 | 0 unchecked | 83 | - | no fixture |
| 18 | LevelSet | `LevelSet` | supported | 100.0 | 100.0 | 100 (8/8) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 18 | VolumeRays | `VolumeRays` | supported | 36.7 | 23.2 | 4 (1/24) | 100 (1/1) | - | 100 | - | 0 unchecked | - | - | no fixture |
| 19 | ZSlice | `ZSlice` | supported | 73.8 | 76.7 | 100 (3/3) | - | 50 | 100 | 17 | - | 50 | - | no fixture |
| 20 | Bokeh | - | supported | 0.0 | 0.0 | 0 (0/42) | - | - | 0 | - | - | - | - | no fixture |

### Keyer

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Keyer | `Keyer` | supported | 95.8 | 97.4 | 100 (10/10) | 100 (9/9) | 78 | 100 | - | 100 (11/11) | 78 | - | unmeasured |
| 2 | ChromaKeyer | `ChromaKeyer` | supported | 99.2 | 98.9 | 100 (16/16) | 100 (13/13) | 93 | 100 | - | - | 100 | - | unmeasured |
| 3 | Difference | `Difference` | supported | 100.0 | 100.0 | 100 (4/4) | 100 (4/4) | 100 | 100 | - | 100 (2/2) | 100 | - | unmeasured |
| 4 | HueKeyer | `HueKeyer` | supported | 100.0 | 100.0 | 100 (4/4) | 100 (3/3) | 100 | 100 | 100 | 100 (5/5) | 100 | - | unmeasured |
| 5 | Keylight | `ScreenKeyer` | partial | 97.7 | 98.0 | 98 (34/35) | 100 (27/27) | 97 | 100 | - | - | 91 | - | unmeasured |
| 6 | Primatte | - | missing | 0.0 | 0.0 | 0 (0/26) | - | - | 0 | 0 | - | - | - | no fixture |
| 6 | Ultimatte | - | missing | 0.0 | 0.0 | 0 (0/60) | - | - | 0 | - | - | - | - | no fixture |
| 7 | IBKColor | `IBKColor` | supported | 84.4 | 79.2 | 100 (6/6) | 100 (4/4) | 25 | 50 | - | - | 100 | - | unmeasured |
| 7 | IBKGizmo | `IBKGizmo` | supported | 75.5 | 59.7 | 47 (7/15) | 100 (6/6) | 71 | 100 | - | - | 93 | - | unmeasured |
| 8 | Cryptomatte | `Cryptomatte` | partial | 43.8 | 40.0 | 25 (1/4) | - | - | 100 | - | - | - | - | no fixture |
| 8 | Encryptomatte | `Encryptomatte` | partial | 46.4 | 42.9 | 29 (1/4) | - | - | 100 | - | - | - | - | no fixture |

### Merge

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Merge | `Merge` | supported | 100.0 | 100.0 | 100 (9/9) | 100 (2/2) | 100 | 100 | 100 | 100 (29/29) | 100 | - | unmeasured |
| 2 | Premult | `Premult` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (1/1) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 3 | Unpremult | `Unpremult` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (1/1) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 4 | Switch | `Switch` | supported | 73.7 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | - | 0 unchecked | - | - | no fixture |
| 5 | Dissolve | `Dissolve` | supported | 76.2 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 0 unchecked | - | - | no fixture |
| 6 | KeyMix | `Keymix` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 7 | AddMix | `AddMix` | supported | 76.5 | 100.0 | 100 (6/6) | 100 (2/2) | 100 | 100 | 100 | 0 unchecked | 80 | - | no fixture |
| 8 | Blend | `Blend` | supported | 75.1 | 93.8 | 100 (2/2) | 100 (2/2) | 83 | 100 | 80 | 0 unchecked | 100 | - | no fixture |
| 9 | CopyBBox | `CopyBBox` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 9 | CopyRectangle | `CopyRectangle` | supported | 60.9 | 100.0 | 100 (5/5) | 0 (0/5) | 100 | 100 | 100 | 0 unchecked | 100 | - | no fixture |
| 10 | ContactSheet | `ContactSheet` | supported | 88.6 | 90.9 | 100 (11/11) | - | 45 | 100 | - | - | 86 | - | no fixture |
| 11 | TimeDissolve | `TimeDissolve` | supported | 55.6 | 100.0 | 100 (3/3) | 0 (0/3) | 100 | 100 | - | 0 unchecked | 83 | - | no fixture |
| 12 | ZMerge | `ZMerge` | supported | 97.2 | 100.0 | 100 (3/3) | - | 100 | 100 | - | - | 83 | - | no fixture |

### Transform

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Transform | `Transform` | supported | 100.0 | 100.0 | 100 (15/15) | 100 (10/10) | 100 | 100 | - | 100 (12/12) | 100 | 100 (1/1) | unmeasured |
| 2 | Crop | `Crop` | supported | 97.6 | 97.1 | 100 (5/5) | 100 (2/2) | 75 | 100 | - | 100 (3/3) | 100 | - | unmeasured |
| 3 | Tracker | `Tracker` | supported | 61.1 | 85.3 | 78 (41/54) | - | 79 | 100 | 100 | 0 unchecked | 69 | - | no fixture |
| 4 | PlanarTracker | `PlanarTracker` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 5 | Reformat | `Reformat` | supported | 96.4 | 95.2 | 94 (15/16) | - | 85 | 100 | - | 100 (15/15) | 100 | 100 (1/1) | unmeasured |
| 6 | CornerPin2D | `CornerPin` | supported | 99.4 | 100.0 | 100 (24/24) | - | 100 | 100 | - | 100 (7/7) | 95 | - | unmeasured |
| 7 | Mirror | `Mirror` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | - | 100 (4/4) | 100 | - | unmeasured |
| 8 | Stabilize | `Stabilize` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 9 | Position | `Position` | supported | 100.0 | 100.0 | 100 (1/1) | - | 100 | 100 | - | 100 (3/3) | - | - | unmeasured |
| 10 | AdjustBBox | `AdjustBBox` | supported | 100.0 | 100.0 | 100 (1/1) | - | 100 | 100 | - | 100 (3/3) | - | - | unmeasured |
| 11 | BlackOutside | `BlackOutside` | supported | 100.0 | 100.0 | - | - | - | 100 | - | 100 (1/1) | - | - | unmeasured |
| 12 | STMap | `STMap` | supported | 95.6 | 98.1 | 100 (6/6) | - | 86 | 100 | 100 | - | 83 | - | no fixture |
| 13 | IDistort | `IDistort` | supported | 92.1 | 94.3 | 100 (8/8) | - | 57 | 100 | 100 | - | 88 | - | no fixture |
| 14 | GridWarp | `GridWarp` | supported | 26.3 | 13.1 | 6 (2/34) | - | 0 | 50 | 17 | - | 100 | - | no fixture |
| 14 | SplineWarp | `SplineWarp` | supported | 35.2 | 28.6 | 18 (4/23) | - | 0 | 67 | 50 | - | 75 | - | no fixture |
| 15 | Tile | `Tile` | supported | 100.0 | 100.0 | 100 (5/5) | - | 100 | 100 | - | 100 (1/1) | 100 | - | unmeasured |
| 16 | VectorCornerPin | `VectorCornerPin` | supported | 25.0 | 20.0 | 0 (0/17) | - | - | 100 | - | - | - | - | no fixture |
| 16 | VectorDistort | `VectorDistort` | supported | 57.1 | 44.4 | 33 (2/6) | - | 100 | 100 | 0 | - | 100 | - | no fixture |
| 17 | PointsTo3D | `PointsTo3D` | supported | 46.7 | 48.3 | 60 (6/9) | - | 0 | 50 | - | - | 50 | - | no fixture |
| 18 | Reconcile3D | `Reconcile3D` | supported | 41.3 | 36.7 | 13 (1/14) | - | 100 | 67 | - | - | - | - | no fixture |
| 19 | TVIScale | `TVIScale` | supported | 100.0 | 100.0 | 100 (3/3) | - | 100 | 100 | 100 | - | 100 | - | no fixture |
| 20 | LensDistortion | `Undistort` | supported | 21.0 | 17.8 | 2 (1/58) | - | 0 | 100 | - | - | - | - | no fixture |

### Metadata

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ViewMetaData | `ViewMetaData` | supported | 100.0 | 100.0 | 100 (3/3) | - | - | 100 | - | - | - | - | no fixture |
| 2 | ModifyMetaData | `ModifyMetaData` | supported | 100.0 | 100.0 | 100 (1/1) | - | - | 100 | - | - | - | - | no fixture |
| 3 | CopyMetaData | `CopyMetaData` | supported | 100.0 | 100.0 | 100 (5/5) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 4 | AddTimeCode | `AddTimeCode` | supported | 91.1 | 94.4 | 100 (5/5) | - | 67 | 100 | - | - | 80 | - | no fixture |
| 5 | CompareMetaData | `CompareMetaData` | supported | 100.0 | 100.0 | 100 (1/1) | - | - | 100 | - | - | - | - | no fixture |

### Other

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Dot | `Dot` | supported | 100.0 | 100.0 | - | - | - | 100 | - | - | - | - | no fixture |
| 2 | NoOp | `NoOp` | supported | 100.0 | 100.0 | - | - | - | 100 | - | - | - | - | no fixture |
| 3 | Backdrop | `Backdrop` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 4 | PostageStamp | `PostageStamp` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 5 | Group | `Group` | supported | 100.0 | 100.0 | 100 (1/1) | - | 100 | 100 | - | - | - | - | no fixture |
| 5 | Input | `Input` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 5 | Output | `Output` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 6 | BlinkScript | - | missing | 0.0 | 0.0 | 0 (0/19) | - | - | 0 | - | - | - | - | no fixture |
| 7 | BurnIn | `BurnIn` | supported | 84.6 | 88.3 | 100 (18/18) | - | 30 | 100 | - | - | 78 | - | no fixture |
| 8 | Link | - | missing | 0.0 | 0.0 | - | - | - | 0 | - | - | - | - | no fixture |
| 8 | LiveGroup | - | missing | 0.0 | 0.0 | 0 (0/11) | - | - | 0 | - | - | - | - | no fixture |
| 8 | LiveInput | - | missing | 0.0 | 0.0 | 0 (0/4) | - | - | - | - | - | - | - | no fixture |
| 8 | VariableGroup | - | missing | 0.0 | 0.0 | 0 (0/1) | - | - | 0 | - | - | - | - | no fixture |
| 10 | Precomp | `Precomp` | supported | 97.1 | 100.0 | 100 (9/9) | - | 100 | - | - | - | 86 | - | no fixture |
| 11 | Root | `time and settings` | partial | 30.0 | 28.0 | 20 (7/43) | - | 60 | - | - | - | - | - | no fixture |
| 13 | Assert | `Assert` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 14 | AudioRead | - | missing | 0.0 | 0.0 | 0 (0/10) | - | - | - | - | - | - | - | no fixture |

## The 30 largest gaps

Ranked by `rank weight * (100 - fidelity)`, the share of the overall shortfall each node accounts for. This is the backlog for the following steps.

| # | Nuke node | NodeBased | Weighted gap | Fidelity | What is missing |
|---|---|---|---|---|---|
| 1 | Roto (Draw #1) | `Roto` | 39.13 | 60.9 | 27 secondary knobs missing; behaviour unchecked (no pixel case); knob order differs from Nuke's (8 of 13 in order) |
| 2 | Read (Image #1) | `Read` | 18.26 | 81.7 | knob order differs from Nuke's (20 of 27 in order); 3 labels differ from Nuke's |
| 3 | BlinkScript (Other #6) | - | 16.67 | 0.0 | no node: needs a Blink-equivalent runtime (PARITY_2D.md Other #6) |
| 4 | Write (Image #2) | `Write` | 16.54 | 66.9 | 29 secondary knobs missing; knob order differs from Nuke's (19 of 26 in order); 3 labels differ from Nuke's |
| 5 | Tracker (Transform #3) | `Tracker` | 12.96 | 61.1 | 13 secondary knobs missing; behaviour unchecked (no pixel case); knob order differs from Nuke's (24 of 35 in order) |
| 6 | Blur (Filter #1) | `Blur` | 9.80 | 90.2 | 1 label differs from Nuke's; not splittable like Nuke: `size` |
| 7 | Primatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 8 | Ultimatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 9 | Sharpen (Filter #2) | `Sharpen` | 8.15 | 83.7 | knobs that change no pixels: `quailty`; knob order differs from Nuke's (5 of 7 in order); not splittable like Nuke: `size` |
| 10 | ShuffleCopy (Channel #2) | `ShuffleCopy` | 7.50 | 85.0 | knob order differs from Nuke's (2 of 4 in order); 2 labels differ from Nuke's |
| 11 | Constant (Image #4) | `Constant` | 7.14 | 71.4 | behaviour unchecked (no pixel case); knob order differs from Nuke's (3 of 4 in order) |
| 12 | AudioRead (Other #14) | - | 7.14 | 0.0 | no node: no audio pipeline (PARITY_2D.md Other #14) |
| 13 | Switch (Merge #4) | `Switch` | 6.58 | 73.7 | behaviour unchecked (no pixel case) |
| 14 | Root (Other #11) | `time and settings` | 6.36 | 30.0 | primary knobs missing: `name`, `project_directory`, `Script_directory`, `label`, `lock_connections`, `lock_range`, `format`, `proxy_type` and 4 more; 24 secondary knobs missing |
| 15 | DustBust (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 16 | MarkerRemoval (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 17 | Bokeh (Filter #20) | - | 5.00 | 0.0 | provided as ZDefocus bokeh kernels, not a node (PARITY_2D.md Filter #20) |
| 18 | DropShadow (Filter #7) | `DropShadow` | 4.78 | 66.5 | behaviour unchecked (no pixel case); knobs that change no pixels: `operation`; knob order differs from Nuke's (5 of 9 in order); 2 labels differ from Nuke's |
| 19 | Dissolve (Merge #5) | `Dissolve` | 4.76 | 76.2 | behaviour unchecked (no pixel case) |
| 20 | RotoPaint (Draw #7) | `RotoPaint` | 4.75 | 66.8 | 15 secondary knobs missing; behaviour unchecked (no pixel case); knob order differs from Nuke's (8 of 12 in order) |
| 21 | ScannedGrain (Draw #9) | `ScannedGrain` | 4.70 | 15.4 | primary knobs missing: `fullGrain`; behaviour unchecked (no pixel case) |
| 22 | DirBlur (Filter #9) | `DirBlur` | 4.41 | 60.3 | behaviour unchecked (no pixel case); knobs that change no pixels: `holdout`, `BlurCenter`, `UseTarget`, `Samples`, `PixelAspect`, `Quality` and 1 more; knob order differs from Nuke's (9 of 12 in order); 3 labels differ from Nuke's |
| 23 | LightWrap (Draw #8) | `LightWrap` | 4.24 | 66.1 | behaviour unchecked (no pixel case); knobs that change no pixels: `disableluma`, `enableglow`, `saturation`, `LumaTolerance`, `white`, `gray` and 13 more; knob order differs from Nuke's (21 of 27 in order) |
| 24 | Keyer (Keyer #1) | `Keyer` | 4.23 | 95.8 | 4 labels differ from Nuke's |
| 25 | TimeDissolve (Merge #11) | `TimeDissolve` | 4.04 | 55.6 | behaviour unchecked (no pixel case); knobs that change no pixels: `in`, `out`, `lut`; 1 label differs from Nuke's |
| 26 | LensDistortion (Transform #20) | `Undistort` | 3.95 | 21.0 | primary knobs missing: `distortionModelPreset`, `lens`; 55 secondary knobs missing |
| 27 | Cryptomatte (Keyer #8) | `Cryptomatte` | 3.52 | 43.8 | primary knobs missing: `manifestSource`, `sidecarFilepath`, `previewEnabled` |
| 28 | AddMix (Merge #7) | `AddMix` | 3.35 | 76.5 | behaviour unchecked (no pixel case); 2 labels differ from Nuke's |
| 29 | Encryptomatte (Keyer #8) | `Encryptomatte` | 3.35 | 46.4 | primary knobs missing: `matteNameType`, `mergeOperation`; 1 secondary knob missing: `exportWrite` |
| 30 | Erode (filter) (Filter #3) | `ErodeFilter` | 3.33 | 90.0 | not splittable like Nuke: `size` |

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
| `clamp_to_values` | guide | pass | within tolerance | - |
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
| `cc_master_chain` | convention | pass | within tolerance | - |
| `cc_gamma_true_power` | convention | pass | within tolerance | - |
| `cc_saturation_zero` | convention | pass | within tolerance | - |
| `cc_saturation_two` | convention | pass | within tolerance | - |
| `cc_master_contrast_pivot` | convention | pass | within tolerance | - |
| `cc_shadows_contrast_pivot` | convention | pass | within tolerance | - |
| `cc_midtones_contrast_pivot` | convention | pass | within tolerance | - |
| `cc_highlights_contrast_pivot` | convention | pass | within tolerance | - |
| `cc_shadows_offset` | convention | pass | within tolerance | - |
| `cc_midtones_offset` | convention | pass | within tolerance | - |
| `cc_highlights_offset` | convention | pass | within tolerance | - |
| `cc_shadows_saturation_gamma_gain` | convention | pass | within tolerance | - |
| `cc_midtones_saturation_gamma_gain` | convention | pass | within tolerance | - |
| `cc_highlights_saturation_gamma_gain` | convention | pass | within tolerance | - |
| `cc_ranges_sum_to_one` | guide | pass | within tolerance | - |
| `cc_master_woven_into_ranges` | convention | pass | within tolerance | - |
| `cc_everything` | convention | pass | within tolerance | - |
| `cc_stored_colour_not_straight` | guide | pass | within tolerance | - |
| `cc_mix_half` | guide | pass | within tolerance | - |
| `cc_mask` | guide | pass | within tolerance | - |
| `saturation_rec709_zero` | convention | pass | within tolerance | - |
| `saturation_rec709_half` | convention | pass | within tolerance | - |
| `saturation_ccir601_zero` | convention | pass | within tolerance | - |
| `saturation_ccir601_half` | convention | pass | within tolerance | - |
| `saturation_average_zero` | convention | pass | within tolerance | - |
| `saturation_average_half` | convention | pass | within tolerance | - |
| `saturation_maximum_zero` | convention | pass | within tolerance | - |
| `saturation_maximum_half` | convention | pass | within tolerance | - |
| `saturation_rec709_boost` | convention | pass | within tolerance | - |
| `saturation_rec709_negative` | convention | pass | within tolerance | - |
| `saturation_default_is_identity` | guide | pass | within tolerance | - |
| `saturation_mix_half` | guide | pass | within tolerance | - |
| `saturation_mask` | guide | pass | within tolerance | - |
| `huecorrect_defaults` | guide | pass | within tolerance | - |
| `huecorrect_sat_0` | convention | pass | within tolerance | - |
| `huecorrect_sat_0.5` | convention | pass | within tolerance | - |
| `huecorrect_sat_2` | convention | pass | within tolerance | - |
| `huecorrect_lum_2` | convention | pass | within tolerance | - |
| `huecorrect_red_gain` | convention | pass | within tolerance | - |
| `huecorrect_green_gain` | convention | pass | within tolerance | - |
| `huecorrect_blue_gain` | convention | pass | within tolerance | - |
| `huecorrect_r_sup_0` | convention | pass | within tolerance | - |
| `huecorrect_g_sup_0` | convention | pass | within tolerance | - |
| `huecorrect_b_sup_0.5` | convention | pass | within tolerance | - |
| `huecorrect_g_sup_0.25` | convention | pass | within tolerance | - |
| `huecorrect_red_gain_around_red` | convention | pass | within tolerance | - |
| `huecorrect_g_sup_around_green` | convention | pass | within tolerance | - |
| `huecorrect_mix_half_masked` | guide | pass | within tolerance | - |
| `huecorrect_sat_thrsh` | convention | not scored: the only published measurements are two points (threshold 0.5: a red gain of 2 became 1.4 at saturation 0.5 and 1.64 at saturation 0.83), which do not fix the fade's shape | - | - |
| `hsv_defaults_leave_colour` | convention | pass | within tolerance | - |
| `hsv_hue_rotation_40` | convention | pass | within tolerance | - |
| `hsv_hue_rotation_in_range` | convention | pass | within tolerance | - |
| `hsv_hue_rotation_negative` | convention | pass | within tolerance | - |
| `hsv_saturation_adjust_down` | convention | pass | within tolerance | - |
| `hsv_saturation_adjust_up_in_range` | convention | pass | within tolerance | - |
| `hsv_saturation_force` | convention | pass | within tolerance | - |
| `hsv_brightness_adjust_down` | convention | pass | within tolerance | - |
| `hsv_brightness_force_in_range` | convention | pass | within tolerance | - |
| `hsv_all_three_ranges` | convention | pass | within tolerance | - |
| `hsv_hue_rolloff_reaches_past_the_range` | convention | pass | within tolerance | - |
| `hsv_rolloff_shape` | convention | not scored: neither the guide nor the sources say whether the fade is linear, smooth or something else, so a pixel inside the rolloff band has no stated value | - | - |
| `hsv_default_output_matte` | convention | not scored: the 17.0 page lists the output checkbox as enabled with rgba.alpha and 'is' hue, which reads as a default HSVTool writing the hue range matte into alpha (Natron's port does); NodeBased keeps the source alpha. Whether Nuke's default really overwrites alpha cannot be confirmed without a render | - | - |
| `hsv_color_replacement` | convention | not scored: in Nuke the two pickers are a helper that sets the ranges and adjustments (Natron: 'Changing this parameter sets the hue, saturation and brightness ranges for this color'); the ranges and rolloffs it chooses are not published, so no per-pixel value can be written down | - | - |
| `hsv_masks` | convention | not scored: the masks are channel knobs of the node in Nuke and extra input sockets in NodeBased, and how a mask combines with the range weights is not published | - | - |
| `hsv_hdr_values` | convention | not scored: the sources describe HSV for 0 to 1 colour; what Nuke does to hue, saturation and value outside it is not published | - | - |
| `toe_black_to_lift_white_untouched` | guide | pass | within tolerance | - |
| `toe_default_lift_is_none` | guide | pass | within tolerance | - |
| `toe_rolloff_shape` | guide | not scored: the guide says only that the lift fades toward the whites; the shape of the rolloff, where it ends, and what a negative value does are not published (NodeBased fades it by (1 - x / 0.2) squared up to 0.2, a width its own `toe` knob sets) | - | - |
| `softclip_none` | guide | pass | within tolerance | - |
| `softclip_log_endpoints` | guide | pass | within tolerance | - |
| `softclip_log_from_zero` | guide | pass | within tolerance | - |
| `softclip_preserve_hue_and_saturation` | convention | pass | within tolerance | - |
| `softclip_preserve_hue_and_brightness` | convention | pass | within tolerance | - |
| `softclip_log_curve` | guide | not scored: the page says 'a logarithmic curve between softclip min and 1.0' and warns of artefacts when softclip min is not 0 or 1, but publishes no formula; only the end points are written down (the cases above) | - | - |
| `posterize_colors` | guide | not scored: the page reads as a bit depth (8 gives 256 colors per channel, so the default 16 would give 65536 and show nothing), while the node is used with the knob as a level count and its default of 16 as visible banding; no formula is published and the two readings disagree (NodeBased uses round(v * (n - 1)) / (n - 1) clamped to 0 to 1, n levels per channel) | - | - |
| `log2lin_default_to_linear` | convention | pass | within tolerance | - |
| `log2lin_black_is_zero_white_is_one` | guide | pass | within tolerance | - |
| `log2lin_to_log` | convention | pass | within tolerance | - |
| `log2lin_ignore_black` | convention | pass | within tolerance | - |
| `log2lin_ignore_black_to_log` | convention | pass | within tolerance | - |
| `log2lin_custom_points` | convention | pass | within tolerance | - |
| `log2lin_custom_points_to_log` | convention | pass | within tolerance | - |
| `log2lin_alpha_untouched` | guide | pass | within tolerance | - |
| `log2lin_soft_clip` | convention | not scored: Kodak's Cineon specification describes an optional soft clip below the black point, but the Log2Lin page has no soft clip knob and states no formula for one, and NodeBased's node has none either, so the region below black is left out rather than guessed | - | - |
| `ploglin_defaults` | convention | pass | within tolerance | - |
| `ploglin_reference_point` | guide | pass | within tolerance | - |
| `ploglin_to_log` | convention | pass | within tolerance | - |
| `ploglin_custom_reference` | convention | pass | within tolerance | - |
| `ploglin_negative_gamma_and_density` | convention | pass | within tolerance | - |
| `ploglin_custom_to_log` | convention | pass | within tolerance | - |
| `colorspace_srgb_to_linear` | convention | pass | within tolerance | - |
| `colorspace_linear_to_srgb` | convention | pass | within tolerance | - |
| `colorspace_gamma22` | convention | pass | within tolerance | - |
| `colorspace_gamma24_out` | convention | pass | within tolerance | - |
| `colorspace_srgb_to_gamma22` | convention | pass | within tolerance | - |
| `colorspace_primaries_709_to_2020` | convention | pass | within tolerance | - |
| `colorspace_primaries_2020_to_709` | convention | pass | within tolerance | - |
| `colorspace_primaries_709_to_p3d65` | convention | pass | within tolerance | - |
| `colorspace_primaries_with_curves` | convention | pass | within tolerance | - |
| `colorspace_illuminant_keeps_xyz` | guide | pass | within tolerance | - |
| `colorspace_illuminant_keeps_xyz_dci` | guide | pass | within tolerance | - |
| `colorspace_bradford_adapts` | convention | pass | within tolerance | - |
| `colorspace_bradford_equal_illuminants` | guide | pass | within tolerance | - |
| `colorspace_does_not_divide_by_alpha` | guide | pass | within tolerance | - |
| `colorspace_hsv` | guide | inexpressible | - | - |
| `colorspace_ycbcr` | guide | inexpressible | - | - |
| `colorspace_cie` | guide | inexpressible | - | - |
| `colorspace_cineon` | guide | inexpressible | - | - |
| `colorspace_rec709_curve` | guide | inexpressible | - | - |
| `colorspace_other_illuminants` | guide | inexpressible | - | - |
| `colorlookup_default_is_identity` | guide | pass | within tolerance | - |
| `colorlookup_red_curve` | guide | pass | within tolerance | - |
| `colorlookup_green_blue_curves` | guide | pass | within tolerance | - |
| `colorlookup_alpha_curve` | guide | pass | within tolerance | - |
| `colorlookup_master_curve` | guide | pass | within tolerance | - |
| `colorlookup_range_does_not_scale_the_axis` | guide | pass | within tolerance | - |
| `colorlookup_without_precomputed_table` | guide | pass | within tolerance | - |
| `colorlookup_outside_the_curve` | guide | not scored: the page does not say whether a curve holds its end values or continues its slope past the keys (NodeBased extends the end segments linearly) | - | - |
| `colorlookup_smooth_keys` | guide | not scored: the page names the curve editor but not the spline Nuke fits through keys added there (smooth is its default); the cases above use linear segments, whose shape is the same in any editor, and NodeBased's smooth and Hermite segments cannot be compared without that spline | - | - |
| `colorlookup_master_with_channel` | guide | not scored: the page says the master curve represents all channels but not how it combines with a channel's own curve (NodeBased runs the master first, then the channel) | - | - |
| `colorlookup_precomputed_table_steps` | guide | not scored: the page says a table 'between 0 and the value specified in the range field' is used but not how many entries it holds, so its rounding cannot be reproduced (NodeBased evaluates the curves exactly, as Nuke does with the table off) | - | - |
| `expression_blank_is_zero` | guide | pass | within tolerance | - |
| `expression_arithmetic_precedence` | convention | pass | within tolerance | - |
| `expression_conditional` | convention | pass | within tolerance | - |
| `expression_equality` | guide | pass | within tolerance | - |
| `expression_layer_dot_channel` | guide | pass | within tolerance | - |
| `expression_functions_rounding` | guide | pass | within tolerance | - |
| `expression_functions_clamp_min_max_lerp` | guide | pass | within tolerance | - |
| `expression_functions_power` | guide | pass | within tolerance | - |
| `expression_functions_trig` | guide | pass | within tolerance | - |
| `expression_functions_step` | guide | pass | within tolerance | - |
| `expression_functions_rint` | guide | pass | within tolerance | - |
| `expression_temp_variables` | guide | pass | within tolerance | - |
| `expression_channel_row` | guide | pass | within tolerance | - |
| `expression_channel_set_rgb` | guide | pass | within tolerance | - |
| `expression_32_bit_float` | guide | pass | within tolerance | - |
| `expression_pixel_coordinates` | guide | not scored: the page says x and y are the pixel coordinate but not whether a pixel's x is its left edge or its centre (a half pixel in), nor which row counts as zero (Nuke counts up from the bottom row; NodeBased's new node counts up from the bottom and uses pixel centres, its older node counted down from the top) | - | - |
| `expression_centred_coordinates` | guide | not scored: the page defines cx and cy by their geometry (0,0 at the centre, -1 and 1 at the left and right edges, 1,1 on the diagonal of the right edge) but, as with x and y, not whether the value belongs to a pixel's centre or its edge, so the half-pixel shift in every value is left unchecked | - | - |
| `expression_noise_functions` | guide | not scored: the page describes Perlin noise and pseudo-random values only in words; Nuke's own table and seed are not published, so no value can be reproduced (NodeBased refuses these) | - | - |
| `expression_colour_conversion_functions` | guide | not scored: the page words 'color component' as a value the function takes and returns without saying whether that is 0 to 1 or 0 to 255 (to_byte returns an 8-bit value, from_byte's text repeats from_sRGB's), so the scale cannot be pinned down (NodeBased refuses these) | - | - |
| `expression_atan2_and_mantissa` | guide | not scored: the page names atan2's arguments (x, y) but not which one is the numerator, and gives mantissa and exponent as the frexp pair without saying whether logb matches it or the C logb | - | - |
| `expression_knob_references` | guide | not scored: the page says any control on any node can be named in an expression; NodeBased's Expression node has no node-knob references (its parameter expressions are a separate system) | - | - |
| `expression_percent_operator` | guide | not scored: the page lists no operators; NodeBased reads % as C's fmod, which the table's fmod describes | - | - |
| `crosstalk_default_is_identity` | convention | pass | within tolerance | - |
| `crosstalk_bleed_red_into_green` | guide | pass | within tolerance | - |
| `crosstalk_nine_lines_make_a_matrix` | guide | pass | within tolerance | - |
| `crosstalk_sum_of_three_lookups` | guide | pass | within tolerance | - |
| `crosstalk_remove_a_channel_from_itself` | guide | pass | within tolerance | - |
| `crosstalk_outside_zero_to_one` | guide | not scored: the page does not say how a crossover curve continues past its end keys (NodeBased extends the end segments linearly), so the HDR and negative values are left out | - | - |
| `crosstalk_smooth_curves` | guide | not scored: the page does not name the spline the curve editor fits through added keys, so only linear segments, whose shape is the same in any editor, are compared | - | - |
| `crosstalk_alpha` | guide | not scored: the page says the output colour channels are sums of lookups and says nothing about alpha (NodeBased leaves it alone) | - | - |
| `histeq_flat_histogram` | guide | not scored: the page states the aim, a flat histogram, and no algorithm: not which channels are equalised together (each on its own or the luminance), the range the histogram covers (the data's minimum to maximum, or 0 to 1, and what happens to values outside it), nor where inside a bin a value lands, so no pixel can be written down (tests/test_2d_behaviour_color_b.py checks the property instead: the output's histogram is flatter and the order of the values is kept) | - | - |
| `histeq_histogram_entries` | guide | not scored: the page names the knob and nothing else about it: how many entries are better than few, and how Nuke maps values into the entries, is not published | - | - |
| `mincolor_adds_the_delta` | guide | not scored: the delta that is added is calculated by the Find Pixel Delta button over a range of frames, not by rendering: the node adds the stored delta (tests/test_2d_behaviour_color_b.py runs the button's calculation and the addition against this statement) | - | - |
| `keyer_red_trapezoid` | convention | pass | within tolerance | - |
| `keyer_green_trapezoid` | convention | pass | within tolerance | - |
| `keyer_blue_trapezoid` | convention | pass | within tolerance | - |
| `keyer_max_trapezoid` | convention | pass | within tolerance | - |
| `keyer_min_trapezoid` | convention | pass | within tolerance | - |
| `keyer_triangle_range` | convention | pass | within tolerance | - |
| `keyer_hdr_range` | convention | pass | within tolerance | - |
| `keyer_alpha_input` | convention | pass | within tolerance | - |
| `keyer_invert` | convention | pass | within tolerance | - |
| `keyer_output_channel` | convention | pass | within tolerance | - |
| `keyer_output_none` | convention | pass | within tolerance | - |
| `keyer_luminance_operation` | guide | not scored: the page says the key is on 'luminance' and shows the graph running 0 to 1, but not the weights of the red, green and blue; Foundry's Rec 709 form (0.2125 / 0.7154 / 0.0721) and the exact Rec 709 form (0.2126 / 0.7152 / 0.0722) differ by up to 3e-4 of the value and the page names neither | - | - |
| `keyer_saturation_operation` | guide | not scored: the page names 'saturation' as a keyed property and gives no formula: HSV saturation (max - min) / max, HSL saturation and (max - min) all fit the words | - | - |
| `keyer_redscreen_operation` | guide | not scored: the page names no formula for the screen operations: 'red minus the larger of green and blue', 'red minus their mean' and 'red minus green' are all in use | - | - |
| `keyer_greenscreen_operation` | guide | not scored: same as redscreen: no formula for the green screen difference | - | - |
| `keyer_bluescreen_operation` | guide | not scored: same as redscreen: no formula for the blue screen difference | - | - |
| `keyer_combine` | guide | not scored: the page gives no formula for intersect and union: the smaller and the larger of the two mattes, their product and A + B - AB all read as 'intersect' and 'union' (tests/test_2d_behaviour_keyer.py checks that NodeBased's choice is the smaller and the larger) | - | - |
| `huekeyer_default_keys_nothing` | guide | pass | within tolerance | - |
| `huekeyer_amount_green_to_black` | guide | pass | within tolerance | - |
| `huekeyer_amount_curve_shape` | guide | pass | within tolerance | - |
| `huekeyer_invert_off_is_the_amount` | guide | pass | within tolerance | - |
| `huekeyer_output_channel` | convention | pass | within tolerance | - |
| `huekeyer_sat_thrsh` | guide | not scored: the page does not say how a pixel below the threshold is rejected (a hard cut, or a fade in over some range), nor whether saturation is the HSV or the HSL one (they differ widely for bright colours); tests/test_2d_behaviour_keyer.py checks that a higher threshold removes the keying from the lower-saturation hues | - | - |
| `huekeyer_curve_smoothing` | guide | not scored: the page does not name the spline the curve editor fits through added points, so only the straight segments, whose shape is the same in any editor, are compared | - | - |
| `huekeyer_channels` | guide | not scored: the page does not say what hue a pixel has when a channel is left out (as 0, or hue from the remaining two); NodeBased's channels knob limits the channels the node may write, like every node with a channels knob | - | - |
| `difference_identical_inputs` | guide | pass | within tolerance | - |
| `difference_offset_subtracts` | guide | pass | within tolerance | - |
| `difference_metric` | guide | not scored: the page does not say how the three colour channels become one matte value: the largest channel difference, their sum, their mean, the luminance of the difference and the length of the difference vector are all in use | - | - |
| `difference_gain` | guide | not scored: the sentence says what a white pixel becomes and not how the other values scale, nor whether the gain is applied before or after the offset is subtracted | - | - |
| `difference_clamp` | guide | not scored: the page does not say whether the matte is clamped to 0 to 1, so the values below 0 (after the offset) and above 1 are left alone | - | - |
| `difference_output_colour` | guide | not scored: the page says the node 'produces the difference as a matte' and not what the colour channels hold | - | - |
| `chromakeyer_screen_colour` | guide | not scored: the page says a pixel of the screen colour becomes transparent and gives no distance measure for the pixels near it: how far a colour is from the screen colour, and how that distance becomes an alpha, is not published (tests/test_2d_behaviour_keyer.py checks that the screen colour itself is transparent and a neutral grey stays opaque) | - | - |
| `chromakeyer_screen_gain` | guide | not scored: the page states the direction and no formula for the gain (tests/test_2d_behaviour_keyer.py checks that a higher gain never makes the matte more opaque) | - | - |
| `chromakeyer_screen_balance` | guide | not scored: the page gives the three values to try and no formula for what a balance weighs against what | - | - |
| `chromakeyer_chroma_gain` | guide | not scored: the page states the effect and no formula (tests/test_2d_behaviour_keyer.py checks that a higher chroma gain never makes the matte more transparent) | - | - |
| `chromakeyer_white_black_point` | guide | not scored: both sentences are relative to the matte before the points, which the page does not give a formula for, so no independent matte can be written down (tests/test_2d_behaviour_keyer.py checks them against the node's own matte before the points) | - | - |
| `chromakeyer_alpha_bias` | guide | not scored: the page describes the use and gives no formula for how a picked colour moves the matte | - | - |
| `chromakeyer_despill` | guide | not scored: the page gives no formula for the despill: which channel is limited, and to what mix of the others | - | - |
| `chromakeyer_replace_modes` | guide | not scored: the page words the four modes ('a corresponding amount of the replace color added for any increase in alpha', 'reintroduced/removed if the alpha is changed') without the amount's formula, and the matte they act on is itself not published (tests/test_2d_fidelity_h_keyers.py and tests/test_2d_behaviour_keyer.py check the stated relations) | - | - |
| `chromakeyer_premultiply` | guide | not scored: the colour that is multiplied is the despilled colour and its matte is not published, so the output colour cannot be written down independently (tests/test_2d_behaviour_keyer.py checks that the colour equals the unpremultiplied colour times the alpha) | - | - |
| `chromakeyer_inside_mask` | guide | not scored: the page says the results add up and does not say how (a sum, a union or a maximum) or how the sum is kept within 1 | - | - |
| `keylight_screen_colour_gain_balance` | guide | not scored: Keylight is a third-party plug-in; the Reference Guide lists its knobs and the introduction says only that the screen colour is picked with the eyedropper and 'many options you can adjust'. Its screen matte formula is not published, so no pixel can be written down | - | - |
| `keylight_clip_black_white` | guide | not scored: both sentences are relative to the screen matte, which has no published formula (tests/test_2d_behaviour_keyer.py checks them against the node's own screen matte) | - | - |
| `keylight_despill_bias` | guide | not scored: no formula is published for either bias | - | - |
| `keylight_replace_methods` | guide | not scored: the methods are named and not defined by any page of the guide | - | - |
| `ibkcolor_screen_type` | guide | not scored: neither page defines 'not green' (a ratio of the channels, a difference, a threshold) or what replaces it, so the processed screen cannot be written down (tests/test_2d_behaviour_keyer.py checks that a green screen plate keeps its green and that a foreground inside it is filled from the screen around it) | - | - |
| `ibkcolor_size` | guide | not scored: the page states the direction and the filter (a Gaussian) without its width in pixels for a given size | - | - |
| `ibkcolor_darks_lights_patch_black` | guide | not scored: the tutorial names what each control cures and gives no formula for any of them, nor the erode | - | - |
| `ibkgizmo_screen_type_difference` | guide | not scored: the matte formula is not published: the pages say 'essentially a difference matte' without the weights of red, blue and green in it. NodeBased has green and blue and no pick or C-red | - | - |
| `ibkgizmo_red_blue_green_weights` | guide | not scored: no page of the guide or the tutorial defines the weights beyond their names | - | - |
| `ibkgizmo_luminance_match_level` | guide | not scored: no page of the guide or the tutorial describes any of the four knobs | - | - |
| `ibkgizmo_screen_subtraction_despill` | guide | not scored: no page of the guide or the tutorial describes this knob | - | - |
| `ibkgizmo_unimplemented_knobs` | guide | not scored: eight of the gizmo's fifteen knobs (color, screen range, enable for the luminance level, autolevels, yellow, cyan, magenta, use bkg chroma) have no NodeBased param, and no page says what they do; they stay unpaired until the gizmo's own definition can be read | - | - |
| `copy_red_from_a_alpha` | guide | pass | within tolerance | - |
| `copy_alpha_from_a_red` | guide | pass | within tolerance | - |
| `copy_four_channels` | guide | pass | within tolerance | - |
| `copy_layer_copy_rgb` | guide | pass | within tolerance | - |
| `copy_layer_copy_alpha` | guide | pass | within tolerance | - |
| `copy_mix_dissolves` | guide | pass | within tolerance | - |
| `copy_mask` | guide | pass | within tolerance | - |
| `copy_mask_and_mix` | guide | pass | within tolerance | - |
| `copy_bounding_box` | guide | not scored: the cases run on the 16 x 12 fixture, where both pictures have the same box | - | - |
| `copy_metadata_and_range` | guide | not scored: they pick which input's metadata and frame range pass down the tree, which no pixel shows | - | - |
| `copy_unpremult_by` | guide | not scored: the page says the image is divided by the channel before the copy and multiplied after; tests/test_2d_behaviour_keyer.py checks the identity it implies on pictures where that channel is 1 | - | - |
| `channelmerge_absminus` | guide | pass | within tolerance | - |
| `channelmerge_b_if_not_a` | guide | pass | within tolerance | - |
| `channelmerge_from` | guide | pass | within tolerance | - |
| `channelmerge_max` | guide | pass | within tolerance | - |
| `channelmerge_min` | guide | pass | within tolerance | - |
| `channelmerge_minus` | guide | pass | within tolerance | - |
| `channelmerge_plus` | guide | pass | within tolerance | - |
| `channelmerge_union` | guide | pass | within tolerance | - |
| `channelmerge_xor` | guide | pass | within tolerance | - |
| `channelmerge_multiply` | guide | pass | within tolerance | - |
| `channelmerge_divide` | guide | pass | within tolerance | - |
| `channelmerge_other_channels_selected` | guide | pass | within tolerance | - |
| `channelmerge_other_channels_are_b` | convention | pass | within tolerance | - |
| `channelmerge_in` | guide | not scored: the page writes a lower-case a or b, which in Merge are the alphas of A and B; ChannelMerge's A and B are one channel each with no alpha of their own, and the page does not say whether the lower-case letter then means the channel's own value or the alpha of its input (NodeBased reads the channel's own value) | - | - |
| `channelmerge_out` | guide | not scored: the page writes a lower-case a or b, which in Merge are the alphas of A and B; ChannelMerge's A and B are one channel each with no alpha of their own, and the page does not say whether the lower-case letter then means the channel's own value or the alpha of its input (NodeBased reads the channel's own value) | - | - |
| `channelmerge_stencil` | guide | not scored: the page writes a lower-case a or b, which in Merge are the alphas of A and B; ChannelMerge's A and B are one channel each with no alpha of their own, and the page does not say whether the lower-case letter then means the channel's own value or the alpha of its input (NodeBased reads the channel's own value) | - | - |
| `shuffle_default_passes_through` | guide | pass | within tolerance | - |
| `shuffle_swap_red_and_green` | guide | pass | within tolerance | - |
| `shuffle_rotate_channels` | guide | pass | within tolerance | - |
| `shuffle_alpha_to_every_channel` | guide | pass | within tolerance | - |
| `shuffle_black_and_white` | guide | pass | within tolerance | - |
| `shuffle_from_the_second_input` | guide | pass | within tolerance | - |
| `shuffle_mix_of_both_and_constants` | guide | pass | within tolerance | - |
| `shuffle_second_output_layer` | guide | not scored: the second layer is a named layer and the 16 x 12 fixture holds only rgba (tests/test_2d_behaviour_keyer.py checks the layer and its routing) | - | - |
| `blur_size_zero_is_identity` | guide | pass | within tolerance | - |
| `blur_box_1` | convention | pass | within tolerance | - |
| `blur_box_2` | convention | pass | within tolerance | - |
| `blur_box_quality_not_limiting` | guide | pass | within tolerance | - |
| `blur_gaussian_weights` | convention | not scored: the guide does not state the gaussian's sigma for a given size (only that gaussian is the default and the smoothest), so its weights cannot be written down without guessing; the shape facts (separable, normalised, symmetric, size 0 the identity) are tested in tests/test_2d_behaviour_filter_a.py | - | - |
| `blur_triangle_weights` | convention | not scored: the guide names the triangle filter but not its weights or how size maps to its width; shape facts are tested instead | - | - |
| `blur_quadratic_weights` | convention | not scored: the guide names the quadratic filter but not its weights or how size maps to its width; shape facts are tested instead | - | - |
| `blur_quality_scaled` | guide | not scored: the guide does not give the scale factor, the down-scaling filter or the block alignment, so the pixels cannot be written down; tests check that the result is smooth, bounded and tile-exact | - | - |
| `erode_fast_positive_1_grows_light` | convention | pass | within tolerance | - |
| `erode_fast_negative_1_grows_dark` | convention | pass | within tolerance | - |
| `erode_fast_positive_2_grows_light` | convention | pass | within tolerance | - |
| `erode_fast_negative_2_grows_dark` | convention | pass | within tolerance | - |
| `erode_fast_zero_is_identity` | guide | pass | within tolerance | - |
| `erode_filter_exact_response` | convention | not scored: the guide states the direction (positive size lets darker areas expand, negative lets brighter ones) and that the filter trades speed for quality, but not the response curve or the footprint; the direction facts are tested in tests/test_2d_behaviour_filter_a.py | - | - |
| `median_size_3` | convention | pass | within tolerance | - |
| `median_size_5` | convention | pass | within tolerance | - |
| `median_ignore_top_line` | guide | pass | within tolerance | - |
| `sharpen_amount_response` | convention | not scored: the guide says only that Sharpen/Soften 'use Laplacian' (blur minus the original) 'to sharpen/soften specific channels' and gives amount, minimum and maximum no description, so the scale of amount, the sign convention and the range's role cannot be written down without guessing; tests/test_2d_behaviour_filter_a.py checks the behaviour that follows from the names (flat is unchanged, sharpen raises contrast, soften lowers it) | - | - |
| `soften_amount_response` | convention | not scored: the guide says only that Sharpen/Soften 'use Laplacian' (blur minus the original) 'to sharpen/soften specific channels' and gives amount, minimum and maximum no description, so the scale of amount, the sign convention and the range's role cannot be written down without guessing; tests/test_2d_behaviour_filter_a.py checks the behaviour that follows from the names (flat is unchanged, sharpen raises contrast, soften lowers it) | - | - |
| `glow_exact_response` | convention | not scored: the guide describes size only as 'values less than 15 increase the dark pixels' (garbled) and does not say how the blurred bright part is scaled, tinted or added, so the glow's pixels cannot be written down | - | - |
| `glow_tolerance_above_everything` | guide | pass | within tolerance | - |
| `glow_brightness_zero` | guide | pass | within tolerance | - |
| `glow_tint_black` | guide | pass | within tolerance | - |
| `glow_effect_only_without_glow` | guide | pass | within tolerance | - |
| `convolve_filter_weights` | guide | not scored: the shared fixtures carry no small filter image and the evaluator's single-kernel path has no filter input, so the case is run on a 3 x 3 filter in tests/test_2d_behaviour_filter_a.py | - | - |
| `matrix_edge_detect_example` | guide | pass | within tolerance | - |
| `matrix_normalize_box` | guide | pass | within tolerance | - |
| `matrix_asymmetric_orientation` | convention | not scored: the guide's description ('multiplying each pixel by the matching matrix value') does not say whether the weights are flipped, so only symmetric matrices are scored | - | - |
| `laplacian_box_1` | guide | pass | within tolerance | - |
| `laplacian_box_2` | guide | pass | within tolerance | - |
| `edgedetect_sobel_response` | convention | not scored: the guide lists the three detectors and says nothing of their kernels, normalisation, or how the channels combine; tests check that a flat image gives no edge, that a step gives an edge on both sides of it, and that the threshold, blur and erode knobs act | - | - |
| `edgedetect_prewitt_response` | convention | not scored: the guide lists the three detectors and says nothing of their kernels, normalisation, or how the channels combine; tests check that a flat image gives no edge, that a step gives an edge on both sides of it, and that the threshold, blur and erode knobs act | - | - |
| `edgedetect_simple_response` | convention | not scored: the guide lists the three detectors and says nothing of their kernels, normalisation, or how the channels combine; tests check that a flat image gives no edge, that a step gives an edge on both sides of it, and that the threshold, blur and erode knobs act | - | - |
| `emboss_traditional_response` | convention | not scored: the guide says Emboss 'offsets the original input using the Angle and Width controls' and gives no formula, shading origin or scale; tests check that the picture's relief is symmetric when the angle turns by 180 degrees and that angle and width act | - | - |
| `transform_identity_exact` | guide | pass | within tolerance | - |
| `transform_translate_right_up` | convention | pass | within tolerance | - |
| `transform_translate_negative` | convention | pass | within tolerance | - |
| `transform_black_outside_off_takes_edge_pixels` | guide | pass | within tolerance | - |
| `transform_black_outside_on_is_black` | guide | pass | within tolerance | - |
| `transform_scale_two_about_center` | guide | pass | within tolerance | - |
| `transform_rotate_ninety_counter_clockwise` | convention | pass | within tolerance | - |
| `transform_invert_undoes_translate` | guide | pass | within tolerance | - |
| `transform_filter_keys_half_pixel` | convention | pass | within tolerance | - |
| `transform_filter_rifman_half_pixel` | convention | pass | within tolerance | - |
| `transform_filter_mitchell_half_pixel` | convention | pass | within tolerance | - |
| `transform_filter_parzen_half_pixel` | convention | pass | within tolerance | - |
| `transform_filter_cubic_half_pixel` | convention | not scored: the guide names Cubic without its coefficients; tests/test_2d_behaviour_transform.py checks that cubic smooths without overshoot at a step | - | - |
| `transform_filter_simon_half_pixel` | convention | not scored: the guide says Simon gives 'medium sharpening' between Keys (minor) and Rifman (significant) and publishes no coefficient; the order of the three is tested in tests/test_2d_behaviour_transform.py | - | - |
| `transform_filter_lanczos4_half_pixel` | convention | not scored: the guide says only that Lanczos4, Lanczos6 and Sinc4 sharpen (Lanczos4 least, Sinc4 most); whether the taps are renormalised to sum to 1 is not stated, so the weights are not scored; the order and constant-picture facts are tested in tests/test_2d_behaviour_transform.py | - | - |
| `transform_filter_lanczos6_half_pixel` | convention | not scored: the guide says only that Lanczos4, Lanczos6 and Sinc4 sharpen (Lanczos4 least, Sinc4 most); whether the taps are renormalised to sum to 1 is not stated, so the weights are not scored; the order and constant-picture facts are tested in tests/test_2d_behaviour_transform.py | - | - |
| `transform_filter_sinc4_half_pixel` | convention | not scored: the guide says only that Lanczos4, Lanczos6 and Sinc4 sharpen (Lanczos4 least, Sinc4 most); whether the taps are renormalised to sum to 1 is not stated, so the weights are not scored; the order and constant-picture facts are tested in tests/test_2d_behaviour_transform.py | - | - |
| `transform_filter_notch` | convention | not scored: the guide says only that Notch is flat smoothing and gives neither its width nor its curve, so no kernel can be written down without guessing; NodeBased does not offer it | - | - |
| `transform_skew_matrix` | convention | not scored: the guide says only 'Skews the image on the x axis' and gives neither the unit of the knob nor the sign; tests/test_2d_behaviour_transform.py checks that the two axes shear different ways and that skew order matters | - | - |
| `transform_matrix_order` | convention | not scored: the guide does not give the matrix order; NodeBased applies scale, then skew, then rotate, then translate about the centre (tests/test_2d_behaviour_transform.py pins what it does) | - | - |
| `transform_motion_blur_shutter` | guide | not scored: the case needs an animated transform over time and the harness runs one frame; tests/test_2d_behaviour_transform.py checks the shutter windows the guide gives (centred, start, end, custom) and that a still transform stays sharp | - | - |
| `reformat_distort_doubles` | guide | pass | within tolerance | - |
| `reformat_fit_letterboxes` | convention | pass | within tolerance | - |
| `reformat_fill_crops` | convention | pass | within tolerance | - |
| `reformat_width_scales_to_the_width` | guide | pass | within tolerance | - |
| `reformat_height_scales_to_the_height` | guide | pass | within tolerance | - |
| `reformat_none_does_not_resize` | guide | pass | within tolerance | - |
| `reformat_center_off_lines_up_the_lower_left` | guide | pass | within tolerance | - |
| `reformat_flip_is_upside_down` | guide | pass | within tolerance | - |
| `reformat_flop_is_left_right` | guide | pass | within tolerance | - |
| `reformat_turn_is_counter_clockwise` | guide | pass | within tolerance | - |
| `reformat_scale_type_halves` | guide | pass | within tolerance | - |
| `reformat_scale_split_scales_each_direction` | guide | pass | within tolerance | - |
| `reformat_force_this_shape_off_keeps_the_shape` | guide | pass | within tolerance | - |
| `reformat_black_outside_off_takes_edge_pixels` | guide | pass | within tolerance | - |
| `reformat_black_outside_on_is_black` | guide | pass | within tolerance | - |
| `reformat_pixel_aspect` | guide | not scored: the guide says only that it sets the output's pixel aspect ratio, not what happens to the pixels; NodeBased carries the value and works in square pixels | - | - |
| `cornerpin_identity` | convention | pass | within tolerance | - |
| `cornerpin_translation` | guide | pass | within tolerance | - |
| `cornerpin_scale_two` | guide | pass | within tolerance | - |
| `cornerpin_perspective` | guide | pass | within tolerance | - |
| `cornerpin_invert` | guide | pass | within tolerance | - |
| `cornerpin_black_outside_off_takes_edge_pixels` | guide | pass | within tolerance | - |
| `cornerpin_from_points_choose_the_part_that_is_pinned` | guide | pass | within tolerance | - |
| `crop_keeps_the_box_and_blacks_the_rest` | guide | pass | within tolerance | - |
| `crop_reformat_makes_the_box_the_format` | guide | pass | within tolerance | - |
| `crop_black_outside_off_takes_edge_pixels` | guide | pass | within tolerance | - |
| `crop_softness_vignette` | convention | not scored: the guide states that softness fades the edges toward black and that 0 is no vignette, but not the ramp's shape or how softness converts to pixels; tests/test_2d_behaviour_transform.py checks that 0 changes nothing and that a larger value darkens more | - | - |
| `mirror_horizontal_flops` | guide | pass | within tolerance | - |
| `mirror_vertical_flips` | guide | pass | within tolerance | - |
| `mirror_both_turns_the_picture_half_way_round` | guide | pass | within tolerance | - |
| `mirror_reflects_a_partial_picture_about_the_format_centre` | guide | pass | within tolerance | - |
| `position_moves_up_and_right_and_repeats_the_edge` | guide | pass | within tolerance | - |
| `position_negative_moves_down_and_left` | guide | pass | within tolerance | - |
| `position_moves_a_partial_picture_and_fills_the_gap` | guide | pass | within tolerance | - |
| `adjustbbox_grows_and_repeats_the_edge` | guide | pass | within tolerance | - |
| `adjustbbox_negative_crops_the_box` | guide | pass | within tolerance | - |
| `adjustbbox_width_and_height_split` | guide | pass | within tolerance | - |
| `blackoutside_fills_outside_the_box_with_black` | guide | pass | within tolerance | - |
| `tile_one_by_one_is_the_picture` | convention | pass | within tolerance | - |
| `tile_rows_and_columns_scale_the_picture_into_tiles` | convention | not scored: the page lists the filters and the knobs but states neither which source pixel an impulse-filtered tile takes (the scaled-down pixel's centre lies on the boundary between two source pixels) nor how the tile edges round off when the size does not divide; tests/test_2d_behaviour_transform.py checks the layout (every tile is the same scaled picture), the mirrors and the filters | - | - |
| `rectangle_fills_the_area_over_the_input` | convention | pass | within tolerance | - |
| `rectangle_invert_draws_everything_else` | guide | pass | within tolerance | - |
| `rectangle_opacity` | guide | pass | within tolerance | - |
| `rectangle_replace_clears_the_input` | guide | pass | within tolerance | - |
| `rectangle_ramp_linear` | convention | pass | within tolerance | - |
| `rectangle_ramp_smooth0_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `rectangle_ramp_smooth1_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `rectangle_ramp_smooth_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `rectangle_ramp_smooth_is_even_at_the_middle` | convention | pass | within tolerance | - |
| `rectangle_ramp_smooth0_curve` | convention | not scored: the page says only which end smooth0 eases into ('colors in the point 0 end are spread wider') and gives no curve; tests/test_2d_behaviour_draw.py checks the end it favours and that it keeps both end colours | - | - |
| `rectangle_ramp_smooth1_curve` | convention | not scored: the page says only which end smooth1 eases into ('colors in the point 1 end are spread wider') and gives no curve; tests/test_2d_behaviour_draw.py checks the end it favours and that it keeps both end colours | - | - |
| `rectangle_softness_falloff` | convention | not scored: the page says only that softness 'sets the softness of the rectangle edges where 0 is a hard edge' (default 1); it gives neither the unit nor the curve, so the soft edge is not scored; tests/test_2d_behaviour_draw.py checks that 0 is a hard edge, that more softness widens the fade and that the fade is symmetric | - | - |
| `radial_softness_zero_is_a_hard_ellipse` | guide | pass | within tolerance | - |
| `radial_invert` | guide | pass | within tolerance | - |
| `radial_opacity` | guide | pass | within tolerance | - |
| `radial_ramp_linear` | convention | pass | within tolerance | - |
| `radial_soft_falloff` | convention | not scored: the page says only that the area's centre is the originating point and that 0 is a hard edge; it gives no curve for the soft ramp, so it is not scored; tests/test_2d_behaviour_draw.py checks that it is full at the centre, falls to nothing at the edge and never rises | - | - |
| `radial_perceptually_linear` | convention | not scored: the page says only that the ramp 'appears more linear to the naked eye'; the transfer curve is not given, so the pixels are not scored; tests/test_2d_behaviour_draw.py checks that the switch changes the ramp, keeps its centre and edge and darkens the middle | - | - |
| `ramp_linear` | convention | pass | within tolerance | - |
| `ramp_smooth0_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `ramp_smooth1_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `ramp_smooth_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `ramp_plinear_ends_are_the_two_colours` | guide | pass | within tolerance | - |
| `ramp_smooth_is_even_at_the_middle` | convention | pass | within tolerance | - |
| `ramp_smooth0_curve` | convention | not scored: the page names the type and says in words which end it eases into (or, for plinear, that it looks more linear to the eye) but gives no curve; tests/test_2d_behaviour_draw.py checks the ends, the direction of the easing and that plinear differs from linear | - | - |
| `ramp_smooth1_curve` | convention | not scored: the page names the type and says in words which end it eases into (or, for plinear, that it looks more linear to the eye) but gives no curve; tests/test_2d_behaviour_draw.py checks the ends, the direction of the easing and that plinear differs from linear | - | - |
| `ramp_plinear_curve` | convention | not scored: the page names the type and says in words which end it eases into (or, for plinear, that it looks more linear to the eye) but gives no curve; tests/test_2d_behaviour_draw.py checks the ends, the direction of the easing and that plinear differs from linear | - | - |
| `noise_fbm_values` | convention | not scored: the page says the node is 'based on the Perlin noise function' and how octaves, lacunarity, gain and gamma combine it, but not the function's values (its gradient table and permutation are Foundry's), so the pixels cannot be written down; tests/test_2d_behaviour_draw.py checks what the page states: gain 0 leaves one octave, gamma is the output power, translate, rotate and scale move the pattern as the transform tab says, the x and y sizes stretch it, zoffset moves through it smoothly and the same settings give the same pixels | - | - |
| `noise_turbulence_values` | convention | not scored: the page says the node is 'based on the Perlin noise function' and how octaves, lacunarity, gain and gamma combine it, but not the function's values (its gradient table and permutation are Foundry's), so the pixels cannot be written down; tests/test_2d_behaviour_draw.py checks what the page states: gain 0 leaves one octave, gamma is the output power, translate, rotate and scale move the pattern as the transform tab says, the x and y sizes stretch it, zoffset moves through it smoothly and the same settings give the same pixels | - | - |
| `grid_number_makes_squares_with_empty_middles` | guide | pass | within tolerance | - |
| `grid_line_positions` | convention | not scored: the page gives the number of squares, the line thickness in pixels and a translate but not where in a square a line of a given thickness sits (starting at the grid position, centred on it or ending at it) or what the last line does at the frame's edge; tests/test_2d_behaviour_draw.py checks the spacing (the frame divided by the number), the thickness, that translate moves the whole grid and that lines are drawn in the colour and the ramp | - | - |
| `grain_minimum_rgb_is_the_floor` | convention | pass | within tolerance | - |
| `grain_apply_only_through_alpha_leaves_empty_alpha_alone` | guide | pass | within tolerance | - |
| `grain_invert_alpha_leaves_full_alpha_alone` | guide | pass | within tolerance | - |
| `grain_pattern` | convention | not scored: the page describes the controls in words (size shrinks or enlarges, irregularity is 'the random quality', intensity is the contrast of the effect against the image) and gives no grain function or amplitude; tests/test_2d_behaviour_draw.py checks what follows from the words: zero intensity adds nothing, the grain is zero-mean, a larger size gives a coarser grain, the pattern changes with the frame and holds when the seed is minus the frame, black sets the grain at black, and each channel is controlled by its own knobs | - | - |
| `dither_amount_zero_adds_nothing` | guide | pass | within tolerance | - |
| `dither_pattern` | convention | not scored: the page says the noise is random, up to 0.5 of the amount either way, and gives neither the generator nor its distribution; tests/test_2d_behaviour_draw.py checks the bound, that the mean is the input's, that monodither gives every channel the same offset, and that the seed, the frame and static seed change the pattern as the page says | - | - |
| `defocus_disc_average` | convention | pass | within tolerance | - |
| `defocus_aspect_ratio_widens_the_disc` | guide | pass | within tolerance | - |
| `defocus_scaling_scales_both_axes` | guide | pass | within tolerance | - |
| `defocus_scaling_split_into_width_and_height` | guide | pass | within tolerance | - |
| `defocus_channels_limit_the_effect` | guide | pass | within tolerance | - |
| `defocus_quality_and_method` | guide | not scored: NodeBased has no quality or method knob: its disc is always summed exactly, which is what fullprecision means; the page gives no interpolation to compare, so a quality below the default cannot be scored | - | - |
| `zdefocus_direct_is_clamped_to_maximum` | guide | pass | within tolerance | - |
| `zdefocus_direct_size_zero_is_no_blur` | guide | pass | within tolerance | - |
| `zdefocus_blur_by_distance` | convention | not scored: the page gives the depth-to-distance conversions and says the blur is 'the size of the blur at infinite depth' and grows nearer the camera, but not the formula between a distance, the focus plane, the depth of field and the size, nor the layer spacing (automatic layer spacing, depth layers, layer curve), the blend between layers, fill foreground or the filter shapes (disc, bladed, image, with shape, roundness, inner size and the rest); the properties it does state are tested in tests/test_2d_behaviour_draw.py: direct blur grows with depth to the maximum, -direct reverses the order, a larger maximum allows more blur, the focus plane stays sharp, and the filter choices change the pixels | - | - |
| `edgeextend_solid_matte_keeps_its_colour` | convention | pass | within tolerance | - |
| `edgeextend_premultiply_multiplies_by_alpha` | guide | pass | within tolerance | - |
| `edgeextend_edge_colour` | convention | not scored: the page says the node 'corrects unpremultiplied foreground color at the edge of soft mattes by eroding or dilating the sample region, pulling pixels from deeper inside or outside your matte' with an algorithm 'independent of the size of the fill regions', but gives neither that algorithm nor what detail amount recovers; the edge fill here is a ring-by-ring average, an approximation. tests/test_2d_behaviour_draw.py checks what the page states: positive erode takes colour from deeper inside the matte and negative erode from outside it, source is premultiplied unpremultiplies the source first, and premultiply multiplies the output | - | - |

## Requirements of 2026-11-02, per node

### Knobs that change no pixels

Each paired knob below was moved off its default to every value its `LIMITS` and `CHOICES` allow the probe to try, alone and with each other paired knob moved first, and NodeBased's output did not change on the fixture. Either the knob is not implemented, or it acts only in a state the probe does not reach (two other knobs moved together, a payload the default node does not hold). Implement it, or unpair it in `docs/parity/nodebased_to_nuke.json` with a reason.

- LightWrap (`LightWrap`): `disableluma`, `enableglow`, `saturation`, `LumaTolerance`, `white`, `gray`, `contrast`, `gamma`, `gain`, `offset`, `shadows_gamma`, `shadows_gain`, `shadows_offset`, `midtones_gamma`, `midtones_gain`, `midtones_offset`, `highlights_gamma`, `highlights_gain`, `highlights_offset`
- Flare (`Flare`): `chroma_shift`
- Dither (`Dither`): `static_seed`
- Sharpen (`Sharpen`): `quailty`
- Soften (`Soften`): `quailty`
- DropShadow (`DropShadow`): `operation`
- DirBlur (`DirBlur`): `holdout`, `BlurCenter`, `UseTarget`, `Samples`, `PixelAspect`, `Quality`, `pixeloffset`
- EdgeDetect (`EdgeDetect`): `blurquality`
- Laplacian (`Laplacian`): `quailty`, `crop`
- CopyRectangle (`CopyRectangle`): `area`, `area#2`, `area#3`, `area#4`, `softness`
- TimeDissolve (`TimeDissolve`): `in`, `out`, `lut`

Not probed, because a one-frame pixel probe cannot see them: acts across frames (frame range, retiming, motion blur): `after`, `before`, `first`, `frame`, `frame_mode`, `last`, `motionblur`, `origfirst`, `origlast`, `origset`, `shutter`, `shuttercustomoffset`, `shutteroffset`; acts on metadata, not pixels: `metainput`, `rangeinput`; acts on the bounding box only, outside the fixture's format: `intersect`, `pbb`; Blur's `crop` (crop to format acts on the bounding box only); ChromaKeyer's `_replaceAlphaDiff` (acts only with a replace mode other than none); ChromaKeyer's `_replaceColor` (acts only with a replace mode other than none); ChromaKeyer's `_replaceAmount` (acts only with a replace mode other than none); ColorLookup's `source` (feeds the Set RGB, Set RGBA and Set A buttons, which add curve points; the kernel reads the curves, not these two); ColorLookup's `target` (feeds the Set RGB, Set RGBA and Set A buttons, which add curve points; the kernel reads the curves, not these two); Crop's `reformat` (makes the box the format, which changes the window and not the fixture's pixels (tests/test_2d_behaviour_transform.py)); Crop's `crop` (black outside acts only where the box reaches past the picture, with intersect off; the probe's box does not (tests/test_2d_behaviour_transform.py)); Glow's `crop` (crop to format acts on the bounding box only); HSVTool's `srccolor` (acts only with Color Replacement on (source and destination colour)); HSVTool's `dstcolor` (acts only with Color Replacement on (source and destination colour)); HSVTool's `huerolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); HSVTool's `satrolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); HSVTool's `brtrolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); IBKColor's `Size` (acts only with a screen to fill into, which needs the erode and darks moved off their defaults); IBKColor's `mult` (acts only once the darks and the erode leave the screen estimate unresolved); IBKGizmo's `ublu` (acts only with a background wired and luminance matching on); KeyMix's `invertMask` (inverts the mask channel, so it acts only with a mask wired); Keylight's `despillBias` (acts only with the despill bias ganged to the alpha bias off); Keylight's `insideReplaceMethod` (acts only with an inside matte wired); Keylight's `insideReplaceColour` (acts only with an inside matte wired); Keylight's `insideComponent` (acts only with an inside matte wired); Keylight's `outsideComponent` (acts only with an outside matte wired); Keylight's `SourceXMethod` (acts only with the source cropped, which is another knob moved); Keylight's `SourceYMethod` (acts only with the source cropped, which is another knob moved); Merge's `screen_alpha` (acts only for operations that do not already composite by alpha (plus, minus, multiply and the like), not the default over); OCIOFileTransform's `invert` (runs a file backwards, so it acts only with a file named); OCIOLookTransform's `invert` (runs a look backwards, so it acts only with a look named); Sharpen's `crop` (crop to format acts on the bounding box only); Shuffle's `fromInput1` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); Shuffle's `fromInput2` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); Shuffle's `out2` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); SoftClip's `softclip_min` (acts only with a conversion chosen; the default conversion is none); Soften's `crop` (crop to format acts on the bounding box only); Toe's `invert` (inverts the mask channel, so it acts only with the mask channel on); Transform's `skew_order` (orders the two skews, so it acts only when skew X and skew Y are both set; the probe moves one at a time (tests/test_2d_behaviour_transform.py)). Whole nodes not probed: Roto (shapes live in the node's payload and the default node holds none); RotoPaint (strokes live in the node's payload and the default node holds none); Tracker (tracks are solved across frames; a one-frame probe cannot move them). Taps, whose behaviour is their input and which have neither `acts` nor `math`: `Viewer`, `Write`, `NoOp`, `Dot`, `PostageStamp`, `Output`, `Profile`, `PlanarTracker`.

### Layout differences

Labels below are NodeBased's, then the guide's.

- Read (`Read`): 20 of 27 paired knobs in Nuke's order; `file` reads "Path", Nuke "File"; `proxy` reads "Proxy file", Nuke "Proxy"; `reload` reads "Reload count", Nuke "reload"
- Write (`Write`): 19 of 26 paired knobs in Nuke's order; `file` reads "Path", Nuke "file"; `proxy` reads "Proxy file", Nuke "proxy"; `reload` reads "Reload count", Nuke "Reload"
- Constant (`Constant`): 3 of 4 paired knobs in Nuke's order
- CurveTool (`CurveTool`): 8 of 11 paired knobs in Nuke's order
- Roto (`Roto`): 8 of 13 paired knobs in Nuke's order
- Text (`Text`): 27 of 40 paired knobs in Nuke's order; `premult` reads "Premult by", Nuke "premult"
- RotoPaint (`RotoPaint`): 8 of 12 paired knobs in Nuke's order
- LightWrap (`LightWrap`): 21 of 27 paired knobs in Nuke's order
- Flare (`Flare`): 2 of 3 paired knobs in Nuke's order
- Glint (`Glint`): 3 of 4 paired knobs in Nuke's order; `repeat` reads "Rays", Nuke "no. of rays"
- AppendClip (`AppendClip`): 3 of 5 paired knobs in Nuke's order
- TimeBlur (`TimeBlur`): 3 of 4 paired knobs in Nuke's order
- TimeEcho (`TimeEcho`): 1 of 2 paired knobs in Nuke's order; `mode` reads "Method", Nuke "TimeEcho Method"; `framesbehind` reads "Frames", Nuke "Frames to look at"
- Kronos (`Kronos`): 3 of 4 paired knobs in Nuke's order; `timingOutputSpeed` reads "Speed", Nuke "Output Speed"
- OFlow (`OFlow`): `timingOutputSpeed` reads "Speed", Nuke "Output Speed"; `timingFrame2` reads "Speed", Nuke "Frame"
- ShuffleCopy (`ShuffleCopy`): 2 of 4 paired knobs in Nuke's order; `in` reads "In1 layer", Nuke "1 in"; `in2` reads "In2 layer", Nuke "2 in"
- Clamp (`Clamp`): 7 of 8 paired knobs in Nuke's order
- ColorLookup (`ColorLookup`): 4 of 6 paired knobs in Nuke's order; `setA` reads "Alpha", Nuke "Set A"
- Exposure (`Exposure`): 5 of 6 paired knobs in Nuke's order
- OCIOColorspace (`OCIOColorspace`): `in_colorspace` reads "Src", Nuke "input"; `out_colorspace` reads "Dst", Nuke "output"
- OCIODisplay (`OCIODisplay`): `invert` reads "Transform Direction", Nuke "invert direction"
- OCIOLookTransform (`OCIOLookTransform`): 3 of 4 paired knobs in Nuke's order; `in_colorspace` reads "Src", Nuke "input"; `invert` reads "Transform Direction", Nuke "invert direction"; `out_colorspace` reads "Dst", Nuke "output"
- OCIOFileTransform (`OCIOFileTransform`): `file` reads "Path", Nuke "file"
- MatchGrade (`MatchGrade`): 2 of 3 paired knobs in Nuke's order
- Vectorfield (`Vectorfield`): `vfield_file` reads "3D LUT (.cube, .3dl)", Nuke "vectorfield file"
- Blur (`Blur`): `size` reads "Radius", Nuke "size"
- Sharpen (`Sharpen`): 5 of 7 paired knobs in Nuke's order
- Glow (`Glow`): 9 of 11 paired knobs in Nuke's order; `tolerance` reads "Threshold", Nuke "tolerance"
- Soften (`Soften`): 5 of 7 paired knobs in Nuke's order
- DropShadow (`DropShadow`): 5 of 9 paired knobs in Nuke's order; `softness` reads "Size", Nuke "softness"; `operation` reads "Operation (over bg)", Nuke "input"
- ZDefocus (`ZDefocus`): `center` reads "Focus plane", Nuke "focus plane (C)"
- DirBlur (`DirBlur`): 9 of 12 paired knobs in Nuke's order; `BlurType` reads "Type", Nuke "BlurType"; `BlurLength` reads "Length", Nuke "BlurLength"; `BlurAngle` reads "Angle", Nuke "BlurAngle"
- EdgeBlur (`EdgeBlur`): `edge_mult` reads "Edge multiplier", Nuke "edge mult"
- Denoise (`Denoise`): `amount` reads "Strength", Nuke "Denoise Amount"
- DegrainSimple (`DegrainSimple`): `rVal` reads "Red", Nuke "R blur"; `gVal` reads "Green", Nuke "G blur"; `bVal` reads "Blue", Nuke "B blur"
- MotionBlur (`MotionBlur`): 1 of 2 paired knobs in Nuke's order; `shutterSamples` reads "Samples", Nuke "Shutter Samples"; `shutterTime` reads "Shutter", Nuke "Shutter Time"
- VectorBlur (`VectorBlur`): 3 of 4 paired knobs in Nuke's order; `uv` reads "Vector layer", Nuke "uv channels"
- Bilateral (`Bilateral`): `size` reads "Spatial size", Nuke "Size"
- EdgeDetect (`EdgeDetect`): 4 of 5 paired knobs in Nuke's order; `edgedetector` reads "Edge Type", Nuke "Edge Detector"
- GodRays (`GodRays`): 2 of 3 paired knobs in Nuke's order
- ZSlice (`ZSlice`): `Zchan` reads "Depth layer", Nuke "Z"
- Keyer (`Keyer`): `range#2` reads "Range", Nuke "A"; `range#3` reads "Range", Nuke "B"; `range#4` reads "Range", Nuke "C"; `range#5` reads "Range", Nuke "D"
- Keylight (`ScreenKeyer`): 28 of 34 paired knobs in Nuke's order
- IBKGizmo (`IBKGizmo`): `ublu` reads "Use BG luminance", Nuke "use bkg luminnance"
- AddMix (`AddMix`): `A` reads "A channels", Nuke "A"; `B` reads "B channels", Nuke "B"
- ContactSheet (`ContactSheet`): 8 of 11 paired knobs in Nuke's order
- TimeDissolve (`TimeDissolve`): `lut` reads "Ease", Nuke "Curves"
- ZMerge (`ZMerge`): `z_channel` reads "Depth layer", Nuke "Z channel"
- Tracker (`Tracker`): 24 of 35 paired knobs in Nuke's order
- CornerPin2D (`CornerPin`): `invert` reads "Direction", Nuke "invert"
- STMap (`STMap`): 4 of 6 paired knobs in Nuke's order
- IDistort (`IDistort`): 6 of 8 paired knobs in Nuke's order
- SplineWarp (`SplineWarp`): 2 of 4 paired knobs in Nuke's order
- PointsTo3D (`PointsTo3D`): 3 of 6 paired knobs in Nuke's order
- AddTimeCode (`AddTimeCode`): 4 of 5 paired knobs in Nuke's order; `startcode` reads "Start timecode", Nuke "startcode"
- BurnIn (`BurnIn`): 10 of 18 paired knobs in Nuke's order
- Precomp (`Precomp`): 5 of 7 paired knobs in Nuke's order

### Split knobs

Nuke's knob class for each knob: AColor_Knob (one slider that splits into r g b a) and WH_Knob (one slider that splits into w h), as Nuke writes them in .nk scripts (for example `multiply {1 1 1 1}`, `size {4 4}`) and as the Nuke Python API names them (nuke.AColor_Knob, nuke.WH_Knob). The Reference Guide's tables do not state knob classes, so this table is not read from the guide. Curated, not exhaustive. Only knobs whose class is long-established are listed; a node that is not here has no splits component and is not penalised. To be confirmed against a Nuke install (scripts from tools/make_nuke_reference_pack.py can print each node's knob classes).

- Noise (`Noise`): `size` (wh). all split.
- Grade (`Grade`): `blackpoint` (rgba), `whitepoint` (rgba), `black` (rgba), `white` (rgba), `multiply` (rgba), `add` (rgba), `gamma` (rgba). all split.
- ColorCorrect (`ColorCorrect`): `saturation` (rgba), `contrast` (rgba), `gamma` (rgba), `gain` (rgba), `offset` (rgba), `shadows.saturation` (rgba), `shadows.contrast` (rgba), `shadows.gamma` (rgba), `shadows.gain` (rgba), `shadows.offset` (rgba), `midtones.saturation` (rgba), `midtones.contrast` (rgba), `midtones.gamma` (rgba), `midtones.gain` (rgba), `midtones.offset` (rgba), `highlights.saturation` (rgba), `highlights.contrast` (rgba), `highlights.gamma` (rgba), `highlights.gain` (rgba), `highlights.offset` (rgba). all split.
- Multiply (`Multiply`): `value` (rgba). all split.
- Add (`Add`): `value` (rgba). all split.
- Gamma (`Gamma`): `value` (rgba). all split.
- Clamp (`Clamp`): `minimum` (rgba), `maximum` (rgba). `minimum` not splittable, `maximum` not splittable.
- Blur (`Blur`): `size` (wh). `size` not splittable.
- Sharpen (`Sharpen`): `size` (wh). `size` not splittable.
- Erode (filter) (`ErodeFilter`): `size` (wh). `size` not splittable.
- Glow (`Glow`): `size` (wh). `size` not splittable.
- Soften (`Soften`): `size` (wh). `size` not splittable.
- Transform (`Transform`): `scale` (wh). all split.
- Reformat (`Reformat`): `scale` (wh). all split.

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

