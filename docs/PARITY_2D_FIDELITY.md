# 2D node fidelity against Nuke

**Overall fidelity: 93.5 / 100** (167 nodes scored, rank-weighted; 95.9 counting only the nodes NodeBased has).

Under the previous formula (knob presence, defaults, sockets, chrome and formula cases only, used from 2026-10-25 to 2026-11-02) the same tree scores 93.2 (95.5 counting only the nodes NodeBased has). The difference is the requirements added on 2026-11-02: knobs that act, Nuke's layout, per-channel splits, and behaviour that counts against a node until a pixel case checks it.

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
| Image | 86.7 | 90.6 | 86.0 | 6 |
| Draw | 91.4 | 83.2 | 90.9 | 17 |
| Time | 96.1 | 78.9 | 96.1 | 13 |
| Channel | 99.1 | 98.4 | 98.9 | 5 |
| Color | 95.6 | 84.8 | 94.6 | 34 |
| Filter | 93.5 | 80.9 | 93.2 | 32 |
| Keyer | 91.6 | 77.2 | 90.7 | 11 |
| Merge | 99.4 | 98.6 | 99.5 | 12 |
| Transform | 93.9 | 84.2 | 93.4 | 20 |
| Metadata | 99.0 | 98.2 | 99.4 | 5 |
| Other | 80.1 | 42.6 | 80.3 | 12 |

Behaviour checked: 82 of the 167 scored nodes have at least one scored pixel case against what the node computes (the `math` column below); 0 nodes whose kernel runs on the fixture have none and score 0 on `math`.

Knobs that act: 351 of the 364 paired knobs probed change the output pixels; 13 change nothing on the fixture (listed per node below). Splits: 14 of the 14 nodes with split knobs in the table split all of them.

## Per node

Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages (`knobs` with the count of Nuke knobs covered, `acts` with acting/probed knobs, `defaults`, `sockets`, `chrome`, `math` with passed/total cases, `unchecked` when no case exists, `layout`, `splits` with fully split/listed knobs) and the pixel-truth column. `Prev.` is the node under the previous formula.

### Image

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Read | `Read` | supported | 87.1 | 87.1 | 100 (32/32) | - | 35 | - | - | - | 100 | - | no fixture |
| 2 | Write | `Write` | supported | 70.9 | 67.2 | 61 (27/56) | - | 42 | - | 100 | - | 100 | - | no fixture |
| 3 | Viewer | `Viewer` | supported | 90.0 | 89.7 | 83 (20/24) | - | - | 100 | 100 | - | - | - | no fixture |
| 4 | Constant | `Constant` | supported | 97.6 | 97.2 | 100 (4/4) | 100 (2/2) | 75 | - | 100 | 100 (4/4) | 100 | - | unmeasured |
| 5 | UDIM Import | - | missing | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 6 | CurveTool | `CurveTool` | supported | 98.1 | 100.0 | 100 (12/12) | - | 100 | 100 | 100 | - | 86 | - | no fixture |
| 7 | Profile | `Profile` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | - | 100 | - | - | - | no fixture |

### Draw

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Roto | `Roto` | supported | 89.6 | 86.8 | 67 (49/75) | - | 100 | 100 | 100 | 100 (18/18) | 100 | - | unmeasured |
| 2 | Text | `Text` | supported | 95.4 | 98.2 | 98 (40/41) | - | 93 | 100 | 100 | - | 80 | - | no fixture |
| 3 | Rectangle | `Rectangle` | supported | 98.6 | 98.7 | 100 (13/13) | - | 87 | 100 | 100 | 100 (9/9) | 100 | - | unmeasured |
| 4 | Ramp | `Ramp` | supported | 98.4 | 98.5 | 100 (10/10) | - | 85 | 100 | 100 | 100 (6/6) | 100 | - | unmeasured |
| 5 | Radial | `Radial` | supported | 98.7 | 98.8 | 100 (14/14) | - | 88 | 100 | 100 | 100 (4/4) | 100 | - | unmeasured |
| 6 | Noise | `Noise` | supported | 99.6 | 99.5 | 100 (26/26) | - | 96 | 100 | 100 | - | 100 | 100 (1/1) | unmeasured |
| 7 | RotoPaint | `RotoPaint` | supported | 95.2 | 93.9 | 85 (74/88) | - | 100 | 100 | 100 | 100 (5/5) | 100 | - | unmeasured |
| 8 | LightWrap | `LightWrap` | supported | 95.4 | 98.9 | 100 (27/27) | 79 (21/26) | 89 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 9 | Grain | `Grain` | supported | 96.7 | 95.0 | 88 (13/14) | 100 (13/13) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 9 | ScannedGrain | `ScannedGrain` | supported | 100.0 | 100.0 | 100 (1/1) | - | - | 100 | - | 100 (6/6) | - | - | unmeasured |
| 10 | DustBust | - | supported | 0.0 | 0.0 | 0 (0/5) | - | - | 0 | - | - | - | - | no fixture |
| 10 | MarkerRemoval | - | supported | 0.0 | 0.0 | 0 (0/5) | - | - | 0 | - | - | - | - | no fixture |
| 11 | Flare | `Flare` | supported | 77.2 | 62.9 | 24 (10/50) | 100 (10/10) | 90 | 100 | - | 100 (7/7) | 100 | - | unmeasured |
| 11 | Glint | `Glint` | supported | 94.0 | 90.8 | 77 (10/13) | 100 (10/10) | 100 | 100 | 100 | 100 (12/12) | 100 | - | unmeasured |
| 11 | Sparkles | `Sparkles` | supported | 80.5 | 68.4 | 36 (9/32) | 100 (9/9) | 89 | 100 | - | 100 (9/9) | 100 | - | unmeasured |
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
| 2 | ShuffleCopy | `ShuffleCopy` | supported | 100.0 | 100.0 | 100 (4/4) | - | - | 100 | - | - | 100 | - | no fixture |
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
| 8 | Clamp | `Clamp` | supported | 99.5 | 99.4 | 100 (8/8) | 100 (8/8) | 94 | 100 | 100 | 100 (3/3) | 100 | 100 (2/2) | unmeasured |
| 9 | HueCorrect | `HueCorrect` | supported | 82.4 | 80.0 | 50 (1/2) | - | 100 | 100 | 100 | 100 (15/15) | - | - | unmeasured |
| 10 | ColorLookup | `ColorLookup` | supported | 92.1 | 90.0 | 75 (6/8) | - | 100 | 100 | 100 | 100 (7/7) | 100 | - | unmeasured |
| 11 | Exposure | `Exposure` | supported | 95.7 | 93.6 | 86 (6/7) | 100 (6/6) | 93 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 12 | HSVTool | `HSVTool` | partial | 88.4 | 80.8 | 59 (14/25) | 100 (9/9) | 100 | 100 | - | 100 (11/11) | 100 | - | unmeasured |
| 13 | ColorMatrix | `ColorMatrix` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 14 | Colorspace | `Colorspace` | supported | 87.7 | 83.6 | 78 (7/9) | 100 (7/7) | 100 | 100 | 100 | 70 (14/20) | 100 | - | unmeasured |
| 14 | OCIOColorspace | `OCIOColorspace` | supported | 81.4 | 71.4 | 29 (2/11) | 100 (2/2) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 14 | OCIODisplay | `OCIODisplay` | supported | 75.5 | 61.5 | 31 (4/17) | 100 (4/4) | 50 | 100 | - | 100 (5/5) | 100 | - | unmeasured |
| 14 | OCIOFileTransform | `OCIOFileTransform` | supported | 88.7 | 83.0 | 60 (2/4) | 100 (2/2) | 90 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 14 | OCIOLogConvert | `OCIOLogConvert` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 100 (2/2) | - | - | unmeasured |
| 14 | OCIOLookTransform | `OCIOLookTransform` | supported | 84.3 | 76.0 | 40 (4/14) | 100 (4/4) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 15 | Log2Lin | `Log2Lin` | supported | 100.0 | 100.0 | 100 (5/5) | 100 (5/5) | 100 | 100 | 100 | 100 (8/8) | 100 | - | unmeasured |
| 15 | PLogLin | `PLogLin` | supported | 97.1 | 96.7 | 100 (5/5) | 100 (5/5) | 100 | 67 | 100 | 100 (6/6) | 100 | - | unmeasured |
| 16 | ColorTransfer | - | supported | 0.0 | 0.0 | 0 (0/1) | - | - | 0 | - | - | - | - | no fixture |
| 16 | MatchGrade | `MatchGrade` | supported | 44.5 | 26.0 | 8 (3/33) | - | - | 100 | - | - | 100 | - | no fixture |
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
| 21 | Vectorfield | `Vectorfield` | supported | 52.6 | 42.3 | 38 (4/10) | - | 0 | 100 | - | - | 100 | - | no fixture |

### Filter

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Blur | `Blur` | supported | 99.2 | 99.0 | 100 (4/4) | 100 (3/3) | 90 | 100 | 100 | 100 (4/4) | 100 | 100 (1/1) | unmeasured |
| 2 | Sharpen | `Sharpen` | supported | 98.5 | 97.9 | 100 (7/7) | 100 (5/5) | 85 | 100 | 100 | - | 100 | 100 (1/1) | unmeasured |
| 3 | Erode (filter) | `ErodeFilter` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | - | 100 | 100 (1/1) | unmeasured |
| 4 | Median | `Median` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 5 | Glow | `Glow` | supported | 99.1 | 98.8 | 100 (11/11) | 100 (10/10) | 88 | 100 | 100 | 100 (4/4) | 100 | 100 (1/1) | unmeasured |
| 6 | Soften | `Soften` | supported | 95.1 | 97.9 | 100 (7/7) | 83 (5/6) | 85 | 100 | 100 | - | 100 | 100 (1/1) | unmeasured |
| 7 | DropShadow | `DropShadow` | supported | 96.8 | 96.1 | 100 (9/9) | 100 (8/8) | 67 | 100 | - | 100 (7/7) | 100 | - | unmeasured |
| 8 | Defocus | `Defocus` | supported | 88.6 | 82.9 | 60 (3/5) | 100 (3/3) | 89 | 100 | 100 | 100 (5/5) | 100 | - | unmeasured |
| 8 | ZDefocus | `ZDefocus` | supported | 74.1 | 69.2 | 31 (9/36) | - | 67 | 100 | 100 | 100 (2/2) | 93 | - | unmeasured |
| 9 | DirBlur | `DirBlur` | supported | 89.1 | 96.0 | 100 (12/12) | 57 (4/7) | 60 | 100 | 100 | 100 (4/4) | 100 | - | unmeasured |
| 10 | EdgeBlur | `EdgeBlur` | supported | 97.1 | 95.6 | 89 (8/9) | 100 (4/4) | 100 | 100 | 100 | 100 (8/8) | 100 | - | unmeasured |
| 10 | EdgeExtend | `EdgeExtend` | supported | 82.6 | 75.0 | 50 (3/6) | 100 (3/3) | 100 | 50 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 11 | Erode (fast) | `Erode` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 100 (5/5) | - | - | unmeasured |
| 12 | DegrainSimple | `DegrainSimple` | supported | 88.1 | 95.6 | 100 (3/3) | - | 67 | 100 | 100 | - | 50 | - | no fixture |
| 12 | Denoise | `Denoise` | supported | 25.4 | 18.1 | 6 (2/37) | - | 50 | 33 | - | - | 50 | - | no fixture |
| 13 | MotionBlur | `MotionBlur` | supported | 25.0 | 26.4 | 33 (2/6) | - | 0 | 25 | - | - | 25 | - | no fixture |
| 13 | MotionBlur2D | `MotionBlur2D` | supported | 68.1 | 63.9 | 75 (3/4) | - | 33 | 50 | - | - | 100 | - | no fixture |
| 13 | MotionBlur3D | `MotionBlur3D` | supported | 63.9 | 55.6 | 50 (3/6) | - | 33 | 100 | - | - | 100 | - | no fixture |
| 13 | VectorBlur | `VectorBlur` | supported | 56.1 | 57.9 | 29 (4/15) | - | 67 | 100 | 100 | - | 38 | - | no fixture |
| 14 | Bilateral | `Bilateral` | supported | 100.0 | 100.0 | 100 (5/5) | - | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 15 | Convolve | `Convolve` | supported | 100.0 | 100.0 | 100 (4/4) | - | 100 | 100 | 100 | - | 100 | - | unmeasured |
| 15 | Matrix | `Matrix` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (2/2) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 16 | BumpBoss | `BumpBoss` | supported | 44.4 | 42.2 | 0 (0/6) | - | 100 | 67 | 100 | - | - | - | unmeasured |
| 16 | EdgeDetect | `EdgeDetect` | supported | 84.7 | 80.6 | 64 (5/7) | 86 (4/5) | 100 | 100 | 100 | - | 100 | - | unmeasured |
| 16 | Emboss | `Emboss` | supported | 71.1 | 54.5 | 18 (2/13) | 100 (2/2) | 86 | 100 | 100 | - | 100 | - | unmeasured |
| 16 | Laplacian | `Laplacian` | supported | 91.3 | 100.0 | 100 (4/4) | 50 (2/4) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 17 | Inpaint | `Inpaint` | supported | 98.7 | 98.8 | 100 (11/11) | - | 91 | 100 | 100 | - | 100 | - | no fixture |
| 18 | GodRays | `GodRays` | supported | 98.9 | 98.8 | 100 (10/10) | 100 (10/10) | 88 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 18 | LevelSet | `LevelSet` | supported | 100.0 | 100.0 | 100 (8/8) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 18 | VolumeRays | `VolumeRays` | supported | 80.5 | 65.3 | 48 (11/24) | 100 (11/11) | 100 | 100 | - | - | 100 | - | unmeasured |
| 19 | ZSlice | `ZSlice` | supported | 73.8 | 76.7 | 100 (3/3) | - | 50 | 100 | 17 | - | 50 | - | no fixture |
| 20 | Bokeh | - | supported | 0.0 | 0.0 | 0 (0/42) | - | - | 0 | - | - | - | - | no fixture |

### Keyer

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Keyer | `Keyer` | supported | 97.9 | 97.4 | 100 (10/10) | 100 (9/9) | 78 | 100 | - | 100 (11/11) | 100 | - | unmeasured |
| 2 | ChromaKeyer | `ChromaKeyer` | supported | 99.2 | 98.9 | 100 (16/16) | 100 (13/13) | 93 | 100 | - | - | 100 | - | unmeasured |
| 3 | Difference | `Difference` | supported | 100.0 | 100.0 | 100 (4/4) | 100 (4/4) | 100 | 100 | - | 100 (2/2) | 100 | - | unmeasured |
| 4 | HueKeyer | `HueKeyer` | supported | 100.0 | 100.0 | 100 (4/4) | 100 (3/3) | 100 | 100 | 100 | 100 (5/5) | 100 | - | unmeasured |
| 5 | Keylight | `ScreenKeyer` | partial | 98.8 | 98.0 | 98 (34/35) | 100 (27/27) | 97 | 100 | - | - | 100 | - | unmeasured |
| 6 | Primatte | - | missing | 0.0 | 0.0 | 0 (0/26) | - | - | 0 | 0 | - | - | - | no fixture |
| 6 | Ultimatte | - | missing | 0.0 | 0.0 | 0 (0/60) | - | - | 0 | - | - | - | - | no fixture |
| 7 | IBKColor | `IBKColor` | supported | 84.4 | 79.2 | 100 (6/6) | 100 (4/4) | 25 | 50 | - | - | 100 | - | unmeasured |
| 7 | IBKGizmo | `IBKGizmo` | supported | 76.4 | 59.7 | 47 (7/15) | 100 (6/6) | 71 | 100 | - | - | 100 | - | unmeasured |
| 8 | Cryptomatte | `Cryptomatte` | partial | 100.0 | 100.0 | 100 (4/4) | - | 100 | 100 | - | - | 100 | - | no fixture |
| 8 | Encryptomatte | `Encryptomatte` | partial | 92.9 | 90.5 | 86 (3/4) | - | 100 | 100 | - | - | 100 | - | no fixture |

### Merge

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Merge | `Merge` | supported | 100.0 | 100.0 | 100 (9/9) | 100 (2/2) | 100 | 100 | 100 | 100 (29/29) | 100 | - | unmeasured |
| 2 | Premult | `Premult` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (1/1) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 3 | Unpremult | `Unpremult` | supported | 100.0 | 100.0 | 100 (2/2) | 100 (1/1) | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 4 | Switch | `Switch` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | - | 100 (2/2) | - | - | unmeasured |
| 5 | Dissolve | `Dissolve` | supported | 100.0 | 100.0 | 100 (1/1) | 100 (1/1) | 100 | 100 | 100 | 100 (4/4) | - | - | unmeasured |
| 6 | KeyMix | `Keymix` | supported | 100.0 | 100.0 | 100 (2/2) | - | 100 | 100 | 100 | 100 (1/1) | 100 | - | unmeasured |
| 7 | AddMix | `AddMix` | supported | 100.0 | 100.0 | 100 (6/6) | 100 (2/2) | 100 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 8 | Blend | `Blend` | supported | 96.8 | 95.3 | 100 (2/2) | 100 (2/2) | 83 | 100 | 80 | 100 (3/3) | 100 | - | unmeasured |
| 9 | CopyBBox | `CopyBBox` | supported | n/a | n/a | - | - | - | - | - | - | - | - | no fixture |
| 9 | CopyRectangle | `CopyRectangle` | supported | 100.0 | 100.0 | 100 (5/5) | 100 (5/5) | 100 | 100 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 10 | ContactSheet | `ContactSheet` | supported | 88.6 | 90.9 | 100 (11/11) | - | 45 | 100 | - | - | 86 | - | no fixture |
| 11 | TimeDissolve | `TimeDissolve` | supported | 100.0 | 100.0 | 100 (3/3) | - | 100 | 100 | - | 100 (3/3) | 100 | - | unmeasured |
| 12 | ZMerge | `ZMerge` | supported | 97.2 | 100.0 | 100 (3/3) | - | 100 | 100 | - | - | 83 | - | no fixture |

### Transform

| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Transform | `Transform` | supported | 100.0 | 100.0 | 100 (15/15) | 100 (10/10) | 100 | 100 | - | 100 (12/12) | 100 | 100 (1/1) | unmeasured |
| 2 | Crop | `Crop` | supported | 97.6 | 97.1 | 100 (5/5) | 100 (2/2) | 75 | 100 | - | 100 (3/3) | 100 | - | unmeasured |
| 3 | Tracker | `Tracker` | supported | 90.7 | 89.0 | 78 (41/54) | - | 79 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
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
| 14 | GridWarp | `GridWarp` | supported | 70.2 | 63.2 | 15 (5/34) | - | 73 | 100 | 100 | 100 (3/3) | 100 | - | unmeasured |
| 14 | SplineWarp | `SplineWarp` | supported | 82.1 | 79.7 | 67 (12/23) | - | 64 | 67 | 100 | 100 (2/2) | 100 | - | unmeasured |
| 15 | Tile | `Tile` | supported | 100.0 | 100.0 | 100 (5/5) | - | 100 | 100 | - | 100 (1/1) | 100 | - | unmeasured |
| 16 | VectorCornerPin | `VectorCornerPin` | supported | 93.3 | 91.0 | 81 (13/17) | - | 100 | 100 | - | 100 (4/4) | 100 | - | unmeasured |
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
| 1 | BlinkScript (Other #6) | - | 16.67 | 0.0 | no node: needs a Blink-equivalent runtime (PARITY_2D.md Other #6) |
| 2 | Write (Image #2) | `Write` | 14.53 | 70.9 | 29 secondary knobs missing |
| 3 | Read (Image #1) | `Read` | 12.94 | 87.1 | nothing missing that the guide lists |
| 4 | Roto (Draw #1) | `Roto` | 10.39 | 89.6 | 26 secondary knobs missing |
| 5 | Primatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 6 | Ultimatte (Keyer #6) | - | 8.33 | 0.0 | no node: commercial keying algorithm (PARITY_2D.md Keyer #6) |
| 7 | AudioRead (Other #14) | - | 7.14 | 0.0 | no node: no audio pipeline (PARITY_2D.md Other #14) |
| 8 | Root (Other #11) | `time and settings` | 6.36 | 30.0 | primary knobs missing: `name`, `project_directory`, `Script_directory`, `label`, `lock_connections`, `lock_range`, `format`, `proxy_type` and 4 more; 24 secondary knobs missing |
| 9 | DustBust (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 10 | MarkerRemoval (Draw #10) | - | 5.00 | 0.0 | provided as a RotoPaint preset, not a node (PARITY_2D.md Draw #10) |
| 11 | Bokeh (Filter #20) | - | 5.00 | 0.0 | provided as ZDefocus bokeh kernels, not a node (PARITY_2D.md Filter #20) |
| 12 | LensDistortion (Transform #20) | `Undistort` | 3.95 | 21.0 | primary knobs missing: `distortionModelPreset`, `lens`; 55 secondary knobs missing |
| 13 | Viewer (Image #3) | `Viewer` | 3.33 | 90.0 | primary knobs missing: `gl_buffer_depth`, `useGPUForInputs`, `disableGPUDitherForViewer`, `GPUDitherAlgoithmForViewer` |
| 14 | Reconcile3D (Transform #18) | `Reconcile3D` | 3.26 | 41.3 | 13 secondary knobs missing; sockets missing: `img` |
| 15 | PointsTo3D (Transform #17) | `PointsTo3D` | 3.14 | 46.7 | primary knobs missing: `cameraMovType`; 2 secondary knobs missing: `point3D`, `point2D`; sockets missing: `img`; knob order differs from Nuke's (3 of 6 in order) |
| 16 | Link (Other #8) | - | 3.12 | 0.0 | no node (PARITY_2D.md Other #8) |
| 17 | LiveGroup (Other #8) | - | 3.12 | 0.0 | no node (PARITY_2D.md Other #8) |
| 18 | LiveInput (Other #8) | - | 3.12 | 0.0 | no node (PARITY_2D.md Other #8) |
| 19 | VariableGroup (Other #8) | - | 3.12 | 0.0 | no node (PARITY_2D.md Other #8) |
| 20 | ColorTransfer (Color #16) | - | 3.12 | 0.0 | covered by MatchGrade, no separate node (PARITY_2D.md Color #16) |
| 21 | Denoise (Filter #12) | `Denoise` | 3.11 | 25.4 | primary knobs missing: `type`, `model`, `lift_blacks`, `profile`, `output`, `rolloff`, `smoothness`, `lumablend` and 1 more; 26 secondary knobs missing; sockets missing: `Motion`, `Noise`; 1 label differs from Nuke's |
| 22 | Tracker (Transform #3) | `Tracker` | 3.09 | 90.7 | 13 secondary knobs missing |
| 23 | SmartVector (Time #8) | `SmartVector` | 2.59 | 17.1 | primary knobs missing: `strengthReg`, `matteChannel`, `output`, `inpainting`, `matteDilation`; 5 secondary knobs missing: `export_write`, `flickerCompensation`, `weightRed`, `weightGreen`, `weightBlue`; sockets missing: `Matte` |
| 24 | Text (Draw #2) | `Text` | 2.29 | 95.4 | 1 secondary knob missing: `group_animation`; knob order differs from Nuke's (27 of 40 in order); 1 label differs from Nuke's |
| 25 | AddTimeCode (Metadata #4) | `AddTimeCode` | 2.22 | 91.1 | knob order differs from Nuke's (4 of 5 in order); 1 label differs from Nuke's |
| 26 | BurnIn (Other #7) | `BurnIn` | 2.20 | 84.6 | knob order differs from Nuke's (10 of 18 in order) |
| 27 | TimeEcho (Time #7) | `TimeEcho` | 2.18 | 54.2 | primary knobs missing: `framesfade`; knob order differs from Nuke's (1 of 2 in order); 2 labels differ from Nuke's |
| 28 | Keyer (Keyer #1) | `Keyer` | 2.12 | 97.9 | nothing missing that the guide lists |
| 29 | Kronos (Time #8) | `Kronos` | 2.09 | 33.1 | primary knobs missing: `input.first`, `input.last`, `resetInputRange`, `retimedChannels`, `timing2`, `timingInputSpeed`, `motionEstimation`, `vectorDetail` and 3 more; 11 secondary knobs missing: `shutterTime`, `autoShutterTime`, `output`, `matteChannel`, `flickerCompensation` ...; sockets missing: `FgVecs BgVecs`, `MoSrc`, `Matte`; knob order differs from Nuke's (3 of 4 in order); 1 label differs from Nuke's |
| 30 | HueCorrect (Color #9) | `HueCorrect` | 1.96 | 82.4 | primary knobs missing: `mix_luminance` |

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
| `ocio_colorspace_gain` | convention | pass | within tolerance | - |
| `ocio_colorspace_encode` | convention | pass | within tolerance | - |
| `ocio_colorspace_decode` | convention | pass | within tolerance | - |
| `ocio_logconvert_log_to_lin` | convention | pass | within tolerance | - |
| `ocio_logconvert_lin_to_log` | convention | pass | within tolerance | - |
| `ocio_look_off` | convention | pass | within tolerance | - |
| `ocio_look_on` | convention | pass | within tolerance | - |
| `ocio_look_invert` | convention | pass | within tolerance | - |
| `ocio_display_mon_plain` | convention | pass | within tolerance | - |
| `ocio_display_mon_bright` | convention | pass | within tolerance | - |
| `ocio_display_alt_plain` | convention | pass | within tolerance | - |
| `ocio_display_input_space` | convention | pass | within tolerance | - |
| `ocio_display_invert` | convention | pass | within tolerance | - |
| `ocio_file_forward` | convention | pass | within tolerance | - |
| `ocio_file_invert` | convention | pass | within tolerance | - |
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
| `roto_bezier_square_fill` | convention | pass | within tolerance | - |
| `roto_bezier_ellipse_fill` | convention | pass | within tolerance | - |
| `roto_edge_pixel_coverage` | convention | not scored: the guide does not say how Roto antialiases a pixel an outline cuts through (the covered fraction of the pixel is the usual choice and what NodeBased draws: this pixel reads 0.5); interior and exterior pixels are scored by the cases above | - | - |
| `roto_feather_widens_outward_linearly` | guide | pass | within tolerance | - |
| `roto_negative_feather_falls_inward` | guide | pass | within tolerance | - |
| `roto_feather_smooth_is_half_way_at_half_the_feather` | guide | pass | within tolerance | - |
| `roto_feather_smooth_curve` | guide | not scored: the guide says where each curve's centre lies ('closer to the shape edge', 'closer to the feather edge', 'halfway') but gives no curve; only the halfway property of smooth is scored, by the case above. NodeBased draws smooth as a smoothstep and has no smooth0 or smooth1 (listed in the step E report) | - | - |
| `roto_shape_opacity_scales_fill` | guide | pass | within tolerance | - |
| `roto_node_opacity_scales_fill` | guide | pass | within tolerance | - |
| `roto_shape_invert_complements` | guide | pass | within tolerance | - |
| `roto_overlap_blends_over` | convention | pass | within tolerance | - |
| `roto_overlap_blends_plus` | convention | pass | within tolerance | - |
| `roto_overlap_blends_minus` | convention | not scored: the guide names the mode but not its direction (the shape taken from what is below, or what is below taken from the shape: Merge's minus and from are opposites), so no pixel is scored; NodeBased subtracts the shape from what is below | - | - |
| `roto_transform_translate_moves_the_shape` | guide | pass | within tolerance | - |
| `roto_visible_off_draws_nothing` | guide | pass | within tolerance | - |
| `roto_output_alpha_keeps_the_colour` | guide | pass | within tolerance | - |
| `roto_output_rgb_keeps_the_alpha` | guide | pass | within tolerance | - |
| `roto_existing_alpha_kept_outside_the_shapes` | guide | pass | within tolerance | - |
| `roto_replace_clears_the_existing_alpha` | guide | pass | within tolerance | - |
| `roto_premultiply_rgb_multiplies_by_the_matte` | guide | pass | within tolerance | - |
| `roto_premultiply_rgba_squares_the_alpha` | guide | pass | within tolerance | - |
| `rotopaint_stroke_is_a_disc_of_the_brush_diameter` | guide | pass | within tolerance | - |
| `rotopaint_a_larger_brush_paints_a_larger_disc` | guide | pass | within tolerance | - |
| `rotopaint_opacity_scales_the_stroke` | guide | pass | within tolerance | - |
| `rotopaint_hardness_falloff_curve` | guide | not scored: the guide says hardness 'defines the rate of falloff from the brush center to its edge' and that increasing it 'decreases blurring at the stroke edges', but gives no curve; the monotone property (a harder brush paints more of a given pixel, a softer one less at the same distance) is tested in tests/test_2d_behaviour_roto.py | - | - |
| `rotopaint_clone_offset_moves_the_source` | guide | pass | within tolerance | - |
| `rotopaint_reveal_shows_the_second_image` | convention | pass | within tolerance | - |
| `tracker_match_move_follows_the_track` | guide | pass | within tolerance | - |
| `tracker_stabilise_removes_the_track` | guide | pass | within tolerance | - |
| `tracker_reference_frame_is_the_identity` | guide | pass | within tolerance | - |
| `dropshadow_angle_180_casts_right` | convention | pass | within tolerance | - |
| `dropshadow_angle_0_casts_left` | convention | pass | within tolerance | - |
| `dropshadow_distance_moves_the_shadow` | convention | pass | within tolerance | - |
| `dropshadow_opacity_scales_the_shadow` | convention | pass | within tolerance | - |
| `dropshadow_colour_tints_the_shadow` | convention | pass | within tolerance | - |
| `dropshadow_inherit_input_colour` | convention | pass | within tolerance | - |
| `dropshadow_disabled_is_the_input` | guide | pass | within tolerance | - |
| `dropshadow_angle_90_direction` | convention | not scored: the guide gives 0 as left and 180 as right and does not say which way the angle turns, so up or down at 90 and 270 is not stated | - | - |
| `dropshadow_softness_blur` | convention | not scored: the guide says softness 'controls how much blur/feather is applied to the shadow effect' and names no filter or radius; that a larger softness spreads the shadow and keeps its total is tested in tests/test_2d_behaviour_filter_f.py | - | - |
| `dirblur_linear_length_3_along_x` | convention | pass | within tolerance | - |
| `dirblur_linear_length_5_along_x` | convention | pass | within tolerance | - |
| `dirblur_linear_angle_90_runs_down_the_columns` | convention | pass | within tolerance | - |
| `dirblur_linear_length_1_is_the_input` | convention | pass | within tolerance | - |
| `dirblur_zoom_and_radial_formulas` | convention | not scored: the guide says BlurLength is 'in pixels' for every type and gives no path for zoom or radial (NodeBased reads the zoom length as a percent of the distance to the centre and the radial length as degrees); what is stated (the centre pixel stays, rays stay on their line, flat areas stay) is tested in tests/test_2d_behaviour_filter_f.py | - | - |
| `lightwrap_intensity_zero_adds_nothing` | convention | pass | within tolerance | - |
| `lightwrap_generate_wrap_only_without_wrap_is_black` | guide | pass | within tolerance | - |
| `lightwrap_wrap_amount_at_the_edge` | convention | not scored: the guide describes the effect ('a reflection of light around the edges of your foreground object by blending in whatever is in the background') and gives neither the falloff nor the blur's filter, so the amount cannot be written down without guessing | - | - |
| `bilateral_size_1_compares_only_the_pixel` | convention | pass | within tolerance | - |
| `bilateral_tiny_colour_sigma_keeps_every_edge` | convention | pass | within tolerance | - |
| `bilateral_gaussian_weights` | convention | not scored: the guide names no weight formula and no pixel scale for the positional sigma; NodeBased uses the classic Gaussian form with sigma positionalSigma x size pixels (nodebased/bilateralknobs.py), and tests/test_2d_behaviour_filter_g.py shows a step edge staying sharp while flat noise smooths | - | - |
| `bilateral_median_filter` | convention | not scored: the guide does not say whether the middle value is a weighted median or a plain one; NodeBased takes the weighted median under the gaussian weights | - | - |
| `vectorcornerpin_pins_on_the_frame_are_the_identity` | convention | pass | within tolerance | - |
| `vectorcornerpin_pins_moved_carry_the_picture` | convention | pass | within tolerance | - |
| `vectorcornerpin_swapped_pins_mirror_the_picture` | convention | pass | within tolerance | - |
| `vectorcornerpin_from_quad_picks_the_source` | convention | pass | within tolerance | - |
| `gridwarp_interior_key_moves_the_pixel_under_it` | convention | pass | within tolerance | - |
| `gridwarp_mix_zero_is_the_source` | convention | pass | within tolerance | - |
| `gridwarp_output_source_is_the_source` | convention | pass | within tolerance | - |
| `gridwarp_background_over_the_source` | convention | not scored: the guide's page for this node is a video transcript with no formula for the background and its mix; the reading is NodeBased's | - | - |
| `splinewarp_curve_moves_the_pixels_on_it` | convention | pass | within tolerance | - |
| `splinewarp_mix_zero_is_the_input` | convention | pass | within tolerance | - |
| `switch_which_0` | guide | pass | within tolerance | - |
| `switch_which_1` | guide | pass | within tolerance | - |
| `switch_which_past_last` | convention | not scored: the guide says only that which = N shows input N and does not say what a value past the last input shows; NodeBased shows the last connected input (tests/test_2d_behaviour_merge.py), and no Foundry source to cite for Nuke's own answer was found | - | - |
| `switch_which_fraction` | convention | not scored: the guide gives only whole numbers for which; what Nuke does with 0.5 (round, floor, or a mix as Dissolve does) is not stated, so NodeBased's whole-number reading is not scored | - | - |
| `dissolve_which_0.0` | guide | pass | within tolerance | - |
| `dissolve_which_0.25` | guide | pass | within tolerance | - |
| `dissolve_which_0.5` | guide | pass | within tolerance | - |
| `dissolve_which_1.0` | guide | pass | within tolerance | - |
| `timedissolve_before_in` | guide | pass | within tolerance | - |
| `timedissolve_after_out` | guide | pass | within tolerance | - |
| `timedissolve_middle` | guide | pass | within tolerance | - |
| `addmix_default_curves` | convention | pass | within tolerance | - |
| `addmix_moved_curves` | guide | pass | within tolerance | - |
| `addmix_premultiplied` | guide | pass | within tolerance | - |
| `addmix_nonlinear` | convention | not scored: the guide says only that the zero-to-infinity range is remapped logarithmically into 0-1; it gives no formula, so the log(1 + a) / (1 + log(1 + a)) remap NodeBased uses cannot be scored | - | - |
| `blend_equal` | guide | pass | within tolerance | - |
| `blend_unequal` | guide | pass | within tolerance | - |
| `blend_unequal_total` | guide | pass | within tolerance | - |
| `blend_all_zero_weights` | convention | not scored: the weights sum to zero, so dividing by the total is undefined and the guide does not say what Nuke returns; NodeBased returns black (tests/test_2d_behaviour_merge.py) | - | - |
| `copyrectangle_default_channels` | guide | pass | within tolerance | - |
| `copyrectangle_channels` | guide | pass | within tolerance | - |
| `copyrectangle_softness` | convention | not scored: the guide gives no fade width or curve for softness, so the linear fade is NodeBased's own and cannot be scored against Nuke | - | - |
| `constant_color` | guide | pass | within tolerance | - |
| `constant_color_hdr` | guide | pass | within tolerance | - |
| `constant_format` | guide | pass | within tolerance | - |
| `constant_color_alpha` | convention | not scored: NodeBased stores the colour multiplied by its alpha (0.25 for the case here); the guide does not say whether Nuke's colour knob is written as typed or premultiplied, so the stored value is not scored | - | - |
| `constant_channels_rgb` | guide | pass | within tolerance | - |
| `edgeblur_far_from_edge_s2_m1` | guide | pass | within tolerance | - |
| `edgeblur_far_from_edge_s1_m1` | guide | pass | within tolerance | - |
| `edgeblur_far_from_edge_s2_m0.5` | guide | pass | within tolerance | - |
| `edgeblur_band_is_blur_s1` | guide | pass | within tolerance | - |
| `edgeblur_band_is_blur_s2` | guide | pass | within tolerance | - |
| `edgeblur_tint_multiplies_rgb` | guide | pass | within tolerance | - |
| `edgeblur_brightness_multiplies_all` | guide | pass | within tolerance | - |
| `edgeblur_size_zero_is_identity` | convention | pass | within tolerance | - |
| `edgeblur_gaussian_weights` | convention | not scored: the guide does not state the gaussian's sigma for a given size (only that gaussian is the default and the smoothest), so its weights cannot be written down without guessing; tests/test_2d_behaviour_extra_filter.py checks that the band equals a plain Blur of the same filter and size | - | - |
| `edgeblur_quality_scaled` | guide | not scored: the guide does not give the scale factor, the down-scaling filter or the block alignment, so the pixels cannot be written down; tests check that the result is bounded and tile-exact | - | - |
| `godrays_default_is_identity` | guide | pass | within tolerance | - |
| `godrays_max_streaks_along_the_ray` | convention | pass | within tolerance | - |
| `godrays_from_and_to_colour_at_the_ends` | guide | pass | within tolerance | - |
| `godrays_average_weights` | convention | not scored: the guide says 'averages together many samples' without saying whether steps counts the samples or the gaps between them (2 or 3 samples at steps 2), so the weight of each streak pixel cannot be written down without guessing; tests check that the average is the mean of the same samples taken singly | - | - |
| `godrays_translate_direction` | convention | not scored: the guide names translate but not its sign (clockwise or not, up or down; Nuke's y runs up), the order it combines with the other controls (its note says translate and skew do not merge correctly with scale), or its units; tests check that the samples move along the ray the knob names and that the knob acts | - | - |
| `godrays_rotate_direction` | convention | not scored: the guide names rotate but not its sign (clockwise or not, up or down; Nuke's y runs up), the order it combines with the other controls (its note says translate and skew do not merge correctly with scale), or its units; tests check that the samples move along the ray the knob names and that the knob acts | - | - |
| `godrays_skew_direction` | convention | not scored: the guide names skew but not its sign (clockwise or not, up or down; Nuke's y runs up), the order it combines with the other controls (its note says translate and skew do not merge correctly with scale), or its units; tests check that the samples move along the ray the knob names and that the knob acts | - | - |
| `godrays_gamma_space` | convention | not scored: the guide gives gamma one sentence ('the gamma space in which colors are interpolated') and no formula; the ends are unchanged by it and the middle sample is not scored | - | - |
| `volumerays_rays_radiate_from_the_volume_centre` | guide | not scored: the node is a gizmo (a group of nodes) whose rays are built from the image's luminance or alpha through a style, a noise flicker, a radial falloff and a ray length in the gizmo's own units, none of which the guide states as a formula; NodeBased's rays come from the image colour weighted by a wired matte, so a pixel reference for either cannot be written down without guessing (tests/test_2d_behaviour_extra_filter.py checks that the rays lie on the line through the volume centre) | - | - |
| `bumpboss_emboss_of_b_by_a` | guide | not scored: the guide states what each knob does and no lighting formula (how A's slope, the light's offset from the centre, height, intensity and the minimum shadow combine into B's pixels), and NodeBased's BumpBoss is a one-input emboss lit from an angle; the node's remaining gap is recorded as unsupported knobs in docs/parity/nodebased_to_nuke.json | - | - |
| `flare_ring_color_at_the_middle_radius` | guide | pass | within tolerance | - |
| `flare_black_from_the_outer_radius` | guide | pass | within tolerance | - |
| `flare_inner_color_fills_the_hole` | guide | pass | within tolerance | - |
| `flare_brightness_scales_the_flare` | guide | pass | within tolerance | - |
| `flare_brightness_zero_is_invisible` | guide | pass | within tolerance | - |
| `flare_size_mult_scales_the_radii_together` | guide | pass | within tolerance | - |
| `flare_position_is_the_centre` | guide | pass | within tolerance | - |
| `flare_fade_curve` | convention | not scored: the fade between the radii (outer falloff and inner falloff 1 'sets the blending or falloff'): the guide gives no curve, so the pixels between the radii cannot be written down without guessing; tests check that the fade is monotonic and tile-exact | - | - |
| `flare_anamorph` | convention | not scored: anamorph 'simulates anamorphic lens flares by altering the default 1:1 flare shape' with no formula for the stretch | - | - |
| `flare_chroma_shift` | convention | not scored: chroma shift 'moves color components in the direction specified in the angle control' with no distance or sign convention given; NodeBased moves red along the angle and blue the opposite way | - | - |
| `flare_alpha` | convention | not scored: the guide does not say what a flare does to the alpha channel | - | - |
| `glint_four_rays` | guide | pass | within tolerance | - |
| `glint_default_two_rays` | guide | pass | within tolerance | - |
| `glint_single_ray_up` | convention | pass | within tolerance | - |
| `glint_rotation_90` | guide | pass | within tolerance | - |
| `glint_rotation_-90` | guide | pass | within tolerance | - |
| `glint_rotation_180` | guide | pass | within tolerance | - |
| `glint_length` | guide | pass | within tolerance | - |
| `glint_from_and_to_colour` | guide | pass | within tolerance | - |
| `glint_max_off_adds_the_steps` | guide | pass | within tolerance | - |
| `glint_max_on_takes_the_brightest` | guide | pass | within tolerance | - |
| `glint_tolerance_excludes_dim_pixels` | guide | pass | within tolerance | - |
| `glint_effect_only` | guide | pass | within tolerance | - |
| `glint_odd_ray_length` | convention | not scored: the guide says odd ray length 'gives every other ray a different length and sets that length' without saying whether the first or the second ray is the odd one, or whether the value is a multiple or a pixel length; tests check that it shortens the odd rays | - | - |
| `glint_highlight_strength` | convention | not scored: the guide says only that pixels above the threshold bloom; whether a ray carries the pixel's value or only the part above the threshold is not stated, so the ray values cannot be written down without guessing (NodeBased carries the part above) | - | - |
| `sparkles_origin_holds_the_ray_colour` | guide | pass | within tolerance | - |
| `sparkles_nothing_beyond_size_4` | guide | pass | within tolerance | - |
| `sparkles_nothing_beyond_size_6` | guide | pass | within tolerance | - |
| `sparkles_no_rays_is_identity` | guide | pass | within tolerance | - |
| `sparkles_zero_brightness_is_identity` | convention | pass | within tolerance | - |
| `sparkles_off_the_rays_is_unchanged` | convention | pass | within tolerance | - |
| `sparkles_other_seed_other_rays` | convention | pass | within tolerance | - |
| `sparkles_rotation_turns_the_rays` | convention | pass | within tolerance | - |
| `sparkles_thickness_widens_the_rays` | convention | pass | within tolerance | - |
| `sparkles_ray_fade` | convention | not scored: ray falloff 'sets how quickly rays falloff as distance from the position indicator increases': the guide gives no curve, so the values along a ray cannot be written down without guessing; NodeBased fades them as (1 - d / size) to the power of the falloff, and tests check that they never rise with distance | - | - |
| `sparkles_modes` | convention | not scored: the Sparkle type (Rays, Sparkles, Fireworks) changes how rays fade and break; only Rays is implemented, and the guide gives no formulae for the other two | - | - |
| `sparkles_random_knobs` | convention | not scored: the random thickness, broken rays, crawl, sparks and chroma knobs 'randomize' within ranges the guide does not state; none is implemented | - | - |
| `scannedgrain_adds_plate_minus_offset_times_amount` | guide | pass | within tolerance | - |
| `scannedgrain_amount_per_channel` | guide | pass | within tolerance | - |
| `scannedgrain_offset_is_the_neutral_grey` | guide | pass | within tolerance | - |
| `scannedgrain_grain_ignores_the_input_level` | guide | pass | within tolerance | - |
| `scannedgrain_saturation_zero_is_monochrome` | convention | pass | within tolerance | - |
| `scannedgrain_low_clip_stops_subtracting` | guide | pass | within tolerance | - |
| `scannedgrain_frame_selection` | convention | not scored: the guide says frames are 'selected from a loop' of about 50 scanned frames and that 'a different grain pattern is produced for each frame' without saying which frame a given frame number and seed pick; NodeBased has one plate image and translates it by seed + frame instead (a seed of minus the frame freezes it), so the pattern cannot be compared with Nuke's without guessing | - | - |

## Requirements of 2026-11-02, per node

### Knobs that change no pixels

Each paired knob below was moved off its default to every value its `LIMITS` and `CHOICES` allow the probe to try, alone and with each other paired knob moved first, and NodeBased's output did not change on the fixture. Either the knob is not implemented, or it acts only in a state the probe does not reach (two other knobs moved together, a payload the default node does not hold). Implement it, or unpair it in `docs/parity/nodebased_to_nuke.json` with a reason.

- LightWrap (`LightWrap`): `disableluma`, `enableglow`, `LumaTolerance`, `white`, `gray`
- Dither (`Dither`): `static_seed`
- Soften (`Soften`): `quailty`
- DirBlur (`DirBlur`): `holdout`, `PixelAspect`, `Quality`
- EdgeDetect (`EdgeDetect`): `blurquality`
- Laplacian (`Laplacian`): `quailty`, `crop`

Not probed, because a one-frame pixel probe cannot see them: acts across frames (frame range, retiming, motion blur): `after`, `before`, `first`, `frame`, `frame_mode`, `last`, `motionblur`, `origfirst`, `origlast`, `origset`, `shutter`, `shuttercustomoffset`, `shutteroffset`; acts on metadata, not pixels: `metainput`, `rangeinput`; acts on the bounding box only, outside the fixture's format: `intersect`, `pbb`; Blur's `crop` (crop to format acts on the bounding box only); ChromaKeyer's `_replaceAlphaDiff` (acts only with a replace mode other than none); ChromaKeyer's `_replaceColor` (acts only with a replace mode other than none); ChromaKeyer's `_replaceAmount` (acts only with a replace mode other than none); ColorLookup's `source` (feeds the Set RGB, Set RGBA and Set A buttons, which add curve points; the kernel reads the curves, not these two); ColorLookup's `target` (feeds the Set RGB, Set RGBA and Set A buttons, which add curve points; the kernel reads the curves, not these two); Crop's `reformat` (makes the box the format, which changes the window and not the fixture's pixels (tests/test_2d_behaviour_transform.py)); Crop's `crop` (black outside acts only where the box reaches past the picture, with intersect off; the probe's box does not (tests/test_2d_behaviour_transform.py)); DirBlur's `BlurCenter` (the origin of a zoom or radial blur (a linear blur uses it only with UseTarget on); the default type is linear (tests/test_2d_behaviour_filter_f.py)); DirBlur's `Samples` (the number of taps of a zoom or radial blur; the default type is linear (tests/test_2d_behaviour_filter_f.py)); DirBlur's `pixeloffset` (offsets the samples of a zoom blur; the default type is linear (tests/test_2d_behaviour_filter_f.py)); DropShadow's `operation` (merges the shadowed picture over the bg input, so it acts only with bg wired (tests/test_2d_behaviour_filter_f.py)); EdgeBlur's `controlchannel` (picks the channel whose edges are blurred; the fixture's channels all change at every pixel, so the band is full whichever is read (tests/test_2d_behaviour_extra_filter.py shows it on a hard matte)); EdgeBlur's `edge_mult` (sets the width of the edge band; the fixture's alpha changes at every pixel, so the band is full at every value the probe tries (tests/test_2d_behaviour_extra_filter.py shows the band narrow)); EdgeBlur's `quality` (acts only when size exceeds quality, which is the size knob moved too (tests/test_2d_behaviour_filter_a.py shows the scaling for Blur; the cases here do the same for EdgeBlur)); EdgeBlur's `crop` (crop to format acts on the bounding box only (tests/test_2d_behaviour_extra_filter.py shows the window grow)); Glow's `crop` (crop to format acts on the bounding box only); HSVTool's `srccolor` (acts only with Color Replacement on (source and destination colour)); HSVTool's `dstcolor` (acts only with Color Replacement on (source and destination colour)); HSVTool's `huerolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); HSVTool's `satrolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); HSVTool's `brtrolloffrange` (acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved); IBKColor's `Size` (acts on a plate with a keyed-out patch to fill (screen erode and patch black set, as in tests/test_2d_behaviour_knobs_color.py); the fixture is not such a plate); IBKColor's `mult` (acts on a plate with a keyed-out patch to fill, as above); IBKGizmo's `ublu` (acts only with a background wired and luminance matching on); KeyMix's `invertMask` (inverts the mask channel, so it acts only with a mask wired); Keylight's `despillBias` (acts only with the despill bias ganged to the alpha bias off); Keylight's `insideReplaceMethod` (acts only with an inside matte wired); Keylight's `insideReplaceColour` (acts only with an inside matte wired); Keylight's `insideComponent` (acts only with an inside matte wired); Keylight's `outsideComponent` (acts only with an outside matte wired); Keylight's `SourceXMethod` (acts only with the source cropped, which is another knob moved); Keylight's `SourceYMethod` (acts only with the source cropped, which is another knob moved); LightWrap's `shadows_gain` (grades only the darkest part of the wrap; the fixture's wrap has none (tests/test_2d_behaviour_filter_f.py moves it on a dark wrap)); Merge's `screen_alpha` (acts only for operations that do not already composite by alpha (plus, minus, multiply and the like), not the default over); Sharpen's `crop` (crop to format acts on the bounding box only); Sharpen's `quailty` (caps the filter's size and so acts only once the size passes it; the default size 1 never does (tests/test_2d_behaviour_filter_g.py moves it at size 40)); Shuffle's `fromInput1` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); Shuffle's `fromInput2` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); Shuffle's `out2` (routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph); SoftClip's `softclip_min` (acts only with a conversion chosen; the default conversion is none); Soften's `crop` (crop to format acts on the bounding box only); TimeDissolve's `in` (acts across frames (the dissolve range and curve turn a frame number into the mix; tests/test_2d_behaviour_merge.py shows the start, middle and end)); TimeDissolve's `out` (acts across frames (the dissolve range and curve turn a frame number into the mix; tests/test_2d_behaviour_merge.py shows the start, middle and end)); TimeDissolve's `lut` (acts across frames (the dissolve range and curve turn a frame number into the mix; tests/test_2d_behaviour_merge.py shows the start, middle and end)); Toe's `invert` (inverts the mask channel, so it acts only with the mask channel on); Transform's `skew_order` (orders the two skews, so it acts only when skew X and skew Y are both set; the probe moves one at a time (tests/test_2d_behaviour_transform.py)); VectorCornerPin's `to1` (the baked pin at the current frame, read only while Use baked pins is on (the Bake Corners button sets it); tests/test_2d_behaviour_filter_g.py); VectorCornerPin's `to2` (the baked pin at the current frame, read only while Use baked pins is on (the Bake Corners button sets it); tests/test_2d_behaviour_filter_g.py); VectorCornerPin's `to3` (the baked pin at the current frame, read only while Use baked pins is on (the Bake Corners button sets it); tests/test_2d_behaviour_filter_g.py); VectorCornerPin's `to4` (the baked pin at the current frame, read only while Use baked pins is on (the Bake Corners button sets it); tests/test_2d_behaviour_filter_g.py). Whole nodes not probed: GridWarp (the grids live in the node's payload and the default node holds none (tests/test_2d_behaviour_warp_h.py moves each knob on a warp)); Roto (shapes live in the node's payload and the default node holds none); RotoPaint (strokes live in the node's payload and the default node holds none); SplineWarp (the curve pairs live in the node's payload and the default node holds none (tests/test_2d_behaviour_warp_h.py moves each knob on a warp)); Tracker (tracks are solved across frames; a one-frame probe cannot move them). Taps, whose behaviour is their input and which have neither `acts` nor `math`: `Viewer`, `Write`, `NoOp`, `Dot`, `PostageStamp`, `Output`, `Profile`, `PlanarTracker`.

### Layout differences

Labels below are NodeBased's, then the guide's.

- CurveTool (`CurveTool`): 8 of 11 paired knobs in Nuke's order
- Text (`Text`): 27 of 40 paired knobs in Nuke's order; `premult` reads "Premult by", Nuke "premult"
- AppendClip (`AppendClip`): 3 of 5 paired knobs in Nuke's order
- TimeBlur (`TimeBlur`): 3 of 4 paired knobs in Nuke's order
- TimeEcho (`TimeEcho`): 1 of 2 paired knobs in Nuke's order; `mode` reads "Method", Nuke "TimeEcho Method"; `framesbehind` reads "Frames", Nuke "Frames to look at"
- Kronos (`Kronos`): 3 of 4 paired knobs in Nuke's order; `timingOutputSpeed` reads "Speed", Nuke "Output Speed"
- OFlow (`OFlow`): `timingOutputSpeed` reads "Speed", Nuke "Output Speed"; `timingFrame2` reads "Speed", Nuke "Frame"
- ZDefocus (`ZDefocus`): `center` reads "Focus plane", Nuke "focus plane (C)"
- Denoise (`Denoise`): `amount` reads "Strength", Nuke "Denoise Amount"
- DegrainSimple (`DegrainSimple`): `rVal` reads "Red", Nuke "R blur"; `gVal` reads "Green", Nuke "G blur"; `bVal` reads "Blue", Nuke "B blur"
- MotionBlur (`MotionBlur`): 1 of 2 paired knobs in Nuke's order; `shutterSamples` reads "Samples", Nuke "Shutter Samples"; `shutterTime` reads "Shutter", Nuke "Shutter Time"
- VectorBlur (`VectorBlur`): 3 of 4 paired knobs in Nuke's order; `uv` reads "Vector layer", Nuke "uv channels"
- ZSlice (`ZSlice`): `Zchan` reads "Depth layer", Nuke "Z"
- ContactSheet (`ContactSheet`): 8 of 11 paired knobs in Nuke's order
- ZMerge (`ZMerge`): `z_channel` reads "Depth layer", Nuke "Z channel"
- CornerPin2D (`CornerPin`): `invert` reads "Direction", Nuke "invert"
- STMap (`STMap`): 4 of 6 paired knobs in Nuke's order
- IDistort (`IDistort`): 6 of 8 paired knobs in Nuke's order
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
- Clamp (`Clamp`): `minimum` (rgba), `maximum` (rgba). all split.
- Blur (`Blur`): `size` (wh). all split.
- Sharpen (`Sharpen`): `size` (wh). all split.
- Erode (filter) (`ErodeFilter`): `size` (wh). all split.
- Glow (`Glow`): `size` (wh). all split.
- Soften (`Soften`): `size` (wh). all split.
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

