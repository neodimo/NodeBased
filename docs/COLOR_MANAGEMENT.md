# Color management

NodeBased's processing contract is explicit from document schema v7 onward:

- built-in OCIO ACES CG Config v4.0, using ACES 2.0 transforms;
- scene-linear **ACEScg** (AP1) working space;
- float32 premultiplied RGBA throughout evaluation, tiles, and caches;
- ACES 2.0 SDR 100-nit Rec.709 as the default display view for new projects;
- pure black as the default viewer background.

Open **Edit → Project settings…** (`S`) to inspect the config, processing space,
and display, and to change the saved default view or viewer background. The
bundled config, ACEScg working space, and sRGB display are deliberately fixed in
this release. The schema records them now so support for external OCIO configs
can be added without another implicit global.

## Input

Read converts color into ACEScg before pixels enter the graph. Explicit Read
choices are sRGB, Linear Rec.709, ACEScg, ACES2065-1, and Raw. Raw is the
intentional no-transform choice for data passes.

`Auto` honors recognized file color-space metadata. Untagged EXRs fall back to
Linear Rec.709, preserving the convention used by NodeBased releases before the
ACEScg working-space migration; other untagged image formats fall back to sRGB.
EXR alpha defaults to premultiplied and other image alpha defaults to straight.

## Processing and display

Every graph operation receives scene-linear ACEScg float32 pixels. RGB is stored
premultiplied by alpha. A display transform therefore unpremultiplies RGB,
applies the selected view to straight color, and re-associates alpha before
compositing the viewer background. Applying a nonlinear view to premultiplied
RGB is incorrect and was fixed as part of this migration.

The `ACES 2.0` viewer choice means OCIO display `sRGB - Display`, view
`ACES 2.0 - SDR 100 nits (Rec.709)`. `sRGB` is available as a simpler display
conversion and `Linear` is a diagnostic passthrough.

## Output

EXR exports remain linear float32 RGBA and are tagged ACEScg. PNG exports are
unpremultiplied and converted from ACEScg through the OCIO sRGB output transform
before 8-bit encoding. Viewer exposure and channel controls never alter exports.

## Upgrade behavior

Schema v1–v6 documents upgrade to v7 on open. Existing v6 projects retain the
old `sRGB` viewer choice, while new v7 projects default to the ACES 2.0 Rec.709
view. Both use the ACEScg processing pipeline after upgrade. Procedural RGB
constants are numbers in the working space, so their interpretation changes
from Linear Rec.709 to ACEScg; file-backed color inputs are converted on ingest.

