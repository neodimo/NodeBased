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

`nodebased/color.py::display_rgb` runs the `ACES 2.0` view transform on the GPU
(`nodebased/gpudisplay.py`, an OCIO `GpuShaderDesc`-generated GLSL shader) whenever a
working `QOpenGLContext` is available, and the `sRGB` view on a thread-chunked CPU path
otherwise. `NODEBASED_DISPLAY_GPU=0` forces CPU for every view; a GPU failure falls back
to CPU per call rather than crashing. Measured numbers, accuracy tolerance, and the
design rationale (including why `sRGB` deliberately stays CPU-only) are in
`docs/BENCHMARKS-v0.16-display.md`.

## Viewer gain, gamma, clipping warning and display

The viewer toolbar carries four display-only controls, saved with the document as
`settings.viewer.look` (absent while they are all at their defaults, so older files load
unchanged):

- **Gain**, in f-stops (-10 to 10, the old Exposure control under Nuke's name), and **Gamma**
  (0.2 to 5). Each has a reset button. The picture is `v * 2 ** gain`, then
  `v ** (1 / gamma)` on the straight (unpremultiplied) colour, then the display transform. Neither
  is written into the graph's pixels, an export or a Write. Nuke has no default hotkeys for these
  controls, so none are bound.
- **Zebra**, a clipping warning: pixels whose gained, gamma-adjusted scene-linear value is above
  1.0 get red diagonal stripes, pixels below 0.0 get blue ones. The thresholds are shown on the
  toggle. It paints the displayable buffer only.
- **Display**: `Project view` follows the project's default view; otherwise any view the fixed
  ACES config offers on the `sRGB - Display` display, read from the config at run time
  (currently ACES 2.0 SDR 100 nits Rec.709, Un-tone-mapped, Video (colorimetric) and Raw). `Raw`
  shows a scene-linear value with no transform, so it can be inspected directly.

The pixel readout always reports the scene-linear floats the graph produced, whatever gain,
gamma, zebra or display is chosen, as Nuke's does.

## Output

EXR exports are linear RGBA tagged ACEScg, written as 16-bit half with ZIPS
compression by default; `bits='float'` keeps linear float32 for data passes. Half
conversion clamps finite over-range values to 65504 so a bright highlight cannot
become `inf`, and leaves author-supplied `inf`/`nan` untouched. PNG exports are
unpremultiplied and converted from ACEScg through the OCIO sRGB output transform
before 8-bit encoding. Viewer exposure and channel controls never alter exports.

## Upgrade behavior

Schema v1–v6 documents upgrade to v7 on open. Existing v6 projects retain the
old `sRGB` viewer choice, while new v7 projects default to the ACES 2.0 Rec.709
view. Both use the ACEScg processing pipeline after upgrade. Procedural RGB
constants are numbers in the working space, so their interpretation changes
from Linear Rec.709 to ACEScg; file-backed color inputs are converted on ingest.

