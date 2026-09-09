# NodeBased 0.3.0 — Wire rewire and four new Nuke-parity nodes

Third desktop release, immediately following 0.2.0's EXR/color-management work.

## What changed since 0.2.0

- **Wire pick-up/rewire.** Clicking an already-connected input port unhooks its
  wire into a pending connection from the same source, which can then be dropped
  on a different input, or on empty space to disconnect. Real unhook/rehook, not
  delete-and-recreate.
- **Four new Nuke-parity nodes**, with keyboard shortcuts and inspector controls:
  **ColorCorrect** (lift/gamma/gain/saturation, sign-safe under fractional gamma
  on negative HDR values), **Blur** (separable box blur, O(n) via a cumulative
  sum, edge padding avoids darkened borders), **Crop** (masks to a rectangle
  without resizing the canvas), and **Shuffle** (remaps output channels from any
  input channel or a constant 0/1).
- All new node types and the null-source disconnect/rewire semantics are exposed
  automatically to the agent protocol's `describe` operation, driven by the same
  SPECS/LIMITS/CHOICES tables the desktop UI reads.
- Projects saved by 0.2.0 load unchanged; the graph format did not change.

## What changed since 0.1.0 (also in 0.2.0)

- **EXR read and write** through OpenImageIO. Float RGBA in, zip-compressed float
  RGBA out. HDR and negative values survive a round trip exactly.
- **Read node color controls.** Input space (Auto, sRGB, Linear Rec.709, ACEScg,
  ACES2065-1, Raw), alpha interpretation (Auto, Straight, Premultiplied), EXR layer
  selection and EXR subimage/part selection. Auto reads EXR as linear/premultiplied
  and PNG/JPEG/TIFF as sRGB/straight.
- **OpenColorIO display views.** sRGB, ACES 2.0 SDR (Rec.709) and Linear, using the
  built-in `cg-config-v4.0.0_aces-v2.0_ocio-v2.5` ACES config. No external OCIO
  config file or `OCIO` environment variable is required, in packages or from source.
- **TIFF input** alongside EXR, PNG and JPEG.
- **Export image** replaces Export PNG and offers float EXR or 8-bit sRGB PNG. The
  headless agent `render` operation follows the same rule by file extension.
- **Darker theme.** Neutral charcoal surfaces with teal, amber, lavender and
  soft-blue node accents, focus rings on inputs and a distinct update button.
- Data windows are composited onto the display window, so cropped EXRs keep their
  position instead of shifting to the origin.
- Projects saved by 0.1.0 load unchanged; Read nodes gain the new controls at their
  Auto defaults.

## Downloads

- **Windows portable ZIP:** extract the entire `NodeBased` folder to a writable
  location and run `NodeBased.exe`. No installer or administrator access required.
  Keep `_internal` and `portable.marker` beside the executable. Update cache and
  rollback files stay under its `portable-data` folder. No registry installation.
- **Windows installer:** per-user installation; no elevation requested.
- **Linux AppImage:** mark executable and run. No installation or Python setup.
  Requires an x86_64 Linux desktop with glibc 2.35+ and the relevant graphics
  libraries. If FUSE is unavailable, run with `APPIMAGE_EXTRACT_AND_RUN=1`.

These early binaries are unsigned. Workplace application-control policies still
apply, including to portable programs.

## Update button

Same explicit flow as GameStore: **Check for updates → Download v… → progress →
Restart to update**. There are no automatic downloads or surprise restarts.
Unsaved work can be saved, discarded or used to cancel the restart. Downloads use
GitHub's published asset SHA-256 digests and are rechecked before installation.
Windows portable builds update in place using a separate helper; installer builds
run the per-user installer. AppImages replace their original file atomically.
A previous portable bundle/AppImage is kept for manual rollback. Keep the app in
a writable location for updates. Source checkouts use Git/pip instead.

## Scope and known limits

Still an M0-class prototype, not a production compositor. Color management runs on
the CPU per preview: on the development machine a 960 × 540 frame took ~58 ms
through the sRGB view and ~202 ms through ACES 2.0, so display transforms are not
yet interactive at high resolution and need a GPU/LUT path. Deep EXR is rejected
rather than flattened. Reads are full-frame with an 8192-per-axis cap. Embedded ICC
profiles are ignored; set Input space explicitly for non-sRGB LDR sources.
Sequences, timeline, 3D and AI generation remain roadmap work. macOS arm64 comes
later. Offscreen tests and packaged launch checks do not establish native GPU or
display performance, or workplace-policy acceptance.
