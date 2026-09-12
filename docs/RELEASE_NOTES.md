# NodeBased 0.13.0 — branch-insert node placement and viewer format guides

## What changed since 0.12.0

- **Adding a node with a selection wires it into that node's branch.**
  Tab-search or a node hotkey used to always drop the new node at the last
  click position, unconnected. If a node is selected and the new node type
  has an input, it now lands near the selection, connects from its output,
  and takes over any existing downstream connection the selected node had —
  the same splice used by the existing Ctrl-drag-a-noodle-midpoint gesture —
  so it's inserted inline in the branch rather than forking a dead end.
  Generators with no input (Read/Constant/Checker) are unaffected and keep
  click-position placement.
- **Viewer format guides, Nuke-style.** A dotted outline now traces the
  display window at any zoom level, and a resolution readout ("3840 x
  2160") sits just outside its bottom-right corner. Implemented as a
  `drawForeground` paint rather than scene items, specifically so it can
  never perturb `itemsBoundingRect()` — the measurement the 0.12.0
  stuck-corner regression test depends on staying exact.

# NodeBased 0.12.0 — playback caching and proxy-resolution playback

## What changed since 0.11.0

- **EXR writes default to 16-bit half at ZIPS (1-scanline) compression.**
  Applies to export and the agent's `render` op alike; both bit depth and
  compression remain explicitly overridable. Half/ZIPS is now verified to
  match the tile path exactly against the reference evaluator on real 4K
  source (max abs delta 0.0, zero fallbacks).
- **Display-ready frame cache.** Looping playback, scrubbing back onto a
  frame already shown, or returning a paused parameter to a value it held
  before is now a cache hit instead of paying the ACES 2.0 view transform
  again — measured 3.14s cold vs 0.08s on an identical repeat at 4K. Keyed
  on the whole document plus frame/tier/view/exposure/channel/background, so
  a graph edit is correctly a miss and undoing it is correctly a hit again.
- **Proxy-resolution playback.** Sources above HD now play back at an
  automatically chosen lower tier — the same technique every NLE and
  compositor uses for this — and restore full quality the instant playback
  stops. Never overrides a tier chosen manually. Cuts a cold 4K frame from
  ~3.1s to ~1.1s. Read-ahead now also warms the transformed display cache,
  not just the raw composite, ahead of the playhead.
- **Fixed: viewer stuck zoomed into a stale corner.** Reconnecting the
  viewer to a differently sized source used to clamp its first frame to
  whatever fraction of the old image's viewport happened to overlap the new
  canvas, so the image could get stuck zoomed into a small corner with no
  way to recenter it. The first frame of a resized source now always
  requests the complete canvas.
- **Fixed: a schema-upgrade step could silently skip its own successor.**
  One upgrade step stamped the generic `SCHEMA_VERSION` constant instead of
  the literal version it actually produced — harmless while that step
  happened to be the last one, but the next schema version added would have
  landed a document tagged current while missing its newest section. Fixed
  and guarded by a test that checks no upgrade step makes this mistake.

## Known scope

Native 4K playback at full (untier'd) quality is not real-time — the ACES
2.0 display transform's CPU cost is the remaining bottleneck for that
specific case. Proxy-resolution playback above closes most of the practical
gap; moving the transform itself off the CPU is unscoped follow-up work.

# NodeBased 0.11.0 — ACEScg color pipeline and project settings

- **Linear ACEScg float32 working space.** Read inputs convert from their tagged
  or selected source space into ACEScg before graph evaluation. Untagged EXRs
  fall back to Linear Rec.709 instead of being assumed to already match the
  working space. EXR output is tagged ACEScg; PNG output converts from ACEScg
  through OCIO rather than applying only an sRGB transfer curve.
- **Correct premultiplied display.** Viewer transforms now unpremultiply,
  transform straight color, then re-associate alpha. Semi-transparent pixels no
  longer disagree between the viewer and PNG output.
- **ACES 2.0 Rec.709 default view.** New projects use the ACES 2.0 SDR 100-nit
  Rec.709 view on the sRGB display.
- **Project settings, schema v7.** `Edit → Project settings…` (`S`) exposes the
  bundled OCIO config, ACEScg working space, display, saved default view, and
  black/checker viewer background. Settings edits are validated, atomic,
  undoable, saved in the project, and available through the Dispatcher protocol.
- **Deterministic slow-playback coverage.** The transport regression now delays
  the tile executor actually used by the viewer and renders a tiny graph, so
  hosted-runner raster speed cannot decide whether the test passes.

# NodeBased 0.10.0 — animation curves

## What changed since 0.9.1

- **Animation curves (document schema v6).** Any numeric node parameter can hold
  a curve of keys with `constant` or `linear` interpolation. Curves are an
  evaluation-time overlay: a node's stored parameters are never mutated, so a
  graph with no curves renders byte-identically to a v5 document.
- **Endpoint hold.** Before the first key and after the last, a curve holds that
  key's value, matching Nuke's extrapolation.
- **Atomic curve editing.** `set_key`, `delete_key` and `clear_curve` are
  Dispatcher operations with undo/redo, and deleting a node drops its curves in
  the same transaction, so a document can never reference a curve on a node that
  is gone. The agent CLI exposes the same operations.
- **Animation reaches the tile viewport.** Curves resolve at the tile executor's
  API boundary, so animated parameters render identically through the tile path
  and the reference evaluator, at every proxy tier.
- **Transparency displays over black.** The viewer composites transparent and
  partially transparent regions over pure black instead of a checkerboard. The
  checker tinted every pixel it showed through, so soft alpha edges displayed
  brighter than the graph produced them.

## Upgrade

Documents at schema v1 through v5 upgrade in place on open, chained through to
v6, gaining an empty animation section. No stored parameter changes, so an
upgraded comp renders exactly as it did before.

## Acceptance evidence

357 tests pass offscreen on the release commit. Animated parameters are pinned
against the reference evaluator at frames 1/5/10 across proxy tiers 1/2/4, and
the frames are asserted to differ, so a parameter frozen at its base value
cannot pass by matching an equally frozen reference.

## Known scope

Animated `Transform` and `Crop` render correctly but take the full-frame
fallback rather than the tile path; those kernels are not tile-supported yet.
Desktop verification is offscreen, so interactive playback feel on real
hardware is unverified.

# NodeBased 0.9.1 — Windows cache-root repair

## What changed since 0.9.0

- **Windows release-test repair.** Disk-cache initialization now survives
  sanitized Windows environments with no profile variables, using `TEMP` or
  the process directory as a safe fallback. A regression test covers it.

# NodeBased 0.9.0 — tile-native viewport and EXR data windows

## What changed since 0.8.0

- **Bounded tile evaluation.** Supported graphs evaluate in 256px tiles, with
  halo-aware Blur and explicit fallback telemetry for unsupported kernels.
- **EXR data windows survive.** Read/evaluation/tile requests preserve offsets
  and overscan beyond the display frame, including negative origins.
- **Bounded source Reads.** Full-resolution tile requests acquire the requested
  source region instead of decoding then slicing a full image.
- **Visible viewport scheduling.** The Viewer requests its visible data-window
  rectangle; panning requests newly exposed pixels. Export remains a complete
  full-resolution reference render.
- **Cache/runtime foundation.** Adaptive 8GiB-max RAM cache, persistent disk
  spill, full/half/quarter proxy tiers, and 24fps delivery controls.

## Acceptance evidence

At 4K on the recorded Linux CPU host, a centered 1920×1080 tile viewport on a
supported graph measured 312.232ms cold and 208.855ms grade-edit p50, versus
2404.175ms and 2052.728ms for the full reference frame. See
`docs/BENCHMARKS-v0.9-4k.md` for scope and machine details.

# NodeBased 0.8.0 — bounded playback and read-ahead

## What changed since 0.7.0

- **Forward playback.** Space or the transport button plays the inclusive comp
  range at document FPS and loops at the end.
- **Bounded read-ahead.** One display request is prioritized ahead of at most
  three future frames. Evaluation stays serial so one worker owns the existing
  LRU cache.
- **Deadline behavior.** The playhead derives from elapsed wall time. Slow
  frames are counted and skipped instead of building a latency spiral.
- **Cancellation and display safety.** Scrubs, edits, channel/view changes and
  stop cancel obsolete work. A result must match both the active generation and
  current frame before it may enter the Viewer; prefetch results only warm cache.
- **Undo-safe transport.** Playback updates the persisted playhead through the
  validated command boundary without consuming the 100-slot artist undo stack.
- **Explicit proxy boundary.** This release only schedules full-quality frames.
  Unimplemented proxy tiers are rejected rather than faking a proxy by scaling
  an already fully-evaluated Viewer image.

## Acceptance evidence

The playback contract in `docs/PLAYBACK.md` fixes a 16 ms enqueue budget, a
three-frame queue cap, cooperative cancellation, stale-frame exclusion, and
wall-clock deadline behavior. Automated coverage exercises the queue, real EXR
sequence cache warming, UI transport, undo preservation and wrong-frame rejection.

# NodeBased 0.7.0 — time foundation and image sequences

## What changed since 0.6.5

- **Time is now first-class.** Schema v5 stores a composition frame range,
  current frame, and frame rate. The evaluator receives an explicit timeline
  frame, leaving a clean boundary for future clips, retimes, tracks, nested
  timelines, and animated parameters.
- **Image-sequence Read.** Read accepts printf (`plate.%04d.exr`) and hash
  (`plate.####.exr`) patterns, a source-frame offset, and explicit missing-frame
  behavior: error, hold nearest, or transparent black at the sequence format.
- **Time-aware caching.** A sequence frame changes the Read fingerprint and its
  dependent branch. Static sources and unrelated branches retain their cached
  pixels while the playhead moves.
- **Minimal timeline strip.** First/current/last controls and scrubbing sit under
  the Viewer. Left/Right step; Home/End jump to the range boundaries.
- **Agent time control.** `describe` advertises time/sequence capabilities,
  `time` edits the range atomically, and headless `render` accepts a frame.
- **Windows installer identity.** The NSIS installer/uninstaller and Start Menu
  shortcut now explicitly use the NodeBased icon.

## Validation

Schema migration, sequence resolution, frame offsets, missing-frame policies,
static/temporal cache behavior, timeline UI/undo synchronization, and a real
frame-specific agent render are covered by the full automated suite.

# NodeBased 0.6.5 — the NodeBased icon

## What changed since 0.6.4

- **NodeBased has its app icon.** A folded green ribbon forming an `N`, on a
  dark rounded tile, with real transparent alpha outside the tile. Used by the
  desktop window, Windows executable/installer, and Linux AppImage.

# NodeBased 0.6.4 — visible reroute handles

## What changed since 0.6.3

- **Ctrl now visibly exposes every noodle's reroute handle.** Each connected
  noodle gets a high-contrast circular midpoint marker while Ctrl is held. The
  marker responds immediately to the global Ctrl key event, even when Qt focus
  is elsewhere in the workspace.
- **Dots behave as reroutes.** A Dot is a compact circular node with a visible
  selected outline. Its body is selectable and draggable after creation; its
  sockets remain at the top and bottom rather than covering the center.
- **App icon plumbing landed, placeholder graphic.** The desktop window,
  Windows executable/installer, and Linux AppImage now load an icon file from
  `assets/`, so a final mark drops in with no code change. The mark bundled in
  this release is an early draft that was rejected for being too detailed;
  design is still in progress and this asset will be replaced.

## Validation

141 automated tests cover Ctrl-handle state, Dot selection and movement,
transactional noodle insertion, source asset loading, and the existing graph,
compositor, media, agent, updater, and packaging behavior.

# NodeBased 0.6.3 — Dot crash repair and reliable graph sockets

## What changed since 0.6.2

- **Fixed the Dot/Switch graph crash.** Their missing theme colors caused a
  scene rebuild exception after creation, leaving a partially cleared graph and
  apparently vanished noodles. Both nodes now render safely.
- **Dots are compact reroute points.** Ctrl-dragging a noodle midpoint shows a
  live Dot preview that follows the cursor and only changes the graph on drop.
- **Larger socket hit targets.** Visible ports remain small, but their grab area
  is 26 px; the graph recognizes the visible socket child as its port target so
  active wires cannot be accidentally dropped on it.

## Validation

138 automated tests cover Dot creation without clearing graph edges, live Ctrl
preview placement, atomic insertion, symmetric wiring, and the full compositor
and packaging suite.

# NodeBased 0.6.2 — safe noodles and viewer controls

## What changed since 0.6.0

- **Safe input rewiring.** Picking up an existing input preserves its original
  connection until a valid output is dropped. Esc or a drop on empty canvas
  leaves the comp intact; a completed gesture replaces only that input.
- **Ctrl-drag a noodle midpoint to insert a Dot.** Holding Ctrl reveals a center
  handle on every connected noodle. Drag one to place a Dot that is atomically
  spliced into the connection, so it cannot leave a half-disconnected graph.
- **Viewer hotkeys:** `R`, `G`, `B`, and `A` solo their channel; pressing the
  currently soloed channel again returns to RGB. `F` fits the image; `H` is the
  matching home/frame alias for the single-image 2D viewer.

## Validation

137 automated tests cover direct/reverse wiring, transactional input rewire,
Ctrl midpoint Dot insertion, channel toggles, and framing keys, in addition to
the compositor, media, schema, agent, and updater suites.

# NodeBased 0.6.0 — masks, mix, graph routing and selection

## What changed since 0.5.1

- **Reusable mask + mix controls** now apply to Grade, ColorCorrect, Blur,
  Transform and Crop. Their optional `mask` input gates the processed result by
  mask alpha, and `mix` blends it back with the source. Existing projects retain
  the exact prior behavior: an absent mask and `mix: 1.0` are the defaults.
- **Dot** adds a zero-cost graph-routing passthrough. **Switch** selects between
  two image inputs with its `which` control; both are available to keyboard node
  search and to the agent interface.
- **Document schema v4** upgrades v3 projects by adding optional filter-mask
  slots and default mix values. v1 through v3 projects continue through the
  tested upgrade chain.
- Agents can discover Dot, Switch, filter mask ports and mix parameters through
  `describe`, and build/render schema-v4 graphs through the local JSON-lines
  interface.

## Validation

135 automated tests cover schema migration, premultiplied mask/mix behavior,
Dot/Switch evaluation, desktop graph behavior, and agent graph construction.
Native display validation of the new optional mask ports remains a human QA item.

# NodeBased 0.5.1 — graph interaction hotfix

## What changed since 0.5.0

- Restores normal **round-capped noodles** at a more readable 3.25 px stroke.
  Arrowheads are painted separately, preventing Qt from filling the curve into
  a ribbon.
- Inputs now wire in **either direction**: drag an output to an input, or drag
  an empty/connected top input to an output. A 24-screen-pixel magnetic target
  makes ports much less finicky at any graph zoom.
- **Tab is pointer-contextual.** When the mouse is over the node graph, Tab is
  captured before Qt focus traversal and opens node search even if a dock or
  control currently owns keyboard focus.

# NodeBased 0.5.0 — graph interaction repair

## What changed since 0.4.0

- **Direct noodle drag wiring.** Drag from any output port directly onto an
  input port to connect. The connection preview follows the pointer and the
  destination resolves on mouse release, so wires no longer depend on a fragile
  click sequence. Existing click-to-connect and click-a-connected-input-to-pick-
  up-and-rewire remain available.
- **Legible directional connections.** Saved and pending noodles now terminate
  in small arrowheads that point into their input. Ports stay above noodles in
  the drawing order, making them easy to grab.
- **Centered top inputs.** A single input is exactly at a node's top centre;
  multiple inputs fan symmetrically around that centre. Outputs stay centered
  at the bottom.
- **Nuke-style Tab search.** Tab opens a keyboard-first popup search at the
  last graph click. Enter creates the selected node there. The placement code
  finds the nearest vacant slot, so a new node never lands on top of an
  existing node. Single-key creation still works.

The interaction coverage now includes direct drag-to-connect, input-centering,
non-overlapping placement, and Tab-search filtering, alongside the existing
rewire/disconnect tests.

# NodeBased 0.4.0 — Full Merge operations, real Transform, Premult/Unpremult

Fourth desktop release. Where 0.3.0 added new nodes, this one makes the
compositing core itself honest: the operations an artist actually reaches for
now behave the way they do in Nuke.

## What changed since 0.3.0

- **All 16 Nuke-style Merge operations.** Merge gains an `operation` choice:
  over, under, plus, minus, multiply, screen, max, min, difference, divide,
  mask, stencil, in, out, atop, xor. Inputs are premultiplied, A is the
  foreground and B is the background, matching Nuke's wiring convention.
  `divide` guards against zero. The 0.3.0 `over`/mix behaviour is preserved
  byte-identical, so existing comps render exactly as before.
- **Transform is a real transform.** Float `translate_x`/`translate_y`,
  `rotate` in degrees, `scale`, and a `center`, with a filter choice of
  nearest, bilinear or cubic. It is inverse-mapped and sub-pixel filtered, and
  does not wrap at the edges. Defaults reduce to identity, so 0.3.0 projects
  render unchanged.
- **Premult and Unpremult** are new single-input nodes. Unpremult leaves RGB
  untouched wherever alpha is 0, so it never emits NaN or inf into the rest of
  the graph.
- **Document schema v3, with a tested upgrade path.** Opening a v2 project
  converts old integer Transform `x`/`y` into float `translate_x`/
  `translate_y`, defaults rotate/scale/center/filter to identity, and gives
  Merge `operation: "over"` where it is missing. The v1 → v2 → v3 chain is
  covered by tests. The .nbcomp compatibility promise holds: projects saved by
  0.1.0, 0.2.0 and 0.3.0 all still open.
- The new node types, parameters and choices reach `agent.describe` and the
  inspector automatically, from the same SPECS/LIMITS/CHOICES tables the
  desktop UI reads. No separate agent-protocol update was needed.

## What changed in 0.3.0

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
