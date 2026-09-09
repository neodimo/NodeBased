# NodeBased 0.1.0 — first desktop release

Native 2D compositing prototype with a node graph, dockable properties, viewer,
seven image nodes, scene-linear float RGBA, undo/redo, project saving and optional
live agent control.

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

This is the first release: the update button will report **Up to date** until a
newer stable version is published.

## Scope

M0 prototype, not a production compositor. PNG/JPEG in, PNG out. EXR/OCIO,
sequences, production caching/GPU performance, 3D and AI generation remain roadmap
work. macOS arm64 release comes later. Offscreen tests and packaged launch checks
do not establish native GPU/display performance or workplace-policy acceptance.
