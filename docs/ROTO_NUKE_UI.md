# Roto and RotoPaint UI parity with Nuke

Baseline facts are sourced from the Nuke 17.0v3 RotoPaint chapter, its linked tool pages, the official Reference Guide, and the Viewer keyboard reference. Roto uses the same tools where the guide says so. Each inventory row is measured against NodeBased code. Score: match = 1, differs = 0.5, missing = 0; this is a coverage estimate, not a pixel/usability rating. See `docs/parity/nuke_roto_ui.json` and `docs/parity/nodebased_roto_ui.json` for complete inventories and citations.

## Measured counts

| Section | Match | Differs | Missing | Score |
|---|---:|---:|---:|---:|
| Toolbar and tools | 32 | 0 | 0 | 100.0 |
| Tool settings | 189 | 1 | 0 | 99.7 |
| Stroke/shape list | 15 | 4 | 1 | 85.0 |
| Viewer editing | 27 | 3 | 1 | 91.9 |
| Timing and animation | 1 | 8 | 1 | 50.0 |
| Overall | — | — | — | 96.1 |
## Largest gaps

1. Transform-box center/pivot behavior still differs from Nuke.
2. B-Spline tension gestures remain open; the official guide names the gesture but does not specify a tension mapping.
3. Clone view selection depends on multi-view projects, which NodeBased does not support.
4. Timing and animation controls do not reproduce Nuke’s complete per-shape/per-point workflow.
6. The stroke/shape list still lacks the View column, Nuke’s full blending list, a Color cell for RotoPaint shapes and an Invert cell for strokes.
7. Layer group transforms and Nuke’s newest-on-top list order are not reproduced.
## Proposed implementation steps

| Step | Side | Scope |
|---|---|---|
| R1 | **viewer side** | Shape tools, selection/point tools, drawing/editing gestures and bindings. |
| R2 | **viewer side** | Per-tool settings strip and brush/clone/reveal settings. |
| R3 | **panel side** | Stroke/shape list columns, ordering, commands, folders and item controls. |
| R4 | **panel side** | Timing, animation keys, lifetime, autokey and ripple behavior. |

Each step is a bounded 40–60 minute implementation slice; the ledger below is the source of its row-level scope.

## Source and scope notes

- The official Reference Guide p.212 gives the list order: Name, Visible, Lock, Overlay color, Color, Invert, Blending, Motion blur, View (stereo-only), Lifetime, Source. The reference inventory records the full order and commands.
- Guide pages do not state a fixed default tool for each toolbar group. NodeBased chooses Bezier for a fresh Roto viewer and Brush for RotoPaint; the Nuke reference default remains unspecified.
- The official RotoPaint chapter and Reference Guide searched here do not describe ripple editing; the inventory marks its reference as undocumented in these official sources, not as a claim that no version has it.
- “View” is stereo-only in the guide; NodeBased has no stereo view control, so it cannot match in the current product scope.
- All linked chapter pages and the official `hotkeys_studio.html` keyboard page loaded. Guessed keyboard-page aliases returned 404; the actual official page is included. The Reference Guide pages were extracted from the official 17.0v3 PDF.

## Gap ledger

| Rank | Step / side | Nuke item | Status | NodeBased location | Exact difference |
|---:|---|---|---|---|---|
| 1 | R1 · viewer side | Select All | match | `RotoViewerToolbar`; `Viewer.hit_press`; `Viewer.hit_release` | Click and marquee selection are restricted to the active selection mode. |
| 2 | R1 · viewer side | Select Splines | match | `RotoViewerToolbar`; `Viewer.hit_press`; `Viewer.hit_release` | Click and marquee selection are restricted to the active selection mode. |
| 3 | R1 · viewer side | Select Points | match | `RotoViewerToolbar`; `Viewer.hit_press`; `Viewer.hit_release` | Click and marquee selection are restricted to the active selection mode. |
| 4 | R1 · viewer side | Select Feather Points | match | `RotoViewerToolbar`; `Viewer.hit_press`; `Viewer.hit_release` | Click and marquee selection are restricted to the active selection mode. |
| 5 | R1 · viewer side | Select several points | match | `Viewer.hit_press`; Ctrl/Cmd+LMB adds points; marquee selection is also supported. |  |
| 6 | R1 · viewer side | Select tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 7 | R1 · viewer side | Add Points | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 8 | R1 · viewer side | Feather add | match | `Viewer.keyPressEvent` E adds per-point feather offsets; Shift+E removes feather. |  |
| 9 | R1 · viewer side | Remove Points | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 10 | R1 · viewer side | Cusp Points | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 11 | R1 · viewer side | Feather reset | match | `Viewer.keyPressEvent` Shift+E | Shift+E removes feather from the selected points. |
| 12 | R1 · viewer side | Smooth Points | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 13 | R1 · viewer side | Remove Feather | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 14 | R1 · viewer side | Tangent move both | match | `Viewer.mouseMoveEvent` Shift-drag | Shift snaps the opposite tangent to the same length. |
| 15 | R1 · viewer side | Open/Close Curve | match | `RotoViewerToolbar`; `Viewer.hit_press` | Clicking with this active tool performs the point operation. |
| 16 | R1 · viewer side | Bezier | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 17 | R1 · viewer side | Cusped Bezier | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 18 | R1 · viewer side | B-Spline | match | `RotoViewerToolbar`; `Viewer.begin_roto_draw` and `Viewer.finish_roto_draw`; `roto.flatten_points` evaluates uniform cubic B-splines. | B-Spline shapes rasterize through the new evaluator; Bezier shapes keep their existing path. |
| 19 | R1 · viewer side | Transform box | match | `Viewer.keyPressEvent` T; `Viewer._roto_box_geometry` | T toggles the transform box for an active Roto selection. |
| 20 | R1 · viewer side | Ellipse | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 21 | R1 · viewer side | Transform-box center/pivot | differs | `Viewer._roto_box_result`; `RotoPanel.build_panel` shape transform center controls; interaction differs from Nuke handles. |  |
| 22 | R1 · viewer side | Rectangle | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 23 | R1 · viewer side | Transform-box pivot modify | match | `Viewer.mousePressEvent` maps Ctrl/Cmd+Shift drag on the transform handle to pivot modification. |  |
| 24 | R1 · viewer side | Cusped Rectangle | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 25 | R1 · viewer side | Nudge point | match | `Viewer.keyPressEvent` numeric keypad arrows | A keypad arrow nudges the selected point by one pixel. |
| 26 | R1 · viewer side | Open Spline | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 27 | R1 · viewer side | Draw Bezier/B-Spline freely | match | Viewer.hit_press; Viewer.hit_move; Viewer.hit_release; Viewer.finish_roto_draw | Ctrl/Cmd+drag sketches and closes the selected Bezier or B-Spline. |
| 28 | R1 · viewer side | Shapes tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 29 | R1 · viewer side | Bezier/B-Spline handles/tension | differs | `Viewer.mouseReleaseEvent` drawing drag creates cubic handles; no B-Spline tension tool or Ctrl/Cmd+Alt tension drag. |  |
| 30 | R1 · viewer side | Brush | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 31 | R1 · viewer side | Sharp exit while drawing | match | `Viewer.hit_release` Shift+LMB | Shift+LMB commits a sharp open-spline end. |
| 32 | R1 · viewer side | Eraser | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 33 | R1 · viewer side | B-Spline tension | differs | Uniform cubic evaluation is present; Ctrl/Cmd+Alt tension adjustment is still absent. |  |
| 34 | R1 · viewer side | Brushes tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 35 | R1 · viewer side | Close shape | match | `Viewer.keyPressEvent` Enter; `Viewer.mouseReleaseEvent` | Enter or clicking the first point closes the shape. |
| 36 | R1 · viewer side | Clone | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 37 | R1 · viewer side | Leave shape open | match | `Viewer.keyPressEvent` Escape with Open Spline selected | Escape commits the spline as open. |
| 38 | R1 · viewer side | Reveal | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 39 | R1 · viewer side | Tool change | match | `RotoViewerToolbar.choose` | Changing tools finalizes an open spline and activates the selected tool. |
| 40 | R1 · viewer side | Clone/Reveal tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 41 | R1 · viewer side | Escape to selection | match | `Viewer.keyPressEvent` Escape with Open Spline active | Escape commits the open spline and returns to Select Points. |
| 42 | R1 · viewer side | Blur | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 43 | R1 · viewer side | Brush size | match | `Viewer.hit_press`, `Viewer.hit_move`, `Viewer.hit_release`; Shift+drag adjusts brush size and mirrors the panel field. |  |
| 44 | R1 · viewer side | Sharpen | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 45 | R1 · viewer side | Clone offset | match | `Viewer.hit_press`, `Viewer.hit_move`, `Viewer.hit_release`; Ctrl/Cmd+drag sets the offset; Ctrl/Cmd+Shift resets it. |  |
| 46 | R1 · viewer side | Smear | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 47 | R1 · viewer side | Reveal onion skin | match | RotoViewerToolbar.settings_strip; Viewer._reveal_onion_image; Viewer._paint_reveal_onion | The selected Reveal input is shown at the configured onion opacity; T toggles it. |
| 48 | R1 · viewer side | Blur/Sharpen/Smear tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 49 | R1 · viewer side | Selection tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar shortcut selects the group and repeated presses cycle its tools. |
| 50 | R1 · viewer side | Dodge | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 51 | R1 · viewer side | Tool mode cycle | missing | No S tool-mode cycle in `Viewer.keyPressEvent`. |  |
| 52 | R1 · viewer side | Burn | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 53 | R1 · viewer side | Paint tool cycles | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar shortcut selects the group and repeated presses cycle its tools. |
| 54 | R1 · viewer side | Dodge/Burn tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 55 | R2 · viewer side | tool · Label points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 56 | R2 · viewer side | tool · Hide splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 57 | R2 · viewer side | tool · Show paint stroke splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 58 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 59 | R2 · viewer side | tool · Hide transform handles | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 60 | R2 · viewer side | tool · Hide transform handles on move | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 61 | R2 · viewer side | tool · Constant selection mode | match | RotoViewerToolbar.settings_strip; Viewer.hit_press | Empty-space clicks preserve a selection while constant selection mode is enabled. |
| 62 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 63 | R2 · viewer side | tool · Label points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 64 | R2 · viewer side | tool · Hide splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 65 | R2 · viewer side | tool · Show paint stroke splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 66 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 67 | R2 · viewer side | tool · Hide transform handles | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 68 | R2 · viewer side | tool · Hide transform handles on move | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 69 | R2 · viewer side | tool · Constant selection mode | match | RotoViewerToolbar.settings_strip; Viewer.hit_press | Empty-space clicks preserve a selection while constant selection mode is enabled. |
| 70 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 71 | R2 · viewer side | tool · Label points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 72 | R2 · viewer side | tool · Hide splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 73 | R2 · viewer side | tool · Show paint stroke splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 74 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 75 | R2 · viewer side | tool · Hide transform handles | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 76 | R2 · viewer side | tool · Hide transform handles on move | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 77 | R2 · viewer side | tool · Constant selection mode | match | RotoViewerToolbar.settings_strip; Viewer.hit_press | Empty-space clicks preserve a selection while constant selection mode is enabled. |
| 78 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 79 | R2 · viewer side | tool · Label points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 80 | R2 · viewer side | tool · Hide splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 81 | R2 · viewer side | tool · Show paint stroke splines | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 82 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 83 | R2 · viewer side | tool · Hide transform handles | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 84 | R2 · viewer side | tool · Hide transform handles on move | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 85 | R2 · viewer side | tool · Constant selection mode | match | RotoViewerToolbar.settings_strip; Viewer.hit_press | Empty-space clicks preserve a selection while constant selection mode is enabled. |
| 86 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 87 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 88 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 89 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 90 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 91 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 92 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 93 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 94 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 95 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 96 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 97 | R2 · viewer side | tool · Hide points | match | RotoViewerToolbar.settings_strip; Viewer.paint_overlays |  |
| 98 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 99 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 100 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 101 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 102 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 103 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 104 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 105 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 106 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 107 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 108 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 109 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 110 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 111 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 112 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 113 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 114 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 115 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 116 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 117 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 118 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 119 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 120 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 121 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 122 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 123 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 124 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 125 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 126 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 127 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 128 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 129 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 130 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 131 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 132 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 133 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 134 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 135 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 136 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 137 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 138 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 139 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 140 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 141 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 142 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 143 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 144 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 145 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 146 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 147 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 148 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 149 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 150 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 151 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 152 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 153 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 154 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 155 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 156 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 157 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 158 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 159 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 160 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 161 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 162 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 163 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 164 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 165 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 166 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 167 | R2 · viewer side | tool · Source | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 168 | R2 · viewer side | tool · Blending mode | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 169 | R2 · viewer side | tool · Feather | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 170 | R2 · viewer side | tool · Feather on | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 171 | R2 · viewer side | tool · Feather falloff | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 172 | R2 · viewer side | tool · Feather type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 173 | R2 · viewer side | tool · Feather link | match | RotoViewerToolbar.settings_strip; Viewer._commit_roto_drag | With feather link off, moving a shape point keeps its feather point in place. |
| 174 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 175 | R2 · viewer side | tool · Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 176 | R2 · viewer side | tool · Color | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 177 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 178 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 179 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 180 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 181 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 182 | R2 · viewer side | tool · Pressure alters opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 183 | R2 · viewer side | tool · Pressure alters size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 184 | R2 · viewer side | tool · Pressure alters hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 185 | R2 · viewer side | tool · Build up | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 186 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 187 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 188 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 189 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 190 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 191 | R2 · viewer side | tool · Clone source transform translate X/Y | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 192 | R2 · viewer side | tool · Rotate | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 193 | R2 · viewer side | tool · Scale | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 194 | R2 · viewer side | tool · Skew X | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 195 | R2 · viewer side | tool · Skew Y | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 196 | R2 · viewer side | tool · Skew order | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 197 | R2 · viewer side | tool · Center X/Y | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 198 | R2 · viewer side | tool · Round to pixel | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 199 | R2 · viewer side | tool · Filter | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 200 | R2 · viewer side | tool · Black outside | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 201 | R2 · viewer side | tool · Time offset | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 202 | R2 · viewer side | tool · Time offset type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 203 | R2 · viewer side | tool · View | differs | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint | Selected-item and tool-specific parity remains incomplete. |
| 204 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 205 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 206 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 207 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 208 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 209 | R2 · viewer side | tool · Paint source | match | RotoViewerToolbar.settings_strip; Viewer._finish_paint | Paint source applies to new strokes and the selected stroke. |
| 210 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 211 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 212 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 213 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 214 | R2 · viewer side | tool · Time offset | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 215 | R2 · viewer side | tool · Time offset type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 216 | R2 · viewer side | tool · Onion skin | match | RotoViewerToolbar.settings_strip; Viewer._paint_reveal_onion | Reveal displays the selected source input as an onion-skin overlay. |
| 217 | R2 · viewer side | tool · Onion skin opacity | match | RotoViewerToolbar.settings_strip; Viewer._paint_reveal_onion | Onion-skin opacity controls the overlay opacity. |
| 218 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 219 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 220 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 221 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 222 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 223 | R2 · viewer side | tool · Effect | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 224 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 225 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 226 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 227 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 228 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 229 | R2 · viewer side | tool · Effect | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 230 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 231 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 232 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 233 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 234 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 235 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 236 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 237 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 238 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 239 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 240 | R2 · viewer side | tool · Opacity | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 241 | R2 · viewer side | tool · Brush size | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 242 | R2 · viewer side | tool · Brush spacing | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 243 | R2 · viewer side | tool · Brush hardness | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 244 | R2 · viewer side | tool · Lifetime type | match | RotoViewerToolbar.settings_strip; Viewer.finish_roto_draw / Viewer._finish_paint |  |
| 245 | R3 · panel side | Name | match | `rotolist.ShapeListTree` Name column (`roto-shape-row-name-*`, `rotopaint-row-name-*`, `*-layer-name-*`) |  |
| 246 | R3 · panel side | Visible | match | `rotolist.ShapeListTree` Visible column (`roto-shape-row-visible-*`, `rotopaint-row-visible-*`, `*-layer-visible-*`) |  |
| 247 | R3 · panel side | Lock | match | `rotolist.ShapeListTree` Lock column (`roto-shape-row-locked-*`, `rotopaint-row-locked-*`, `*-layer-locked-*`) |  |
| 248 | R3 · panel side | Overlay color | match | `rotolist.ShapeListTree` Overlay color column (`*-row-overlay-*`); stored as `overlay_color`, drawn by the Viewer outline, never rendered |  |
| 249 | R3 · panel side | Color | differs | `rotolist.ShapeListTree` Color column (`*-row-color-*`) | RotoPaint shape items carry no colour of their own, so their Color cell is disabled; strokes and Roto shapes edit theirs. |
| 250 | R3 · panel side | Invert | differs | `rotolist.ShapeListTree` Invert column (`*-row-invert-*`) | A RotoPaint stroke has no invert setting, so its cell is disabled; shapes edit theirs. |
| 251 | R3 · panel side | Blending | differs | `rotolist.ShapeListTree` Blending column (`*-row-blend-*`) | Roto offers ten blending modes and RotoPaint four, a subset of Nuke's list. |
| 252 | R3 · panel side | Motion blur | match | `rotolist.ShapeListTree` Motion blur column (`*-row-motion_blur-*`) |  |
| 253 | R3 · panel side | View | missing | No multiview support or view list in `RotoPanel.build_panel` / `Window.build_node_panel`. | Stereo-only column; NodeBased has no multi-view project support, so it stays out of scope. |
| 254 | R3 · panel side | Lifetime | match | `rotolist.ShapeListTree` Lifetime column (`*-row-lifetime-*`): shows the type and range, opens the panel's lifetime editor |  |
| 255 | R3 · panel side | Source | match | `rotolist.ShapeListTree` Source column (`*-row-source-*`) |  |
| 256 | R3 · panel side | Layers and folders | differs | `rotolist.ShapeListTree` tree: layers as folders, drag in/out and reorder, a layer's Visible and Lock apply to its children; `shapes.layer_states` | Group transforms of a layer and Nuke's newest-on-top ordering are not reproduced; the list order stays the render order. |
| 257 | R3 · panel side | Add new layer | match | `rotolist.ShapeListTree`.cmd_add_layer, context menu `roto-list-add-layer` |  |
| 258 | R3 · panel side | Duplicate | match | `rotolist.ShapeListTree`.cmd_duplicate, context menu `roto-list-duplicate` (items and layers with their contents) |  |
| 259 | R3 · panel side | Copy > Curve | match | `rotolist.ShapeListTree`.cmd_copy, context menu Copy > Curve | Strokes have no curve and are skipped. |
| 260 | R3 · panel side | Cut > Curve | match | `rotolist.ShapeListTree`.cmd_cut, context menu Cut > Curve | Strokes have no curve and are skipped. |
| 261 | R3 · panel side | Paste > Spline | match | `rotolist.ShapeListTree`.cmd_paste, context menu Paste > Spline (pastes across Roto and RotoPaint) |  |
| 262 | R3 · panel side | Delete | match | `rotolist.ShapeListTree`.cmd_delete, context menu `roto-list-delete` (a layer takes its contents with it) |  |
| 263 | R3 · panel side | Set lifetime / frame range | match | `rotolist.ShapeListTree`.cmd_set_lifetime, context menu dialog `roto-list-set-lifetime` |  |
| 264 | R3 · panel side | Planar-track | match | `rotolist.ShapeListTree`.cmd_planar_track, context menu `roto-list-planar-track` (the panel's existing track action) |  |
| 265 | R4 · panel side | Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 266 | R4 · panel side | Set key: spline | differs | `RotoPanel.build_panel` / shape key button; `Window.make_key_button` and `Window.build_curve_menu`. |  |
| 267 | R4 · panel side | Delete key: spline | differs | `Window.build_curve_menu` / Delete key at frame; `Window.make_key_button`. Key control presentation differs. |  |
| 268 | R4 · panel side | Set/delete point keys | differs | `RotoPanel.build_panel` `key_shape` keys the selected shape’s point scalars together; no individual point key control. |  |
| 269 | R4 · panel side | Set key: transform | differs | `RotoPanel.build_panel` per-transform key buttons; no RotoPaint spline-key control. |  |
| 270 | R4 · panel side | Lifetime types | differs | `RotoPanel.build_panel` / `roto-shape-lifetime-type`; `Window.build_node_panel` / `rotopaint-layer-lifetime`. |  |
| 271 | R4 · panel side | Lifetime endpoints | differs | Roto shape lifetime start/end fields; RotoPaint panel exposes range endpoints, with different control placement. |  |
| 272 | R4 · panel side | Ripple editing | missing | No ripple timing operation in `RotoPanel.build_panel` or `Window.build_node_panel`; Nuke chapter has no such workflow page. |  |
| 273 | R4 · panel side | Per-shape keyframe controls | differs | `RotoPanel.build_panel` / shape key button and scalar key controls; no same selected-item key scope menu. |  |
| 274 | R4 · panel side | Motion blur keys | differs | `RotoPanel.build_panel` / motion blur samples/shutter/offset controls; `Window.build_node_panel` selected-item fields. |  |
