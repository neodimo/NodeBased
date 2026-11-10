# Roto and RotoPaint UI parity with Nuke

Baseline facts are sourced from the Nuke 17.0v3 RotoPaint chapter, its linked tool pages, the official Reference Guide, and the Viewer keyboard reference. Roto uses the same tools where the guide says so. Each inventory row is measured against NodeBased code. Score: match = 1, differs = 0.5, missing = 0; this is a coverage estimate, not a pixel/usability rating. See `docs/parity/nuke_roto_ui.json` and `docs/parity/nodebased_roto_ui.json` for complete inventories and citations.

## Measured counts

| Section | Match | Differs | Missing | Not applicable | Score |
|---|---:|---:|---:|---:|---:|
| Toolbar and tools | 32 | 0 | 0 | 0 | 100.0 |
| Tool settings | 189 | 0 | 0 | 1 | 100.0 |
| Stroke/shape list | 19 | 0 | 0 | 1 | 100.0 |
| Viewer editing | 31 | 0 | 0 | 0 | 100.0 |
| Timing and animation | 10 | 0 | 0 | 0 | 100.0 |
| Overall | — | — | — | — | 100.0 |

Not applicable rows stay in the inventory and are counted, but take no part in the score: they are neither a hit nor a miss. The two are the Clone tool's View control and the list's View column, which Nuke shows only for multi-view (stereo) projects; NodeBased has no multi-view projects.

## Largest gaps

None are left against the traversed guide pages. What the ledger does not cover: the tools past Nuke (edge snapping, point tracking, in-betweens, the GPU matte), playback and ahead-of-time evaluation.

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
| 21 | R1 · viewer side | Transform-box center/pivot | match | `Viewer._roto_box_action`, `_roto_pivot_updates` and `_commit_roto_drag`; centre drag preserves rendered points and is covered by `tests.test_roto_toolbar_r1`. |  |
| 22 | R1 · viewer side | Rectangle | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 23 | R1 · viewer side | Transform-box pivot modify | match | `Viewer.mousePressEvent` maps Ctrl/Cmd+Shift drag on the transform handle to pivot modification. |  |
| 24 | R1 · viewer side | Cusped Rectangle | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 25 | R1 · viewer side | Nudge point | match | `Viewer.keyPressEvent` numeric keypad arrows | A keypad arrow nudges the selected point by one pixel. |
| 26 | R1 · viewer side | Open Spline | match | `RotoViewerToolbar`; `Viewer.hit_release`; `Viewer.finish_roto_draw` | The selected drawing tool commits the corresponding Bezier geometry. |
| 27 | R1 · viewer side | Draw Bezier/B-Spline freely | match | Viewer.hit_press; Viewer.hit_move; Viewer.hit_release; Viewer.finish_roto_draw | Ctrl/Cmd+drag sketches and closes the selected Bezier or B-Spline. |
| 28 | R1 · viewer side | Shapes tool cycle | match | `RotoViewerToolbar`; `Viewer.keyPressEvent` | The toolbar button selects its group and its shortcut cycles the tools. |
| 29 | R1 · viewer side | Bezier/B-Spline handles/tension | match | `Viewer.mouseReleaseEvent` drawing drag creates cubic handles; `Viewer.hit_press` Ctrl/Cmd+Alt-drag on a B-Spline point sets its animatable `tension`; `roto.tensioned_controls`, `roto.flatten_points` | The guide names the gesture but gives no scale or formula; this is a convention (see RW2 conventions below). |
| 30 | R1 · viewer side | Brush | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 31 | R1 · viewer side | Sharp exit while drawing | match | `Viewer.hit_release` Shift+LMB | Shift+LMB commits a sharp open-spline end. |
| 32 | R1 · viewer side | Eraser | match | `RotoViewerToolbar`; `Viewer.paint_tool` drives the viewer gesture. | The selected viewer tool is the operation used by paint gestures. |
| 33 | R1 · viewer side | B-Spline tension | match | `Viewer.hit_press` Ctrl/Cmd+Alt-drag on a B-Spline point sets its animatable `tension`; `roto.tensioned_controls`, `roto.flatten_points` | The guide names the gesture but gives no scale or formula; this is a convention (see RW2 conventions below). |
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
| 51 | R1 · viewer side | Tool mode cycle | match | `Viewer.keyPressEvent`: S (no modifiers, Viewer focus) steps through the tools of the active group. | The guide: "S cycles through the modes of the currently selected tool" (working_toolbars.html). |
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
| 203 | R2 · viewer side | tool · View | not applicable | No view list: a project here has one view (Nuke shows the control only for multi-view projects). | Not applicable: the Clone tool's View control picks the stereo or multi-view source and the guide shows it only when the project has multiple views; NodeBased has no multi-view projects. It takes no part in the score. |
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
| 249 | R3 · panel side | Color | match | `rotolist.ShapeListTree` Color column (`*-row-color-*`); `paint.rasterise` draws a RotoPaint shape in its own colour (its alpha scales the coverage) |  |
| 250 | R3 · panel side | Invert | match | `rotolist.ShapeListTree` Invert column (`*-row-invert-*`); shapes invert their matte in `roto.rasterise`, strokes invert their footprint in `paint.apply_stroke` |  |
| 251 | R3 · panel side | Blending | match | `rotolist.ShapeListTree` Blending column (`*-row-blend-*`); `shapes.BLEND_MODES` (Roto) and `shapes.PAINT_BLEND_MODES` (RotoPaint); `roto.blend_onto` |  |
| 252 | R3 · panel side | Motion blur | match | `rotolist.ShapeListTree` Motion blur column (`*-row-motion_blur-*`) |  |
| 253 | R3 · panel side | View | not applicable | No View column: a project here has one view (Nuke shows the column only for multi-view projects). | Not applicable: the View column lists the views an item appears in and the guide shows it only for multi-view projects; NodeBased has no multi-view projects. It takes no part in the score. |
| 254 | R3 · panel side | Lifetime | match | `rotolist.ShapeListTree` Lifetime column (`*-row-lifetime-*`): shows the type and range, opens the panel's lifetime editor |  |
| 255 | R3 · panel side | Source | match | `rotolist.ShapeListTree` Source column (`*-row-source-*`) |  |
| 256 | R3 · panel side | Layers and folders | match | `rotolist.ShapeListTree` tree, front-most item first at every level as in Nuke (`rotolist.reverse_tree`; the stored order stays render order, so saved shots render as before): layers as folders, drag in/out and reorder, a layer's Visible and Lock apply to its children; a layer's own transform (`shapes.resolve_layer_transform`) is applied to every shape inside it, edited from the panel's Layer transform controls and by the transform box with the layer row selected |  |
| 257 | R3 · panel side | Add new layer | match | `rotolist.ShapeListTree`.cmd_add_layer, context menu `roto-list-add-layer` |  |
| 258 | R3 · panel side | Duplicate | match | `rotolist.ShapeListTree`.cmd_duplicate, context menu `roto-list-duplicate` (items and layers with their contents) |  |
| 259 | R3 · panel side | Copy > Curve | match | `rotolist.ShapeListTree`.cmd_copy, context menu Copy > Curve | Strokes have no curve and are skipped. |
| 260 | R3 · panel side | Cut > Curve | match | `rotolist.ShapeListTree`.cmd_cut, context menu Cut > Curve | Strokes have no curve and are skipped. |
| 261 | R3 · panel side | Paste > Spline | match | `rotolist.ShapeListTree`.cmd_paste, context menu Paste > Spline (pastes across Roto and RotoPaint) |  |
| 262 | R3 · panel side | Delete | match | `rotolist.ShapeListTree`.cmd_delete, context menu `roto-list-delete` (a layer takes its contents with it) |  |
| 263 | R3 · panel side | Set lifetime / frame range | match | `rotolist.ShapeListTree`.cmd_set_lifetime, context menu dialog `roto-list-set-lifetime` |  |
| 264 | R3 · panel side | Planar-track | match | `rotolist.ShapeListTree`.cmd_planar_track, context menu `roto-list-planar-track` (the panel's existing track action) |  |
| 265 | R4 · panel side | Autokey | match | `RotoViewerToolbar.settings_strip`; node-level `tool_settings.__global__.autokey` drives viewer edits. |  |
| 266 | R4 · panel side | Set key: spline | match | `RotoPanel.build_panel` and `build_paint_timing_controls`; selected spline count, navigation, and set controls. |  |
| 267 | R4 · panel side | Delete key: spline | match | `RotoPanel.build_panel` and `build_paint_timing_controls`; selected spline keys delete at the current frame. |  |
| 268 | R4 · panel side | Set/delete point keys | match | `RotoPanel.build_panel` and `build_paint_timing_controls`; selected-point set/delete controls for both nodes. |  |
| 269 | R4 · panel side | Set key: transform | match | `RotoPanel.build_panel` and `Window.build_node_panel`; per-field keys on Roto and RotoPaint transforms. |  |
| 270 | R4 · panel side | Lifetime types | match | Roto and RotoPaint panels expose All, Start to, Single, To end and Range for shapes and strokes. |  |
| 271 | R4 · panel side | Lifetime endpoints | match | Roto and RotoPaint panels show From/To only when used by the selected lifetime mode. |  |
| 272 | R4 · panel side | Ripple editing | match | Select tools' settings strip: Ripple edit, Ripple mode (all, from start, to end, range) and Ripple from/to; `rotoripple.ripple_shapes` applied in `Viewer._commit_roto_shapes` for drags and keypad nudges; a red border on the Viewer while on. |  |
| 273 | R4 · panel side | Per-shape keyframe controls | match | Roto and RotoPaint panels expose All, Spline, Transform and Attributes key scopes for the selected shape item. |  |
| 274 | R4 · panel side | Motion blur keys | match | Samples, shutter and offset key buttons in the Roto panel; the list's Motion blur cell shows the shutter's key state and, with autokey on, sets a shutter key at the current frame (`rotolist.ShapeListTree.edit_motion_blur`) | A shutter key of 0 is blur off from that frame. |

## RW2 conventions

Step RW2 closed four differences the Roto walkthrough (`tools/roto_walkthrough.py`) found. Where the guide gives no formula the choice below is a *convention*, marked as such.

- **B-Spline tension** (ledger rows 29 and 33). Ctrl/Cmd+Alt-drag on a B-Spline point changes that point's `tension` (a number from -1 to 1, default 0, animatable like the other point scalars, one undo step). The guide names the gesture and gives no scale or formula, so *convention*: 100 screen pixels of horizontal drag is a tension of 1, and a point's effective control point is `c = P + 0.5 * t * (P - M)`, M the midpoint of the point's two neighbours (an open spline's end point is its own outside neighbour). Tension 1 puts the curve through the point, tension -1 pulls the control halfway to M for a looser curve, tension 0 is the plain uniform cubic B-Spline, so a point without tension renders exactly as before. Open B-Spline strokes now spread each point's feather offset over the curve by the same weights.
- **S key** (row 51). With the pointer over the Viewer, S steps to the next tool in the active group (the guide's "modes of the currently selected tool"); Select is the group when none is active. Shortcut clash check: the Node Graph binds S to create a Shuffle, but only when the graph has the keyboard, so the Viewer's S does not reach it; no other Viewer binding uses an unmodified S (Ctrl+S is Save, and a modified S does not cycle). Listed in the Viewer shortcut table.
- **Layer transform** (row 256). A layer stores an optional `transform` with the shape transform's fields (translate, rotate, scale, skew, skew order, centre), all animatable, identity by default; a layer without one renders bit-identically to before. It is applied to every shape inside the layer after the shape's own transform, then each enclosing layer from the innermost out. The centre defaults to the middle of the points the layer holds, taken before any layer transform (*convention*; a stored centre wins). It is edited from the Layer transform group in the Roto panel when a layer row is selected, and by the transform box (move, scale, rotate, pivot) which then frames every point of the layer. As with a shape's own transform, dragging a single point of a transformed shape in the Viewer sets the point to the position under the cursor without undoing the transform.
- **Blending list** (row 251). Roto's Blending cell offers the first ten modes as before plus the rest of the Merge node's operations: under, divide, mask, stencil, in, out, atop, xor, average, hypot, matte, disjoint-over, conjoint-over, copy, geometric, overlay, hard-light, soft-light, color-dodge and color-burn. *Convention*: neither the guide nor the parity inventory lists Roto's modes, so the list is Merge's. Each runs through Merge's own `_merge_op` on premultiplied RGBA (A is the shape's colour times its coverage with the coverage as alpha, B is what is below), with no new blend arithmetic.
- **Motion blur cell** (row 274). The list's Motion blur cell shows the item's shutter key state (a coloured cell: keyed on this frame, or animated with a key elsewhere). With autokey on, ticking it sets a key at the current frame on the item's motion-blur shutter (its value, or the default 0.5 when that is 0) and ticks the flag; clearing it keys the shutter to 0, blur off from that frame; on an item that is not animated yet, clearing it just clears the flag. With autokey off it only sets the flag. Ripple editing stays out: the guide documents no ripple workflow (row 272).
- **Transform box fix.** The Viewer's transform box was built from `QRectF.united` over one-point rectangles, which Qt treats as empty, so the box collapsed to a point and could not be grabbed with the mouse. It now frames the selected points.

## RW3 conventions

Step RW3 made the shape list read as Nuke's does and added the "Past Nuke" tour (walkthrough scenes 17 to 19).

- **List order** (row 256). The top row of the list is the front-most item, at every level (layers and the items inside them), and a new shape or stroke appears at the top of its layer. The stored order stays render order, back first, so no schema changed and saved shots render bit-identically; only the view and the edits that come from it read the other way round. Dropping an item above another puts it in front of that one; Duplicate puts the copy directly above the original, in front of it; Paste goes in front of the selected item; an item dropped into a layer lands at the top of it; Move up brings the item forward. Selection sync and the undo steps are unchanged. The View column (row 253) stays out of scope: Nuke shows it for multi-view projects only and NodeBased has none, so the walkthrough reports it as not applicable.
- **Past Nuke tour.** Scenes 17 (Snap to edges and Fit to edges), 18 (Track points) and 19 (Fill in-betweens) run on pictures computed from formulas, with a caption under each screenshot in the pack's README.

## RW4 conventions

Step RW4 closed the last six rows.

- **Ripple editing** (row 272). Nuke's "Deleting or Rippling Keyframes" page puts a ripple edit button in the Select tools' settings with a menu of all, from start, to end and range. The settings strip of the four Select tools now has Ripple edit (off), Ripple mode (all) and, for the range choice, Ripple from and Ripple to; they are stored once per node with Autokey and, being defaults, are absent from older documents. With it on, a point edit made in the Viewer (a drag, a transform box edit or a keypad nudge) moves each of that point's other keys inside the range by the offset the edit gave it at the current frame; keys outside the range stay put, a point with no keys simply takes the new value, and a shape that gained or lost points is left as edited. The whole change is one command, so one undo restores every key. All is the project's first to last frame, from start is the first frame to the current one, to end is the current frame to the last. A red border around the Viewer shows the mode is on. Fit to edges ripples like a hand edit (see ROTO_TRACKING.md); Track points and Fill in-betweens write keys of their own over a range, so they do not ripple.
- **RotoPaint shape Color and Invert** (rows 249 and 250). A RotoPaint shape item stores an optional `color`, drawn as the shape's colour (its alpha scales the coverage); a shape without one still paints white. Invert already inverted a shape's matte; a stroke now takes an `invert` flag too (the guide's list inverts "shape/stroke/group") and then affects everything outside its own footprint. The list's Color and Invert cells are live on both kinds.
- **RotoPaint blending list** (row 251). `shapes.PAINT_BLEND_MODES` is the four classic modes (over, add, multiply, screen, whose formulas and results are unchanged) followed by every other mode of Roto's list. Those are drawn by `roto.blend_onto`, which holds Roto's own formulas, so a RotoPaint shape over a picture reads the same as the second shape in Roto; like a Roto shape, the mode acts only inside the shape's bounds (or the stroke's reach), and an inverted item reaches the whole frame. The settings strip's Blending mode menu follows the node kind.
- **View rows** (rows 203 and 253). Both are stereo-only and marked not applicable; see the counts above.

