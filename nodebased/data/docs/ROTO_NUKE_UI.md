# Roto and RotoPaint UI parity with Nuke

This baseline records guide facts and compares them with the current viewer and properties-panel implementation. The numeric score is a coverage estimate, not a pixel-perfect usability rating: each inventory entry contributes 1 for match, 0.5 for differs, and 0 for missing. Nuke page details still needing extraction are explicitly represented as provisional entries and should be verified before implementation.

## Measured counts

| Section | Match | Differs | Missing | Score |
|---|---:|---:|---:|---:|
| Toolbar and tools | 0 | 23 | 3 | 44.2 |
| Tool settings | 0 | 0 | 26 | 0.0 |
| Stroke/shape list | 0 | 11 | 0 | 50.0 |
| Viewer editing | 2 | 7 | 0 | 61.1 |
| Timing and animation | 0 | 6 | 0 | 50.0 |
| Overall | 2 | 47 | 29 | 32.7 |

## Largest gaps

1. No vertical RotoPaint tool groups in the viewer; tools live in a properties-panel dropdown.
2. No per-tool settings strip across the top of the viewer.
3. Brush settings are not context-switched by tool and appear in the node panel.
4. Nuke's shape-drawing tool group (Bezier variants, B-Spline, ellipse and rectangle variants) is absent as a toolbar group.
5. Nuke's point-edit tool group (add/remove/cusp/smooth/feather) is absent as a toolbar group.
6. Stroke/shape list is represented as a simpler layer editor rather than a columnar item list.
7. List-level overlay color and item color controls do not match Nuke's column interactions.
8. Shape invert, blending, motion blur, visibility, lock and lifetime are not presented as Nuke list columns.
9. Nuke's selection-mode cycling and tool-group cycling bindings differ from NodeBased's reduced tool switching.
10. Animation key workflows and shape/point context commands have incomplete parity.

## Proposed implementation steps

| Step | Side | Rows | Scope |
|---|---|---|---|
| R1 | viewer side | toolbar groups; Bezier/B-Spline/ellipse/rectangle; point tools; selection cycling | 40–60 minutes |
| R2 | viewer side | per-tool settings bar; brush controls; tool-specific settings | 40–60 minutes |
| R3 | viewer side | point gestures, feather/tangent editing, transform handles and bindings | 40–60 minutes |
| R4 | panel side | list columns, visibility/lock/overlay/color/invert/blend/motion blur/lifetime | 40–60 minutes |
| R5 | panel side | shape/stroke properties, grouping, context commands and keyframe workflow | 40–60 minutes |

Several Nuke details (exact list column ordering, individual tool-setting defaults, and some ripple/key commands) remain unverified from the cached guide text; they are excluded from definitive claims pending a page-by-page source extraction. The reference inventory names its source page for every recorded item.

## Gap ledger

| Rank | Step / side | Nuke entry | Status | NodeBased location | Gap |
|---:|---|---|---|---|---|
| 1 | R3 · viewer side | Add point | differs | nodebased/app.py: Viewer keyPressEvent, mousePressEvent/mouseMoveEvent and Roto overlay handlers | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 2 | R3 · viewer side | Delete point | differs | nodebased/app.py: Viewer keyPressEvent, mousePressEvent/mouseMoveEvent and Roto overlay handlers | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 3 | R3 · viewer side | Feather | differs | nodebased/app.py: Viewer keyPressEvent, mousePressEvent/mouseMoveEvent and Roto overlay handlers | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 4 | R3 · viewer side | Tangent handles | differs | nodebased/app.py: Viewer keyPressEvent, mousePressEvent/mouseMoveEvent and Roto overlay handlers | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 5 | R3 · viewer side | Transform box | differs | nodebased/app.py: Viewer keyPressEvent, mousePressEvent/mouseMoveEvent and Roto overlay handlers | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 6 | R3 · viewer side | Shape motion blur | differs | nodebased/app.py: RotoPaint layer editor and item edit actions | The reduced layer editor does not expose Nuke-equivalent per-shape motion-blur controls or spline clipboard context commands. |
| 7 | R3 · viewer side | Copy/cut/paste spline positions | differs | nodebased/app.py: RotoPaint layer editor and item edit actions | The reduced layer editor does not expose Nuke-equivalent per-shape motion-blur controls or spline clipboard context commands. |
| 8 | R1 · viewer side | Select All | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 9 | R1 · viewer side | Select Splines | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 10 | R1 · viewer side | Select Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 11 | R1 · viewer side | Select Feather Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 12 | R1 · viewer side | Add Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 13 | R1 · viewer side | Remove Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 14 | R1 · viewer side | Cusp Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 15 | R1 · viewer side | Smooth Points | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 16 | R1 · viewer side | Remove Feather | missing | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 17 | R1 · viewer side | Open/Close Curve | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 18 | R1 · viewer side | Bezier | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 19 | R1 · viewer side | Cusped Bezier | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 20 | R1 · viewer side | B-Spline | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 21 | R1 · viewer side | Ellipse | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 22 | R1 · viewer side | Rectangle | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 23 | R1 · viewer side | Cusped Rectangle | missing | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 24 | R1 · viewer side | Open Spline | missing | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 25 | R1 · viewer side | Brush | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 26 | R1 · viewer side | Eraser | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 27 | R1 · viewer side | Clone | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 28 | R1 · viewer side | Reveal | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 29 | R1 · viewer side | Blur | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 30 | R1 · viewer side | Sharpen | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 31 | R1 · viewer side | Smear | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 32 | R1 · viewer side | Dodge | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 33 | R1 · viewer side | Burn | differs | nodebased/app.py: Viewer.paint_tool and Roto/Properties viewer handlers; no Nuke-equivalent toolbar tool button | Tool is selected through a reduced properties-panel control or has no matching dedicated tool button. |
| 34 | R2 · viewer side | Select All | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 35 | R2 · viewer side | Select Splines | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 36 | R2 · viewer side | Select Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 37 | R2 · viewer side | Select Feather Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 38 | R2 · viewer side | Add Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 39 | R2 · viewer side | Remove Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 40 | R2 · viewer side | Cusp Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 41 | R2 · viewer side | Smooth Points | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 42 | R2 · viewer side | Remove Feather | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 43 | R2 · viewer side | Open/Close Curve | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 44 | R2 · viewer side | Bezier | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 45 | R2 · viewer side | Cusped Bezier | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 46 | R2 · viewer side | B-Spline | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 47 | R2 · viewer side | Ellipse | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 48 | R2 · viewer side | Rectangle | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 49 | R2 · viewer side | Cusped Rectangle | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 50 | R2 · viewer side | Open Spline | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 51 | R2 · viewer side | Brush | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 52 | R2 · viewer side | Eraser | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 53 | R2 · viewer side | Clone | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 54 | R2 · viewer side | Reveal | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 55 | R2 · viewer side | Blur | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 56 | R2 · viewer side | Sharpen | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 57 | R2 · viewer side | Smear | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 58 | R2 · viewer side | Dodge | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 59 | R2 · viewer side | Burn | missing | nodebased/app.py: Properties panel builder; no top-of-viewer per-tool settings bar | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 60 | R4 · panel side | Name | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 61 | R4 · panel side | Visible | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 62 | R4 · panel side | Lock | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 63 | R4 · panel side | Overlay | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 64 | R4 · panel side | Color | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 65 | R4 · panel side | Invert | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 66 | R4 · panel side | Blending | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 67 | R4 · panel side | Motion blur | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 68 | R4 · panel side | Life | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 69 | R4 · panel side | Groups/folders | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 70 | R4 · panel side | Context menu | differs | nodebased/app.py: RotoPaint properties builder, rotoppaint-layer-list and layer edit widgets; no columnar Stroke/Shape list | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 71 | R5 · panel side | Lifetime types | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 72 | R5 · panel side | Autokey | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 73 | R5 · panel side | Set key | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 74 | R5 · panel side | Delete key | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 75 | R5 · panel side | Ripple editing | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
| 76 | R5 · panel side | Per-shape keys | differs | nodebased/app.py: paint lifetime combo, paint item editor and Roto point key handling | NodeBased exposes a reduced or differently placed control/gesture; see the named implementation location. |
