# New look (UI redesign)

The approved design is a set of HTML mockups (layout, node graph look and wires). This file records what
each stage of the restyle changed in the real Qt app. Stages: 1 colours, spacing, type and the top bar;
2 node graph look and wires; 3 drag a free node onto a wire to insert it; 4 icons and the left column;
5 viewer strip and timeline; 6 Properties; 7 motion.

## Lane 2 step notes

### NL1: colours, spacing, type and the top bar

- **Tokens.** `nodebased/theme.py` holds the design tokens: `TOKENS` (surfaces `bg0`–`bg3`, hairlines
  `line` / `line2`, four text levels `tx0`–`tx3`, accent `acc` #5ee0b5), `FAMILY_COLORS` (the thirteen
  family colours, no yellow), `RADIUS` (9–12 px on controls and panels), `SPACING` and `TYPE` (Noto Sans,
  13 px base, Noto Sans Mono for values). The default theme is `NodeBased`, built from them; Charcoal,
  Graphite, Slate and Ash remain selectable and pick up the same stylesheet.
- **One stylesheet.** `theme.build_style()` styles the whole application, including the top bar (by
  objectName). Chrome widgets carry no colour literals; `tests/test_topbar.py` scans `topbar.py` for hex
  colours.
- **Top bar.** `nodebased/topbar.py`: logo mark, project / shot with a saved dot (project is the folder,
  shot is the file name; the dot dims while there are unsaved changes), workspace tabs as a segmented
  control, the search box, the GPU pill (adapter name read from the GL renderer, last frame time from the
  viewer status) and the Check for updates button as the primary button. There is no Export button.
- **Menus.** The menu bar is hidden. Its menus sit under the ☰ button in the top bar, together with the
  quick actions the old toolbar carried (Open image, Add node, Save project, Export image, 3D viewport,
  Slice viewer, Cache inspector). Every menu action is also added to the window so its shortcut keeps
  working while the bar is hidden.
- **Search.** The box opens the existing node search under it; Tab in the node graph is unchanged. The
  search lists nodes only for now; commands and settings are not searched yet.
- **Workspace tabs.** Composite and Render show the 2D viewer, 3D and Simulate the 3D viewport. Simulate
  also opens the Slice viewer and Cache inspector, Render opens the Conditioning queue. The tabs never
  close a panel, and the highlighted tab follows the viewer's own 2D / 3D switch.
- **Narrow windows.** Sizes come from font metrics. When the bar is narrower than its contents it drops
  the shot name first, then the GPU adapter name (the light and the frame time stay).
- **Screenshot.** `docs/images/ui-redesign/nl1.png`: the real app, offscreen at 1920 x 1080, with a viewer
  image and a dozen nodes of several families. The project label there ("hot_pour / comp_v012") is set
  for the picture only. The offscreen run has no GL, so the GPU pill reads CPU there.

### NL2: node graph look and wires

- **Families.** `nodecatalog.node_family(kind)` gives every registered node type one of the thirteen
  families. The catalog's categories are the families; the plumbing under "Other" (Dot, Group, Backdrop,
  NoOp ...) is filed by `nodecatalog.OTHER_KIND_FAMILY` (neutral Metadata, or Image for ContactSheet and
  PostageStamp). `theme.family_color(kind)` is its colour. `tests/test_graph_look.py` fails on any node
  type without a family.
- **Nodes.** A node is a rounded card (10 px corner; 3D nodes keep their pill and circle shapes): dark
  interior tinted with the family colour, a 2 px outline in the colour, a faint inner glow, a highlight
  along the top edge, a soft halo and the family icon in the colour beside a light name, then a hairline
  and a monospace details line. The details line shows the label and state when there are any and
  otherwise the first knobs, those changed from their defaults first. A selected node has a brighter
  halo and ring. The Viewer wears the interface accent. Postage stamps get rounded corners. The halo is
  a separate child item, so placement and framing see the card alone. At far zoom the glow is skipped.
  Name and details are cut with an ellipsis from the font's own metrics.
- **Wires.** A wire takes its source node's family colour with a faint glow. Mask wires and the Viewer
  tap are dashed (the tap thin and faint). Default shape is the curve of the layout mockup.
- **Port directions.** The main output faces down and the main input up; the A input faces left and a
  mask input faces right (every node that accepts a mask: Grade, Blur and the rest), so a mask wire
  enters it horizontally from the right. Rim sockets of the circle nodes face outward. A wire leaves and
  enters along the direction its port faces in both modes (`graphlook.wire_path`).
- **Curved / Right angle.** The toggle sits in the graph's bottom-right corner next to the zoom control
  and the minimap (`nodebased/graphcorner.py`). Right angle draws horizontal and vertical runs with
  slightly rounded 90 degree turns; on a drop longer than 80 px the horizontal run sits 22 px above the
  input. The choice is saved in the settings (`interface/wire_mode`) and restored at start.
- **Minimap and zoom.** The corner holds a minimap (one rectangle per node in its family colour, the
  view outlined in the accent; click or drag to move the view) and a zoom control (minus, the current
  percentage which resets to 100%, plus).
- **Tests changed because the design changes them.** `test_3d_node_shapes`: a 2D card now has a rounded
  corner, so the square-corner assertion became "rounded card, 3D pill rounder still".
  `test_desktop` title test: the name is 13 px, left aligned after the icon, not 14 pt centred; the
  Viewer-tap test accepts the mockup's custom dash pattern.
- **Screenshot.** `docs/images/ui-redesign/nl2.png` (curved wires) and `nl2-right-angle.png` (the same
  comp with the toggle on Right angle): the real app, offscreen at 1920 x 1080, with a viewer image and a
  dozen nodes of several families, a Roto mask wire into the Grade's right-hand socket, postage stamps
  on the Checker and the Constant. The graph panel is given more height for the picture only.
- **Speed.** The halo is drawn once per shape, colour and selection state and reused (a pixmap cache),
  so panning and dragging in a large graph repaint a picture per node instead of a dozen strokes.

### NL3: drag a free node onto a wire to insert it

- **Gesture.** Drag a node that has no connections over a wire: the wire lights up (a brighter, thicker
  stroke with a soft halo). Drop it there and the node is spliced in: the wire's source feeds the node's
  main input and the node's output feeds the wire's old destination input, a mask input included. The
  main input is the one a node made from a selection takes (`creation_slot`: B on a Merge, bg on a Roto,
  otherwise the first input that accepts the source's type). A wire out of a ShuffleCopy's second output
  keeps that output choice.
- **When it applies.** Only a node with no connections at all inserts, and only one node at a time. A
  connected node dragged over a wire just moves, as does a node with no input or no output (Read, Input,
  Viewer, Output, Write and the other Write nodes). A node that was not moved is never inserted, so a
  click on a node resting on a wire does nothing. A wire only counts when both ends can take the node by
  type, so a 2D node does not light a geometry wire.
- **Which wire.** The wire has to pass through the node's card within 8 screen pixels (so the reach is
  the same at any zoom); with several, the one nearest the card's centre. It works the same in Curved and
  Right angle, because the test uses the wire's actual path.
- **Undo.** The move and the two new connections are one batch: a single undo restores the original wire
  and the node's position. If the batch is refused the node is just moved.
- **Code.** `Graph.wire_to_insert_into` (hit test and type check), `Graph.set_insert_target` (highlight,
  `Edge.set_highlight`), `Graph.commit_moves` (the batch). Tests: `tests/test_wire_insert.py`.
- **Screenshot.** `docs/images/ui-redesign/nl3.png`: the real app, offscreen at 1920 x 1080, mid-drag with
  a free Blur held over the Constant's wire into the Merge, the wire lit. `nl3-inserted.png` shows the
  result after the drop.

### NL4: icons and the left column

- **The column.** The category shelf that sat as a second row under the top bar is now the mockup's tall
  icon column on the left (`nodebased/nodeshelf.py`, `NodeRail`, in a left tool bar named
  `node-rail-toolbar`): Favourites, Recent, a hairline, one button per family with its family colour dot,
  Settings pinned at the bottom (it opens the existing Settings dialog). The family whose panel is open is
  lit and shows a 3 px bar on the column's edge in the family colour (the accent for Favourites and
  Recent). Sizes come from font metrics: the column is a button plus two margins wide (60 px at the 13 px
  Noto Sans, wider on a font that is wider), and when the window is short the buttons and then the gaps
  shrink so all seventeen stay on screen (1280 x 720 shows them at about 31 px; the floor is 22 px, which
  is what the 800 x 500 minimum window gets).
- **The panel.** Clicking a family opens `NodePanel`, a rounded (14 px) panel with a blurred copy of the
  window behind a 92% tint (a grab of the window region under it, shrunk and enlarged) and the mockup's
  soft drop shadow: the family colour square, its name and node count, a filter box that has focus on
  open, a two column grid (22 px tile with the node's glyph in its family colour, then the name), and a
  footer with "N more" and the Enter key hint. Six rows show; the rest scroll (wheel, arrow keys, or a click
  on "N more"). The common nodes, those with a glyph of their own, come first, in catalog order. Enter adds
  the highlighted node, a click adds it, dragging a row into the graph adds it at the drop point (the same
  `application/x-nodebased-kind` drag the NODES dock uses); Up/Down/PageUp/PageDown (and Left/Right while
  the filter is empty) move the highlight; Escape, a click anywhere outside the panel and the column, a
  resize of the window or the window losing focus close it, and clicking the lit family again closes it
  too. The filter looks at node names and descriptions within the family. Favourites and Recent open the
  same panel with the starred and the recently added nodes.
- **Icons.** SVG files in `nodebased/data/icons/`, on the mockup's 24 x 24 grid with a 1.6 stroke in
  `currentColor`: `family-<Family>.svg` (the thirteen families and Other, the glyphs of the mockup's rail),
  `ui-*.svg` (favourites, recent, search, settings) and `node-<Kind>.svg` for 146 common nodes. A node
  with no file of its own wears its family's glyph. `nodebased/nodeicons.py` renders a file in any colour
  (pixmaps at 1x and 2x for high-DPI screens). The files are listed in `pyproject.toml`'s package data and
  the PyInstaller build already bundles `nodebased/data`.
- **Minimum window width.** `TopBar.floor_width()` measures the bar's narrowest form from the fonts (search
  as an icon, no shot name, GPU light only; the logo, tabs, menu and Check for updates button whole) and
  the window sets its minimum width to that plus the column's width (never below 800 px), again whenever
  the font or style changes. On Windows' wider fonts the bar is no longer cut off at the minimum. The
  narrowest-window top bar tests now drag the window to its real minimum instead of `max(800, floor + 40)`.
- **Saved layouts.** The workspace layout version went from 3 to 4 (the toolbar changed shape, so a layout
  saved with the family row under the top bar is dropped for the new default) and the layout revision
  from 4 to 5.
- **Tests changed because the design changes them.** `test_workspace_layout`: the family buttons are the
  column's 17 buttons and the node panel lists a family's nodes (the old per-family menus are gone); the
  stacked-column check compares only the docks on screen (the column moved every visible dock 60 px right,
  and a hidden dock keeps its old position). `test_topbar`: see above.
- **Screenshot.** `docs/images/ui-redesign/nl4.png`: the real app, offscreen at 1920 x 1080, the Color panel
  open beside its lit button; `nl4-filter.png` has "gr" typed into the filter. The graph panel is given
  more height for the picture only.
- **Not done.** The panel's blur is a still copy of what was under it when it opened, not a live
  backdrop blur (Qt has no blur-behind for a child widget).

### NL5: viewer strip and timeline

- **Strip.** The viewer's two rows of controls are one rounded strip floating at the top of the viewer
  (`nodebased/viewerstrip.py`): 2D / 3D, the channel buttons RGB R G B A (R, G and B in their own
  colours), the display transform as a menu, gain and gamma, ROI and Fit, and a "more" menu. The more
  menu holds what the old rows held beyond that: Proxy (Full, 1/2, 1/4, 1/8), Proxy while playing, Zebra,
  Format mask and Mask mode, 1:1 pixels, Reset gain, Reset gamma and the viewer inputs (Show input 1 to 9,
  B buffer, Compare). The display menu has two sections: Project view (sRGB, ACES 2.0, Linear) and This
  viewer (the project's default or any view of the ACES config).
- **Same state as before.** The widgets keep their names and signals (`channels`, `display_view`,
  `viewer_display`, `proxy`, `playback_proxy`, `exposure`, `gamma`, `zebra`, `roi_button`, `mask_choice`,
  `mask_mode`, `viewer_info`). The combo boxes and check boxes that have no place in the strip stay as
  hidden models, and menu picks go through them (`currentIndex` then `activated`), so the document, the
  shortcuts and the agent bridge read and write exactly what they did. `channels` is now `ChannelButtons`
  with the combo's methods (`currentText`, `setCurrentText`, `currentTextChanged`).
- **Corner readouts.** Bottom left: resolution, colour space and proxy state ("proxy 1/2", "proxy while
  playing" or "full resolution"). Bottom right: cursor position (Nuke's y), the pixel value with its
  swatch and the A/B buffer tag, and the zoom. The pixel readout is no longer a box that hunts for a free
  spot over the picture; the 2D picture is fitted into the band between the strip and the corners
  (`Viewer.insets`). `pixel_readout.label` still holds the whole machine-readable line (raw floats,
  alpha included). The nine-button input strip is shown on the bottom edge between the two corners when
  they leave it room, and its choices are always in the more menu.
- **Time row.** One slim row: previous frame, the accent play button, next frame; the monospace frame
  field (still the largest field); the track; In and Out; the rate and its presets; the real-time light.
  The light is green and reads "real time" while playback keeps the clock and turns warm and reads
  "dropping frames" once playback has dropped frames. When the viewer is narrow or the font is wide the row
  hides the status text, then the light's words, then the rate presets (`TimeRow.fit`); a field that is
  being edited is never hidden.
- **Track.** The cached range is a green bar along the bottom (uncached frames show the quiet track), keys
  are diamonds in the key blue between the frame numbers and the bar, and the playhead is an accent line
  with a glow and its frame number on a tab. The cache and key marks are pushed in exactly as before.
- **Graph "?" overlay.** The line of shortcuts above the node graph is gone. The same items, word for word
  (`nodebased/shortcuthelp.py`), open as a panel from the "?" button in the graph's corner and from the ?
  key; Escape, ? or a click outside closes it.
- **Dock title bars.** VIEWER, NODE GRAPH, PROPERTIES and the other docks have a slim, quiet title bar
  (small muted caps over a hairline, a transparent float and close pair). They float, close and rearrange
  as before.
- **Lock panels (final pass, Gonzo 10/10).** The mockup has no panel title bars, so docked panels now
  hide theirs by default. Workspace → Lock panels (on) controls it and is remembered per machine
  (`interface/lock_panels`); unlock to drag, float or close panels by their bars. A floating panel always
  keeps its bar. The Window menu still shows and hides every panel (`tests/test_new_look_final_pass.py`).
- **Input row only when in use (Gonzo 10/10).** The nine-button input row and its B and compare menus
  stay hidden until two or more viewer inputs are wired or a B buffer is chosen; with a single input the
  bottom edge is clear as in the mockup. The more menu holds the same choices at all times.
- **One-line colour and xyz rows (Gonzo 10/10).** A wrapping knob row now sizes a numeric field for
  the number it shows (plus a spare digit) instead of the widest number its range allows. At the default
  Properties width a colour's swatch and four fields sit on one line (before: four lines, each field
  squeezed to "0…"), and Translate, Rotate, Scale and Pivot keep x, y and z on one line.
- **Sizes.** Every width is from font metrics. At 1280 x 720 and 1440 x 920, also with the font a quarter
  larger, nothing in the strip, the corners or the time row is clipped (`tests/test_new_look_*_nl5.py`).
  The strip turns ROI and Fit into icons before it would clip.
- **Tests changed because the design changes them.** `test_no_toolbar_widget_is_clipped_at_1440x920` read
  the two old rows and now reads the strip; `test_menu_shortcuts_and_graph_hint_remain_current` and
  `test_the_table_covers_every_shortcut_in_the_hint_row` read the "?" overlay instead of the hint label;
  the key mark test reads the diamond rectangle (`key_mark_rect`) instead of the old bottom-standing tick;
  the time row clipping test counts nine widgets instead of ten (the TIME caption is gone).
- **Screenshot.** `docs/images/ui-redesign/nl5.png`: the real app, offscreen at 1920 x 1080, the pointer
  over the picture, frames 1 to 61 cached (the green range), four keys on the selected Grade (the
  diamonds); `nl5-help.png` has the "?" overlay open. The graph panel is given less height than its
  default for the picture only.
- **Not done.** The strip is translucent, not a live backdrop blur (Qt has no blur-behind for a child
  widget). The picture is fitted below the strip when the app fits it (first picture, Fit, a new format);
  resizing a dock afterwards does not refit, as before, so the picture can reach under a corner readout
  until Fit is pressed.

## End-to-end sprint, stage 5: punch list and workspace layouts (Gonzo, 10/10)

From the real-display QA of the new look (`FINDINGS.md`, findings 1 to 6), plus the Agent and Queue panels.

- **Family panel ghost (finding 1).** The green block over the list rows was the timeline's cache bar and
  playhead, blurred by the panel's frosted backdrop and shown through its translucent tint. The panel is
  now opaque and paints no backdrop. `tests/test_e2e5_ui_punchlist.py` opens every family panel and checks
  that no row, widget or item overlays another row and that the panel ignores what is behind it.
- **Simulate overlaps (finding 2).** The 2D corner readouts (size, colour space, proxy, pixel, zoom) hide
  while the 3D viewport is shown; the viewport's help and status lines start below the viewer strip
  (`Viewport3D.top_inset`); and the strip is raised above the page the stack shows (switching pages had
  put the 3D viewport over it).
- **Workspace layouts (finding 3).** `Window.select_workspace_tab` switches dock layouts. Composite and 3D
  share the artist's layout. **Simulate**: 3D viewport and timeline on top, the cache inspector and slice
  viewer side by side where the node graph sat, Properties at its normal width. **Render**: the queue and
  the new **output preview** (`nodebased/outputpreview.py`: the viewed Write node's name, file, format,
  bit depth and the viewer's current picture) side by side there, the viewer in 2D on the Write node, its
  Properties on the right. Leaving a layout saves its dock state in QSettings (`tab_layouts/<name>`);
  entering restores it, so Composite returns exactly. A layout is built the first time it is entered.
  The workspace saved at close is always the Composite arrangement. The queue panel sits in a scroll area
  so its tall form never squeezes the viewer.
- **Top-bar search (finding 4).** "Search nodes, commands, settings…": one list of node types, menu
  commands (every enabled menu action, with its menu path) and settings names (which open Settings).
  Enter runs the first result. Tab in the graph still searches nodes only.
- **Status bar state word (finding 5).** Ready, Rendering, Playing or Saving at the far left (a Write
  render or a viewport render shows Rendering; saving wins over the rest). Not done: the live render pill
  in the viewer ("Grade1 updated · 9 ms on GPU").
- **Letterbox (finding 6).** The viewer no longer draws a dotted frame, or the "640 x 360" label under it,
  around the canvas; the size is in the corner readout.
- **Agent panel and Conditioning Queue.** Surfaces, type and spacing from the tokens: accent primary
  buttons (Start Claude Code, Start bounded loop), mono caption activity log, quiet table headers,
  small-caps section headings, 14 px margins and 10 px gaps. Behaviour and object names the tests use are
  unchanged.
- **Screenshots** (offscreen, 1920 x 1080, `examples/mixed_scene`): `docs/images/ui-redesign/`
  `stage5-composite.png`, `stage5-simulate.png`, `stage5-render.png`, `stage5-agent.png`, `stage5-queue.png`.

## New look, punch list 2 (10/11, DiMo's review of the stage 5 screenshots)

Screenshots (offscreen, 1920 x 1080, `examples/mixed_scene`): `docs/images/ui-redesign/pl2-*.png`, each part with a
`-before` and an `-after`.

- **Viewer surround.** The area around the picture is pure black (`theme.VIEWER_SURROUND`, and the 3D
  viewport's clear colour) in every theme, as in Nuke. The grid and the corner readouts are unchanged.
- **Properties empty state.** With nothing selected the panel is empty: no hint, no format lines, no blurb.
  The family badge and the title appear with a selection.
- **Transport.** Start, previous keyframe, step back, play backwards, stop, play, step forward, next
  keyframe, end, and a loop mode toggle (play once, loop, ping-pong; remembered per machine). Icon-only, in the
  left column's stroke language. Keys: Left / Right step, Alt+Left / Alt+Right previous / next keyframe,
  Home / End, Space play / stop, in the viewer L plays, J plays backwards (while playing forwards J still
  stops and steps back, as before) and K stops.
- **Accent.** The default accent is an amber in the family of Nuke's selection orange, `#e0a24a` (HSL 35, 71, 58;
  the old mint `#5ee0b5` was 160, 68, 62), with `acc2` `#c0842f`, ink `#2a1a05` and washes at 12 and 40 percent.
  DiMo chose it on 10/11 over the muted teal `#66b7ac` first proposed; both the mint and the muted teal stay as
  named choices in Settings. The other themes' own accents were already pastel and are unchanged.
  The first candidates are compared in `pl2-4-accent-candidates.png`; the amber is in `pl2-4-accent-amber.png`.
- **Playback unit.** The viewer strip and the timeline are one quiet unit: one tone (`bg1`), 28 px controls on an
  8 px rhythm, no drawn separators. The ruler spans the full width on top (1 px ticks, thin key ticks, a 2 px
  cached-frames line, a 1 px playhead with a small flag); under it one row: the frame field, the transport
  centred, and In, Out, rate and the real-time light. The status text and the rate presets go first when the
  row is narrow.

## Wiring (10/11, DiMo: connecting a noodle must feel natural, never fussy)

Every radius is in screen pixels, converted through the view transform, so wiring feels the same at 40 and
150 percent. The constants live next to `Graph` in `nodebased/app.py`.

- **Dropping a wire** (`WIRE_REACH` = 40 px). The target is the nearest compatible socket within 40 px of the
  pointer; if none, the first socket of the card under the pointer. Dragging an output onto a card takes its
  first free input that accepts the wire (a mask input only when nothing else is free); dragging an input onto a
  card takes the card's output. A card with no free input takes nothing. Compatible means the right direction,
  another node, a socket type that accepts the wire and no cycle. With nothing compatible in range the wire is
  dropped ("Wire dropped") and every existing connection stays.
- **The snap.** As soon as a target is in range the pending wire's end glides onto the socket (80 ms), the wire,
  the socket (drawn larger, brighter) and the card light up in the same tone as the insert hint, and the status
  bar says "to Merge1.B". Leaving the range lets the wire follow the pointer again. Esc cancels and clears it.
- **Starting a wire** (`GRAB_REACH` = 20 px). Pressing within 20 px of a socket starts a wire from it: on its
  label, on the card edge beside it, or on empty space. Pressing in the bottom 10 px band of a card
  (`OUTPUT_BAND`) starts an output wire. Pressing deeper than 8 px inside a card is a node drag, as ever
  (`GRAB_EDGE`). The socket a press would grab draws a little larger as the pointer nears it. Dragging from a
  connected input still picks it up and keeps the old connection until a valid drop. A click-click wire ends at
  the second click with the same reach.
- Start reach is 20 px and not 40 px: with 40 px every press on a card title would start a wire instead of
  moving the node.

Captures (offscreen, `docs/images/ui-redesign/wire-*.png`): `wire-1a/1b/1c` a drop 34 px off the socket (following,
snapped, connected); `wire-2a/2b` a drop on a card's body; `wire-3a/3b/3c` hover, larger socket, a wire started
from a near miss; `wire-4-zoom40` and `-zoom150` the same snap at 40 and 150 percent.
