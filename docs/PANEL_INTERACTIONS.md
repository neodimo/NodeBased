# Panels that work like Nuke

The fidelity audit (`docs/PARITY_2D_FIDELITY.md`) scores how a Properties panel *looks*: knob order, labels, shared rows
and widget types. This page is about how the panel *works*: what happens when an artist types, drags, scrubs, splits,
picks a colour, keys or resets a knob.

- **Reference:** `docs/parity/nuke_knob_interactions.json`, one entry per behaviour, each with the widget it belongs to,
  what Nuke does, and the sentence from Foundry's Nuke 17.0 User Guide that says so. Where the guide pages do not say
  what Nuke does, the entry is `undocumented` with one line and no claim.
- **Probes:** `tools/nuke_panel_interactions.py` builds real panels in a real window (Grade for number and split knobs,
  Constant for colour) and drives them with Qt key and mouse events, then reads what the document stores and the undo
  step it took.
- **Tests:** `tests/test_panel_interactions.py` asserts every behaviour that matches Nuke and names the open gaps.
- **Live numbers:** the section "Panels that work like Nuke" in `docs/PARITY_2D_FIDELITY.md`, one row per widget type.
  It is reported beside the fidelity score and is not part of its formula.

## Sources

- Properties Panels (Nuke 17.0 User Guide):
  <https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/properties_panels.html>
- Using the Color Controls (Nuke 17.0 User Guide):
  <https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/using_color_controls.html>
- Animating Parameters (Nuke 17.0 User Guide):
  <https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/animating_parameters.html>
- Linking Expressions (Nuke 17.0 User Guide):
  <https://learn.foundry.com/nuke/17.0/content/comp_environment/expressions/linking_expressions.html>
- Working with Nodes and Using the Tab Menu (Nuke 17.0 User Guide), for the Node Graph:
  <https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/working_nodes.html>,
  <https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/tab_menu.html>

## What the number field now does (lane 10, 2026-11-16)

- A formula typed into a field is worked out on Enter (`378/2` gives 189). Before, the slash was silently dropped and
  `3/2` stored 32.
- Up/Down and the mouse wheel change the digit just after the text cursor (before the 0 of 20.51 steps by ones); with no
  digit after the cursor they keep the field's own step.
- Alt+left drag on the value scrubs it, and a middle drag is Nuke's virtual slider (Shift faster, Alt slower). The value
  is committed once, on release, as one undo step.
- `=` still opens the expression editor below the knob, which then marks the knob as expression-driven.

Other fixes in the same step:

- The knob's animation menu "Clear animation" now keeps the value the knob had on the current frame (Nuke's No
  animation), in one undo step.
- The panel menu gained "Set key on all knobs" (one undo step).
- A colour knob's swatch now works as Nuke's: a click turns on the eye dropper (the swatch gets an accent border), Ctrl+click
  in the viewer samples one linear pixel into the knob, Ctrl+Shift+drag samples the average of a region, and a second
  click on the swatch turns it off. Ctrl+right-click in the viewer discards the samples and puts back the colour the
  knob had when the eye dropper was turned on. A plain click in the viewer still does what it always did. The "Pick" button beside
  the swatch opens the colour picker (wheel, RGB, HSV and TMI sliders), which the swatch used to open.
- A colour swatch dragged onto another colour swatch copies the colour (one undo step).
- Closing a split knob's channel fields keeps every channel's value, as the guide says ("click the channel selector
  button again to close and save your settings"). Channels that hold different values fold into one line that lists
  them; clicking the button again shows the fields. Equal values still close to one value. Before, closing asked and then
  kept only the first channel.

## Channel pickers (lane 10, step T2, 2026-11-17)

Source: Calling Channels (Nuke 17.0 User Guide),
<https://learn.foundry.com/nuke/17.0/content/comp_environment/channels/calling_channels.html>.

- **Channel set** (a node's `channels` knob, and Merge's A channels, B channels and output, a Draw node's output): a
  layer menu (rgb, rgba, alpha, all, none, or a typed layer name) followed by one check box per channel of the chosen
  layer, as the guide says ("The layer's channels appear with check boxes. Deselect those channels which you don't wish
  to process"). Unticking a box stores the channels left ticked (`rgba.red rgba.blue`); with every box ticked the knob
  stores the layer's own word again. The panel then shows the rgba layer with the left-out channel unticked: the guide
  does not show how Nuke names such a set in its menu.
- With the menu at none, a channel field appears on the right, the guide's "right most channel field", and picking a
  channel there processes that one channel (`rgba.alpha`).
- **Single channel** (the mask channel, unpremult by, premult by, and the other knobs that take one channel): a layer
  menu and a channel menu side by side, as Nuke draws `rgba` and `alpha`, stored as `layer.channel`. At none the channel
  part is greyed out; choosing the rgba layer picks alpha, the guide's default matte.
- The document stores what it stored before (a word, a layer name or a channel list the renderer already read), so old
  documents load unchanged and no schema change was needed. Copy's and ChannelMerge's channel menus store their own
  `A.a`-style values and keep their one-menu widget for now.

## One mask strip (lane 10, step T2)

Every node with Nuke's mask controls draws them the same way, from one builder (`knobs.panel_layout`): a row with the mask
box, the mask channel picker (layer and channel), and the Inject, Invert and Fringe boxes each carrying its name; a second
row with (Un)premult by and its Invert box; then Mix. The words and order are the guide's knob tables' (mask, inject,
invert, fringe, (un)premult by, invert, mix), and a node whose page words a knob differently keeps the page's word
(Keymix and STMap: Mask channel; HistEQ: Invert mask; Grain: Invert alpha). Ticking the mask box picks the alpha channel;
unticking sets the mask to none.

## Node tab (lane 10, step T2)

In the order the Properties Panels page lists it: label; font, with bold, italic, size and the text colour on the font's
row; hide input; cached; disable; dope sheet; bookmark; postage stamp; static frame; lifetime range (start, end, use
lifetime). The node colour and the Viewer highlight colour, which sit in Nuke's panel header, stay at the end of the tab.

## Copy and ChannelMerge channel menus (lane 10, step T3, 2026-11-17)

Copy's four from/to menus and ChannelMerge's A, B and output menus are Nuke single-channel knobs, so they now show as the
layer menu and channel menu side by side (`A` + `red`, `rgba` + `alpha`). The document keeps the words it always held
(`A.r`, `B.a`, `R`, `none`), so older scripts load unchanged; a pick stores the matching word, and a channel the menu does
not offer is refused and the stored one shown again.

## On-screen controls (lane 10, step T3, 2026-11-17)

- **Reference:** `docs/parity/nuke_viewer_controls.json`, one entry per on-screen control of a 2D node, each quoting its
  source: the User Guide's [Transforming in 2D](https://learn.foundry.com/nuke/17.0/content/comp_environment/transforming_elements/transforming_2d.html)
  page for the Transform jack, and the Reference Guide node pages for Crop, Text, Tracker, GridWarp and the rest. A control
  or rule the pages read do not state is `undocumented`; Blur and Defocus have no on-screen control.
- **Probes:** `tools/nuke_viewer_controls.py` presses, drags and releases on NodeBased's Viewer at image coordinates through
  the viewer's own press, move and release handlers, then checks the knob values, that the drag is one undo step and that an
  animated knob gets a key at the current frame. `tests/test_viewer_controls.py` asserts every result.
- **Live numbers:** the section "On-screen controls that work like Nuke" in `docs/PARITY_2D_FIDELITY.md`, one row per node.

What Nuke documents for the Transform jack: drag the centre to translate, Shift+drag to keep the move on x or y, Ctrl/Cmd+drag
to move the pivot; drag the circle to scale both ways, its left and right points to scale x, top and bottom points to scale y;
drag the horizontal bar to rotate (snapping to typical values, Shift turns the snapping off); drag the vertical bar to skew;
with invert on, a second small handle appears. Crop's box edges drag in the Viewer; Text's box moves and resizes there and
Ctrl/Cmd+drag moves its centre; the Tracker places an anchor per track; GridWarp shows its source and destination grids with
handles and a transform overlay. The pages read state no Viewer control for CornerPin2D's pins, Reformat, Radial, Ramp,
Rectangle, Position or Flare, so those are listed as undocumented.

Fixed in this step: Ctrl-drag on the jack's centre now moves the pivot as in Nuke (it needed Ctrl+Shift before; Ctrl+Shift
still works), and Shift-drag keeps a translate on the axis the pointer moved furthest along. Before the step 3 of the 16
documented controls worked like Nuke's; after it, 5.

Finished in the step's finish pass (all 16 documented controls now pass):

- **Transform jack.** The left and right points scale x alone and the top and bottom points y alone, storing scale as two
  values (Nuke's split scale); a split scale keeps its handle, and the corners then scale both factors together. The jack is
  drawn through the same matrix as the picture (split scale and skew included). Rotating snaps to the nearest multiple of 45
  degrees within 2 degrees (the guide says "typical values" and names none, so the values are NodeBased's); Shift, pressed
  at any time during the drag, turns the snapping off. A vertical bar rises from the pivot: dragging its top sideways sets
  skew X so the bar passes through the pointer. With invert on, a small second handle (circle and cross) is drawn beside the
  pivot; the guide says only that it appears.
- **Crop.** The box is drawn with grips at its corners and edge midpoints; dragging an edge or corner moves that side and
  keeps the opposite one, in whole pixels. Inside the box is left to the Viewer (the guide names only the edges).
- **Text.** The box moves when dragged inside and resizes from its edges and corners; Ctrl-drag moves the centre (the
  offset from the box's centre that rotation and scaling turn about), marked with a cross.
- **Tracker.** Add track places a new anchor, drawn with its pattern and search boxes; dragging the anchor's centre moves the
  track. Zoomed out, the pattern box's edges came within grab range of the centre and took the click; the centre now wins
  within half the pattern box (or 4 screen pixels), and the edges still resize further out.
- **GridWarp.** The source and destination grids' lines and points are drawn whenever the node is viewed and selected
  (Nuke's per-grid visible switch has no NodeBased counterpart); where the two grids share a point, a drag takes the
  source's. A transformation overlay sits at the grid transform's centre: dragging the centre translates the whole grid,
  dragging the end of its x or y axis turns and scales it. It edits the source transform while "use source transform" is on,
  the destination's otherwise. Grid points take a click before the overlay.

Every drag is one undo step and keys an animated knob at the current frame.

**ColorTransfer's ROI box (lane 10, step T8, 2026-11-18).** The [ColorTransfer](https://learn.foundry.com/nuke/17.0/content/reference_guide/color_nodes/colortransfer.html)
page: "You can also adjust the ROI in the Viewer by dragging its edges", and "you may want to connect the Viewer to input A
rather than the ColorTransfer node". NodeBased now draws the ROI as the same box Crop and Text use: corner, edge and body
grips, one undo step per drag, a key at the current frame on an animated knob. The knobs stay Nuke's x, y, r, t from the
bottom left; the default (all zero, the whole input) draws over the whole frame. The box also shows with the Viewer on input A.
The two controls are probed as `ColorTransfer.roi_box` and `ColorTransfer.roi_on_input_a`.

## The Properties bin and linking knobs (lane 10, step T4, 2026-11-17)

- **Reference:** the `properties_bin` and `linking` entries of `docs/parity/nuke_knob_interactions.json`, quoting the User
  Guide's [Properties Panels](https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/properties_panels.html)
  page (opening a panel, the maximum-panels field, lock, remove all panels, a panel's close, float and hide buttons) and its
  [Linking Expressions](https://learn.foundry.com/nuke/17.0/content/comp_environment/expressions/linking_expressions.html) page
  (Add expression, Node.knob references, Ctrl/Cmd+drag, the blue knob and green Node Graph arrow, the Link menu), plus the
  Animating Parameters page's Link to > Tracker linking dialog.
- **Probes:** the same `tools/nuke_panel_interactions.py`, now clicking and double-clicking nodes in the real Node Graph,
  pressing Return there, clicking a panel's close button with Alt and Ctrl, and Ctrl-dragging one knob's field onto another's.
  `tests/test_l10_t4_bin_linking.py` asserts every passing one by name.

What the guide states and NodeBased now does: inserting a node shows its panel; double-click, Ctrl/Cmd+click, or selecting
a node and pressing Return opens its panel in the bin (newest on top); the Max panels field in the bin caps how many stay
open; Clear all empties the bin; a panel's close closes it, Alt+click closes every panel, Ctrl/Cmd+click every other panel
(the close entry sits in the panel's menu, so Alt or Ctrl is held while choosing it); the header arrow hides a panel's pages.
Ctrl/Cmd+double-click on a node opens its panel in a floating window, and the panel menu's Float moves a panel out of the bin
into one (choosing it again in the floating window docks it back); a floating panel follows edits like the bin and closes
when its node is deleted. The bin's Lock button keeps the bin as it is: every panel opened while it is on floats.
Right-click > Add expression (or =) opens the expression editor; `Node.knob` links one knob to another; Ctrl/Cmd+dragging a
knob's number field onto another knob's field links the second to the first, stored as an expression (`grade1.gamma`, or
`knob("id", "gamma")` for an id that is not a plain name), in one undo step. A linked knob is drawn light blue, lighter than
the key blue, so linked and keyed knobs stay apart. The Node Graph draws a dashed green arrow from each node an expression
reads to the node it drives (it takes no clicks, so wires and nodes under it behave as before). Edit expression shows the
link.

Fixed in this step: Return and Ctrl/Cmd+click open a panel (Return did nothing and Ctrl+click only toggled the selection);
Alt+click and Ctrl/Cmd+click on close; floating panels, the float entry and the bin's lock (none existed); the menu entry reads
Add expression as in Nuke (it read Enter expression); Ctrl/Cmd+drag linking (it did not exist); linked knobs turn blue (they
were purple); the green Node Graph arrow (there was none). Before the step 6 of the 13 documented bin behaviours and 2 of the
9 documented linking behaviours worked like Nuke's; after it, 13 of 13 and 6 of 9. Older documents load unchanged: the
document format is the same. Ctrl/Cmd+Alt+click, the guide's other way to float a panel, stays NodeBased's wire-cutting knife.

Not yet like Nuke: Ctrl/Cmd+drag between animation buttons to link several values at once (split knobs cannot carry an
expression yet); the Link menu of channel, layer and format knobs; and Link to > Tracker linking dialog. Undocumented, so not scored: where a new panel goes in the bin, the field's default and which panel
the limit closes, pinning a single panel, removing a link from a number knob (NodeBased: Clear expression on the knob menu),
copying and pasting a knob's value or animation from the panel, and reading another frame in a link (part of the expression
language).

## Open gaps

- Knob widgets: none among the documented behaviours read so far (31 of 31 pass). The bin and linking gaps are listed
  in the section above. Not documented, so not scored:
- What the folded line of a split knob shows is NodeBased's own choice: the guide does not say what Nuke displays there.
- Enumeration, check box, XY and file knobs, the mask fringe box, a typed value outside the slider range, and the
  Animation menu's Copy/Paste and Set to default have no interaction rule on the User Guide pages read, so they are listed
  as undocumented.

## Knob linking finished and the Node Graph (lane 10, step T5, 2026-11-17)

- **Reference:** the `linking` and `node_graph` entries of `docs/parity/nuke_knob_interactions.json`. The Node Graph entries
  quote the 17.0 User Guide's [Working with Nodes](https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/working_nodes.html)
  and [Using the Tab Menu](https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/tab_menu.html) pages, the
  [Backdrop](https://learn.foundry.com/nuke/17.0/content/reference_guide/other_nodes/backdrop.html) Reference Guide page and the
  Node Graph preferences in [Appendix A](https://learn.foundry.com/nuke/17.0/content/misc/available_preferences_studio.html).
  Each probe in `tools/nuke_panel_interactions.py` sends real key and mouse events to the built window's Node Graph and reads the
  document, the undo step and the selection; `tests/test_l10_t5_node_graph.py` and `tests/test_l10_t5_linking.py` assert them.
- **Linking, 8 of 9 (6 before):** Ctrl/Cmd+drag of one knob's animation button onto another's links the whole knob (every
  channel of an r g b a knob) in one undo step; the knob menu's Link to lists every Tracker and links the knob to its translate x
  or y. Left: the Link menu of channel, layer and format knobs (those knobs hold names, which expressions do not drive yet), and
  split knobs have no animation button, so a split knob cannot be the drop target.
- **Node Graph, 14 of 25 documented (7 before):** fixed here: Tab opens the Tab menu whenever the graph has focus (before, only with
  the pointer over it); Tab then Return makes the last node made from the Tab menu, and nodes used often from it list first
  (keyboard shortcuts such as B do not count, as the guide says); N opens a rename field on top of the node (Return renames in one
  undo step, Escape keeps the name); Ctrl/Cmd+D disconnects the selected nodes; Shift+click adds a node to the selection (before
  it replaced it; Ctrl+click still adds, so nothing was lost); a paste connects to the node selected before it. Already like Nuke:
  creating a node connects it to the selected one, D, Ctrl/Cmd+A, Ctrl/Cmd+X, colours by class and the tile colour, a Backdrop
  carrying the nodes on it.
- **Still open:** Clone with an active link (NodeBased's Alt+C makes an independent copy), Ctrl/Cmd+Alt+A Select Connected Nodes,
  and not yet probed: Shift to start a new branch, Ctrl/Cmd to replace, inserting by dragging onto a pipe, dragging a pipe's end to
  connect or disconnect, Shift+drag to duplicate a pipe, Ctrl/Cmd+click to add a Dot, Ctrl/Cmd+drag to select upstream, snap to
  grid. Undocumented on the pages read: swapping inputs with Shift, a downstream-only selection, an Alt+C duplicate.

## The rest of the Node Graph (lane 10, step T6, 2026-11-17)

- **Probes:** ten more in `tools/nuke_panel_interactions.py`, each dragging or keying the built window's Node Graph and checking the
  document's wires, the single undo step and the selection; `tests/test_l10_t6_node_graph.py` asserts them.
- **Node Graph, 24 of 25 documented (14 before).** Built here: Shift+Return in the Tab menu adds the node as a new branch off the
  selection and leaves the nodes below wired as they were; Ctrl/Cmd+Return replaces the selected node, the new one taking its
  inputs and everything that read it, in one undo step; Shift+drag on a wire draws a second wire from the same output onto
  another node and keeps the first (a Shift+click without moving still adds the wire to the selection); Ctrl/Cmd+drag on a node
  selects it and every node feeding it and moves them together (a Ctrl/Cmd+click still opens the panel); Ctrl/Cmd+Alt+A and
  Edit > Select connected nodes select every node wired to the selection, up or down; Edit > Snap to grid (off by default, as in
  Nuke) lands a dragged node on the graph's 32-unit dot grid. Fixed: pressing the middle of a Dot started a wire from its socket,
  so a Dot could not be dragged; the Dot's body now always moves it and its wires start from its rim. Already like Nuke, now
  measured: dragging a pipe's input end to empty space disconnects it and onto the middle of another node reconnects it,
  dropping a free node onto a wire places it between the two nodes, and Ctrl/Cmd+click on a wire's handle adds a Dot that passes
  its input on.
- **No shortcut taken away.** Shift+click on a wire still toggles its selection (now decided on release, so a drag can start); the
  Tab menu's plain Return is unchanged.
- **Closed in the finish pass below:** Clone, and the Link menu of channel, layer and format knobs.

## Clone and the Link menu (lane 10, step T6 finish, 2026-11-17)

- **Node Graph, 25 of 25 documented (24 before). Linking, 9 of 9 (8 before).**
- **Clone.** Edit > Clone and Alt+K make a copy of each selected node that keeps an active link to its parent, as the guide
  says; a solid orange line joins each clone to its parent in the Node Graph. Every knob edit on any member of a clone group
  (a typed value, a key, a curve, an expression, bypass) is applied to all of them in the same undo step, so their knobs stay
  identical. A clone of a clone joins the same parent. Edit > Declone and Alt+Shift+K make the selected nodes independent
  again, keeping their values; deleting a parent hands the group to its first clone, and moving a clone into a Group drops its
  link. The links are kept in an optional `clones` entry of the project settings (clone id to parent id), so older documents
  load unchanged and the schema stays where it was. Groups, Viewers and Dots are not cloned.
- **Link menu.** Channel set, single channel and format knobs (`channels`, `maskChannelInput`, `format`, a lens format and
  the rest of their families) carry a Link menu button beside them: Set link lists every knob of the same family in the
  graph, Edit link takes a typed `Node.knob`, Remove link brings back the knob's own value, and Ctrl/Cmd+drag of the button
  onto another such knob links that knob to this one. A linked knob follows the word its target holds (through a chain of
  links), shows it greyed out, and draws the green expression arrow in the Node Graph; a Reformat whose format is linked takes
  that format's size. The link is stored as a bare reference in the document's expressions section, so it shares their
  cycle check and one undo step; a formula, or a link to a knob of another family, is refused by name.
- **No shortcut taken away.** Alt+K and Alt+Shift+K were free in the Node Graph.
- **Tests:** `tests/test_l10_t6_clone.py` and `tests/test_l10_t6_link_menu.py`.

## The Curve Editor and the Dope Sheet (lane 10, step T7, 2026-11-18)

Source: Animating Parameters (Nuke 17.0 User Guide), sections Using the Dope Sheet and Using the Curve Editor
(<https://learn.foundry.com/nuke/17.0/content/getting_started/using_interface/animating_parameters.html>). The reference holds 21
documented Curve Editor behaviours and 12 documented Dope Sheet behaviours, each with the guide's sentence; deleting a key in the
Curve Editor, its undo steps, dragging and scaling keys in the Dope Sheet and its frame ruler are listed as undocumented.

NodeBased's curve canvas edits curve knobs (ColorLookup's lookups) and, from a knob's animation menu (Curve editor…), an
animated knob's keys; keyframed knobs show in the Dope Sheet. Measured with `tools/nuke_panel_interactions.py`: the Curve
Editor went from 3 to 15 of 21 in step T7 and to 21 of 21 in its finish pass, the Dope Sheet from 2 to 7 of 12 and then to
12 of 12.

What the curve canvas now does:

- **Ctrl/Cmd+Alt+click adds a key** where you click, on the curve or off it. Double-click still adds one on the curve.
- **A plain drag moves a key on one axis** (the larger motion wins); **Ctrl/Cmd+drag moves it freely.** One undo step per drag.
- **Right-click > Interpolation** lists Constant, Linear, Smooth, Catmull-Rom, Cubic, Horizontal and Break. Smooth takes the
  neighbours' slope but goes flat at a peak or dip, so the curve never overshoots; Catmull-Rom takes it regardless; Cubic solves
  the natural spline so the second derivative is continuous; Horizontal sets zero slope; Break lets the handles move apart.
- **A** frames all, **F** centres the selected key, **+** and **-** zoom, and **Alt+drag** pans (as well as the middle button).
- **Delete/Backspace** deletes the selected keys.
- **Several keys** (finish pass): Shift+click adds or removes a key, a drag on empty space draws a marquee (Shift adds or
  removes), Ctrl/Cmd+A selects every key, and dragging one selected key moves them all, one undo step.
- **Several curves**: a list beside the canvas names the node's curve knobs (Master, Red, Green, Blue, Alpha on
  ColorLookup). Shift+click shows a run of them, Ctrl/Cmd+click one more; the others are drawn thin under the current one,
  which is the one that edits.
- **Edit > Move…** (right-click) opens Move Animation Keys: x and y are expressions of each selected key's own x and y,
  such as `x+10` or `y*2`. One undo step; a bad expression is named in the dialog and changes nothing.
- **The expression field** at the bottom sets the curve from an expression of x (numbers, x, pi, e, + - * / ** and abs,
  min, max, pow, sqrt, exp, log, sin, cos, tan, floor, ceil, clamp, lerp, smoothstep). The curve is sampled into 17 keys
  whose tangents follow the expression, so the stored curve format is unchanged; Revert puts the keys back.
- **Frame snap**: the knob animation menu's Curve editor… shows that knob's keys with the frame along x. A moved key's x
  rounds to the nearest whole frame and stays a frame from its neighbours; with Shift it follows the pointer while
  dragging, and lands on a whole frame because the document keeps whole-frame keys. One interpolation per curve
  (constant, linear or smooth) is what an animated knob stores, so tangent handles are hidden there.

What the Dope Sheet now does:

- **Click a key** to select it (it turns orange and shows its frame number); Shift+click adds or removes; clicking a node's name
  selects all its keys.
- **Delete/Backspace** erases the selected keys on every keyed knob of that node, one undo step.
- **Ctrl/Cmd+Alt+click** on an empty spot keys every animated knob of that row at the curve's current value.
- **Drag selected keys in time**, one undo step; a key stops beside the next one rather than pass it.
- **Project range lines** (finish pass): the time area spans a few frames either side of the project range, and the first
  and last frames have light lines of their own.
- **Marquee**: a drag on empty space selects the keys inside it; Shift adds or removes them.
- **Move field**: the row under the sheet holds a Move number and < > buttons that move the selected keys by that many
  frames (negative moves them back), one undo step.
- **Copy and paste**: Ctrl/Cmd+C copies the selected keys (every knob keyed on those frames); Ctrl/Cmd+V pastes them with
  the earliest on the current frame, one undo step.
- **Read clip bars**: a Read whose panel is open shows its clip (first to last frame) as a gray bar; dragging the bar
  repositions the clip by changing the Read's frame offset, one undo step.

Older documents load unchanged: no stored format changed, and framing never edits a curve.

Every documented Curve Editor and Dope Sheet behaviour now passes. Older documents load unchanged: no stored format changed.

- **Tests:** `tests/test_l10_t7_curves.py`.
