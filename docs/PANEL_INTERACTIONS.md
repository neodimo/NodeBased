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

Still open, named in the report's table: the jack's scale-x and scale-y points, rotate snapping, the skew bar and the invert
handle; on-screen boxes for Crop and Text; probes for the Tracker anchor and GridWarp's grids, which NodeBased does draw.

## Open gaps

- None among the documented behaviours read so far (31 of 31 pass). Not documented, so not scored:
- What the folded line of a split knob shows is NodeBased's own choice: the guide does not say what Nuke displays there.
- Enumeration, check box, XY and file knobs, the mask fringe box, a typed value outside the slider range, and the
  Animation menu's Copy/Paste and Set to default have no interaction rule on the User Guide pages read, so they are listed
  as undocumented.
