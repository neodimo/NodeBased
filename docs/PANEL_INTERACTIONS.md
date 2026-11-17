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

## Open gaps

- Every documented behaviour now passes. What the folded line of a split knob shows is NodeBased's own choice: the guide
  does not say what Nuke displays there.
- Enumeration, check box, channel and layer pickers, XY, file, and the mask/mix rows have no interaction rule on the
  User Guide pages read, so they are listed as undocumented.
