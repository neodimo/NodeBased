# The Nuke parity review book

`docs/PARITY_2D_FIDELITY.md` scores every node with numbers. This book is what a person looks at instead: one page per
node, to judge by eye whether NodeBased's node looks and behaves like Nuke's. The book is build output; it is written
to `docs/parity/review/out/`, which git ignores. Only the generators, their tests and this file are in the repository.

## Build it

From the repository root, with the project's Python:

    python tools/parity_review_book.py --ten                 # the ten high-use nodes (Blur, Grade, Merge, ...)
    python tools/parity_review_book.py                       # every scored node
    python tools/parity_review_book.py --nodes Blur,Grade    # chosen nodes

Open `docs/parity/review/out/index.html` in a browser. Qt draws the panels offscreen, so on a machine without a
display set `QT_QPA_PLATFORM=offscreen`. Your saved NodeBased layout is not read or changed.

## What a page shows

1. **Does the panel look like Nuke's?** On the left, the Nuke 17.0 Reference Guide's knob table for each tab (label,
   widget type, default) with a link to Foundry's page for the node; Foundry's screenshots stay on Foundry's site. On
   the right, NodeBased's real Properties panel, every tab, rendered by the app's own panel builder. Below them, one
   row per Nuke knob: the paired NodeBased param, or *missing*, or *unsupported* with the reason, and a mark for each
   difference in order, label, widget type or default. The guide does not say which knobs share a row, so rows are
   shown and not marked. Per-widget behaviours appear once `docs/parity/nuke_knob_interactions.json` exists.
2. **Does it compute what Nuke computes?** Every behaviour case of the node: the input, NodeBased's output, what the
   case says the result should be, the difference (black is equal, white is 0.1 or more), the audit's verdict and
   whether the case comes from the guide's text or from an established convention. The audit runs on a 16 x 12
   fixture; the pictures use 256 x 256 plates made for the node's kind (a ramp and colour bars for colour nodes, a step
   edge and a white square for filters, a grid for transforms, a screen-colour plate for keyers). Each case says
   which one it shows. Then "knobs that act": the node at its defaults and with each main-tab knob moved.

## Compare against Nuke's own pixels

So far every expectation is what Nuke's guide states. With Nuke you can render the real thing, and the book picks it up.

1. Export the scripts (one folder per case: the case's input plates as 32-bit float EXRs and a `.nk` script):

       python tools/nuke_script_export.py                  # into docs/parity/review/out/nuke_scripts
       python tools/nuke_script_export.py --nodes Blur,Grade

   Each script reads its plates, applies the Nuke node with the case's knob values under Nuke's own knob names, and
   writes `nuke_render.exr` (32-bit float, raw) beside itself. Cases whose input is animated or built inside NodeBased
   (the Time-group frame stamps, Roto and Tracker documents) have no still plate and are listed under `skipped` in
   `MANIFEST.json`.

2. Render them with Nuke 17:

   - **Nuke, NukeX or Nuke Studio (commercial licence):** `nuke -t docs/parity/review/out/nuke_scripts/render_all.py`
     renders every script in one pass. A single one: `nuke -x path/to/case.nk`.
   - **Nuke Indie or Non-commercial:** these licences restrict command-line rendering (Non-commercial renders only
     encrypted `.nknc` scripts from the command line; check what your Indie licence allows). Open a case's `.nk` in the
     GUI and render its `Write_render` node for frame 1. For a whole pass under Non-commercial, the older Python pack
     (`tools/make_nuke_reference_pack.py`, run with `nuke -t`) renders the same cases, and
     `tools/audit_nuke_fidelity.py --reference-dir` scores those renders directly; copy them into
     `docs/parity/nuke_renders/` for the book.

3. Collect the renders where the book looks for them, and rebuild:

       python tools/nuke_script_export.py --collect docs/parity/review/out/nuke_scripts
       python tools/parity_review_book.py

   `--collect` copies each `nuke_render.exr` to `docs/parity/nuke_renders/<case>.exr`. Each page then adds a "Nuke
   rendered" column and NodeBased's difference against it (pass within 1e-4), and the index counts the nodes verified
   against Nuke's own pixels. Do not commit the renders without asking first: the repository is public.

These scripts have not yet been opened in Nuke. Before trusting a whole folder, render one Merge case and check by eye
that input 0 is B and input 1 is A, and that the colours match the plates.

## The worklist

    python tools/parity_review_book.py --worklist

writes `WORKLIST.md` and `worklist.json` beside this file (both committed): every mismatch the book marks (default,
label, widget type, order, knob not laid out, missing knob, failing or missing behaviour case, failing widget
behaviour), one row each, with the node, its group, the lane that owns it, Nuke's value and NodeBased's, sorted by the
node's rank weight. A default the pairing skips on purpose (`skip_default`) is listed as accepted with its reason.
The book's index counts the open and accepted items per lane, and a node page with no open item carries a "clean"
badge. `tests/test_parity_review_book.py` fails when the committed `WORKLIST.md` is out of date: rerun the command.
