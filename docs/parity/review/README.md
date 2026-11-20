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
   - **Nuke Non-commercial:** terminal mode runs Python files only. It refuses the `.nk` scripts above ("Please specify
     an existing .nknc script or a Python file"), so use the Python pack instead, which renders the same cases:

         python tools/make_nuke_reference_pack.py PACK_DIR
         NUKE=/path/to/nuke-nc.sh PACK_DIR/render_all.sh RENDER_DIR     # nuke-nc.sh runs Nuke17.0 --nc "$@"

     Each case is one script run with `nuke -t` (about 2 seconds each; several can run side by side). Non-commercial's
     other limits shape the scripts: at most 10 nodes reachable from Python per script, and output at most 1920x1080
     (each script crops the node's output to its format before the Write). The pack builds the Time group's frame
     stamp inside Nuke (an Expression node whose red is the frame) and renders each case at its own frame.
     Knob values are written for each knob's kind in Nuke 17, read from `tools/nuke17_knobs.json`; rerun
     `python tools/nuke17_knob_table.py --nuke /path/to/nuke-nc.sh` after a Nuke upgrade to refresh it.
     `MANIFEST.json` lists the cases left out (`skipped`: not pixel behaviours, or nodes NodeBased does not have) and
     the cases Nuke Non-commercial cannot render, each with its reason (`not_renderable`: Primatte, Ultimatte and
     GenerateLUT are not offered; Roto shapes, RotoPaint strokes, Tracker tracks and DustBust spots are not written
     into Nuke yet; OCIO cases whose colourspaces are not in Nuke's default config; fractional frames).
     Score the renders with `python tools/audit_nuke_fidelity.py --reference-dir RENDER_DIR` and copy the EXRs into
     `docs/parity/nuke_renders/` for the book. A failed case leaves a `.log` beside its render that says why; fix the
     generator, delete that case's log and EXR, and run its script again.
   - **Nuke Indie:** check what your licence allows from the command line; the GUI can render a case's `.nk`
     (`Write_render`, frame 1).

3. Collect the renders where the book looks for them, and rebuild:

       python tools/nuke_script_export.py --collect docs/parity/review/out/nuke_scripts
       python tools/parity_review_book.py

   `--collect` copies each `nuke_render.exr` to `docs/parity/nuke_renders/<case>.exr` (the Python pack's renders go
   there directly). Each page then adds a "Nuke rendered" column and NodeBased's difference against it (pass within
   1e-4), and the index counts the nodes verified against Nuke's own pixels. `docs/parity/nuke_renders/` is ignored by
   git: the repository is public, so the renders stay on the machine that made them.

Before trusting a whole folder, render one Merge case and check that input 0 is B and input 1 is A, and that the
colours match the plates. Checked on 11/20 with Nuke 17.0v6 Non-commercial: Merge `over` of the fixtures A and B came
out exactly A + B(1 - a) in all four channels, negative and HDR values untouched.

## The worklist

    python tools/parity_review_book.py --worklist

writes `WORKLIST.md` and `worklist.json` beside this file (both committed): every mismatch the book marks (default,
label, widget type, order, knob not laid out, missing knob, failing or missing behaviour case, failing widget
behaviour), one row each, with the node, its group, the lane that owns it, Nuke's value and NodeBased's, sorted by the
node's rank weight. A default the pairing skips on purpose (`skip_default`) is listed as accepted with its reason.
The book's index counts the open and accepted items per lane, and a node page with no open item carries a "clean"
badge.
Rerun the command after a panel change; the committed copy is a snapshot.
