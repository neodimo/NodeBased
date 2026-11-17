# Parity decisions for DiMo

Every item on the review book's worklists (all lanes) that no lane can close without a product decision: a Nuke node
NodeBased does not have, a project-wide setting, or a knob that only means something once NodeBased grows a feature it
lacks. Nothing here is built; each line is a choice.

**Cost** is the fidelity the item takes away: the node's rank weight (the audit's 1 / rank, shared among the nodes of
one row) times its gap (1 minus the node's score for a missing node or Root; for knobs inside a node, their share of
the node's knob component). **Points** is that cost as points of the overall score (cost / total weight × 100; total
weight 31.80, overall 97.3 on the tree this was measured on, Tuesday 11/17 at 10:05 AM). Closing every line here
would add about 2.2 points. Sorted by cost.

| Cost | Points | Item | What it costs | What building it would take |
|---|---|---|---|---|
| 0.167 | 0.52 | BlinkScript (node) | the whole node: 19 knobs, a user-written GPU kernel per node | A kernel language (Blink or a WGSL subset) compiled at edit time on the wgpu device, with its parameters turned into knobs; the largest item here and a security surface (user code on the GPU). |
| 0.083 | 0.26 | Primatte (node) | the whole node: 27 knobs | A third keyer built on Primatte's published polyhedron colour-space model, with its sample-and-shrink tools in the Viewer; Keylight, IBK and ChromaKeyer already cover the job. |
| 0.083 | 0.26 | Ultimatte (node) | the whole node: 60 knobs, five inputs | Ultimatte's algorithm is proprietary and unpublished, so only a look-alike screen keyer with its knob names is possible; scoring it as Nuke's would be invented. |
| 0.071 | 0.22 | AudioRead (node) | the whole node: 10 knobs | An audio decoder (WAV first), a waveform in the Dope Sheet and the curve the node outputs; NodeBased has no audio anywhere yet. |
| 0.050 | 0.16 | Bokeh (node) | the whole node: 42 knobs, and its behaviour case | A depth-driven bokeh filter beside ZDefocus, reusing its depth and focal-plane handling with an image-shaped aperture; mostly a kernel and its tile rule. |
| 0.045 | 0.14 | Root (project settings) | about half of Root's score: 26 of 43 knobs (colour management and LUTs, views, proxy settings, render mode) and 8 defaults | A Project Settings window over the document's settings, then a decision per section: OCIO config per project, the LUT table, the proxy policy; the stereo views are their own line below. |
| 0.031 | 0.10 | ColorTransfer (node) | the whole node | A colour-transfer kernel matching the source to a reference image; read the guide page first for which statistics Nuke matches, then it is a small kernel. |
| 0.031 | 0.10 | LiveGroup (node) | the whole node: 11 knobs | Groups that load from and publish to a separate file, with an editable flag and reload; a document-format decision (references between projects). |
| 0.031 | 0.10 | LiveInput (node) | the whole node: 4 knobs | Only meaningful with LiveGroup; comes with it. |
| 0.031 | 0.10 | VariableGroup (node) | the whole node | Graph-scoped variables (Nuke's GSVs) that expressions read and groups override; an expression-engine and document decision. |
| 0.031 | 0.10 | Link (node) | the whole node | A node that mirrors another node's knobs by reference; needs a link model in the document and the Properties panel. |
| 0.015 | 0.05 | Viewer GPU options (3 knobs) | use GPU for inputs, GPU dither and its algorithm | Decide whether the Viewer exposes its GPU path as user switches at all; NodeBased picks the adapter itself. |
| 0.012 | 0.04 | Stereo views (Roto's view, stereo offset, source view; Root's views table) | three Roto knobs here, the Root views knobs inside the Root line above | A views model in the document (a list of named views, a current view, per-view knob values); every node would carry it, so it is a schema-wide change. |
| 0.004 | 0.01 | Non-square pixels (Reformat's box pixel aspect) | one knob; `pixel_aspect` elsewhere is carried and changes no pixels | Anamorphic support: pixel aspect read by Reformat, Transform and the Viewer's display scale; NodeBased works in square pixels throughout. |
| 0.001 | 0.00 | Lens solver (LensDistortion's solve, solve error, preview) | three knobs | A grid-detection and distortion-fit solver inside LensDistortion; today the coefficients are typed in. |
| (not scored) | — | UDIM Import (rank 5) | the audit cannot score it (the guide lists no counted knobs), weight 0.2 | Reading a UDIM tile set as one image (a file-pattern reader plus a tiled UV layout); decide whether NodeBased reads UDIM sequences. |

How the numbers were made: the audit's own units (`tools/audit_nuke_fidelity.py`, `build_units`), weighted as the
report weights them. A knob line uses the audit's knob component (primary knobs weigh 2, secondary 1) and leaves out
the knob's effect on the acts and defaults components, so it is a slight underestimate.
