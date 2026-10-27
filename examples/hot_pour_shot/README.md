# The Hot pour in a shot

This example puts the Hot pour preset into a camera-matched tabletop shot. The 64-cell liquid and 96-cell steam
simulations span 120 frames and bake in under 20 minutes on the RTX 3080 Ti. The glass, liquid and table render
together in 3D under a 1 × 1 Rect key, which gives the tabletop shadow a soft edge. The spout starts above the
camera's frame; a narrow continuity bridge follows its animated path into the glass.

`make_shot.py` regenerates `hot_pour_shot.nbcomp`; `make_plate.py` renders the camera-matched table and backdrop.
Both scripts use the project's Python environment. The plate is committed as `plate.png`. The comp graph renders
the live glass, liquid and table over that plate, then the steam volume. `hot_pour_shot.nbcomp` is the artist-facing graph.

`render_evidence.py --adapter default --workdir <cache-directory>` compares CPU/GPU output on frames 1, 60
and 120, then writes the 960 × 540 showcase stills at frames 60, 90 and 120. The showcase uses 32 fixed
path samples and final denoising. Source noise and added turbulence are off; the 96-cell, slow-rising steam source
and smaller raymarch step produce the visible column. Before/after frames and timings are recorded in
`docs/FLUIDS_SPIKE.md`. Volume shadows remain disabled to avoid the coarse one-step estimate that darkened the transparent liquid. Keep
the work directory on a volume with enough room for the simulation cache; `--small-only` skips the stills.
