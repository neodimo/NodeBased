# The Hot pour in a shot

This example puts the Hot pour preset into a camera-matched tabletop plate. The 32-cell liquid and steam
simulation spans 120 frames and is sized to bake in under 20 minutes on this machine (the measured time is
recorded in `docs/FLUIDS_SPIKE.md`). The tabletop is a shadow catcher; the foreground cup is a holdout. A
directional key light casts the fluid's shadow onto the plate.

`make_shot.py` regenerates `hot_pour_shot.nbcomp`; `make_plate.py` renders the solid-only plate from the same
camera and set. Both scripts use the project's Python environment. The plate is committed as `plate.png`.
The comp graph renders the fluid scene over that plate. `hot_pour_shot.nbcomp` is the artist-facing graph.

`render_evidence.py --adapter default --workdir <cache-directory>` compares CPU/GPU output on frames 1, 60
and 120, then writes the 960 × 540 showcase stills at frames 60, 90 and 120. The showcase uses adaptive
sampling (8 minimum, 32 maximum, 0.01 relative-noise threshold), a four-bounce transmission cap and no steam cast shadows (the volumetric steam itself stays enabled). This avoids the coarse one-step volume-shadow estimate that darkened the transparent liquid; the path-traced steam keeps density-aware free-flight sampling. Keep the work
directory on a volume with enough room for the simulation cache; `--small-only` skips the stills.
