# The Hot pour in a shot

This example puts the Hot pour preset into a camera-matched tabletop plate. The 32-cell liquid and steam
simulation spans 120 frames and is sized to bake in under 20 minutes on this machine (the measured time is
recorded in `docs/FLUIDS_SPIKE.md`). The tabletop is a shadow catcher; the foreground cup is a holdout. A
soft rectangular key light casts the fluid's shadow onto the plate.

`make_shot.py` regenerates `hot_pour_shot.nbcomp`; `make_plate.py` renders the solid-only plate from the same
camera and set. Both scripts use the project's Python environment. The plate is committed as `plate.png`.
The comp graph renders the fluid scene over that plate. `hot_pour_shot.nbcomp` is the artist-facing graph.

`render_evidence.py --adapter default --workdir <cache-directory>` compares CPU/GPU output on frames 1, 60
and 120, then writes the 960 × 540 showcase stills at frames 60, 90 and 120. The showcase uses adaptive
sampling (16 minimum, 64 maximum, 0.01 relative-noise threshold), a four-bounce transmission cap, a rising turbulent steam plume and broad ambient fill. The point key light retains the stream highlight; ambient fill softens the tabletop shadow contrast while the volume remains free of the coarse one-step shadow estimate that darkened the transparent liquid. Keep the work
directory on a volume with enough room for the simulation cache; `--small-only` skips the stills.
