# Fire shot

Run `make_shot.py` to regenerate the editable NodeBased graph: a 64-cell fuel fire at 128 path-traced samples with a
seeded sideways wobble on the source velocity (`make_shot.build_document(division_size=.0625, pt_samples=32, curl=0)`
rebuilds the earlier 32-cell straight shot).

`render_evidence.py` has three modes, all taking `--workdir` for a disposable simulation cache outside the repository:

- default: bakes to frame 24, renders frames 8, 16 and 24, writes a frame-16 comparison with fire lighting on meshes
  switched off, and records flame occupancy and grain in `benchmarks/fire_shot/fire-look-shot-<tag>.json`;
- `--compare`: 32 and 64 cells at 32 and 128 samples, with a contact sheet and image differences;
- `--curl-sweep`: 64 cells at 128 samples with several wobble strengths;
- `--grain`: renders the shot with the old (collision-summed) and the new (integrated) fire-emission estimator, each in its own
  process, and writes grain, render time, per-component grain at frame 16 and the mean agreement at 2048 samples to
  `benchmarks/fire_shot/fire-grain-<tag>.json`, with a before/after contact sheet and a zoom of the upper plume.

The shot contains a floor, a stone beside the fire and one dim blue fill light. The simulation cache is disposable; only
the graph, scripts, benchmark files and rendered stills are deliverables.
