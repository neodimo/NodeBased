# Lane 8 E3 — queue chains and LAN workers

In progress on `openclaw/nb-2d-parity-b`. Scope: durable dependency-aware job chains, queue panel,
TCP remote workers with artifact synchronization, docs and CPU tests. Start commit: `ca956d4`.

# Lane 6 H2 finish — branch handoff

All requested code, measurements, and three-adapter targeted checks are committed on `openclaw/nb-fluids-spike`. `docs/FLUIDS_SPIKE.md` records the sparse/dense 256³ and 512³ comparisons and the 256³ path-traced plume before/after timing and 64-sample noise check. The retained tiles are compact; existing `Volume` consumers still expand them for rendering. Awaiting Gonzo's review, integration suite, and merge. No Windows or real-display run by this lane.
