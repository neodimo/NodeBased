# ADR-0004 — Preserve native intent and account for fidelity

- **Status:** accepted for experiment
- **Decision:** portable results carry ordered fidelity records; unsupported
  representation-local semantics remain versioned `Native` intents rather than
  being flattened into portable claims.
- **Context:** conversions enable composition but can silently create a
  lowest-common-denominator system; topology and some model operations have no
  portable meaning.
- **Alternatives:** forbid native operations; silently rasterize; expose every
  native operation as a core capability.
- **Why:** honest refusal and visible loss preserve rich backends and durable
  graph intent.
- **Revisit when:** fidelity records cannot guide a planner or native intents
  make Scores practically non-portable.
