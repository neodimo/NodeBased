# ADR-0005 — Reduction separates family generation from the fold

- **Status:** accepted for E2 experiment
- **Decision:** model ray reduction as two stages: a capability adapter generates
  an ordered family of operation-local contributions; a portable combiner folds
  that family. The initial portable family contains depth, premultiplied RGBA,
  and an active mask only.
- **Context:** mesh visibility requires intersection while volume visibility
  requires density integration. Calling both `sample(ray)` hides the important
  difference, but opaque first-hit and Beer–Lambert accumulation share ordered
  front-to-back algebra after representation-specific work.
- **Alternatives:** put the entire reduction in each adapter; force both through
  a `Sampleable` field; make a universal deep-sample representation.
- **Why:** the split exposes the actual shared law without conflating query
  semantics. The intermediate is lowering IR, not a lowest-common-denominator
  stored representation.
- **Revisit when:** a third reducer cannot use the fold without adding native
  fields, or materializing contribution families is too expensive for GPU or
  streaming execution. A production implementation may fuse both stages while
  preserving their logical contract.
