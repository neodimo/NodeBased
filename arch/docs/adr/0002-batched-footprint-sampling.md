# ADR-0002 — Sampling is batched and footprint-aware

- **Status:** accepted for experiment
- **Decision:** the sole portable sampling shape is
  `sample(FootprintBatch) -> SampleBatch`; no scalar protocol exists.
- **Context:** point queries cannot express filtering under transforms, and
  per-sample dynamic dispatch prevents GPU/SIMD execution.
- **Alternatives:** scalar samples plus optional derivatives; pixel-sized
  samples; backend-specific sampling APIs.
- **Why:** one strict shape makes filtering information and batch execution
  available from the first implementation.
- **Revisit when:** E1 shows footprint transport cannot represent enough useful
  filtering behavior even with declared approximation.

