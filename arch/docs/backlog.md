# Carta backlog

## Now — Phase 0

- [x] Write Round 0 investigation and adversarial review.
- [x] E1: implement the Chart Test with raster, SDF, and splat adapters.
- [x] Record E1 measurements and revise the abstraction.
- [ ] E2: visibility test — mesh first-hit and volume integration through one
  `Reduce` shape, without claiming they are the same sampler.
  Protocol fixed in [`experiments/e2-visibility-reduction.md`](experiments/e2-visibility-reduction.md);
  implementation in progress.

## Next — Phase 1

- [ ] Space/unit mismatch and inference tests.
- [ ] Partial maps and validity propagation.
- [ ] Conservative nonlinear footprint transport.
- [ ] Fidelity composition and floor experiment.
- [ ] Value typing for color, labels, normals, and distributions.
- [ ] Decide whether footprint shape/filter intent belongs in the sample query
  or reconstruction policy; E1's basis alone leaves this adapter-defined.

## Later

- [ ] Score schema/version/migration spike.
- [ ] Capability lowering planner and inspectable plans.
- [ ] Logical versus bitwise cache identity.
- [ ] Streaming/locality contracts for video and out-of-core volume.
- [ ] One learned field adapter after deterministic adapters survive.

## Explicitly deferred

- Diffusion backend and latent-space manipulation.
- Cross-representation topology abstraction.
- Production GPU runtime, DCC UI, and broad interchange.
