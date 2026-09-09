# Technology choice

## Recommendation

- **Prototype:** Python 3.11+ and NumPy, isolated under `arch/`. It minimizes
  experiment cost and aligns with the current compositor without importing it.
- **Core runtime candidate:** Rust after protocol shapes survive Phase 0/1.
  Ownership and enums help explicit resource/fidelity state; C ABI and Python
  bindings keep adapters open. This is a direction, not a commitment.
- **GPU strategy:** WGSL/WebGPU for portable first coverage, with batched kernels
  and an optional native backend only when measurement shows a gap.
- **Plugin boundary:** versioned C ABI for stable runtime mechanics; capability
  and intent descriptors serialized independently; Python for experimental
  adapters and PyTorch/JAX bridges.

## Alternatives

- **C++:** strongest graphics ecosystem and mature native integration, but a
  larger safety/build burden for a new core. Revisit if host-DCC embedding
  dominates.
- **All Python:** right for experiments, wrong for interactive scheduling and
  long-lived ABI guarantees.
- **CUDA-first:** excellent on one vendor, violates the durable portable core
  requirement. Keep as an adapter/backend, never the contract.
- **PyTorch/JAX core:** convenient for learned representations but silently makes
  dense differentiable tensors the architecture. Use only behind adapters.

No runtime-language commitment should be made before E1/E2 determine whether
the contracts are worth porting.
