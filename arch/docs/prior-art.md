# Prior art — ideas to steal, boundaries to keep

This is a design preflight, not a dependency list. Existing systems solve large
parts of the problem, but none is adopted as Carta's core.

- **Nuke:** steal pull-based region evaluation, channels, explicit DAGs, and
  compositing algebra. Do not inherit the assumption that every value is a 2D
  image at evaluation boundaries.
- **Houdini:** steal immutable-ish geometry flows, attributes, proceduralism,
  contexts, and visible cooking. Avoid context-specific mini-languages becoming
  representation silos.
- **Blender/depsgraph:** steal separation of authored state and evaluated state.
  Avoid coupling core meaning to one scene object model.
- **OpenUSD:** steal composed opinions, stable identities, time samples, schemas,
  and non-destructive layering. Treat USD as Phase 8 interchange: its scene
  semantics are broader than an edit Score and its imaging assumptions are not
  Carta's capability protocol.
- **MaterialX/shader graphs:** steal typed ports, portable intent, and lowering to
  targets. Avoid equating all computation with dense shader values.
- **Shader sampling:** steal texture footprints, derivatives, mip selection, and
  batched/SIMD execution. Generalize them beyond pixels rather than hiding them
  behind `sample(point)`.
- **Scene graphs/ECS:** steal identity and transform composition; reject hierarchy
  or component bags as the universal evaluation model.
- **LLVM/MLIR:** steal legalization, target capabilities, canonicalization,
  inspectable IR, and deterministic cost models. This directly motivates Plan
  lowering rather than capability dispatch.
- **Tensor frameworks:** steal lazy graphs, device placement, autodiff as an
  optional transform, and shape discipline. Reject dense tensors as the common
  representation and reject implicit global mutation.
- **Scientific computing/xarray:** steal named dimensions, coordinates, units,
  chunking, and explicit missing data.
- **Trait/protocol systems:** steal structural capability conformance and narrow
  contracts. Prefer a few graded capabilities over interface confetti.
- **Functional programming:** steal pure graph meaning, referential transparency
  downstream of `Realize`, and algebraic laws that authorize scheduling.
- **Differentiable rendering:** steal gradients as a declared capability and
  visibility-aware estimators. Never make differentiability universal.

The result is deliberately a small custom experimental core now. Reusing NumPy
is adequate for falsification; adopting a large scene or ML framework would
front-load constraints before the central hypothesis is tested.

