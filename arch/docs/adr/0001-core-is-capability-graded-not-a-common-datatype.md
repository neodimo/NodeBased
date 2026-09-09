# ADR-0001 — Capability grades, not a common datatype

- **Status:** accepted for experiment
- **Decision:** operations request graded capabilities; representations remain
  opaque and may implement subsets. There is no universal payload datatype.
- **Context:** raster, mesh, measures, fields, and model handles do not share an
  honest storage or query model.
- **Alternatives:** convert everything to RGBA; use a universal tensor; define a
  wide base representation interface.
- **Why:** every alternative either loses semantics or forces false methods.
- **Revisit when:** two independent adapters cannot express a needed shared op
  without representation branches above the adapter boundary.
