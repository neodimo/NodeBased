# ADR-0003 — Capability negotiation is a lowering pass

- **Status:** proposed; implementation deferred past E1
- **Decision:** resolve abstract operations into inspectable Plans through
  legalization and explicit conversions, not direct capability dispatch.
- **Context:** mixed representations require conversion choices with different
  costs and losses.
- **Alternatives:** multimethod dispatch; adapter selection in each op; implicit
  conversion to a common buffer.
- **Why:** compiler-style lowering centralizes legality, cost, fidelity floors,
  and explanation.
- **Revisit when:** Phase 2 examples show deterministic local dispatch is enough
  or the planner cannot remain predictable/pinnable.
