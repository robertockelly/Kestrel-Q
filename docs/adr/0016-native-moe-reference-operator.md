# ADR 0016 — Native MoE reference operator

Status: **Accepted**

## Context

Kestrel-Q now has scalar GDN and QSA reference operators, bounded quantized
expert views, and independently validated scalar numerics.

## Decision

1. Implement MoE as a separate scalar CPU reference operator.
2. Treat router top-k expert IDs/order as exact correctness outputs.
3. Execute only selected routed experts.
4. Keep shared-expert execution separate from routing and combine according to
   the canonical equation.
5. Define correctness against independent pinned Class-C vectors.
6. Validate real expert-stack/view plumbing separately.
7. Defer expert cache/residency/prefetch/scheduler policy.
8. Introduce no SIMD/CUDA optimization before scalar reference stability.
9. Require future optimized MoE implementations to validate against this
   reference.
10. Keep MoE separate from PLE value execution and full-layer composition.

## Evidence gate

Require:
- exact characterized MoE contract;
- reduced oracle PASS;
- canonical 512/top-10 selection PASS;
- calibration/holdout PASS;
- real target bindings/geometry PASS;
- CPU/CUDA regressions PASS.

## Evidence

The pinned Apache-2.0 Transformers oracle generated deterministic expected
values before native comparison. Five Tier-A calibration cases, four disjoint
holdout cases and seven Tier-B 512-expert/top-10 routing cases pass. Expert
IDs/order are exact; routed, shared and final floating checkpoints pass their
separate calibration-only contracts.

All 48 target MoE configs validate the 512-expert axis, ordered gate/up split,
shared bindings and physical types. Bounded views re-derive the three packed
expert-footprint groups while touching zero model-payload bytes. CPU and CUDA
Release regression suites pass.

## Consequences

- Future optimized MoE paths must preserve exact routing and validate against
  this scalar/reference evidence.
- Physical selection and semantic execution remain separate; the reference
  operator does not define a cache or I/O-per-token policy.
- The pinned CPU tie behavior is model-oracle evidence, not a claim that
  arbitrary `torch.topk` implementations promise a portable stable ordering.
- Expert residency, prefetch and scheduling remain deferred. In particular,
  `KQ-BACKLOG-BENCH-002` is still required before final storage policy.
