# ADR 0018 — Native transformer layer reference

Status: **Accepted**

## Context

Kestrel-Q has independently validated scalar reference operators for GDN, QSA,
MoE and PLE value execution, but no canonical complete transformer-layer
composition.

## Decision

1. Compose isolated operators only through a dedicated canonical layer
   reference.
2. Keep layer-owned GR/residual semantics separate from suboperator internals.
3. Make layer family explicit and validated by layer ID.
4. Apply PLE only at the exact canonical layer and position.
5. Compose only persistent suboperator state; do not turn forward-only GR
   activations into cache state.
6. Define correctness against independent complete-layer Class-C vectors.
7. Use an explicit transactional policy for multi-state layer failures.
8. Keep Task 2.10 as a one-layer reference, not a full-model executor.
9. Require future optimized layer paths to validate against this reference.
10. Keep embedding, model loop, final norm, LM head and logits as subsequent
    milestones.

## Evidence gate

Require:
- exact layer/GR contract;
- independent ordinary-GDN, QSA and PLE-enabled layer vectors;
- calibration/holdout PASS;
- persistent state transitions PASS;
- 48/48 target layer configurations valid;
- exact PLE placement;
- state-footprint reconciliation PASS;
- CPU/CUDA regressions PASS.

## Evidence and consequences

The pinned decoder layer proves PLE is added before the layer-1 attention GR,
then the mixer and MoE are each enclosed by a separate four-branch GR. The
native scalar implementation matches independent complete-layer calibration
and disjoint holdout for ordinary GDN, QSA and PLE-GDN families, with a frozen
`2e-6` maximum-absolute boundary contract. Synthetic state tests prove
prefill/decode continuation, reset/replay and rollback after downstream MoE
failure. The verified artifact validates 35 ordinary-GDN, 12 QSA and one
PLE-GDN config with zero payload access.

The selected transactional design keeps committed and staging copies of
persistent suboperator state. This intentionally costs bounded additional
reference memory in exchange for an unambiguous no-partial-advance contract.
Future optimized ownership may change only behind equivalent tests and a
separate measured decision.

ADR 0018 is **ACCEPTED**. It does not authorize embedding lookup, a 48-layer
executor, final norm, LM head/logits, sampling, final scheduler/cache policy,
SIMD or CUDA model kernels.
