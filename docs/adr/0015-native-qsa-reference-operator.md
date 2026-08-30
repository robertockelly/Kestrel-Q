# ADR 0015 — Native QSA reference operator

Status: **Accepted**

## Context

Kestrel-Q now has a scalar GDN reference operator and the underlying container,
semantic, tensor-view and numeric infrastructure. The remaining attention path
is Qwen3.8 QSA, whose sparse selection and growing state require independent
validation before optimization.

## Decision

1. Implement QSA as a separate scalar CPU reference operator.
2. Define sparse selected block/token IDs and ordering as exact correctness
   outputs.
3. Treat K/V and indexer state transitions as first-class golden evidence.
4. Validate against pinned independent Class-C vectors.
5. Use reduced synthetic cases plus a bounded threshold-crossing corpus to
   exercise sparse selection.
6. Validate the real target GGUF bindings separately from the synthetic oracle.
7. Defer final KV-cache allocation/scheduling policy.
8. Introduce no SIMD/CUDA QSA kernel before the scalar reference stabilizes.
9. Require future optimized QSA implementations to validate against this
   reference.
10. Keep QSA separate from MoE and PLE value execution.

The canonical operational contract is fixed by
`docs/QSA-OPERATOR-CONTRACT.md`.  The target uses 24 Q heads, two K/V heads,
256-wide core heads, four 128-wide index-query heads, one raw index key,
four-token blocks and a 512-block limit.  The selected block/token stream is an
exact correctness output.  K, V and raw index keys are explicit bounded stream
state; final capacity/residency policy remains deferred.

The scalar executable path uses F32 synthetic weights and caller scratch.  The
real GGUF gate is structural and reads no payload.  An independently generated
Tier-B selection corpus exercises 512 and 513 candidates directly through the
same checked selector used by the operator, avoiding a full-model or
quadratic target-context run.

## Evidence gate

Require:
- exact characterized QSA contract;
- calibration and disjoint holdout PASS;
- exact sparse selection PASS;
- prefill/decode/state PASS;
- 12/12 target QSA layers valid;
- 36/36 GDN IDs rejected;
- target state-growth reconciliation PASS;
- CPU/CUDA regressions PASS.

## Evidence

- independent pinned Class-C oracle: five calibration cases, six disjoint
  holdout/decode cases and three threshold-selection cases;
- exact block/token selection, prefill/decode continuation and state: PASS;
- floating operator/state/score/probability contracts: calibrated separately,
  holdout PASS;
- 12/12 real QSA configs valid, 36/36 GDN IDs reject;
- BF16 semantic state growth: 2,304 bytes/token/layer and 27,648 bytes/token
  across 12 QSA layers;
- real model payload bytes touched: zero;
- clean CPU and CUDA regression gates: PASS.

## Consequences

Future optimized QSA paths must reproduce the exact discrete selection stream
and meet checkpoint-specific numeric contracts.  This ADR does not authorize
MoE, PLE value execution, a complete layer/model, final KV-cache scheduling,
SIMD or CUDA QSA kernels.  `KQ-BACKLOG-BENCH-002` remains deferred.
