# ADR 0019 — Target quantized layer execution

Status: **Accepted**

## Context

Kestrel-Q has independently validated scalar operator and complete-layer
semantics, plus bounded GGUF views and scalar quantized numerics. Complete
layers have not yet executed against the actual quantized target weights.

## Decision

1. Introduce a semantic target-weight provider between model equations and GGUF
   storage.
2. Implement real quantized linear access through Task 2.2 views and Task 2.5
   numerics.
3. Never materialize a complete target weight matrix in F32.
4. Keep split/transformed physical representation behind the provider.
5. Access only routed experts selected by the canonical router.
6. Access only PLE rows requested by canonical address intents.
7. Validate target quantized layer execution against an independent Class-Q
   decode plus canonical-equation oracle.
8. Treat Task 2.11 timings as characterization, not storage-performance
   evidence.
9. Keep full 48-layer execution as the next milestone.
10. Require future optimized providers to preserve the semantic contract.

## Evidence gate

Require:
- ordinary GDN real layer PASS;
- QSA real layer PASS;
- PLE-GDN real layer PASS;
- calibration/holdout PASS;
- exact discrete selections PASS;
- access invariants PASS;
- 48/48 provider compatibility preflight;
- CPU/CUDA regressions PASS.

## Evidence and consequences

The accepted implementation preserves one operator/layer algorithm and adds a
semantic provider only at weight-access points. It inverts the verified
converter transforms, preserves split bindings, opens selected expert members
and exact PLE rows, and reuses Task 2.5 numerics. No complete target matrix is
materialized in F32.

Task 2.12 sharpened that accepted boundary: the converter's zero-centered
`+1` storage rule applies to HC, QSA/indexer and PLE norms, while the GDN
linear-attention norm is direct gamma. The independent Task 2.11 generator and
provider now both perform that source-proven inverse exactly once.

Pinned llama.cpp decoding plus pinned Transformers equations independently
generate four calibration profiles and a disjoint holdout for target layers 0,
3 and 1. Native floating outputs pass all calibrated contracts; MoE top-10
order, selected-expert access and PLE intents pass exact comparison. Provider
preflight passes 48/48 layers with zero payload dereference.

The accepted correctness run accounts 772,826,304 logical packed bytes,
including two rollback injections, below the 768 MiB ceiling. The largest
simultaneous F32 weight materialization is 163,840 bytes. These values do not
measure disk I/O, residency or throughput.

Embedding, full 48-layer orchestration, final norm, LM head/logits and greedy
decode remain separate work. A production cache/prefetch/scheduler requires
measured benefit and separate governance; `KQ-BACKLOG-BENCH-002` remains
deferred.
