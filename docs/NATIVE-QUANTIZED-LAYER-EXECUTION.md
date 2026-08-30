# Native target-quantized layer execution

Status: **TASK 2.11 COMPLETE / PASS**

Task 2.11 connects the accepted scalar layer equations to the exact registered
GGUF without creating a second model algorithm. Provider-aware adapters in the
GDN, QSA, MoE, PLE-value and gated-residual modules replace only weight access;
their operation order, state transitions and F32 reference numerics are the
same paths accepted by Tasks 2.6–2.10.

## Independent oracle

The research generator uses two authorities that are independent of native
Kestrel-Q output:

1. `llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`
   (MIT) decodes every requested physical block/row;
2. pinned Transformers
   `805a9e939fa8c1bff8d8ffdf041c051b71a914aa` (Apache-2.0) executes the
   canonical Qwen equations for model revision
   `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

The oracle uses the exact 111,334,654,400-byte GGUF identity
`8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`.
It never downloads canonical BF16 weights and never consumes native output as
expected data. Temporary decoded rows/matrices stay below ignored
`.research-cache`; no raw packed or decoded real weight is committed.

Two deterministic calibration profiles and one disjoint holdout cover
one-token prefill followed by one decode step for:

| Family | zero-based layer | Result |
|---|---:|---|
| ordinary GDN | 0 | PASS |
| QSA | 3 | PASS |
| PLE-GDN | 1 | PASS |

All 18 output comparisons (12 calibration and 6 holdout) pass their
per-family/per-phase calibrated absolute-or-relative contracts. Observed
holdout maximum absolute differences are `9.0003014e-6` (GDN decode),
`6.556511e-6` (QSA prefill) and `1.5050173e-6` (PLE-GDN prefill). Large ULP or
relative values near zero are not used alone as acceptance criteria.

Exact-discrete validation passes for all six top-10 route vectors in each
corpus, including order, and for all 32 PLE member/row requests. The QSA
one-token/two-token cases have zero completed historical candidate blocks and
exact selected local tails of one then two tokens. Physical expert access is
the exact selected set, although execution is deliberately grouped by
ascending expert ID for deterministic accumulation.

The continuation decode is evaluated from the state produced by the preceding
prefill, so all three family comparisons are state-dependent. The governed
state asset additionally records the independent QSA K/V/raw-index cache
identity. Full GDN, QSA and PLE state/checkpoint contracts remain covered by
the independent Tasks 2.6, 2.7, 2.9 and 2.10 vectors; Task 2.11 changes only
their weight source.

## Real execution and state

The real integration creates reference-F32 state/config objects from the
verified target semantics, then executes dequantized weights directly from
bounded views. It runs reset/one-token prefill and a continuation decode for
all three families. GDN recurrent/conv, QSA K/V/indexer and PLE address/value
state remain the same explicit transactional substate objects accepted by the
earlier operator tasks.

Representative caller scratch for a one-token call is 4,904,992 bytes (GDN),
1,855,356 bytes (QSA capacity 8) and 4,906,016 bytes (PLE-GDN). Owned double-
slot transactional state is 9,929,008, 110,928 and 11,034,928 bytes
respectively. These are Release characterization measurements, not performance
targets.

The full provider preflight validates 48/48 layer configurations: all semantic
roles/types/transforms, both endpoints of every routed stack, and all PLE
members. It opens no payload accessor. Two real provider failures—before the
first payload read and after 7,332,736 logical packed bytes—leave output,
visible state and position unchanged.

The accepted correctness run touches 765,493,568 logical packed bytes in the
three successful layer executions and 772,826,304 bytes after including the
two rollback injections, below the 805,306,368-byte (768 MiB) ceiling. It
accounts 32,652,832 quantized blocks, 109 unique semantic tensors, 180 selected
expert matrix requests and 32 exact PLE rows. The provider object is 4,232
bytes and the largest simultaneous F32 weight materialization is 163,840
bytes.

One observed KQ-01 Release characterization run reported 0.397–0.492 seconds
per representative one-token layer phase and 2.486 seconds accumulated inside
provider scalar row-dot loops. This single synchronous correctness run is not
a benchmark: cache warmth and physical storage residency were uncontrolled,
and no throughput or performance comparison is claimed.

Clean Release validation passes 34/34 CPU tests and 36/36 CUDA tests. The CUDA
build retains only the previously documented NVCC-generated C4211 warning; the
new C17 code emits no `/W4` warning.

## Scope

This task does not implement token embedding, a 48-layer loop, final norm, LM
head/logits, argmax/sampling, native decode, an expert/PLE cache, prefetch,
scheduler, SIMD or CUDA model kernels. Logical payload counters are not disk
throughput or residency measurements. A longer real prefill would exceed the
bounded correctness budget with this deliberately uncached scalar provider;
the accepted real cases therefore use one-token prefill followed by one decode
step for each family.
