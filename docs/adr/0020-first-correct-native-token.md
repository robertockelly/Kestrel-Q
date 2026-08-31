# ADR 0020 — First Correct Native Token

Status: **Accepted**

## Context

Kestrel-Q has independently validated tokenizer, storage/numerics, PLE,
GDN/QSA/MoE operators, canonical layer composition and execution of real
quantized target layers. The remaining milestone is model-level entry, 48-layer
orchestration and output/logit computation.

## Decision

1. Define M1 as one independently verified native greedy next token from the
   verified real quantized Qwen3.8 model.
2. Compose the existing target provider and 48 canonical layer references
   without rewriting operator equations.
3. Use explicitly bounded short-context model state for M1.
4. Access embedding and LM-head rows/chunks without full F32 matrix
   materialization.
5. Use deterministic greedy argmax only.
6. Require an independent full-model Class-Q oracle using explicit canonical
   token IDs.
7. Use committed canonical tokenizer evidence to establish oracle/native input
   identity.
8. Attach no performance guarantee to M1.
9. Defer scheduler/cache/SIMD/CUDA optimization until correctness is proven.
10. Keep vision, MTP, batching and stochastic sampling outside M1.

## Evidence gate

Require:
- native tokenizer input IDs exact;
- complete native 48-layer execution;
- independent full-model oracle;
- native greedy token ID exactly equal to oracle;
- native decode recorded;
- model-level transactional regression PASS;
- CPU/CUDA regressions PASS.

## Consequences

M1 proves correctness, not throughput. The scalar path rereads weights per
prompt token and therefore uses a governed 64 GiB logical-touch ceiling; its
40,208,768,960-byte result is not physical disk I/O. Future caching, SIMD,
CUDA, batching, sampling, vision and MTP remain separate measured work.

## Accepted evidence

On baseline `0ac3aa04c0e539cbf128c4e259a08c05472845a0`, the native path and
pinned llama.cpp `90c26fcd4b2114b4aa39d09d69318cb8f438d27a` both consume canonical
IDs `[9419,11,710,467,3621,27325,13]` and select token 271 (`\n\n`) from
the immutable registered GGUF. Native execution covers every layer in order,
touches no complete F32 target matrix and passes early/middle/late transactional
failure recovery plus clean CPU/CUDA regressions. Evidence is recorded under
`research/milestones/Qwen3.8-Flash-Next/`.
