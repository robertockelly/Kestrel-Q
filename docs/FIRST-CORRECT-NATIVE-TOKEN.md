# M1 — First Correct Native Token

Status: **PASS**

M1 means one independently verified greedy next token produced by the native
C17 runtime from the registered real quantized artifact. It is a correctness
milestone, not a throughput or usability claim.

## Identity and input

- canonical model: `Qwen/Qwen3.8-Flash-Next` at
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- artifact: `KQ-MODEL-ARTIFACT-001`,
  `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`, 111,334,654,400 bytes, SHA-256
  `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`;
- prompt: `KQ-PROMPT-001`, exact UTF-8 `Hello, Kestrel-Q.`, SHA-256
  `6a013508a5747146c4c37c1cdc104d49d2a75dbaa2950b23e2d72eb8ac2b4787`;
- canonical/native IDs: `[9419, 11, 710, 467, 3621, 27325, 13]`;
- batch 1, context capacity 8, CPU scalar reference, one greedy token.

The native path decoded seven bounded embedding rows, executed all 48 layers
in exact order, applied the final hyper-connection mixer, computed all 248,320
LM-head logits, selected token **271**, and decoded the exact bytes `\n\n`.
The token is not EOG.

## Independent result

The test-only oracle is pinned llama.cpp
`90c26fcd4b2114b4aa39d09d69318cb8f438d27a` (MIT), CPU-only with mmap,
lazy tensor reads and context 8. It received the explicit committed canonical
IDs, bypassing the divergent GGUF tokenizer semantics proved by Task 2.3. It
also selected token **271** (`\n\n`). Native/oracle next-token equality is
therefore **EXACT DISCRETE PASS**, not self-oracle evidence.

Both rank token 271 first and token 353 second. Oracle logits were 16.343235
and 15.5642185; native logits were 16.6189995 and 15.4765329. Lower top-N order
differs under independently calibrated scalar/backend floating behavior, so
top-N floats remain descriptive rather than a newly invented full-model
tolerance. The oracle exposed a final layer-47 branch summary but no final
normalized-hidden vector; the native final normalized summary is recorded and
is not presented as an independent float match.

## Accounting

The successful native run took 151,061,011,900 ns on KQ-01. It recorded:

| Counter | Value |
|---|---:|
| Logical packed bytes | 40,208,768,960 |
| Quantized blocks | 1,693,222,880 |
| Unique semantic tensors | 1,239 |
| Embedding logical bytes | 19,040 |
| Routed expert selections | 3,360 |
| Selected expert matrix requests | 10,080 |
| PLE row requests | 112 |
| LM-head logical bytes | 675,430,400 |
| Owned model/layer state | 359,881,768 |
| Peak model scratch | 8,693,472 |
| Full logits buffer | 993,280 |
| Largest simultaneous provider F32 result/staging output | 993,280 |

The 40.21 GB value respects the 64 GiB correctness ceiling. It counts repeated
logical semantic payload, not disk I/O, page-cache traffic or storage
throughput. The largest provider result is the complete logits vector, not a
weight matrix. No complete target weight matrix or model was materialized in
F32.

At context capacity 8 the released semantic-state accounting is 116,195,328
bytes of fixed GDN state, 221,184 bytes of allocated QSA K/V/raw-index capacity,
32 bytes of PLE address history and 184,320 bytes of PLE value history. Seven
prompt positions retain 193,536 of those QSA semantic bytes. Their 116,600,864
byte capacity total is distinct from the 359,881,768-byte scalar-correctness
container: the remaining 243,280,904 bytes are C-object overhead, F32 reference
representation, transactional duplicate slots and bounded copy workspace.

## Dirty-journey findings

The first native attempt failed after the provider's 256-entry semantic trace
filled during whole-model execution; the per-layer capacity did not cover the
1,294-entry registry. The fixed capacity is 2,048 and the exact token test is
the regression.

The next run selected token 44 because GGUF stores qwen4exp zero-centred
RMS gammas as `1 + delta`, while the canonical primitive also adds one. Moving
the inverse transform to the provider changed the result to token 20. Per-layer
oracle checkpoints then located the remaining first divergence at PLE layer 1:
PLE norms had been converted twice. Removing that duplicate produced token
271. The provider now performs each folded-gamma conversion exactly once.

Early-, middle- and late-layer injected provider failures preserve caller
outputs and position; each is followed by a complete control-equivalent
token-271 recovery. The strengthened test exposed a recovery-order bug: QSA
private staging was still advanced when the next call queried scratch geometry.
Resetting all dirty private layers before that query fixes the root cause and
is covered by the three recoveries. Invalid/padded IDs, context/logits capacity,
non-finite argmax and exact ties also fail closed.

## Limits

M1 has no multi-token decode API, batching, stochastic sampling, vision, MTP,
server, cache/prefetch scheduler, SIMD or CUDA model kernels. The state API is
caller-owned and short-context. `KQ-BACKLOG-BENCH-002` remains deferred before
final scheduler/residency policy.
