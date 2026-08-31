# ADR 0021 — Native multi-token greedy generation

Status: **Accepted**

## Decision

1. Prompt prefill occurs exactly once.
2. Multi-token generation advances through explicit one-token decode steps.
3. Generated token IDs feed model decode directly.
4. Every decode step is transactional.
5. A later failed step preserves the successful generated prefix.
6. The entire greedy token sequence must exactly match an independent
   full-model Class-Q oracle.
7. Correctness runs use small explicit context capacity.
8. Prompt replay is prohibited.
9. Stochastic sampling and runtime optimization remain deferred.
10. M1 one-token evidence remains immutable and backward compatible.

The model executor retains its explicit immutable configuration and bounded
per-session state. Successful prefill makes that state directly continuable;
`decode_one` consumes the previously selected canonical token ID and commits
one new position only after all 48 layers, final mixing, logits, argmax and
native decode succeed. Layer transaction slots provide reverse-order rollback
when a later layer or post-layer stage fails. Weight-provider counters and
diagnostic traces are monotonic audit data, not persistent semantic state.

The governed Task 2.13 case uses context capacity 16. Pinned
`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`, receiving the explicit
canonical prompt IDs, independently produced `[271, 248068, 198, 760]`.
Native Kestrel-Q produced the same four IDs and decoded fragments
`["\n\n", "<think>", "\n", "The"]`. The first-token M1 contract therefore
remains unchanged.

## Evidence gate

Require:
- independent multi-token continuation;
- exact oracle/native token sequence;
- M1 regression;
- state continuity;
- per-step rollback/retry;
- QSA/MoE/PLE exact invariants;
- CPU/CUDA regressions.

All gates passed. The six governed `multi-token-*.json` artifacts under the
existing milestone revision root record independent-oracle provenance, native
state hashes, component invariants, rollback/retry, logical payload and memory.
ADR 0021 is accepted by Task 2.13.

## Consequences

- The public executor supports correctness-first greedy prefill plus one-token
  continuation; it is not a conversation/session product abstraction.
- `kq-run --max-new-tokens N --greedy` supports the governed bounded range and
  never retokenizes accumulated output or replays the prompt.
- Sampling, batching, MTP/speculation, cache/prefetch policy, SIMD and CUDA
  model kernels remain outside this decision.
- `KQ-BACKLOG-BENCH-002` remains deferred and required before final PLE
  disk-backed policy.
