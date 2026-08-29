# ADR 0012 — Native PLE address engine

Status: **Accepted — 2026-08-29**

## Context

Epic 1 pinned deterministic PLE/N-gram address behavior and independent golden
vectors. Task 2.1 exposes canonical PLE members and Task 2.2 can create bounded
physical PLE member views, but storage policy remains intentionally deferred.

## Decision

1. Implement a native C17 Qwen3.8 PLE address engine.
2. Consume canonical token IDs and explicit bounded stream state.
3. Emit deterministic logical PLE member/table + row/index address intents.
4. Define exact fixed-width integer arithmetic.
5. Use identical canonical semantics for prefill and incremental decode.
6. Keep address generation independent from physical GGUF offsets.
7. Open no PLE payload view and perform no disk I/O.
8. Make output consumable by future prefetch logic without initiating prefetch.
9. Keep `PLE_DISK_BACKED_CANDIDATE` preliminary.
10. Keep `KQ-BACKLOG-BENCH-002` required before final storage/cache/scheduler
    policy.

## Acceptance gate (satisfied)

- 7/7 sequence goldens exact;
- 4/4 decode goldens exact;
- independent PLE golden SHA unchanged;
- incremental == recomputation;
- native tokenizer-to-PLE integration;
- fail-closed incompatible PLE configuration;
- zero PLE payload views and zero model tensor payload touch.

## Evidence

- The exact PLE contract is pinned in `docs/PLE-ADDRESS-CONTRACT.md` against
  the official config and Apache-2.0 Transformers revision/source hash.
- The immutable production config validates all 128 fused logical members,
  six PLE dense semantics, three metadata-derived arrays and target topology.
- The bounded 32-byte stream state supports transactional prefill and decode;
  incremental output equals canonical full recomputation.
- All seven original sequence vectors and four decode steps pass exactly and
  the original SHA-256 is unchanged. Independent regeneration is
  byte-identical.
- Expanded independent evidence SHA-256
  `b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`
  passes 12 sequence cases, three stream cases and tokenizer integration.
- Synthetic incompatible-config/state/capacity/overflow tests fail closed.
- Real-artifact validation reports zero PLE payload views and zero tensor
  payload bytes touched.

This ADR accepts the logical address boundary only. It does not accept a PLE
storage, cache, prefetch or scheduling policy.
