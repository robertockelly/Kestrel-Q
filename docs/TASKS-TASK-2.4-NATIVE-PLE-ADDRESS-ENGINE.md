# TASKS-TASK-2.4-NATIVE-PLE-ADDRESS-ENGINE.md

Status: **COMPLETE / PASS — 2026-08-29**

Evidence: original PLE SHA-256
`495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`;
expanded differential SHA-256
`b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`;
clean CPU Release 14/14 and CUDA Release 16/16.

## Baseline

- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.3 COMPLETE/PASS.
- [x] Confirm Task 2.4 NOT STARTED before edits.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization

- [x] Create `docs/PLE-ADDRESS-CONTRACT.md`.
- [x] Record exact n-gram orders.
- [x] Record head/table ordering.
- [x] Record primes/constants.
- [x] Record XOR/multiply/modulo semantics.
- [x] Record fixed-width integer/wrap behavior.
- [x] Record history/state requirements.
- [x] Separate address-state from value-state.
- [x] Record prefill/decode transitions.

## Config/API

- [x] Immutable PLE config.
- [x] Explicit stream state.
- [x] Stable logical address descriptor.
- [x] Reset API.
- [x] Capacity/required-size behavior.
- [x] No global mutable state.

## Model compatibility

- [x] Validate target model identity.
- [x] Validate 128 logical members.
- [x] Validate required metadata-derived semantics.
- [x] Validate n-gram/head/table geometry.
- [x] Reject incompatible near-match config.

## Arithmetic

- [x] Fixed-width integer types.
- [x] Explicit cast/wrap semantics.
- [x] Exact XOR/multiply/modulo order.
- [x] Address bounds.
- [x] No signed-overflow UB.

## Prefill/decode

- [x] Beginning-of-sequence semantics.
- [x] Exact history behavior.
- [x] Exact canonical output order.
- [x] Incremental one-token decode.
- [x] 4/4 golden decode steps.
- [x] Incremental == recomputation.
- [x] Reset/replay.

## Native integration

- [x] UTF-8 -> Task 2.3 tokenizer -> PLE.
- [x] Exact independent PLE results.
- [x] No production Python path.

## Positive tests

- [x] shortest history.
- [x] repeated token.
- [x] alternating token.
- [x] low/high canonical IDs.
- [x] history/order boundaries.
- [x] first/last head/member.
- [x] near-zero/near-max row.
- [x] bounded long prefill + decode.

## Fail closed

- [x] invalid token ID.
- [x] invalid arguments.
- [x] count/arithmetic overflow.
- [x] insufficient output capacity.
- [x] corrupted state.
- [x] wrong member/head/order config.
- [x] invalid row/modulo size.
- [x] missing metadata-derived semantics.
- [x] incompatible model.
- [x] lifetime misuse where detectable.

## Oracle

- [x] 7/7 sequence goldens exact.
- [x] 4/4 decode goldens exact.
- [x] original PLE SHA unchanged.
- [x] independent canonical regeneration PASS.
- [x] expanded differential corpus.
- [x] differential SHA recorded.
- [x] no self-oracle.

## Payload boundary

- [x] PLE payload views opened = 0.
- [x] tensor payload touched = 0.
- [x] no physical offset arithmetic in address engine.

## Metrics

- [x] config construction time.
- [x] config-owned bytes.
- [x] stream state bytes.
- [x] addresses/token.
- [x] prefill time.
- [x] decode-step time.
- [x] no storage benchmark.

## Regression

- [x] Clean CPU Release PASS.
- [x] CPU CTests PASS.
- [x] Clean CUDA Release PASS.
- [x] CUDA CTests PASS.
- [x] Task 2.0-2.3 remain PASS.
- [x] No new warnings.
- [x] `git diff --check` PASS.

## Docs/governance

- [x] `docs/PLE-ADDRESS-CONTRACT.md`.
- [x] `docs/NATIVE-PLE-ADDRESS-ENGINE.md`.
- [x] `docs/KQ-PLE-API.md`.
- [x] ADR 0012 finalized.
- [x] ARCHITECTURE/TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] Task 2.5 NOT STARTED.

## Safety

- [x] no PLE value lookup.
- [x] no disk/cache/prefetch/scheduler.
- [x] no tensor math/inference.
- [x] no model CUDA kernels.
- [x] no production research dependency.
- [x] no tracked model weights/secrets/local paths.
- [x] build trees untracked.
