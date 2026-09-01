# TASKS-TASK-3.1-SAMPLED-INCREMENTAL-GENERATION.md

## Baseline
- [x] HEAD = `9986b934d639b61575018653dfc8a4eabe2965ea`
- [x] Task 3.0 COMPLETE/PASS
- [x] ADR 0022 ACCEPTED
- [x] Task 3.1 NOT STARTED
- [x] Working tree clean

## Contract
- [x] SAMPLED-GENERATION-CONTRACT.md
- [x] Prefill/first-draw lifecycle
- [x] Incremental sampled-step lifecycle
- [x] Combined model/RNG transaction
- [x] EOG stop boundary
- [x] Padded-ID failure boundary
- [x] Retry/replay semantics

## Integration
- [x] Model executor calls Task 3.0 sampler
- [x] No sampler logic duplication
- [x] Explicit sampler config
- [x] Explicit caller-owned RNG state
- [x] Prompt prefill exactly once
- [x] Incremental sampled decode
- [x] No prompt replay

## Primary real trace
- [x] Governed prompt
- [x] Official sampling profile
- [x] Explicit seed/stream
- [x] Small context capacity
- [x] Up to 4 tokens/EOG
- [x] Per-step independent sampler equality
- [x] Exact RNG word/state equality

## Holdout
- [x] Disjoint seed/stream
- [x] No seed cherry-picking
- [x] Independent expected vectors
- [x] State/invariant checks PASS

## Replay
- [x] Same reset/config/seed/stream
- [x] Identical token sequence
- [x] Identical RNG states
- [x] Identical model-state summaries
- [x] Identical stop reason

## Transactions
- [x] Early model failure rollback
- [x] Middle model failure rollback
- [x] Late model failure rollback
- [x] Post-decode/pre-sample failure rollback
- [x] Padded-ID sampler failure rollback
- [x] RNG unchanged on failed step
- [x] Prefix preserved
- [x] Retry deterministic

## Stop/control
- [x] EOG synthetic/control integration
- [x] No decode after EOG
- [x] No RNG draw after EOG
- [x] max_new_tokens=0
- [x] max_new_tokens=1
- [x] context exhaustion
- [x] stopped-state behavior

## Exact lower-level invariants
- [x] QSA exact
- [x] MoE top-10 exact
- [x] Unselected experts untouched
- [x] PLE 16 rows/token exact
- [x] No speculative PLE rows

## Greedy preservation
- [x] M1 token 271 unchanged
- [x] Task 2.13 sequence unchanged
- [x] Historical evidence unchanged

## CLI
- [x] --sample
- [x] explicit seed
- [x] optional stream if governed
- [x] official profile
- [x] --greedy unchanged
- [x] mode conflict fail closed
- [x] bounded diagnostics

## Evidence
- [x] sampled-generation-contract
- [x] sampled-generation-oracle
- [x] sampled-generation-native
- [x] sampled-generation-state
- [x] sampled-generation-validation
- [x] sampled-generation-manifest
- [x] deterministic hashes PASS
- [x] no temporary full logits committed

## Payload/memory
- [x] Prefill counters
- [x] Per-decode counters
- [x] Total correctness ceiling
- [x] Model state
- [x] RNG/generation state
- [x] Transaction overhead
- [x] Sampling scratch
- [x] Peak scratch/logits
- [x] No full F32 weight matrix

## Fail closed
- [x] Missing sampler config
- [x] Corrupt RNG state
- [x] Invalid continuation
- [x] Padded-ID selection
- [x] Sampling failure
- [x] Provider failure
- [x] Context/position overflow
- [x] Stopped-state misuse
- [x] CLI invalid args

## Governance
- [x] ADR 0023 ACCEPTED
- [x] Docs/API updated
- [x] CHANGELOG updated
- [x] CPU clean PASS
- [x] CUDA clean PASS
- [x] Task 3.0 regression PASS
- [x] Repository safety PASS
- [x] git diff --check PASS
- [x] Task 3.1 COMPLETE/PASS only after all gates
- [x] Epic 3 COMPLETE/PASS only after Task 3.1 PASS
- [x] No commit/push
