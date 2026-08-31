# TASKS-TASK-2.13-MULTI-TOKEN-GREEDY-DECODE.md

Status: **COMPLETE / PASS**

## Baseline

- [x] HEAD = `1f2b967d6bf5ec7aaa263192462ede1eb5c9bd28`.
- [x] Initial worktree inspected; four pre-existing untracked Task 2.13
  governance files were preserved and incorporated rather than reset.
- [x] Task 2.12 / M1 PASS.
- [x] Task 2.13 was NOT STARTED in canonical status records.
- [x] KQ-BACKLOG-BENCH-002 DEFERRED.

## Contract

- [x] MULTI-TOKEN-DECODE-CONTRACT.md.
- [x] Prefill exactly once.
- [x] Direct generated-token decode.
- [x] Position progression.
- [x] GDN/QSA/PLE state carry.
- [x] EOG semantics.
- [x] max-new-token semantics.
- [x] Per-step transaction.

## Oracle

- [x] Same canonical prompt IDs.
- [x] Independent pinned llama.cpp oracle.
- [x] Up to 4 greedy tokens or canonical EOG.
- [x] First token remains 271.
- [x] Oracle sequence frozen before native validation.
- [x] No self-oracle.

## Native

- [x] Explicit prefill.
- [x] Explicit decode-one.
- [x] max_new_tokens loop.
- [x] Greedy only.
- [x] EOG stop.
- [x] No prompt replay.
- [x] Small explicit context capacity 16.

## State

- [x] Model position per token.
- [x] GDN state continuity.
- [x] QSA sequence length continuity.
- [x] PLE address continuity.
- [x] PLE value continuity.
- [x] Compact state summaries.
- [x] No unexpected reset.

## Exact invariants

- [x] Full native token sequence == oracle.
- [x] QSA block/tail behavior exact.
- [x] MoE top-10 exact.
- [x] Unselected experts untouched.
- [x] PLE 16 intents/token.
- [x] No speculative PLE rows.

## Transactions

- [x] Later-step injected failure.
- [x] Earlier prefix preserved.
- [x] Pre-step state restored.
- [x] Retry reproduces oracle token.

## CLI

- [x] `--max-new-tokens N`.
- [x] Prompt tokenized once.
- [x] Per-token progress.
- [x] Greedy only.
- [x] EOG stop.

## Evidence

- [x] multi-token contract.
- [x] multi-token oracle.
- [x] multi-token native.
- [x] multi-token state.
- [x] multi-token validation.
- [x] manifest/SHA.
- [x] Six-file deterministic regeneration byte-identical.

## Payload / memory

- [x] Prefill counters.
- [x] Per-token counters.
- [x] Total payload 59,212,012,160 bytes under 96 GiB.
- [x] State/memory/scratch.
- [x] No complete F32 target matrix.

## Fail closed

- [x] max_new_tokens=0.
- [x] max_new_tokens=1 M1 regression.
- [x] EOG early-stop branch and canonical EOG identities.
- [x] Context exhaustion.
- [x] Invalid continuation state.
- [x] Position/QSA capacity overflow.
- [x] Failed-step rollback/retry.
- [x] Invalid selected token.
- [x] Incremental decode failure.
- [x] Prompt replay regression.

## Governance

- [x] ADR 0021 ACCEPTED.
- [x] Docs/API updated.
- [x] TASKS/ROADMAP/Epic 2 updated per existing plan.
- [x] CHANGELOG updated.
- [x] CPU clean Release: 40/40 PASS.
- [x] CUDA clean Release: 42/42 PASS.
- [x] Task 2.0–2.12 regressions PASS.
- [x] No new Kestrel-Q warnings; governed NVCC C4211 only.
- [x] git diff --check PASS.
- [x] No commit/push.
