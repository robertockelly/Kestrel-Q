# TASKS-TASK-2.3A-CANONICAL-TOKENIZER-OVERRIDE.md

Status: **COMPLETE / PASS — Task 2.3 continuation, 2026-08-29**

## Preserve blocker evidence
- [x] Preserve current dirty Task 2.3 characterization delta.
- [x] Keep `TOKENIZER-CONTRACT.md` and GGUF insufficiency finding.
- [x] Keep original golden-validation evidence.

## Decision
- [x] Record canonical override adapter decision.
- [x] Update main Task 2.3 plan/checklist to reference continuation.
- [x] Keep corrected-GGUF and sidecar as rejected/deferred alternatives.

## GGUF substrate gate
- [x] Validate vocabulary strings/IDs.
- [x] Validate ordered merges/ranks.
- [x] Validate padded/unused IDs.
- [x] Validate required special-token substrate.
- [x] Reject incompatible near-match substrate.

## Canonical override
- [x] NFC normalization.
- [x] Canonical marks-excluding pre-tokenization.
- [x] Byte-level BPE.
- [x] Canonical special-token classes.
- [x] IDs 248060-248065.
- [x] BOS absent.
- [x] EOS identity.
- [x] `add_eos_token=false`.
- [x] Canonical skip-special decode.

## Unicode
- [x] Choose deterministic governed NFC method.
- [x] Pin Unicode/library/data version.
- [x] Review license/provenance.
- [x] No host-locale dependency.
- [x] Expanded NFC corpus.

## Native tokenizer/chat
- [x] Immutable `kq_tokenizer`.
- [x] Encode/decode APIs.
- [x] Stable capacity/errors/lifetime.
- [x] Separate `kq_chat`.
- [x] Official canonical template authority.
- [x] generation prompt OFF/ON.
- [x] unsupported branches fail explicitly.

## Original goldens
- [x] 10/10 prompts exact.
- [x] 14/14 segments exact.
- [x] exact decode/round-trip.
- [x] 2/2 chat cases exact.
- [x] original hashes unchanged.
- [x] independent oracle validation PASS.

## Divergence corpus
- [x] decomposed Unicode/NFC.
- [x] combining marks.
- [x] IDs 248060-248065.
- [x] BOS absence.
- [x] EOS no-auto-add.
- [x] skip-special decode.
- [x] safe template divergence.
- [x] canonical oracle only.
- [x] deterministic evidence/hash.

## Fail closed
- [x] changed vocab string.
- [x] wrong token ID.
- [x] changed merge/rank.
- [x] padded-ID misuse.
- [x] conflicting special token.
- [x] invalid tokenizer metadata.
- [x] invalid UTF-8.
- [x] unsupported chat branch.
- [x] arithmetic/capacity errors.

## Safety/regression
- [x] tensor payload touched = 0.
- [x] no sidecar requirement.
- [x] no GGUF mutation.
- [x] no runtime Python.
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.2 PASS.
- [x] no new warnings.
- [x] `git diff --check` PASS.

## Docs/governance
- [x] Update TOKENIZER-CONTRACT.md.
- [x] Create NATIVE-TOKENIZER.md.
- [x] Create KQ-TOKENIZER-API.md.
- [x] Create NATIVE-CHAT-FORMATTING.md.
- [x] Finalize ADR 0011.
- [x] Update ARCHITECTURE/TASKS/ROADMAP/Epic 2.
- [x] Update CHANGELOG.md.

## Final
- [x] Task 2.3 COMPLETE/PASS.
- [x] ADR 0011 ACCEPTED.
- [x] Task 2.4 NOT STARTED.
- [x] KQ-BACKLOG-BENCH-002 DEFERRED.
