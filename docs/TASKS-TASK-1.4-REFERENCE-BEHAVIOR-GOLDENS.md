# TASKS-TASK-1.4-REFERENCE-BEHAVIOR-GOLDENS.md

Status: **COMPLETE / PASS — 2026-08-29**

## A. Baseline
- [x] Read `AGENTS.md` and all Epic 1 outputs.
- [x] Record actual HEAD and working-tree state.
- [x] Confirm Tasks 1.0–1.3 COMPLETE/PASS.
- [x] Confirm ADR 0005/0006 ACCEPTED.
- [x] Confirm Task 2 NOT STARTED.

## B. Oracle hierarchy
- [x] Confirm/pin canonical executable reference.
- [x] Record revision/license/files used.
- [x] Select/pin GGUF reference runtime.
- [x] Verify qwen4exp + exact GGUF type support.
- [x] Record GGUF oracle revision/license/build requirements.
- [x] Separate canonical vs quantized semantics.
- [x] Ensure Kestrel-Q is not sole oracle.

## C. Prompt suite
- [x] Stable `KQ-PROMPT-*` IDs.
- [x] ASCII case.
- [x] Whitespace case.
- [x] Punctuation/numeric case.
- [x] Italian/Spanish accents case.
- [x] Non-Latin Unicode case.
- [x] C code case.
- [x] system+user chat case.
- [x] multi-turn chat case.
- [x] repeated-prefix case.
- [x] bounded longer-prefill case.
- [x] UTF-8 hashes recorded.

## D. Tokenizer goldens
- [x] Exact token IDs.
- [x] Token counts.
- [x] Special-token behavior.
- [x] Decode/round-trip checks where meaningful.
- [x] Exact-match validation.
- [x] Deterministic regeneration PASS.

## E. Chat-template goldens
- [x] Pinned official template.
- [x] Structured messages.
- [x] Exact rendered string.
- [x] UTF-8 SHA-256.
- [x] Exact rendered token IDs.
- [x] `add_generation_prompt` behavior.
- [x] Deterministic regeneration PASS.

## F. PLE-address goldens
- [x] Length 1.
- [x] Length 2.
- [x] Length 3.
- [x] Length >3.
- [x] Repeated-token case.
- [x] Large-valid-token-ID case.
- [x] Bigram/trigram addresses.
- [x] Head/partition data.
- [x] Rolling-history update.
- [x] Decode-step case.
- [x] Exact-match validation.

## G. Operator/checkpoint contract
- [x] GR fields.
- [x] GDN fields.
- [x] QSA/indexer fields.
- [x] MoE/router fields.
- [x] Final hidden/logit fields.
- [x] Greedy token/generation fields.
- [x] Stable schema documented.

## H. Synthetic operator vectors
- [x] Determine which can run through pinned independent references without weights.
- [x] Generate where safely supported.
- [x] Mark others `PLANNED_REFERENCE_REQUIRED`.
- [x] No self-referential Kestrel-Q goldens.

## I. Numeric policy
- [x] Exact-match classes defined.
- [x] Tolerant classes defined.
- [x] Shape/NaN/Inf rules.
- [x] `atol`/`rtol` fields.
- [x] Uncalibrated tolerances explicitly marked.

## J. Canonical full-model plan
- [x] Exact model/reference revisions.
- [x] Batch/text-only/no-vision/no-MTP settings.
- [x] Greedy deterministic settings.
- [x] Prompt IDs.
- [x] Required layer/router/GDN/QSA/logit checkpoints.
- [x] Capable-reference environment contract.
- [x] Unavailable vectors marked deferred.

## K. Quantized GGUF plan
- [x] Exact GGUF SHA.
- [x] Exact reference runtime revision.
- [x] Deterministic build/runtime settings.
- [x] MTP/speculation disabled.
- [x] Greedy prompt suite.
- [x] Offload/mmap configuration recorded.
- [x] KQ-01 resource guard applied.
- [x] Unsafe local execution deferred.

## L. Machine-readable assets
- [x] `manifest.json`.
- [x] `prompt-suite.json`.
- [x] `canonical/tokenizer-vectors.json`.
- [x] `canonical/chat-template-vectors.json`.
- [x] `canonical/ple-address-vectors.json`.
- [x] `canonical/operator-vector-plan.json`.
- [x] `canonical/full-model-vector-plan.json`.
- [x] `quantized/gguf-vector-plan.json`.
- [x] SHA-256 recorded.
- [x] Manifest references/statuses validate.
- [x] Deterministic regeneration PASS.

## M. Documents/ADR
- [x] Create `docs/REFERENCE-ORACLE-STRATEGY.md`.
- [x] Create `docs/GOLDEN-VECTORS.md`.
- [x] Create `docs/REFERENCE-PROMPT-SUITE.md`.
- [x] Finalize ADR 0007.
- [x] Update tools README.
- [x] Update CHANGELOG.

## N. Epic/governance
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update Epic 1 checklist.
- [x] Update this checklist.
- [x] Mark Epic 1 COMPLETE only if exit gate is genuinely met.
- [x] Task 2 remains NOT STARTED.

## O. Safety
- [x] No BF16 checkpoint download.
- [x] No unsafe BF16 execution on KQ-01.
- [x] No unbounded GGUF paging/run.
- [x] Tracked `.safetensors` = 0.
- [x] Tracked `.gguf` = 0.
- [x] Repository model weights = 0.
- [x] No secrets.
- [x] No production runtime/model implementation.
- [x] `git diff --check` PASS.

## P. Final report
Report actual HEAD, both oracle revisions/licenses/statuses, prompt-suite hash, generated safe-vector hashes, tokenizer/chat/PLE coverage, operator-plan status, canonical and quantized full-model plan/status, tolerance policy, ADR 0007 result, Epic 1 result, files changed, safety validation, deferred gates, Task 1.4 status and working-tree status.

## Completion evidence

- Starting HEAD: `f5e5a1052db89e56aa81439efbce08c63e831bf2`.
- Prompt suite SHA-256:
  `ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15`.
- Tokenizer vectors SHA-256:
  `cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153`.
- Chat-template vectors SHA-256:
  `72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6`.
- PLE address vectors SHA-256:
  `495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`.
- Manifest SHA-256:
  `aa572756672f288957d429a60d7180650ffb2d603a792b21cd72def0a14ec0c4`.
- Independent regeneration: **8 / 8 files byte-identical**.
- Full-model Class C/Q statuses: `DEFERRED_CAPABLE_REFERENCE_ENV`.
- Operator floating-vector status: `PLANNED_REFERENCE_REQUIRED`.
- Weight payload downloaded or executed: **false**.
