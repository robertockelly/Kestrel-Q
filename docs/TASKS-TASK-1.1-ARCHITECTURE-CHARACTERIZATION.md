# TASKS-TASK-1.1-ARCHITECTURE-CHARACTERIZATION.md

## A. Governance prelude
- [x] Read `AGENTS.md`.
- [x] Add mandatory canonical `CHANGELOG.md` rule.
- [x] Add reconstructibility principle to `docs/PRINCIPLES.md`.
- [x] Record governance change in `CHANGELOG.md`.
- [x] Confirm no parallel worklog is introduced.

## B. Baseline and provenance
- [x] Record current Kestrel-Q HEAD.
- [x] Confirm Task 1.0 COMPLETE/PASS.
- [x] Confirm canonical HF revision unchanged.
- [x] Confirm canonical Qwen research revision unchanged.
- [x] Reuse Task 1.0 cache; download no weight shards.
- [x] Pin any Tier B implementation source.
- [x] Record Tier B license and exact symbols/files used.

## C. Multimodal/text boundary
- [x] Trace top-level conditional-generation model.
- [x] Identify vision path.
- [x] Identify text/language path.
- [x] Identify multimodal merge/injection path.
- [x] Trace text-only input.
- [x] Prove conditions that bypass vision execution.
- [x] Record image/video special-token handling.

## D. Text backbone
- [x] Verify vocab and hidden dimensions.
- [x] Generate explicit 48-layer type table.
- [x] Verify 36 GDN + 12 QSA/full layers.
- [x] Verify Q/KV heads and head dimension.
- [x] Verify RoPE/position configuration.
- [x] Verify normalization, activation and output gating.
- [x] Document embedding → layers → norm → logits flow.

## E. Gated DeltaNet
- [x] Extract/derive canonical equations.
- [x] Identify projections.
- [x] Characterize recurrent state.
- [x] Characterize local convolution/state if present.
- [x] Characterize prefill update.
- [x] Characterize decode update.
- [x] Record symbolic state shapes/dtypes.
- [x] Separate algorithm from implementation fusion.

## F. Qwen Sparse Attention
- [x] Characterize attention Q/K/V.
- [x] Characterize indexer projections.
- [x] Verify compression ratio.
- [x] Verify indexer heads/KV heads/head dim.
- [x] Verify indexer budget.
- [x] Characterize micro-block construction.
- [x] Characterize selection.
- [x] Characterize trailing incomplete block.
- [x] Characterize causality and positional handling.
- [x] Characterize prefill/decode.
- [x] Record persistent state.

## G. Gated Residual
- [x] Verify four residual branches.
- [x] Verify low-rank gating config.
- [x] Characterize read gate.
- [x] Characterize write-back.
- [x] Verify branch mixing presence/absence.
- [x] Identify placement around attention/MoE.
- [x] Record FP8 residual-state capability.

## H. MoE
- [x] Characterize router.
- [x] Verify 512 experts.
- [x] Verify top-10.
- [x] Verify shared expert.
- [x] Verify intermediate sizes.
- [x] Verify activation.
- [x] Characterize output weighting/combination.
- [x] Separate training load balancing from inference.

## I. N-gram / PLE
- [x] Verify n-gram size.
- [x] Verify 20M vocabulary base.
- [x] Verify PLE layer.
- [x] Verify embedding dimension.
- [x] Verify heads-per-ngram.
- [x] Verify split/partition settings.
- [x] Verify convolution setting.
- [x] Derive deterministic addressing.
- [x] Characterize prefill/decode lookup.
- [x] Identify prefetchable addresses.
- [x] Keep scheduler design deferred.

## J. MTP
- [x] Characterize MTP topology.
- [x] Characterize hidden-state input and record the exact remaining Tier A
  fusion uncertainty.
- [x] Characterize attention choice.
- [x] Characterize embedding sharing.
- [x] Prove whether ordinary next-token logits depend on MTP.
- [x] Characterize speculative use separately.

## K. Runtime state
- [x] Create `docs/MODEL-RUNTIME-STATE.md`.
- [x] Inventory GDN state.
- [x] Inventory QSA KV/index state.
- [x] Inventory GR branch state.
- [x] Inventory PLE token-history state.
- [x] Inventory position/RoPE state.
- [x] Inventory optional MTP state.
- [x] Inventory optional vision state.
- [x] Mark fixed vs context-growing state.

## L. End-to-end flows
- [x] Document prefill.
- [x] Document single-token decode.
- [x] Identify state reads/writes.
- [x] Identify context-dependent operations.
- [x] Identify future prefetch candidates without designing scheduler.

## M. ADR 0005
- [x] Validate text-only boundary.
- [x] Validate vision omission for text-only requests.
- [x] Validate MTP omission for ordinary autoregressive generation.
- [x] Accept ADR because evidence is sufficient.
- [x] Record remaining non-blocking MTP fusion uncertainty for future MTP work.

## N. Evidence
- [x] Create pinned evidence JSON.
- [x] Give every major claim a stable claim ID.
- [x] Record source tier/revision/location.
- [x] Validate JSON syntax.
- [x] Cross-check docs against evidence IDs.

## O. Safety
- [x] Downloaded Safetensors shards = 0.
- [x] Tracked Safetensors = 0.
- [x] Tracked GGUF = 0.
- [x] Runtime/model implementation changes = 0.
- [x] No incompatible copied source.
- [x] No secrets.
- [x] `git diff --check` PASS; untracked Task 1.1 files also pass
  `git diff --no-index --check`.
- [x] Task 1.2 remains NOT STARTED.
- [x] Task 1.3 remains NOT STARTED.

## P. Project records
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update `CHANGELOG.md`.
- [x] Update this checklist with results/evidence.

## Results and evidence

- Starting Kestrel-Q HEAD:
  `aaecee5b38269166f478023430a1208f2a49478f`.
- Canonical HF revision:
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Canonical Qwen research revision:
  `69885871a64393807d988b27b1b5e380e8f28526`.
- Tier B implementation: `huggingface/transformers` commit
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, Apache-2.0; exact files and
  hashes are recorded in `evidence.json`.
- Tier C MTP cross-check: `sgl-project/sglang` commit
  `9579bff86085f886cf6d1ec69349017d0caeced4`, Apache-2.0.
- Evidence:
  `research/model-architecture/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/evidence.json`.
- Evidence JSON SHA-256:
  `E6BFEB3CBBC3EE6565B5D645CAE88B8573A10215506AD994304478EDF6A2B9DB`.
- ADR 0005: **ACCEPTED**.
- Safety result: zero repository `*.safetensors`/`*.gguf`, zero tracked model
  weights, zero strong secret-pattern hits, zero runtime/model implementation
  changes, and diff checks pass.
- Task 1.1: **COMPLETE / PASS**.

## Q. Final report
- [x] Report prepared with:
- current HEAD;
- sources/revisions used;
- Tier B source if used;
- architecture findings by subsystem;
- exact 48-layer pattern;
- GDN state/update model;
- QSA/index state/update model;
- GR semantics;
- MoE semantics;
- N-gram/PLE semantics;
- MTP dependency result;
- ADR 0005 result;
- evidence path/hash;
- files changed;
- CHANGELOG governance-rule status;
- safety validation;
- remaining uncertainties;
- Task 1.1 status;
- working-tree status.
