# TASKS-TASK-1.2-TENSOR-INVENTORY-FOOTPRINT.md

Status: **COMPLETE / PASS**

## A. Baseline
- [x] Read `AGENTS.md` and all Task 1.0/1.1 governed outputs.
- [x] Record actual HEAD and working-tree state.
- [x] Confirm Task 1.1 COMPLETE/PASS and ADR 0005 ACCEPTED.
- [x] Confirm Task 1.3 NOT STARTED.
- [x] Confirm canonical HF revision.

## B. Header-safe tooling
- [x] Derive exact 131-shard allowlist from pinned index.
- [x] Implement bounded HTTP Range-only header retrieval.
- [x] Read only bytes 0-7 for header length.
- [x] Enforce sane per-shard header maximum.
- [x] Retrieve exactly the JSON header range.
- [x] Abort if Range is ignored/unbounded.
- [x] Enforce total-download safety budget.
- [x] Record every request in network audit.
- [x] Never save a `.safetensors` file.
- [x] Document research-tool usage.

## C. Canonical reconciliation
- [x] Parse all 131 headers.
- [x] Reconcile index tensor names vs header tensors.
- [x] Detect duplicates/extras/missing.
- [x] Validate dtype and shape.
- [x] Validate parameter counts.
- [x] Validate data offsets/non-overlap.
- [x] Reconcile aggregate payload bytes.
- [x] Validate shard-size relation where available.
- [x] Produce shard-header manifest.

## D. Deterministic classification
- [x] Define versioned classification rules.
- [x] Classify text backbone.
- [x] Classify PLE/N-gram.
- [x] Classify vision.
- [x] Classify MTP.
- [x] Classify connector/multimodal tensors if distinct.
- [x] Classify embeddings/head/norm.
- [x] Classify GDN.
- [x] Classify QSA/indexer.
- [x] Classify GR parameters.
- [x] Classify routers.
- [x] Classify routed experts.
- [x] Extract layer IDs.
- [x] Record stacked expert IDs and derive exact per-expert values.
- [x] Classify shared experts.
- [x] Zero unexplained UNKNOWN/REVIEW rows.

## E. Machine-readable outputs
- [x] Create `tensor-inventory.csv`.
- [x] Create `tensor-summary.json`.
- [x] Create `shard-header-manifest.json`.
- [x] Create `network-range-audit.json`.
- [x] Deterministic ordering.
- [x] Record SHA-256 hashes.

## F. Static footprint
- [x] Exact full checkpoint stored footprint.
- [x] Exact initial text-only footprint.
- [x] Exact vision-excluded footprint.
- [x] Exact MTP-excluded footprint.
- [x] Exact PLE footprint.
- [x] Exact routed-expert footprint.
- [x] Exact shared-expert footprint.
- [x] Exact dense/router/norm footprint.
- [x] Record weight-sharing/tied-weight implications.

## G. Layer/expert analysis
- [x] Per-layer type and bytes.
- [x] Per-layer non-expert bytes.
- [x] Per-layer router bytes.
- [x] Per-layer routed-expert bytes.
- [x] Per-layer shared-expert bytes.
- [x] Per-expert size distribution.
- [x] Top-10 selected expert parameter payload/layer.
- [x] Aggregate selected expert parameter footprint/token.
- [x] Label clearly as parameter footprint, not measured I/O.

## H. PLE
- [x] Exact PLE tensors.
- [x] Exact PLE parameters.
- [x] Exact PLE bytes.
- [x] Percent of full checkpoint.
- [x] Percent of initial text-only scope.
- [x] Idealized Q8/Q6/Q5/Q4/Q3.
- [x] Relate to Task-1.1 prefetch semantics without designing scheduler.

## I. Idealized quantization
- [x] Q8 lower bounds.
- [x] Q6 lower bounds.
- [x] Q5 lower bounds.
- [x] Q4 lower bounds.
- [x] Q3 lower bounds.
- [x] Explicitly state overhead exclusions.
- [x] Do not call these GGUF sizes.

## J. Runtime state
- [x] GDN recurrent matrix bytes.
- [x] GDN convolution-state bytes.
- [x] QSA K/V bytes per token/context.
- [x] QSA raw-index bytes per token/context.
- [x] Record that no second persistent QSA block-key cache is canonical.
- [x] PLE token-history/state bytes.
- [x] Context table: 1 / 4096 / 16384 / 65536 / 262144.
- [x] Exclude GR current-forward branches from persistent context cache.
- [x] Separate optional MTP/vision state.

## K. KQ-01 feasibility
- [x] Compare full BF16 to budgets.
- [x] Compare text-only BF16 to budgets.
- [x] Compare Q8/Q6/Q5/Q4/Q3 lower bounds.
- [x] Compare PLE/expert/dense families.
- [x] Use measured 8 GiB VRAM / 20 GiB host budgets.
- [x] Reference measured RAM/PCIe/SATA facts correctly.
- [x] No tokens/s claims.

## L. Documentation
- [x] Create `docs/MODEL-TENSOR-INVENTORY.md`.
- [x] Create `docs/MODEL-FOOTPRINT.md`.
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update Epic 1 checklist.
- [x] Update this Task 1.2 checklist.
- [x] Update `tools/README.md`.
- [x] Update `CHANGELOG.md`.

## M. Safety
- [x] 131/131 header request audit PASS.
- [x] Total weight payload bytes fetched = 0.
- [x] Downloaded/tracked `.safetensors` = 0.
- [x] Tracked `.gguf` = 0.
- [x] Repository model weights = 0.
- [x] No secrets.
- [x] No runtime/model implementation added.
- [x] Task 1.3 NOT STARTED.
- [x] `git diff --check` PASS.

## N. Evidence

- actual starting HEAD: `eb089d31e0550e234ca6be9233841cad8235f9f8`;
- canonical revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- headers: 131/131 through 262 bounded HTTP 206 responses;
- total response bytes: 229,760;
- weight payload bytes fetched: 0;
- canonical tensors: 1,658/1,658, unique and fully classified;
- aggregate tensor payload: 359,999,963,128 bytes, equal to index
  `metadata.total_size`;
- independent capture/regeneration reproduced every output SHA-256;
- machine evidence: `research/model-tensors/Qwen3.8-Flash-Next/`
  `de4b8e4d43b917e7706784d8bb445c9af86a3540/`;
- detailed static, quantization, layer/expert and state evidence:
  `docs/MODEL-TENSOR-INVENTORY.md` and `docs/MODEL-FOOTPRINT.md`.

## O. Development findings

1. The first capture-tool draft removed the configured failure-audit path after
   a successful run. Root cause: output cleanup and invalid-evidence lifetime
   were incorrectly coupled. No capture failed and no evidence was removed.
   The final tool creates failure audits exclusively, refuses an existing path
   before network access and never deletes it. A regression check verified the
   existing file's hash and the absence of network/output progress.
2. The first inventory draft placed only the global classification version in
   every row's `classification_rule`. Classification results were correct, but
   per-row traceability was incomplete. The final value includes both version
   and exact component rule; the analyzer asserts all 1,658 rows carry it and
   deterministic outputs were regenerated.

## P. Final status

**TASK 1.2 — TENSOR INVENTORY & FOOTPRINT: COMPLETE / PASS**

Task 1.3 remains **NOT STARTED**. Task 1.2 adds no model/runtime implementation.
