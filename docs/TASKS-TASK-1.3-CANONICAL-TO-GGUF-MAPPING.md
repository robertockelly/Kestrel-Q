# TASKS-TASK-1.3-CANONICAL-TO-GGUF-MAPPING.md

Status: **COMPLETE / PASS**

Evidence root:
`research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/`

## A. Baseline
- [x] Read `AGENTS.md` and Task 1.0–1.2 outputs.
- [x] Record actual HEAD/working tree.
- [x] Confirm Task 1.2 COMPLETE/PASS.
- [x] Confirm Task 1.3 NOT STARTED before changes.
- [x] Confirm canonical and Unsloth revisions.
- [x] Read `KQ_GGUF_PATH`.
- [x] Verify GGUF identity against artifact register.

## B. Read-only GGUF parser
- [x] Parse magic/version.
- [x] Parse metadata directory.
- [x] Parse tensor directory.
- [x] Parse dimensions/types/offsets.
- [x] Derive data-section start/alignment.
- [x] Validate unique names.
- [x] Validate offsets/non-overlap.
- [x] Validate all spans within file.
- [x] Validate split/merged metadata.
- [x] Do not scan tensor payload contents.

## C. GGUF inventory
- [x] Create `gguf-metadata.json`.
- [x] Create `gguf-tensor-inventory.csv`.
- [x] Reproduce tensor/type distribution.
- [x] Calculate packed bytes per tensor.
- [x] Calculate file/header/alignment overhead.
- [x] Calculate packed bytes by GGUF type.
- [x] Calculate effective bits/parameter.

## D. Mapping rules
- [x] Define deterministic mapping-rule version.
- [x] Map embeddings/head/final norm.
- [x] Map all 48 text layers.
- [x] Map GDN.
- [x] Map QSA and indexer.
- [x] Map GR.
- [x] Map routers.
- [x] Map routed experts.
- [x] Map shared experts.
- [x] Map PLE/N-gram.
- [x] Map vision.
- [x] Map MTP.
- [x] Map connector/projector if distinct.
- [x] Account for ties/aliases/fusions/splits/layout transforms.

## E. Coverage
- [x] 1,658 canonical tensors mapped.
- [x] 1,224 GGUF tensors mapped.
- [x] canonical UNRESOLVED = 0.
- [x] unexplained GGUF = 0.
- [x] Reconcile exact 434 tensor-count difference.

## F. Converter/format evidence
- [x] Pin GGUF/GGML format source used.
- [x] Record exact revision/license.
- [x] Pin converter source if needed.
- [x] Record exact revision/license.
- [x] Inspect Unsloth provenance.
- [x] Separate observed/inferred/proven transformations.
- [x] Copy no implementation source.

## G. Quantization analysis
- [x] Count tensors by GGUF type.
- [x] Packed bytes by GGUF type.
- [x] Parameter-weighted type distribution.
- [x] Family-level effective bits/parameter.
- [x] PLE type distribution.
- [x] Routed expert type distribution.
- [x] Router/shared-expert type distribution.
- [x] GDN type distribution.
- [x] QSA/indexer type distribution.
- [x] Embedding/head/norm/GR distribution.
- [x] Vision/MTP distribution.

## H. Imatrix
- [x] Record embedded imatrix provenance.
- [x] Record calibration dataset metadata.
- [x] Determine whether scores are embedded/recoverable.
- [x] Identify only evidence-backed precision decisions.
- [x] Avoid unsupported sensitivity claims.

## I. Actual family footprint
- [x] Full GGUF packed tensor bytes.
- [x] Initial text-only packed bytes.
- [x] Vision packed bytes.
- [x] MTP packed bytes.
- [x] PLE packed bytes.
- [x] Routed experts packed bytes.
- [x] Shared experts packed bytes.
- [x] Dense/non-routed packed bytes.
- [x] Per-expert packed bytes.
- [x] Per-layer packed bytes.
- [x] Top-10 selected packed parameter footprint/layer.
- [x] All-layer selected packed parameter footprint.
- [x] Compare with Task 1.2 BF16/Q4 floors.

## J. 384-byte merged delta
- [x] Reconfirm upstream shard sum.
- [x] Inspect split/merge metadata semantics.
- [x] Use only bounded metadata/header access if upstream queried.
- [x] No upstream GGUF payload download.
- [x] Prove explanation or mark unresolved.

## K. KQ-01 implications
- [x] Compare actual text size to 8/20 GiB budgets.
- [x] Compare actual PLE size.
- [x] Compare actual routed experts.
- [x] Compare actual dense/shared.
- [x] Reference RAM/PCIe/SATA correctly.
- [x] No token-rate claim.
- [x] No selected-footprint value mislabeled as physical I/O.

## L. ADR 0006
- [x] Evaluate direct GGUF.
- [x] Evaluate Kestrel-Q-native container.
- [x] Evaluate staged approach.
- [x] Record evidence-backed decision.
- [x] Do not implement container.

## M. Evidence/output
- [x] Create `gguf-metadata.json`.
- [x] Create `gguf-tensor-inventory.csv`.
- [x] Create `canonical-gguf-mapping.csv`.
- [x] Create `gguf-summary.json`.
- [x] Create `mapping-evidence.json`.
- [x] Deterministic ordering.
- [x] Record SHA-256 hashes.
- [x] Deterministic regeneration PASS.

## N. Documentation
- [x] Create `docs/MODEL-GGUF-MAPPING.md`.
- [x] Create `docs/MODEL-QUANTIZATION-FOOTPRINT.md`.
- [x] Create/finalize ADR 0006.
- [x] Update `docs/MODEL-ARTIFACT-REGISTER.md`.
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update Epic 1 checklist.
- [x] Update this Task 1.3 checklist.
- [x] Update `tools/README.md`.
- [x] Update `CHANGELOG.md`.

## O. Safety
- [x] GGUF unchanged.
- [x] No GGUF copied into repo.
- [x] No Safetensors downloaded/tracked.
- [x] No secrets.
- [x] No user-specific GGUF path committed.
- [x] No production loader/runtime code added.
- [x] No incompatible source copied.
- [x] `git diff --check` PASS.

## P. Final report
Report actual HEAD, GGUF identity, type/count/overhead summary, mapping coverage, 434-count reconciliation, packed footprints, effective bpp, per-expert/top-10 findings, imatrix/provenance findings, 384-byte result, ADR 0006 outcome, evidence paths/hashes, files changed, safety, limitations, Task 1.3 status and working tree.

## Validation evidence

- artifact full-file identity: `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`;
- local inspection: GGUF v3, 67 metadata keys, 1,224 unique structurally valid tensors;
- mapping: 1,658/1,658 canonical, `UNRESOLVED = 0`, 1,224/1,224 GGUF, unexplained = 0;
- upstream split audit: 40 exact Range requests, 342 bytes, tensor payload bytes fetched = 0;
- deterministic regeneration: PASS, all five evidence files byte-identical;
- evidence SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `gguf-metadata.json` | `181ac010d7fc7f236408706801bbc2fe37a1310f115f33c1b5654fdffef84add` |
| `gguf-tensor-inventory.csv` | `10844a7cfe8f29008b43a90ffacc5056163f8edbe604eb26ab03534215894bfd` |
| `canonical-gguf-mapping.csv` | `f5cdd705e110ed0999def41292e4b761d242b5e4c3bd139323c01c597be70e5e` |
| `gguf-summary.json` | `eb58feeaf7b6c321fdf272215322c0df75d6832a23955b43ec892429437e00fa` |
| `mapping-evidence.json` | `f5b169f253d73bed055ac5714e62792238d508b4f6aa258572e6016752908a07` |

- safety: no tracked GGUF/Safetensors, no repository weights, no local path, no production runtime/model changes;
- `git diff --check`: PASS.
