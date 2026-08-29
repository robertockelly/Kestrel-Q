# TASKS-EPIC-1-MODEL-CHARACTERIZATION.md

Status: **COMPLETE / PASS — TASKS 1.0–1.4**

## Task 1.0 — Model source & artifact baseline
- [x] Pin official Qwen upstream revision(s).
- [x] Enumerate official repository files without downloading weight shards.
- [x] Capture hashes/sizes for metadata required by research.
- [x] Record Qwen model license separately from Kestrel-Q Apache-2.0.
- [x] Establish canonical source-of-truth ADR.
- [x] Register the local GGUF Q4 artifact with provenance/hash —
  `KQ-MODEL-ARTIFACT-001` is `REGISTERED_VERIFIED`.
- [x] Record no-full-weight-download rule for current phase.
- [x] Update project backlog/roadmap/changelog.

## Task 1.1 — Architecture characterization
- [x] Characterize top-level multimodal wrapper.
- [x] Characterize text model.
- [x] Characterize 48-layer execution pattern.
- [x] Characterize Gated DeltaNet.
- [x] Characterize Qwen Sparse Attention.
- [x] Characterize QSA indexer.
- [x] Characterize Gated Residual / hyper-connections.
- [x] Characterize MoE routing and shared expert.
- [x] Characterize N-gram / PLE lookup.
- [x] Characterize MTP.
- [x] Record tokenizer/chat-template boundary for later detailed work.
- [x] Determine evidence-backed text-only boundary through accepted ADR 0005.

## Task 1.2 — Tensor inventory & footprint
- [x] Parse the official index and bounded headers without weight payload.
- [x] Build deterministic tensor-family taxonomy.
- [x] Derive and reconcile shapes/dtypes/shards for all 1,658 tensors.
- [x] Calculate exact bytes by tensor family.
- [x] Separate text / vision / PLE / MTP footprints.
- [x] Calculate idealized Q8/Q6/Q5/Q4/Q3 payload floors.
- [x] Compare preliminary placement candidates against KQ-01 budgets.

## Task 1.3 — GGUF mapping
- [x] Register exact GGUF quantization.
- [x] Inspect GGUF metadata.
- [x] Inspect tensor inventory.
- [x] Map canonical names to GGUF names.
- [x] Identify conversions/fusions/omissions.
- [x] Document provenance of the converter/quantizer.
- [x] Accept ADR 0006 staged direct-GGUF-first initial container strategy.

## Task 1.4 — Reference behavior plan
- [x] Select authoritative Class-C and Class-Q reference implementations.
- [x] Pin their versions/revisions and licenses.
- [x] Generate exact tokenizer and chat-template golden vectors.
- [x] Generate exact PLE address golden vectors.
- [x] Define routing/operator/layer/logit checkpoint contracts.
- [x] Define canonical and GGUF greedy-generation contracts.
- [x] Define exact/tolerance and deterministic-regeneration rules.
- [x] Accept ADR 0007 and preserve capable-environment deferrals as future
  correctness gates.

## Epic governance
- [x] Keep `docs/TASKS.md` synchronized with current Epic 1 status.
- [x] Keep `docs/ROADMAP.md` synchronized with current Epic 1 status.
- [x] Update `CHANGELOG.md` for the Task 1.0 milestone.
- [x] Add ADR 0004 for the Task 1.0 source-of-truth decision.
- [x] Keep Task 1.0 upstream source/evidence provenance reproducible.
- [x] Record Task 1.1 completion and ADR 0005 acceptance.
- [x] Keep Task 1.2 header/inventory evidence deterministically reproducible.
- [x] Update `CHANGELOG.md` for the Task 1.2 milestone and research tooling.
- [x] Keep Task 1.3 GGUF mapping evidence deterministically reproducible.
- [x] Update `CHANGELOG.md` for the Task 1.3 milestone and ADR 0006.
- [x] Keep Task 1.4 safe golden evidence deterministically reproducible.
- [x] Update `CHANGELOG.md` for Task 1.4 and ADR 0007.
- [x] Close Epic 1 only after its seven exit gates were satisfied.
