# TASKS-EPIC-1-MODEL-CHARACTERIZATION.md

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
- [ ] Characterize top-level multimodal wrapper.
- [ ] Characterize text model.
- [ ] Characterize 48-layer execution pattern.
- [ ] Characterize Gated DeltaNet.
- [ ] Characterize Qwen Sparse Attention.
- [ ] Characterize QSA indexer.
- [ ] Characterize Gated Residual / hyper-connections.
- [ ] Characterize MoE routing and shared expert.
- [ ] Characterize N-gram / PLE lookup.
- [ ] Characterize MTP.
- [ ] Characterize tokenizer/chat-template semantics.
- [ ] Determine evidence-backed text-only boundary.

## Task 1.2 — Tensor inventory & footprint
- [ ] Parse official Safetensors index only.
- [ ] Build tensor-family taxonomy.
- [ ] Derive shapes/dtypes/shards.
- [ ] Calculate bytes by tensor family.
- [ ] Separate text / vision / PLE / MTP footprints.
- [ ] Calculate idealized quantized footprints.
- [ ] Map candidate storage tiers against KQ-01 budgets.

## Task 1.3 — GGUF mapping
- [ ] Register exact GGUF quantization.
- [ ] Inspect GGUF metadata.
- [ ] Inspect tensor inventory.
- [ ] Map canonical names to GGUF names.
- [ ] Identify conversions/fusions/omissions.
- [ ] Document provenance of the converter/quantizer.
- [ ] Decide whether GGUF is an initial runtime format or only a research artifact.

## Task 1.4 — Reference behavior plan
- [ ] Select authoritative reference implementation(s).
- [ ] Pin their versions/revisions.
- [ ] Define tokenizer golden vectors.
- [ ] Define routing golden vectors.
- [ ] Define layer/logit golden vectors.
- [ ] Define greedy-generation golden vectors.
- [ ] Define tolerances and reproducibility rules.

## Epic governance
- [x] Keep `docs/TASKS.md` synchronized with current Epic 1 status.
- [x] Keep `docs/ROADMAP.md` synchronized with current Epic 1 status.
- [x] Update `CHANGELOG.md` for the Task 1.0 milestone.
- [x] Add ADR 0004 for the Task 1.0 source-of-truth decision.
- [x] Keep Task 1.0 upstream source/evidence provenance reproducible.
