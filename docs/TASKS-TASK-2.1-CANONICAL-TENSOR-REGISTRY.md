# TASKS-TASK-2.1-CANONICAL-TENSOR-REGISTRY.md

Status: **COMPLETE / PASS — 2026-08-29**

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.0 COMPLETE/PASS.
- [x] Confirm Task 2.1 NOT STARTED before edits.
- [x] Keep `KQ-BACKLOG-BENCH-002` DEFERRED.

## Model identity
- [x] Validate `qwen4exp`.
- [x] Validate hidden size 2560.
- [x] Validate vocab 248320.
- [x] Validate 48 layers / 36 GDN / 12 QSA.
- [x] Validate expert count 512 / top-k 10.
- [x] Validate shared expert.
- [x] Validate PLE configuration.
- [x] Fail closed on incompatible near-match.

## Semantic API
- [x] Component enum.
- [x] Layer-type enum.
- [x] Projection/role enum.
- [x] Binding relation enum.
- [x] Runtime-scope enum.
- [x] Placement-hint enum.
- [x] Stable semantic key.
- [x] Immutable semantic descriptor.
- [x] Explicit ownership/lifetime.
- [x] Deterministic semantic-ID formatting.

## Mapping rules
- [x] Direct/renamed mappings.
- [x] GDN mappings.
- [x] QSA mappings.
- [x] QSA indexer mappings.
- [x] GR mappings.
- [x] Router mappings.
- [x] Routed expert-stack mappings.
- [x] Shared expert mappings.
- [x] Embedding/head/norm mappings.
- [x] PLE dense mappings.
- [x] 128 PLE semantics → fused physical tensor.
- [x] 3 PLE metadata-derived semantics.
- [x] 48 MoE gate_up split mappings.
- [x] 12 QSA index_qk split mappings.
- [x] No fuzzy fallback.

## Structural validation
- [x] Rank/shape checks.
- [x] Hidden/intermediate geometry.
- [x] Expert-axis checks.
- [x] Split part count/order/shape.
- [x] PLE fused geometry.
- [x] Compatible physical types.
- [x] No duplicate semantic key.
- [x] No unexplained physical tensor.

## Placement hints
- [x] `ALWAYS_NEEDED_CANDIDATE`.
- [x] `ROUTED_EXPERT_CACHE_CANDIDATE`.
- [x] `PLE_DISK_BACKED_CANDIDATE`.
- [x] `EXCLUDED_INITIAL_SCOPE`.
- [x] Neutral fallback only where evidence does not support more.
- [x] No allocation/cache/prefetch implementation.

## Synthetic tests
- [x] Direct binding.
- [x] Split binding.
- [x] Fused binding.
- [x] Metadata-derived binding.
- [x] Layer topology.
- [x] Expert-stack axis.
- [x] Wrong topology.
- [x] Wrong hidden/vocab/layer/expert values.
- [x] Missing semantic tensor.
- [x] Unknown/ambiguous physical tensor.
- [x] Wrong shape/expert axis.
- [x] Missing/duplicate split part.
- [x] Invalid split geometry.
- [x] Invalid PLE fusion.
- [x] Missing PLE metadata.
- [x] Invalid layer ID/type.
- [x] Failure cleanup.

## Real artifact
- [x] Opt-in `KQ_GGUF_PATH`.
- [x] 1224 physical.
- [x] 1294 semantic entries.
- [x] 3 metadata-derived.
- [x] 1224 unique physical coverage.
- [x] unknown physical = 0.
- [x] unbound required = 0.
- [x] 48 layers / 36 GDN / 12 QSA.
- [x] expert count 512 / top-k 10.
- [x] payload bytes accessed = 0.

## Epic 1 oracle comparison
- [x] Add test/research-only native registry export.
- [x] Compare to `canonical-gguf-mapping.csv`.
- [x] Verify all initial-text canonical relations.
- [x] Verify all GGUF physical relations.
- [x] Verify split/fused/metadata-derived statuses.
- [x] Production runtime reads no research CSV/JSON.

## Inspector
- [x] Add semantic-summary mode.
- [x] Add one-key query if useful.
- [x] Deterministic output.
- [x] No payload access.

## Regression
- [x] Clean CPU Release PASS.
- [x] CPU CTests PASS.
- [x] Clean CUDA Release PASS.
- [x] CUDA CTests PASS.
- [x] Task 2.0 tests remain PASS.
- [x] No new Kestrel-Q warnings.
- [x] `git diff --check` PASS.

## Documentation/governance
- [x] Create `docs/MODEL-SEMANTIC-REGISTRY.md`.
- [x] Create `docs/KQ-SEMANTIC-API.md`.
- [x] Finalize ADR 0009.
- [x] Update `docs/ARCHITECTURE.md`.
- [x] Update `docs/TASKS.md`.
- [x] Update `docs/ROADMAP.md`.
- [x] Update Epic 2 plan/status.
- [x] Update this checklist.
- [x] Update `CHANGELOG.md`.

## Safety
- [x] No tensor payload view.
- [x] No dequantization/inference/tokenizer/PLE execution.
- [x] No scheduler/cache.
- [x] No production research-file dependency.
- [x] tracked `.gguf`/`.safetensors` = 0.
- [x] no model weights/secrets/local path.
- [x] build trees untracked.

## Final report
Report actual HEAD, production files/API, model identity validation, semantic
and binding counts, topology, placement annotations, synthetic/fail-closed
results, real-artifact results, Epic 1 oracle comparison, payload boundary,
CPU/CUDA regression, ADR 0009, root causes/fixes, remaining limitations,
Task 2.1 status and worktree status.

## Recorded evidence

- Entry HEAD: `5db71945806b483c7d8306144a7987ae9f17f320`; only the four
  maintainer-provided governed Task 2.1 documents were untracked at entry.
- Native registry: 1,294 semantics, 1,224/1,224 unique physical coverage,
  three metadata-derived, unknown physical zero and unbound required zero.
- Relations: 847 renamed, 256 transformed, 60 split, 128 fused and three
  metadata-derived. Placement annotations: 1,061 always-needed, 96 routed
  expert-cache and 137 PLE disk-backed candidates.
- Complete native dump versus pinned Epic 1 mapping: PASS and byte-identical;
  dump SHA-256
  `a214013005b4600ade0d4284169d1ab4f70a9329069aed73568720a927898d49`.
- Synthetic suite: one complete positive target plus 20 fail-closed mutations.
- Intermediate `/W4` finding: MSVC C4701/C4703 conservatively flagged pointers
  assigned through status-returning semantic-construction helpers. Explicit
  initialization removed the warnings without changing control flow.
- Intermediate CLI finding: semantic modes were initially dispatched after the
  file open but before GGUF parsing, so a null GGUF reached the model adapter.
  The branch now follows successful `kq_gguf_open`; the CTest oracle exercises
  `--semantic-dump` and guards the ordering regression.
