# TASKS-TASK-2.0-NATIVE-GGUF-MMAP.md

Status: **COMPLETE / PASS — 2026-08-29**

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Epic 1 COMPLETE/PASS.
- [x] Confirm Task 2 begins here.
- [x] Keep `KQ-BACKLOG-BENCH-002` deferred.

## File layer
- [x] Read-only `CreateFileW`.
- [x] Exact `GetFileSizeEx`.
- [x] Read-only `CreateFileMappingW`.
- [x] Allocation-granularity handling.
- [x] Bounded logical `MapViewOfFile`.
- [x] Checked 64-bit arithmetic.
- [x] Deterministic cleanup.

## GGUF parser
- [x] GGUF v3 header.
- [x] Bounded strings.
- [x] Target metadata scalar/array types.
- [x] Tensor directory.
- [x] Rank/dimensions.
- [x] Duplicate detection.
- [x] Alignment/data-section validation.
- [x] EOF/span validation.
- [x] Packed-byte accounting.

## Types
- [x] F32.
- [x] BF16.
- [x] Q5_1.
- [x] Q8_0.
- [x] Q4_K.
- [x] Q5_K.
- [x] IQ4_NL.
- [x] Invalid block geometry fails closed.
- [x] No dequantization.

## CLI
- [x] `kq-inspect`.
- [x] Summary fields.
- [x] Type counts.
- [x] Optional modes explicitly deferred; summary mode is the Task 2.0 surface.
- [x] No payload dump.

## Tests
- [x] Valid synthetic fixtures.
- [x] Bad magic/version.
- [x] Truncation.
- [x] Malformed strings/arrays.
- [x] Overflow.
- [x] Bad rank/dimensions.
- [x] Unsupported type.
- [x] Duplicate tensor.
- [x] Bad alignment/span.
- [x] Invalid block geometry.
- [x] Failure cleanup.

## Real artifact
- [x] Opt-in `KQ_GGUF_PATH`.
- [x] File size 111334654400.
- [x] GGUF v3.
- [x] qwen4exp.
- [x] Metadata 67.
- [x] Tensors 1224.
- [x] Packed bytes 111323630080.
- [x] Overhead 11024320.
- [x] Type counts exactly match Epic 1.
- [x] Artifact unchanged by size/last-write-time check.

## Boundary
- [x] No payload dereference in normal inspection.
- [x] No inference.
- [x] No dequantization.
- [x] No tokenizer/PLE/scheduler.
- [x] No CUDA coupling.
- [x] No llama.cpp runtime dependency.

## Regression
- [x] Clean CPU Release build/test PASS.
- [x] Clean CUDA Release build/test PASS.
- [x] No new Kestrel-Q warnings.
- [x] `git diff --check` PASS.

## Governance
- [x] Create `docs/NATIVE-GGUF-LAYER.md`.
- [x] Create `docs/KQ-INSPECT.md`.
- [x] Finalize ADR 0008.
- [x] Update architecture/TASKS/ROADMAP.
- [x] Update Epic 2 status.
- [x] Update `CHANGELOG.md`.

## Safety
- [x] tracked `.gguf` = 0.
- [x] tracked `.safetensors` = 0.
- [x] repository model weights = 0.
- [x] no secrets/local model path.
- [x] build trees not tracked.

## Final report
Report production files/APIs, Windows mapping behavior, GGUF coverage, synthetic/malformed tests, real-artifact comparison, payload-touch evidence, CPU/CUDA regression, ADR 0008, root causes/fixes, limitations, Task 2.0 status and worktree status.

## Recorded evidence

- Entry HEAD: `4f9f1dc35420d7987d75b0524cc204eb993d7618`.
- Synthetic suite: one deterministic valid fixture containing all seven target
  types plus 21 fail-closed malformed cases; PASS in CPU and CUDA build trees.
- Real artifact: all Epic 1 structural values and type counts match;
  `directory_bytes_parsed = 11024307`, `data_section_offset = 11024320`,
  `payload_bytes_accessed = 0`.
- Intermediate `/W4` build finding: MSVC C4701 conservatively flagged two
  parser locals populated only through status-returning helpers. Root cause was
  compiler data-flow ambiguity, not unhandled input. Explicit initialization
  removed the warnings; subsequent `/W4` CPU/CUDA builds are clean and retain
  the warning-free build gate.
- Final clean validation and safety scans are recorded by the completion report;
  generated build directories remain ignored and untracked.
