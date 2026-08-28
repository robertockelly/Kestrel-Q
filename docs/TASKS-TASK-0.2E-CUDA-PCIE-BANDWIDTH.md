# TASKS-TASK-0.2E-CUDA-PCIE-BANDWIDTH.md

Status: **COMPLETE / PASS**

## A. Governance/backlog

- [x] Read `docs/BACKLOG-UPDATE-CUDA-PORTABILITY.md`.
- [x] Add `KQ-BACKLOG-CUDA-001` to `docs/TASKS.md`.
- [x] Add CUDA release portability item to `docs/ROADMAP.md`.
- [x] Update `CHANGELOG.md`.
- [x] Keep the item DEFERRED until first public binary release.

## B. Baseline protection

- [x] Read `AGENTS.md`.
- [x] Inspect Task 0.3B CUDA smoke implementation.
- [x] Confirm CPU-only baseline still passes before benchmark changes.
- [x] Confirm CUDA smoke still passes before benchmark changes.
- [x] Preserve the validated `native` CUDA architecture behavior.

## C. Benchmark implementation

- [x] Add dedicated CUDA bandwidth benchmark target.
- [x] Implement pageable H2D.
- [x] Implement pinned H2D.
- [x] Implement pageable D2H.
- [x] Implement pinned D2H.
- [x] Gate concurrent bidirectional transfer and explicitly SKIP when unsupported.
- [x] Test default sizes 1/4/16/64/256 MiB.
- [x] Add warm-up iterations.
- [x] Add multiple measured iterations.
- [x] Add high-resolution host timing.
- [x] Add CUDA-event timing where meaningful.
- [x] Validate data correctness for every mode.
- [x] Check every CUDA API call.
- [x] Record allocation/pinning setup costs where appropriate.
- [x] Produce stable machine-readable output.

## D. PCIe under-load evidence

- [x] Implement Windows orchestration to sample current PCIe generation/width while transfers are active.
- [x] Resolve `nvidia-smi.exe` robustly; do not assume Developer Prompt PATH.
- [x] Capture pstate/link/memory samples in CSV.
- [x] Capture reliable Gen4 x16 under-load link samples; fallback limitation path was not needed.

## E. Validation

- [x] Clean CPU-only configure/build/test PASS.
- [x] Clean CUDA configure/build/test PASS.
- [x] CUDA smoke PASS.
- [x] Bandwidth benchmark functional run PASS.
- [x] No transfer correctness failures.
- [x] No CUDA unchecked-error path introduced.
- [x] No stale `/mnt/data` or foreign build-tree paths.
- [x] No generated build directories tracked.

## F. Evidence

- [x] Create `bench/results/raw/KQ-01/2026-08-28/`.
- [x] Save raw benchmark CSV.
- [x] Save console/tool metadata evidence.
- [x] Save PCIe samples CSV.
- [x] Do not edit raw evidence after capture.

## G. Documentation

- [x] Update `docs/MEMORY-BANDWIDTH-BUDGET.md`.
- [x] Replace theoretical-only PCIe discussion with measured-vs-theoretical distinction.
- [x] Compare against RAM ~25.63 GB/s and SATA ~0.44 GB/s.
- [x] Document implications for expert/tensor prefetch.
- [x] Update `docs/TASKS.md` Task 0.2E status.
- [x] Update `CHANGELOG.md`.

## H. Final report

Report:

- files changed;
- backlog update status;
- benchmark methodology;
- root causes/findings;
- CPU test result;
- CUDA test result;
- pageable H2D/D2H results;
- pinned H2D/D2H results;
- bidirectional result;
- allocation/setup findings;
- PCIe generation/width observed under load, if captured;
- raw evidence paths;
- architectural implications;
- unresolved limitations;
- working tree status;
- final Task 0.2E status.
