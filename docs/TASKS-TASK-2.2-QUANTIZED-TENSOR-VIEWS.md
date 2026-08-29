# TASKS-TASK-2.2-QUANTIZED-TENSOR-VIEWS.md

Status: **COMPLETE / PASS — 2026-08-29**

## Baseline
- [x] Record actual HEAD/worktree.
- [x] Confirm Task 2.1 COMPLETE/PASS.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Quant geometry
- [x] F32
- [x] BF16
- [x] Q5_1
- [x] Q8_0
- [x] Q4_K
- [x] Q5_K
- [x] IQ4_NL
- [x] block elements/bytes
- [x] checked block count/packed bytes
- [x] logical range -> block range
- [x] bounds/overflow failures

## Views
- [x] immutable read-only view
- [x] whole tensor
- [x] split relation
- [x] transformed layout metadata
- [x] explicit lifetime/cleanup
- [x] no copy/dequantization

## Expert views
- [x] expert axis validation
- [x] expert id 0..511
- [x] checked stride/span
- [x] prove contiguity
- [x] map only selected member
- [x] fail closed when not representable

## PLE views
- [x] 128 member geometry
- [x] first/middle/last
- [x] checked member span
- [x] no full-PLE mapping
- [x] invalid member fails

## Metadata-derived
- [x] payload request returns explicit no-payload status
- [x] no fake offset/view

## Synthetic payload tests
- [x] exact bytes
- [x] guard boundaries
- [x] unaligned logical offset
- [x] block range
- [x] split parts
- [x] expert member
- [x] PLE member
- [x] close/reopen

## Fail closed
- [x] out-of-range
- [x] overflow
- [x] span past EOF
- [x] invalid block geometry
- [x] bad expert id/axis
- [x] non-contiguous member
- [x] bad PLE member/fusion
- [x] broken split
- [x] metadata-derived payload request
- [x] transformed-layout misuse

## Real artifact
- [x] dense
- [x] Q8_0
- [x] Q4_K
- [x] IQ4_NL
- [x] Q5_1/Q5_K
- [x] expert members incl. layer 2
- [x] split MoE
- [x] split QSA
- [x] PLE first/middle/last
- [x] no hard-coded offsets
- [x] payload touched = 0

## Oracle/regression
- [x] compare Task 1.3 geometry
- [x] production reads no research files
- [x] clean CPU Release PASS
- [x] clean CUDA Release PASS
- [x] Task 2.0/2.1 remain PASS
- [x] no new warnings
- [x] git diff --check PASS

## Docs/governance
- [x] QUANTIZED-TENSOR-VIEWS.md
- [x] KQ-TENSOR-VIEW-API.md
- [x] ADR 0010
- [x] ARCHITECTURE / semantic API updates
- [x] TASKS/ROADMAP/Epic 2
- [x] CHANGELOG.md

## Safety
- [x] no dequant/inference/tokenizer/PLE/scheduler
- [x] no production research dependency
- [x] no tracked model weights
- [x] no secrets/local path
- [x] build trees untracked

## Recorded evidence

- Entry HEAD: `dfc9fea7c9320b691a444ff47d7b2cb692e558ec`; only the four
  maintainer-provided governed Task 2.2 documents were untracked at entry.
- Synthetic payload suite: all seven storage types, whole/split/expert/PLE,
  guarded and non-allocation-granularity spans, two open/close lifecycles and
  1,192 deliberately dereferenced fixture payload bytes.
- Real registered artifact: dense and all-seven-type representatives, layer-2
  expert members, both MoE/QSA split forms and PLE members 0/64/127; payload
  bytes touched by the test are zero.
- Task 1.3 geometry oracle: 1,224/1,224 physical descriptors and
  111,323,630,080 packed bytes match; deterministic dump SHA-256
  `c3e15e3ec379c207629183fd94cb708dc95d92f87f8948882438dda96f6729ab`.
- Implementation review finding: the first view-info draft recorded requested
  elements and physical block bytes but omitted the separate canonical
  unpacked-byte count. Root cause was an incomplete descriptor contract, not
  incorrect mapping arithmetic. `logical_unpacked_bytes` and the copied stable
  semantic ID were added before the clean gate, and the synthetic suite asserts
  the corrected accounting.
- Intermediate `/W4` finding: MSVC C4701 conservatively flagged the checked
  canonical element count returned through a status helper even though failure
  returned before use. Explicit initialization removed the diagnostic without
  changing control flow; the final clean CPU/CUDA builds guard the regression.
- Intermediate test finding: the first non-contiguous-axis mutation also
  changed the canonical element total, so the earlier geometry gate correctly
  rejected it before contiguity was evaluated. The fixture now preserves total
  elements while moving the expert axis off the slowest physical dimension,
  isolating the intended `NONCONTIGUOUS_TENSOR_VIEW` regression.
