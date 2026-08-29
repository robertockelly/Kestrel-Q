# ADR 0008 — Native read-only GGUF container layer

Status: **Accepted — 2026-08-29**

## Context

ADR 0006 selected a staged strategy: use the verified GGUF for initial correctness/runtime work while keeping canonical model semantics independent from physical container format.

## Decision

1. Implement a Kestrel-Q-owned GGUF v3 parser in C17.
2. Use Windows-native read-only memory mapping.
3. Expose bounded logical file views.
4. Keep GGUF physical tensor descriptors separate from canonical semantic descriptors.
5. Parse/validate metadata and tensor directory without touching tensor payload in ordinary inspection.
6. Support the target-required subset first; unsupported forms fail closed.
7. Do not link to llama.cpp/GGML.
8. Use checked 64-bit file/span arithmetic.

## Evidence gate

Task 2.0 must demonstrate:

- valid and malformed synthetic coverage;
- exact reproduction of Epic 1's real-GGUF structural oracle;
- deterministic cleanup/failure behavior;
- no intentional tensor-payload access in normal inspection.

Task 2.0 finalizes this decision with the following evidence.

The evidence gate is satisfied. Task 2.0's deterministic fixture suite covers a
valid seven-type GGUF plus truncation, magic/version errors, bounded-string and
array failures, arithmetic overflow, invalid rank/dimensions/type/alignment,
duplicate tensors, invalid spans and invalid quantized block geometry. Failure
paths release mappings and handles.

The opt-in KQ-01 integration test reproduces the registered artifact oracle:
111,334,654,400 file bytes, GGUF v3, `qwen4exp`, 67 metadata entries, 1,224
tensors, 111,323,630,080 packed bytes, 11,024,320 overhead bytes and all seven
type counts. Parsing stops at directory byte 11,024,307 before the aligned data
section at 11,024,320 and reports zero payload bytes accessed.

## Consequences

- Task 2.1 can build canonical semantic descriptors over validated physical
  descriptors without coupling semantics to GGUF names.
- The initial production runtime remains Windows-first and target-first; other
  GGUF versions, metadata forms and tensor types fail closed.
- The header/directory mapping is virtual and bounded. It does not establish a
  scheduler, residency policy or measured physical-I/O behavior.
- No llama.cpp/GGML runtime dependency, dequantizer, tokenizer, PLE engine,
  expert cache or inference path is introduced.
