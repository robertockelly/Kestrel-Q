# PLAN-TASK-2.0-NATIVE-GGUF-MMAP.md

Status: **COMPLETE / PASS — 2026-08-29**

## Objective

Implement Kestrel-Q's first production runtime component:

> Windows-native, read-only, C17 GGUF v3 parsing backed by a 64-bit-safe memory-mapped file abstraction.

Primary executable:

```text
kq-inspect <model.gguf>
```

No inference, dequantization, tokenizer, PLE engine, expert cache, scheduler or model CUDA kernel is allowed in this task.

## Epic 1 structural oracle

Verified artifact:

- file size: `111334654400`
- GGUF version: `3`
- architecture: `qwen4exp`
- metadata entries: `67`
- tensors: `1224`
- packed tensor bytes: `111323630080`
- overhead: `11024320`

Type counts:

- BF16: 24
- F32: 557
- IQ4_NL: 1
- Q4_K: 94
- Q5_1: 43
- Q5_K: 2
- Q8_0: 503

These are integration-test oracle values, never parser constants.

## Windows file layer

Use read-only native APIs:

- `CreateFileW`
- `GetFileSizeEx`
- `CreateFileMappingW`
- `MapViewOfFile`
- `UnmapViewOfFile`
- `CloseHandle`

Requirements:

- exact 64-bit size/offset handling;
- checked `offset + length`, alignment, count×size and dimension products;
- hide Windows allocation-granularity alignment inside bounded logical views;
- no write access;
- deterministic cleanup on all error paths.

Do not assume the full 111 GB file must be resident merely because it is virtually mapped.

## GGUF parser

Support GGUF v3 target-first.

Parse/validate:

- magic/version;
- metadata count and tensor count;
- bounded length-prefixed strings;
- metadata scalar/array types present in the verified artifact;
- tensor names, ranks, dimensions, types and offsets;
- data-section alignment/start;
- tensor packed spans and aggregate bytes;
- duplicate tensor names;
- spans inside EOF and non-overlap as required.

Unknown or unsupported forms fail closed.

Untrusted strings are not implicit C strings.

## Required tensor types

Recognize and validate storage geometry for:

- F32
- BF16
- Q5_1
- Q8_0
- Q4_K
- Q5_K
- IQ4_NL

For each, know block elements and bytes/block. Do not dequantize.

Pinned format provenance may use `ggml-org/llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a` (MIT), but implementation must remain independent and must not link to llama.cpp.

## Error model

Distinguish at least:

- invalid argument;
- file open/size/map failure;
- truncated file;
- bad magic;
- unsupported version;
- malformed/unsupported metadata;
- malformed tensor;
- unsupported tensor type;
- arithmetic overflow;
- invalid alignment;
- out-of-range span;
- duplicate tensor;
- inconsistent data section.

Library returns status; CLI prints diagnostics.

## Tensor descriptor

Expose immutable physical descriptors containing:

- bounded name view;
- rank/dimensions;
- type;
- element count;
- data offset;
- packed bytes.

Do not attach canonical Qwen semantics yet. That belongs to Task 2.1.

## `kq-inspect`

Minimum default summary:

- file size;
- GGUF version;
- architecture;
- metadata count;
- tensor count;
- data-section offset;
- packed tensor bytes;
- format overhead;
- counts by tensor type.

Useful optional modes may include:

- `--metadata`
- `--tensors`
- `--tensor <name>`
- `--json`

No payload dump or eager payload read.

## Synthetic tests

Create deterministic tiny valid fixtures and malformed cases.

Required fail-closed coverage includes:

- truncation;
- bad magic/version;
- malformed/absurd strings;
- malformed arrays;
- count/size overflow;
- invalid rank/dimensions;
- unsupported tensor type;
- duplicate name;
- invalid alignment;
- span past EOF;
- invalid quantized block geometry.

## Real artifact integration

Opt-in only when `KQ_GGUF_PATH` exists. The 111 GB artifact is not a CI dependency.

Assert the complete Epic 1 structural oracle above.

Do not hash 111 GB on every run and do not silently accept another GGUF.

## No-payload-touch boundary

Normal parse/inspect must not intentionally dereference tensor payload. Mapping virtual address space is not the same as touching resident pages.

Descriptors calculate spans from directory/type geometry.

Do not claim exact physical disk reads without system tracing.

## Build/regression

Task 2.0 has no CUDA dependency.

Validate:

- clean CPU Release build/tests;
- clean CUDA Release build/tests;
- all existing tests remain green;
- `/W4` with no new Kestrel-Q warnings.

## Documentation

Create:

- `docs/NATIVE-GGUF-LAYER.md`
- `docs/KQ-INSPECT.md`

Finalize ADR 0008.

Update:

- `docs/ARCHITECTURE.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- Epic 2 plan/status
- Task 2.0 checklist
- `CHANGELOG.md`

## Definition of done

Task 2.0 is COMPLETE/PASS only when Kestrel-Q has a tested, fail-closed, Windows-native C17 read-only GGUF v3 layer and `kq-inspect` reproduces Epic 1's structural oracle on the verified 111 GB artifact without implementing inference or touching tensor payload during normal inspection.

Do not commit/push automatically.
