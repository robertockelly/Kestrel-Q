# Native read-only GGUF layer

Status: **TASK 2.0 COMPLETE / PASS**

Task 2.0 introduces Kestrel-Q's first production C17 container component. It is
a Windows-native, read-only file/view layer plus a target-first GGUF v3 parser.
It performs no inference, dequantization, tokenization or model scheduling.

## Production boundaries

The public headers are:

- `include/kq_status.h`: stable status categories and bounded diagnostics;
- `include/kq_file.h`: opaque read-only files and bounded logical views;
- `include/kq_gguf.h`: immutable GGUF metadata and physical tensor descriptors.

The Win32 implementation uses `CreateFileW` with `GENERIC_READ`,
`GetFileSizeEx`, read-only `CreateFileMappingW` and `MapViewOfFile`. A logical
view may start at any byte offset. Internally the mapping begins at the previous
Windows allocation-granularity boundary, checks `alignment_delta + length`, and
exposes only the requested logical pointer and length. Every offset, end,
alignment, allocation and dimension product is checked before use.

The caller owns `kq_file`. A `kq_file_view` owns one mapped view, and `kq_gguf`
owns its header/directory view and descriptor arrays. Close a GGUF before its
file. All failed opens release partial descriptor arrays and mapped views; the
malformed-fixture suite verifies cleanup by deleting every temporary file after
failure.

## Supported GGUF subset

The parser accepts GGUF version 3 and the metadata forms required by
`KQ-MODEL-ARTIFACT-001`:

- scalar `UINT16`, `UINT32`, `INT32`, `FLOAT32`, `BOOL` and `STRING`;
- arrays of `INT32`, `UINT64` or `STRING`;
- bounded, structurally valid UTF-8 length-prefixed keys, values and tensor
  names;
- explicit/default power-of-two alignment up to 4096 bytes;
- tensor ranks 1–8 with non-zero dimensions.

Unsupported metadata types, nested arrays and other unimplemented forms fail
closed. Defensive limits cap the mapped header/directory at 512 MiB, individual
strings at 128 MiB and array elements at 2,000,000. These are parser safety
limits, not properties inferred from the target model.

Directory counts are separately capped at 1,024 metadata entries and 8,192
tensors. The verified artifact uses 67 and 1,224 respectively.

The physical tensor registry supports only the seven storage types observed in
the verified artifact:

| Type | ID | Elements/block | Bytes/block |
|---|---:|---:|---:|
| F32 | 0 | 1 | 4 |
| Q5_1 | 7 | 32 | 24 |
| Q8_0 | 8 | 32 | 34 |
| Q4_K | 12 | 256 | 144 |
| Q5_K | 13 | 256 | 176 |
| IQ4_NL | 20 | 32 | 18 |
| BF16 | 30 | 1 | 2 |

For each tensor the parser validates block divisibility of the first physical
dimension, checked element and packed-byte products, aligned relative offset,
absolute EOF bounds and non-overlap. Duplicate metadata keys and tensor names
are rejected. No type is dequantized.

## No-payload-touch boundary

`kq_gguf_open` maps at most the first 512 MiB of virtual file address space so
that file-backed bounded string views remain valid. Virtual mapping is not a
payload read. The parser advances only through the fixed header, metadata and
tensor directory, then calculates tensor spans from dimensions and block
geometry. It exposes data offsets and packed lengths, not payload pointers.

The API reports `directory_bytes_parsed` and
`payload_bytes_accessed`. Synthetic and real-artifact tests require the parsed
cursor to stop before the aligned data section and require
`payload_bytes_accessed == 0`. Windows may independently perform paging or
read-ahead; Task 2.0 does not claim exact physical I/O without system tracing.

Normal inspection does not hash the 111 GB artifact. The opt-in integration
test confirms size and last-write timestamp remain unchanged around inspection.

## Error model

The library distinguishes invalid arguments, file open/size/map failures,
truncation, bad magic, unsupported version, malformed/unsupported metadata,
malformed tensors, unsupported tensor types, arithmetic overflow, defensive
limit violations, invalid alignment, out-of-range spans, duplicate names and
an inconsistent data section. APIs return `kq_status`; optional
`kq_diagnostic` text adds bounded context for the CLI and tests.

## Provenance and limitations

The implementation is Kestrel-Q-owned C17 and has no llama.cpp/GGML runtime or
source dependency. Type IDs and block geometry are format facts verified during
Task 1.3 against the pinned MIT-licensed llama.cpp evidence revision; no
third-party implementation code is copied.

This layer does not attach canonical Qwen semantics to physical GGUF names,
read tensor values, expose arbitrary payload views through the GGUF API, parse
split-model sets, support other GGUF versions/types or implement a production
model loader. Canonical semantic descriptors begin in Task 2.1.
