# NATIVE-GGUF-API-GUIDE.md

Status: **IMPLEMENTED / TASKS 2.0–2.2 COMPLETE**

## File boundary

Task 2.0 exposes opaque read-only Windows file and view objects:

```c
kq_status kq_file_open_readonly(const wchar_t *, kq_file **, kq_diagnostic *);
uint64_t  kq_file_size(const kq_file *);
kq_status kq_file_view_open(kq_file *, uint64_t, uint64_t,
                            kq_file_view **, kq_diagnostic *);
const unsigned char *kq_file_view_data(const kq_file_view *);
void kq_file_view_close(kq_file_view *);
void kq_file_close(kq_file *);
```

A logical view is bounded even when the underlying Windows mapping must be aligned to allocation granularity.

## GGUF boundary

Implemented capabilities:

```c
kq_status kq_gguf_open(kq_file *, kq_gguf **, kq_diagnostic *);
void      kq_gguf_close(kq_gguf *);

uint64_t kq_gguf_metadata_count(...);
uint64_t kq_gguf_tensor_count(...);

const kq_gguf_metadata *kq_gguf_metadata_at(...);
const kq_gguf_metadata *kq_gguf_find_metadata(...);
const kq_gguf_tensor *kq_gguf_tensor_at(...);
const kq_gguf_tensor *kq_gguf_find_tensor(const kq_gguf *, const char *);
```

Use `(ptr,length)` string views for untrusted file-backed strings.

Tensor descriptors contain name, rank, dimensions, GGUF type, element count, offset and packed bytes. They do not require a payload pointer.

Task 2.1 extends metadata descriptors with retained scalar bits and bounded raw
views for fixed-width numeric arrays. Typed helpers expose uint16/uint32/int32
scalars and indexed int32/uint64 array values. String arrays remain validated
and length-addressable but are not flattened or copied. These are still
header/directory views, not tensor payload access.

Task 2.0 is target-first: GGUF v3 and the metadata/tensor types required by the verified Qwen artifact. Unsupported forms fail closed.

`kq_gguf` retains a header/directory view into the caller-owned `kq_file`.
Close the GGUF before the file. Returned metadata/tensor/string views are
immutable and remain valid only until `kq_gguf_close`.

Canonical model semantics are exposed separately by `include/kq_model.h`; see
`docs/KQ-SEMANTIC-API.md`.

Task 2.2 keeps arbitrary payload offsets out of the public GGUF API. Its
semantic tensor-view layer internally verifies that a physical descriptor
belongs to the parsed GGUF and that the requested checked block span remains
inside that tensor before delegating to `kq_file_view_open`. Public consumers
use `include/kq_tensor_view.h`, documented in `docs/KQ-TENSOR-VIEW-API.md`.
