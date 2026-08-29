# NATIVE-GGUF-API-GUIDE.md

Status: **IMPLEMENTED / TASK 2.0 COMPLETE**

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

const kq_gguf_tensor *kq_gguf_tensor_at(...);
const kq_gguf_tensor *kq_gguf_find_tensor(const kq_gguf *, const char *);
```

Use `(ptr,length)` string views for untrusted file-backed strings.

Tensor descriptors contain name, rank, dimensions, GGUF type, element count, offset and packed bytes. They do not require a payload pointer.

Task 2.0 is target-first: GGUF v3 and the metadata/tensor types required by the verified Qwen artifact. Unsupported forms fail closed.

`kq_gguf` retains a header/directory view into the caller-owned `kq_file`.
Close the GGUF before the file. Returned metadata/tensor/string views are
immutable and remain valid only until `kq_gguf_close`.
