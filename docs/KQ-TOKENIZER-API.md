# Kestrel-Q tokenizer API

Status: **Task 2.3 baseline**

The public C17 API is declared in `include/kq_tokenizer.h`.

## Lifecycle

```c
kq_status kq_tokenizer_open_from_gguf(
    const kq_gguf *gguf,
    kq_tokenizer **out_tokenizer,
    kq_diagnostic *diagnostic);

void kq_tokenizer_close(kq_tokenizer *tokenizer);
```

Construction validates the exact physical tokenizer substrate. `kq_tokenizer`
borrows metadata views from `kq_gguf`; close it before closing the GGUF, mapped
file or owning model. The object is immutable after construction and
encode/decode accept a const tokenizer. No global mutable state exists.

## Encoding

`kq_tokenizer_encode` accepts explicit UTF-8 pointer/length input and either
rejects canonical special literals or recognizes them as atomic IDs. It never
adds BOS/EOS. A null output with capacity zero is a size query. Non-empty output
returns `KQ_STATUS_BUFFER_TOO_SMALL` with the exact required count; no partial
result is exposed.

Invalid UTF-8, options, arithmetic, allocation, or substrate states produce a
stable non-OK status and diagnostic. Input `(NULL,0)` is valid; `(NULL,n)` is
not.

## Decoding

`kq_tokenizer_decode` accepts explicit ID pointer/count and supports:

- `KQ_TOKENIZER_DECODE_KEEP_SPECIAL`; or
- `KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL`.

The latter follows official added-token classification, not GGUF token types.
IDs 248077–248319 and all larger IDs fail with
`KQ_STATUS_INVALID_TOKEN_ID`. Required-size/capacity behavior mirrors encode.

## Metrics

`kq_tokenizer_get_metrics` exposes construction time and owned/lookup heap
bytes for characterization.
`kq_tokenizer_nfc_unicode_version()` reports `9.0.0` and
`kq_tokenizer_property_unicode_version()` reports `16.0.0`, matching the two
distinct Unicode table versions in the pinned oracle.

The API does not expose vocabulary mutation, a generic regex engine, Python
objects, tensor payload views or implicit model lifetime ownership.
