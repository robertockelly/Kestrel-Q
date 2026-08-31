# Kestrel-Q target weight-provider API

The public C17 declarations are in `include/kq_weight_provider.h`. This API is
model-specific and accepts only descriptors owned by the same verified model
registry supplied at construction.

## Construction and lifetime

`kq_weight_provider_open` borrows a GGUF and model and fixes a positive logical
payload budget. The file/GGUF/model must outlive the provider. Close active
layer/state objects before their model owners. The provider contains no hidden
global state and opens every bounded tensor view for one operation only.

`kq_weight_provider_preflight_layer` validates all executable semantics and
all physical types for one layer, plus first/last routed-expert member views
and PLE member views. It may create virtual read-only mappings but never calls
the payload data accessor; 48/48 preflight therefore reports zero payload
bytes accessed.

## Semantic operations

`kq_weight_provider_linear_f32` executes one canonical matrix-vector request.
The caller supplies the semantic descriptor, exact binding part, explicit
expert ID or `KQ_WEIGHT_PROVIDER_NO_EXPERT`, canonical rows/columns, output
capacity and scratch. Output is staged and unchanged on error.

`kq_weight_provider_vector_f32` permits only bounded vectors and the explicitly
supported GDN/PLE convolution vectors. The hard element bound is 65,536;
complete matrices fail closed. For the exact qwen4exp artifact it also restores
converter-folded zero-centered deltas for HC, QSA/indexer and PLE norm roles;
the direct GDN linear-attention norm gamma is not transformed.

`kq_weight_provider_ple_lookup_interface` returns the existing Task 2.9
storage-neutral callback contract. It decodes exactly one 160-value fused-table
row per canonical intent. Metadata-derived semantics have no payload path.

`kq_weight_provider_get_metrics` returns provider-owned counters valid until
close. The trace-copy calls expose the bounded diagnostic route order,
physical expert-member access order and PLE member/row order used by Task 2.11
validation. They do not select experts, prefetch data or define scheduling.
`linear_elapsed_nanoseconds` is optional high-resolution host timing around
the scalar row-dot loops. A missing platform counter leaves it at zero; it is
characterization telemetry, not an I/O, residency or throughput measurement.
Counter conversion and accumulation use checked 64-bit arithmetic.

## Errors and memory

The API preserves the specific model/tensor/numeric status produced by the
underlying registry, view and numeric layers. Typical failures include wrong
ownership, absent split part, invalid expert, dimension mismatch, unsupported
type/geometry, insufficient capacity, aliasing, arithmetic overflow and
payload-budget exhaustion. No request is silently truncated.

The caller owns activation and scratch storage. The complete-layer convenience
query `kq_layer_required_quantized_scratch_bytes` adds a fixed 1 MiB
weight-operation arena to the existing operator/layer scratch. This arena is
reused serially; it is not proportional to a matrix or the model.
