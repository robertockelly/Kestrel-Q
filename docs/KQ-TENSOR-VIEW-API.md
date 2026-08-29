# Kestrel-Q tensor-view API

Status: **IMPLEMENTED / TASK 2.2 COMPLETE**

The public C17 interface is `include/kq_tensor_view.h`.

## Quantized geometry

```c
kq_quant_geometry_for_type(type_id, &geometry, &diagnostic);
kq_quant_packed_size(type_id, elements, &bytes, &diagnostic);
kq_quant_element_to_block(type_id, total, index,
                          &block, &element_in_block, &diagnostic);
kq_quant_range_to_blocks(type_id, total, offset, count,
                         &span, &diagnostic);
```

Ranges are non-empty and half-open. `kq_quant_block_span` records the requested
elements, first/block count, physical byte offset/length and any leading or
trailing elements included because quantized storage is block-addressed.

## Open operations

```c
kq_tensor_view_open_binding(gguf, semantic, binding_index,
                            access, &view, &diagnostic);
kq_tensor_view_open_expert_member(gguf, semantic, binding_index,
                                  expert_id, access,
                                  &view, &diagnostic);
kq_tensor_view_open_ple_member(gguf, semantic, access,
                               &view, &diagnostic);
```

`KQ_TENSOR_VIEW_PHYSICAL_LAYOUT` permits explicit inspection/consumption of
validated physical bytes while retaining layout metadata.
`KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS` rejects transformed physical
layouts and incomplete split parts.

Generic binding opens reject fused PLE members so callers cannot accidentally
map the complete table. Expert opens require a proven contiguous last physical
axis. Metadata-derived semantics return `KQ_STATUS_NO_TENSOR_PAYLOAD`.

## Inspection and lifetime

```c
const kq_tensor_view_info *info = kq_tensor_view_get_info(view);
const unsigned char *bytes = kq_tensor_view_physical_data(view);
kq_tensor_view_close(view);
```

The returned info is immutable and valid until close. The byte pointer is a
physical quantized representation, never a dequantized or automatically
canonicalized buffer. Its addressable length is exactly
`info->mapped_logical_length`.

Keep the source semantic registry and parsed GGUF/backing file alive while the
view is open. Close the view first. Opening allocates only small view state and
one read-only Win32 mapping object; it does not copy tensor payload.

## Layout classifications

- `CANONICAL_CONTIGUOUS`: one physical byte sequence has canonical flattened
  ordering despite GGUF's reversed dimension notation;
- `TRANSFORMED_PHYSICAL`: converter transform/reorder is preserved explicitly;
- `SPLIT_SEGMENT`: one ordered physical part of a canonical semantic;
- `FUSED_MEMBER`: one bounded canonical member inside a shared physical tensor.

No accessor promises typed scalar values. Dequantization and execution remain
future, separately tested APIs.
