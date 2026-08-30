# Kestrel-Q numeric API

Status: **IMPLEMENTED / TASK 2.5 COMPLETE**

The public C17 interface is `include/kq_numeric.h`. All functions return a
stable `kq_status` and optionally populate `kq_diagnostic`.

## Packed block decode

```c
kq_dequantize_blocks_f32(type_id, packed, packed_bytes,
                          output, output_capacity,
                          &output_count, &diagnostic);
```

`packed_bytes` must be a non-zero whole number of the type's blocks. On
insufficient output capacity, `output_count` reports the exact requirement and
the function returns `KQ_STATUS_BUFFER_TOO_SMALL`; no silent truncation occurs.
Input/output overlap is forbidden. Supported types are F32, BF16, Q5_1, Q8_0,
Q4_K, Q5_K and IQ4_NL.

```c
kq_dequantize_view_blocks_f32(view, relative_first_block, block_count,
                               order, output, output_capacity,
                               &output_count, &diagnostic);
```

The view overload validates the requested block range against the mapped Task
2.2 member/segment. `KQ_NUMERIC_PHYSICAL_ORDER` explicitly returns physical
GGUF order and is allowed for transformed or split segments.
`KQ_NUMERIC_REQUIRE_CANONICAL_ORDER` rejects transformed and incomplete split
layouts with `KQ_STATUS_TENSOR_LAYOUT_MISMATCH`.

The tensor view and its model/GGUF/file owners must remain alive for the call.
The numeric layer retains no pointer after return.

## Quantized row dot

```c
kq_quantized_row_dot_f32(type_id, packed, packed_bytes,
                          activation, activation_count,
                          &scalar, &diagnostic);
```

The activation length must equal the decoded row length. The function decodes
one block at a time into a fixed 1,024-byte stack scratch and accumulates F32
left-to-right. It is a correctness primitive, not GEMM.

## F32 primitives

The API exposes:

```c
kq_f32_add(...);
kq_f32_multiply(...);
kq_f32_scale(...);
kq_f32_dot(...);
kq_f32_sigmoid(...);
kq_f32_silu(...);
kq_f32_swiglu(...);
kq_f32_rms_norm(...);
kq_f32_softmax(...);
kq_f32_stable_top_k(...);
kq_f32_renormalize(...);
```

All vector lengths are non-zero and contiguous. Input and output buffers may
not overlap. Inputs must be finite; invalid NaN/Inf, dimensions, epsilon,
top-k, capacity, alias or non-default rounding mode fail explicitly. Top-k
orders descending scores and resolves exact ties by lower input index.

`kq_f32_rms_norm` takes the Qwen weight-delta representation and multiplies by
`1 + weight_delta`. It is not the gated/grouped model operator.

## Ownership and exclusions

The numeric APIs allocate no heap memory and keep no global state. They do not
own or extend tensor-view lifetimes. No dequantized tensor object, quantizer,
SIMD kernel, CUDA kernel, model layer, inference graph or sampler is exposed.
