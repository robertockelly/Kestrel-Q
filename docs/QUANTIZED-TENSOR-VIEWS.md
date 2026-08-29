# Quantized tensor views

Status: **TASK 2.2 COMPLETE / PASS**

Task 2.2 establishes the first production payload-view boundary above the
read-only GGUF container and canonical semantic registry. A view maps only a
validated physical block span; it never copies, concatenates, dequantizes or
executes tensor values.

## Supported block geometry

| GGUF type | Elements/block | Bytes/block |
|---|---:|---:|
| `F32` | 1 | 4 |
| `BF16` | 1 | 2 |
| `Q5_1` | 32 | 24 |
| `Q8_0` | 32 | 34 |
| `Q4_K` | 256 | 144 |
| `Q5_K` | 256 | 176 |
| `IQ4_NL` | 32 | 18 |

Checked helpers derive packed bytes, the containing block for one element and
the minimal block-aligned physical byte span for a non-empty logical element
range. Partial tensor geometry, out-of-range requests and every checked
addition/product overflow fail explicitly.

## View descriptor

An immutable `kq_tensor_view_info` records separate facts for:

- the copied stable semantic ID and complete canonical shape;
- the requested whole/segment/member logical shape and element range;
- canonical unpacked byte count, without implying those bytes exist;
- complete physical GGUF shape, type and block geometry;
- first block, block count, tensor-relative byte offset and file offset;
- the bounded Win32 logical mapping offset/length;
- leading/trailing elements included only by block alignment;
- relation, part role, view kind and explicit layout classification.

The data accessor is deliberately named `kq_tensor_view_physical_data`.
Transformed physical bytes are never labeled canonical-contiguous. A caller
may request `KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS`; transformed and
incomplete split representations then fail with
`KQ_STATUS_TENSOR_LAYOUT_MISMATCH`.

Windows allocation-granularity padding stays hidden inside `kq_file_view`.
The public mapped logical span begins exactly at the validated quant block and
cannot address the hidden prefix.

## Whole and split views

A direct, renamed or transformed one-binding semantic may open one complete
physical tensor. The mapped length is exactly `packed_bytes`; no implicit
allocation or copy occurs.

The 48 routed `gate_up` and 12 QSA index `qk` semantics retain their two
ordered bindings. Each gate/up or query/key part opens independently as
`SPLIT_PART`; the API does not expose a concatenated temporary buffer or claim
that one part is the complete canonical tensor.

## Routed expert members

`kq_tensor_view_open_expert_member` validates the semantic expert count and
canonical axis, the binding's physical expert axis and the physical dimension.
The current registered layout places the 512-expert axis last (the slowest
GGML physical axis), so one expert is one contiguous product of all faster
dimensions. The selected member must also begin and end on quant-block
boundaries.

An expert axis that is not the slowest physical axis is not silently strided;
it fails with `KQ_STATUS_NONCONTIGUOUS_TENSOR_VIEW`. No 512-entry permanent
slice registry is created.

Real layer-2 samples derive, rather than hard-code:

- `Q8_0` down stack: 891,289,600 packed bytes / 512 = **1,740,800 bytes** per
  expert;
- each `Q5_K` gate/up stack: 576,716,800 / 512 = **1,126,400 bytes** per
  expert;
- combined selected layer-2 expert representation: **3,993,600 bytes**.

These are view geometry, not measured I/O or execution throughput.

## Fused PLE members

Each of the 128 canonical PLE table descriptors retains its fused member index.
The view layer verifies physical shape `[160, 320001536]`, canonical member
shape `[2500012, 160]`, exact divisibility and quant-block boundaries. One
member maps **225,001,080 bytes**, exactly 1/128 of the 28,800,138,240-byte
`IQ4_NL` physical tensor. Members 0, 64 and 127 pass the real integration gate.

The generic whole-binding API rejects a fused PLE semantic, preventing an
accidental 26.855-GiB mapping when one logical member was requested. This task
does not execute PLE address/hash logic.

The three address-array semantics are metadata-derived. Any payload request
returns `KQ_STATUS_NO_TENSOR_PAYLOAD`; no zero-length or fabricated view is
created.

## Evidence

The deterministic payload-bearing fixture uses known bytes and guard regions.
It validates all seven types, a non-allocation-granularity file offset, exact
whole spans, block expansion, ordered split parts, contiguous expert members,
first/middle/last fused members, cleanup/reopen and fail-closed malformed
requests. It intentionally dereferences 1,192 fixture payload bytes across two
lifecycle passes.

The opt-in real test uses only `KQ_GGUF_PATH`. It constructs and closes dense,
all-seven-type, expert, split and PLE mappings without calling the physical-data
accessor:

```text
payload_bytes_accessed=0
payload_bytes_touched_by_test=0
```

The test/research-only Task 1.3 validator reconciles all 1,224 physical names,
ranks, dimensions, types, block geometry, element counts, offsets and packed
bytes. Aggregate packed bytes are 111,323,630,080. The byte-identical native
geometry dump SHA-256 is
`c3e15e3ec379c207629183fd94cb708dc95d92f87f8948882438dda96f6729ab`.
Production code does not read the evidence CSV or use Python.

## Boundaries

The view, semantic registry, GGUF and backing file must remain open in that
order of dependency; close a view before closing the GGUF/file whose mapping it
owns. All tested failures return no partial view and every successful view has
one explicit close operation.

No dequantization, tensor math, inference, tokenizer, PLE addressing, allocator,
cache, prefetch, scheduler or CUDA model kernel is introduced. Memory mapping
is not a residency or physical-I/O measurement. `KQ-BACKLOG-BENCH-002` remains
deferred before final PLE placement policy.
