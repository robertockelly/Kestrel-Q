# Kestrel-Q semantic registry API

Status: **IMPLEMENTED / TASK 2.1 COMPLETE**

The public C17 interface is `include/kq_model.h`.

## Construction and lifetime

```c
kq_status kq_model_open_from_gguf(const kq_gguf *gguf,
                                  kq_model **out_model,
                                  kq_diagnostic *diagnostic);
void kq_model_close(kq_model *model);
```

The caller retains ownership of the parsed `kq_gguf`. Keep it open until after
`kq_model_close`, because bindings refer to its immutable metadata and physical
descriptors. A failed construction returns no partial model. There is no global
mapping state and no post-construction mutation API.

Construction validates model identity, topology, metadata, all expected
physical names, ranks, dimensions, storage-type compatibility and binding
cardinality. It also rejects every unrecognized physical tensor.

## Model and coverage queries

The API exposes validated model facts:

```c
kq_model_hidden_size(model);
kq_model_vocabulary_size(model);
kq_model_context_length(model);
kq_model_layer_count(model);
kq_model_gdn_layer_count(model);
kq_model_qsa_layer_count(model);
kq_model_expert_count(model);
kq_model_expert_top_k(model);
kq_model_layer_type_at(model, layer_id);
```

Coverage queries include semantic/physical counts, unique physical coverage,
metadata-derived entries, unknown/unbound counts, and per-relation/component/
placement totals. Out-of-range enum or layer queries return zero or
`KQ_MODEL_LAYER_INVALID`.

## Semantic lookup

```c
const kq_semantic_tensor *kq_model_semantic_tensor_at(
    const kq_model *model, uint64_t index);

const kq_semantic_tensor *kq_model_find_semantic_tensor(
    const kq_model *model, const char *semantic_id);
```

Returned descriptors are immutable and valid until `kq_model_close`.
`semantic_id` is the runtime key. `canonical_name` is traceability information;
physical GGUF names live only inside binding descriptors.

Each `kq_tensor_binding` identifies an ordered physical part or exact metadata
entry. Split parts carry their role/order, fused PLE bindings carry member
index/count, and routed stacks carry the validated physical expert axis. A
metadata-derived binding has no physical tensor or payload offset.

## Enumerations

The API makes the following concepts explicit:

- component and projection role;
- `GDN`/`QSA` layer type;
- direct, renamed, transformed, split, fused, metadata-derived and
  absent-scope relations;
- required text, excluded vision and excluded MTP scopes;
- always-needed, routed-cache, PLE-disk-backed, excluded and neutral placement
  annotations;
- canonical BF16/I64 descriptor dtype.

Name helpers produce deterministic diagnostic strings for every enumeration.

## Non-capabilities

This API does not provide tensor payload views, per-expert slices,
dequantization, allocation, placement enforcement, execution, tokenizer/PLE
logic or scheduling. Those require later task-specific correctness gates.
