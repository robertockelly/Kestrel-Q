# MODEL-GGUF-MAPPING-GUIDE.md

## Canonical semantics
Canonical tensor semantics come from the pinned official Qwen Safetensors checkpoint and Task 1.1 architecture evidence. GGUF is a derived physical representation.

## Mapping principles
A canonical tensor may map to one GGUF tensor, a renamed tensor, a fused tensor, multiple GGUF tensors, or no standalone GGUF tensor because of tying/derivation/layout semantics.

Never equate tensor-count difference with semantic loss without evidence.

## Mapping statuses
- `ONE_TO_ONE`
- `RENAMED_ONE_TO_ONE`
- `FUSED_INTO_GGUF`
- `SPLIT_IN_GGUF`
- `TRANSFORMED_LAYOUT`
- `OMITTED_INITIAL_SCOPE_VALID`
- `OMITTED_FORMAT_DERIVED`
- `ALIAS_OR_TIED`
- `NO_DIRECT_TENSOR_EQUIVALENT`
- `UNRESOLVED`

PASS requires `UNRESOLVED = 0`.

`TRANSFORMED_LAYOUT` is the one-to-one converter-transform bucket. The mapping
row must state whether the concrete operation is a dimension/layout reorder,
singleton-dimension removal or value-domain conversion; the status name alone
must never be used to infer the transform.

## Packed bytes
GGUF packed bytes must come from actual tensor type/block semantics and tensor spans. Do not estimate actual GGUF storage with nominal bits/parameter where block formats apply.

## Effective bits per parameter

```text
effective_bpp = packed_tensor_bytes * 8 / parameter_count
```

Keep file-level metadata/alignment overhead separate.

## Evidence confidence
Use:
- `PROVEN_CANONICAL`
- `PROVEN_FORMAT`
- `PROVEN_CONVERTER`
- `OBSERVED_ARTIFACT`
- `INFERRED_STRONG`
- `UNRESOLVED`

Task PASS requires no unresolved semantic mapping. Provenance details may remain `INFERRED_STRONG` only when explicitly documented.

## Payload rule
Tensor payload values are not required for Task 1.3 structural mapping. Do not read payload data by default.
