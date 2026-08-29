# MODEL-TENSOR-INVENTORY-GUIDE.md

## Canonical source

The Task 1.2 inventory derives only from:

`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`

The local Unsloth GGUF must not be used to define canonical tensor identity. GGUF mapping begins in Task 1.3.

## Safetensors metadata rule

Safetensors files begin with:

- bytes `0..7`: little-endian unsigned JSON-header length;
- next `header_length` bytes: JSON metadata;
- tensor payload after the header.

Task 1.2 may retrieve only the first 8 bytes and the exact declared JSON header range.

If HTTP Range is not honored, abort.

## Inventory invariants

For every tensor:

```text
parameter_count = product(shape)
payload_bytes   = data_offsets[1] - data_offsets[0]
```

For fixed-width dtypes, independently verify payload size from parameter count and element size when Safetensors semantics permit.

Data offsets must be valid and non-overlapping.

## Units

Use exact bytes for canonical storage.

Where human-readable units are useful:

- GiB = bytes / 2^30
- GB = bytes / 10^9

Always label the unit.

## Quantization lower bounds

`parameter_count * bits / 8` is an idealized payload lower bound only.

It excludes:

- scales
- zero points
- block metadata
- padding/alignment
- format metadata
- mixed-precision exceptions

Never label these values as GGUF sizes.

## Initial-scope labels

Every tensor receives exactly one:

- `REQUIRED_INITIAL_TEXT`
- `EXCLUDED_INITIAL_VISION`
- `EXCLUDED_INITIAL_MTP`
- `REVIEW`

Task 1.2 PASS requires `REVIEW = 0`.

## Preliminary placement labels

Optional analysis-only labels:

- `ALWAYS_NEEDED_CANDIDATE`
- `ROUTED_EXPERT_CACHE_CANDIDATE`
- `PLE_PREFETCH_CANDIDATE`
- `COLD_BACKING_CANDIDATE`
- `EXCLUDED_INITIAL_SCOPE`

These are not scheduler policy.

## Classification rule

Classification must be deterministic and traceable to canonical tensor naming plus the architecture evidence from Task 1.1.

No classification may be inferred from the local GGUF.

## Evidence

All headline numbers in `docs/MODEL-FOOTPRINT.md` must be reproducible from committed machine-readable outputs and research tooling.
