# Qwen3.8-Flash-Next canonical-to-GGUF mapping

Status: **TASK 1.3 COMPLETE / PASS**

## Authority and scope

The pinned official Qwen Safetensors checkpoint at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` remains canonical for tensor
identity and model semantics. This document maps that canonical inventory to
the derived Unsloth artifact `KQ-MODEL-ARTIFACT-001`; GGUF names do not redefine
the architecture.

The derived source is pinned to
`unsloth/Qwen3.8-Flash-Next-GGUF@c8b5954a88c2775c546b92593eda40ea041d3176`.
Format and converter behavior was checked against
`ggml-org/llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`
under its MIT license. Exact source files, functions and hashes are recorded in
`mapping-evidence.json`; no third-party implementation source is copied here.

## Verified local structure

| Property | Observed value |
|---|---:|
| Filename | `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf` |
| Size | 111,334,654,400 bytes |
| SHA-256 | `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2` |
| Container | GGUF v3 |
| Architecture | `qwen4exp` |
| Metadata keys | 67 |
| Unique tensors | 1,224 |
| Tensor directory end | 11,024,307 |
| Data-section offset | 11,024,320 |
| Pre-data alignment padding | 13 bytes |
| Packed tensor bytes | 111,323,630,080 |
| Inter-tensor/trailing padding | 0 bytes |
| Header/directory/alignment overhead | 11,024,320 bytes |

All ranks, dimensions, type IDs, block divisibility, offsets and spans validate.
No spans overlap or extend past the file. The inspection read header, metadata
and tensor-directory bytes and performed a whole-file SHA-256 pass; it did not
interpret tensor payload values.

## Complete mapping coverage

| Canonical mapping status | Canonical tensors |
|---|---:|
| `RENAMED_ONE_TO_ONE` | 847 |
| `TRANSFORMED_LAYOUT` | 256 |
| `FUSED_INTO_GGUF` | 128 |
| `SPLIT_IN_GGUF` | 60 |
| `OMITTED_INITIAL_SCOPE_VALID` | 364 |
| `OMITTED_FORMAT_DERIVED` | 3 |
| `UNRESOLVED` | **0** |
| Total | **1,658** |

Every one of the 1,224 GGUF tensors maps back to one or more canonical meanings;
unexplained GGUF tensors are **0**. `TRANSFORMED_LAYOUT` includes one-to-one
converter transforms whose stored values/layout are not a pure rename; the CSV
records the exact transform note for every row.

The tensor-count difference is exactly:

```text
1,658 canonical tensors
- 333 vision tensors omitted by ADR 0005
-  31 MTP tensors omitted by ADR 0005
-   3 PLE int64 address tensors moved to GGUF metadata
- 127 net from 128 PLE table shards fused into one tensor
+  48 net from 48 MoE gate_up tensors split into gate and up
+  12 net from 12 QSA index_qk tensors split into q and k
= 1,224 GGUF tensors
```

This is a representation reconciliation, not evidence of 434 missing weights.

## Versioned mapping rules

| Canonical pattern | GGUF representation | Rule/evidence |
|---|---|---|
| token embedding / LM head | `token_embd.weight` / `output.weight` | tensor-name map |
| final and per-layer hyper-connections | `output_hc_*` / `blk.N.hc_{attn,ffn}_*` | tensor-name map |
| GDN projections/state weights | `blk.N.attn_{qkv,gate}`, `blk.N.ssm_*` | tensor-name map plus converter transforms |
| QSA q/k/v/o and norms | `blk.N.attn_*` | tensor-name map |
| QSA `index_qk_proj` | `blk.N.indexer.{q,k}_proj` | converter split |
| router | `blk.N.ffn_gate_inp` | tensor-name map |
| routed expert `down_proj` | `blk.N.ffn_down_exps` | tensor-name map |
| routed expert `gate_up_proj` | `blk.N.ffn_{gate,up}_exps` | converter split |
| shared expert/gate | `blk.N.ffn_*_shexp` | tensor-name map |
| 128 PLE table shards | one `per_layer_token_embd.weight` | numeric-order concatenation |
| PLE int64 address constants | `qwen4exp.ple.*` metadata arrays | exact metadata representation |
| PLE dense weights | `blk.1.ple_*` | tensor-name map plus norm/conv transforms |
| vision and MTP | no GGUF tensor | accepted ADR-0005 initial-scope omission |

GGUF dimension arrays use the reverse physical dimension order from the
canonical PyTorch/Safetensors shapes. That convention alone is not treated as
a semantic transpose. Proven converter transforms are recorded separately:

- GDN `A_log` is stored after the converter's `-exp` transform;
- grouped GDN value-head rows/columns are reordered to tiled GGML order for five
  projections in each of the 36 GDN layers;
- GDN and PLE convolution singleton dimensions are removed;
- three zero-centred PLE norm gammas are stored after adding one;
- QSA index q/k and routed-expert gate/up projections are split;
- PLE table shards are concatenated in numeric shard order.

## Published split versus local merge

Only bounded header ranges were fetched from the four pinned upstream shards:
40 exact HTTP Range requests, 342 response bytes and **zero tensor-payload
bytes**. The fixed headers prove the shard tensor partition `0 + 297 + 752 +
175 = 1,224`; secondary shards also expose exact `split.no`, `split.count = 4`
and `split.tensors.count = 1224` metadata.

The local physical tensor order partitions at those exact boundaries. Subtracting
the corresponding local packed spans from each published shard size gives
format overheads of 10,946,624, 18,720, 48,064 and 11,296 bytes. Therefore:

```text
published split total                    111,334,654,784
- identical packed tensor bytes         111,323,630,080
= aggregate split overhead                  11,024,704

local merged overhead                       11,024,320
split overhead - merged overhead                    384
```

Pinned `gguf-split` merge semantics copy each tensor's exact packed byte count
and recreate only alignment padding. All local tensor packed sizes are already
32-byte multiples and observed tensor padding is zero. The 384-byte difference
is therefore **proven format/header-directory overhead only**, not tensor data.

## Physical locality observations

The PLE table is one contiguous 28.800 GB tensor; the six PLE dense tensors form
a second run. Embedding and LM head are each one run. Per-layer families are
interleaved: routed experts and shared experts occupy 144 runs each, routers 48,
GDN 48, QSA attention 12 and QSA indexer 12. Direct random access is structurally
possible from validated offsets, but this layout does not itself establish an
optimal prefetch or residency policy.

## Machine-readable evidence

The canonical evidence directory is:

`research/model-gguf/Qwen3.8-Flash-Next/c8b5954a88c2775c546b92593eda40ea041d3176/`

- `gguf-metadata.json`: identity, metadata and structural/type summary;
- `gguf-tensor-inventory.csv`: all 1,224 GGUF tensors, spans, types and reverse mappings;
- `canonical-gguf-mapping.csv`: all 1,658 canonical mapping decisions;
- `gguf-summary.json`: reconciliations, footprints, locality and provenance;
- `mapping-evidence.json`: pinned sources, hashes, claims and network-range audit.

| Evidence artifact | SHA-256 |
|---|---|
| `gguf-metadata.json` | `181ac010d7fc7f236408706801bbc2fe37a1310f115f33c1b5654fdffef84add` |
| `gguf-tensor-inventory.csv` | `10844a7cfe8f29008b43a90ffacc5056163f8edbe604eb26ab03534215894bfd` |
| `canonical-gguf-mapping.csv` | `f5cdd705e110ed0999def41292e4b761d242b5e4c3bd139323c01c597be70e5e` |
| `gguf-summary.json` | `eb58feeaf7b6c321fdf272215322c0df75d6832a23955b43ec892429437e00fa` |
| `mapping-evidence.json` | `f5b169f253d73bed055ac5714e62792238d508b4f6aa258572e6016752908a07` |

The local artifact path is intentionally absent. The GGUF remains outside the
repository and is accessed only through `KQ_GGUF_PATH`.
