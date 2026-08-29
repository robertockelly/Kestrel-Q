# Qwen3.8-Flash-Next canonical tensor inventory

Status: **PINNED / VERIFIED** for Task 1.2

Canonical model: `Qwen/Qwen3.8-Flash-Next`

Canonical revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`

This inventory describes the official Safetensors checkpoint. It does not use
GGUF names, packing or quantization to define canonical model composition.

## Evidence and safe capture

The Task 1.0 pinned index supplied the only shard allowlist. The research tool
requested bytes `0-7` and then exactly the declared JSON-header interval from
each of the 131 shard URLs at the pinned revision. Every response was HTTP 206,
had the exact expected `Content-Range`, `Content-Length` and remote total, and
returned no byte beyond the requested interval.

Capture results:

- expected and captured shards: **131 / 131**;
- audited requests: **262**;
- length-prefix bytes: **1,048**;
- JSON-header bytes: **228,712**;
- total response bytes: **229,760**;
- largest response: **39,296 bytes**;
- weight payload bytes fetched: **0**;
- files saved with a `.safetensors` suffix: **0**.

The audit treats the Safetensors payload boundary as byte
`8 + header_length`. No request reached that boundary.

## Canonical reconciliation

| Check | Result |
|---|---:|
| Index tensor names | 1,658 |
| Header tensor descriptors | 1,658 |
| Unique tensor names | 1,658 |
| Missing index names | 0 |
| Extra header names | 0 |
| Duplicate names | 0 |
| BF16 tensors | 1,655 |
| I64 metadata tensors | 3 |
| Unexplained `UNKNOWN` / `REVIEW` classifications | 0 |
| Aggregate tensor payload | 359,999,963,128 bytes |
| Index `metadata.total_size` | 359,999,963,128 bytes |
| Reconciliation | **PASS** |

For every tensor, the analyzer independently checks shape-product parameter
count, fixed-width dtype bytes and the recorded data interval. Within every
shard, intervals begin at zero, are contiguous and non-overlapping. For all 131
shards:

```text
8 + JSON header length + final data offset = pinned remote shard size
```

The aggregate remote shard size is 360,000,192,888 bytes; the 229,760-byte
difference from payload is exactly the aggregate eight-byte prefixes plus JSON
headers.

## Scope and classification

Every row has one deterministic `KQ-TENSOR-CLASSIFICATION-v1` rule and exactly
one ADR-0005 scope label:

| Scope | Tensors | Parameters/elements | Stored bytes |
|---|---:|---:|---:|
| `REQUIRED_INITIAL_TEXT` | 1,294 | 176,943,899,555 | 353,887,799,320 |
| `EXCLUDED_INITIAL_VISION` | 333 | 448,931,056 | 897,862,112 |
| `EXCLUDED_INITIAL_MTP` | 31 | 2,607,150,848 | 5,214,301,696 |
| **Full checkpoint** | **1,658** | **179,999,981,459** | **359,999,963,128** |

The text scope includes PLE because ordinary text logits depend on its layer-2
injection. Vision and MTP remain present in the canonical inventory but are
excluded only from the initial execution scope accepted by ADR 0005.

Canonical expert tensors are stored with all 512 experts stacked in their first
dimension. Their inventory rows therefore use `expert_id = STACKED_0_511`;
per-expert results are derived exactly from that dimension rather than inventing
512 canonical tensor names.

## Exact component inventory

| Component | Tensors | Parameters/elements | Stored bytes |
|---|---:|---:|---:|
| Token embedding | 1 | 635,699,200 | 1,271,398,400 |
| LM head (untied) | 1 | 635,699,200 | 1,271,398,400 |
| GDN | 324 | 2,086,510,464 | 4,173,020,928 |
| QSA attention | 72 | 597,694,464 | 1,195,388,928 |
| QSA indexer | 36 | 19,663,872 | 39,327,744 |
| Layer Gated Residual | 384 | 634,060,800 | 1,268,121,600 |
| Final Gated Residual | 3 | 6,563,840 | 13,127,680 |
| Routers | 48 | 62,914,560 | 125,829,120 |
| Routed experts | 96 | 120,795,955,200 | 241,591,910,400 |
| Shared experts and shared gates | 192 | 236,052,480 | 472,104,960 |
| PLE embedding tables | 128 | 51,200,245,760 | 102,400,491,520 |
| PLE dense weights | 6 | 32,839,680 | 65,679,360 |
| PLE I64 address metadata | 3 | 35 | 280 |
| Vision blocks | 324 | 411,466,608 | 822,933,216 |
| Vision patch embedding | 2 | 1,770,624 | 3,541,248 |
| Vision position embedding | 1 | 2,654,208 | 5,308,416 |
| Multimodal merger | 6 | 33,039,616 | 66,079,232 |
| MTP input fusion | 4 | 13,120,000 | 26,240,000 |
| MTP Gated Residual | 11 | 19,773,440 | 39,546,880 |
| MTP QSA attention/indexer | 9 | 51,446,528 | 102,893,056 |
| MTP router | 1 | 1,310,720 | 2,621,440 |
| MTP routed experts | 2 | 2,516,582,400 | 5,033,164,800 |
| MTP shared expert and gate | 4 | 4,917,760 | 9,835,520 |

The only non-BF16 stored tensors are the three PLE I64 address tensors. Token
embedding and LM head are distinct stored tensors, consistent with untied word
embeddings.

## PLE / N-gram structure

The canonical PLE has 128 BF16 table tensors. Every table has shape
`[2,500,012, 160]`, or 400,001,920 parameters and 800,003,840 stored bytes.
Together the tables occupy 102,400,491,520 bytes. PLE also contains:

- 32,839,680 BF16 dense parameters in projection, normalization and convolution
  tensors;
- I64 `layer_multipliers` with shape `[3]`;
- I64 `ngram_heads_offsets` with shape `[16]`;
- I64 `ngram_heads_vocab_sizes` with shape `[16]`.

Total PLE is 51,233,085,475 elements and 102,466,171,160 bytes: 28.462828% of
the complete checkpoint and 28.954423% of initial text-only static bytes. The
128 physical table partitions and 16 logical bigram/trigram heads are distinct
concepts. Task 1.1 proves that their addresses are known before layer 2; this
inventory only labels them `PLE_PREFETCH_CANDIDATE` and does not choose a cache
or scheduling policy.

## Per-layer and routed-expert inventory

Layer IDs are zero-based canonical checkpoint indices. `Non-expert` includes
attention/GDN, Gated Residual and PLE where present; router, routed and shared
expert columns are separate.

| ID | Type | Non-expert bytes | Router bytes | Routed bytes | Shared bytes | Total bytes | Parameters/elements |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 1 | GDN + PLE | 102,608,507,608 | 2,621,440 | 5,033,164,800 | 9,835,520 | 107,654,129,368 | 53,827,064,579 |
| 2 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 3 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 4 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 5 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 6 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 7 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 8 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 9 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 10 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 11 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 12 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 13 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 14 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 15 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 16 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 17 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 18 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 19 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 20 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 21 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 22 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 23 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 24 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 25 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 26 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 27 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 28 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 29 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 30 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 31 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 32 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 33 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 34 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 35 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 36 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 37 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 38 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 39 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 40 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 41 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 42 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 43 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |
| 44 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 45 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 46 | GDN | 142,336,448 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,187,958,208 | 2,593,979,104 |
| 47 | QSA | 129,312,256 | 2,621,440 | 5,033,164,800 | 9,835,520 | 5,174,934,016 | 2,587,467,008 |

Each layer stores 2,516,582,400 routed-expert parameters, or 4,915,200
parameters and 9,830,400 BF16 bytes per expert. This is uniform across all 48
layers. Selecting ten experts names 49,152,000 parameters / 98,304,000 BF16
bytes per layer. Across 48 layers the selected routed-expert parameter set is
2,359,296,000 parameters / 4,718,592,000 BF16 bytes.

These are **selected parameter payloads**, not measured reads, transfers,
residency, cache misses or throughput. They do not assert that all selected
weights move for every token.

## Machine-readable evidence

Canonical outputs are under:

```text
research/model-tensors/Qwen3.8-Flash-Next/
  de4b8e4d43b917e7706784d8bb445c9af86a3540/
    shard-header-manifest.json
    tensor-inventory.csv
    tensor-summary.json
    network-range-audit.json
```

The CSV is the row-level canonical inventory. `tensor-summary.json` carries all
131 shard reconciliation records, all 48 layer summaries, routed-expert
derivations, idealized quantization floors and runtime-state arithmetic.

| Artifact | SHA-256 |
|---|---|
| `shard-header-manifest.json` | `193BE76B3A3C30F623D3CEDAA8C64E302F6F5D7A672B266B23A888D09615C1FD` |
| `network-range-audit.json` | `33CD7C88C2E80DD153BEDE4494C6C8FE08457226C25DF1923A45211CEB242DCD` |
| `tensor-inventory.csv` | `46E0B550A8A37A48180D96C151B61354FE6251AE91734DF64DAFE5975202177A` |
| `tensor-summary.json` | `B73DE4029BD56260EF36A9F4DBA944C5DBBFEA10178D510BA253A30F047851CA` |

An independent second network capture and offline regeneration under ignored
`.research-cache/` reproduced all four hashes exactly.
