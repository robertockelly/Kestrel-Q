# Native PLE address engine

Status: **TASK 2.4 COMPLETE / PASS**

Task 2.4 implements the weight-independent Qwen3.8-Flash-Next PLE address
stage in production C17. Canonical token IDs and a bounded per-stream history
produce ordered logical address intents. The engine opens no PLE tensor view,
contains no PLE value computation and performs no storage operation.

## Construction and compatibility

`kq_ple_config_open_from_model` consumes the immutable Task 2.1 semantic
registry. Construction requires the exact target topology (2,560 hidden,
248,320 vocabulary, 262,144 context, 48 layers split 36 GDN/12 QSA), PLE on
zero-based layer 1, 128 fused logical table members of 2,500,012 rows by width
160, six PLE dense semantics and all three metadata-derived address arrays.

The multipliers, head offsets and prime moduli are compared with the pinned
canonical adapter values. A right-looking model with a changed count, shape,
layer type, member relation or metadata array fails with
`KQ_STATUS_INCOMPATIBLE_PLE`. Configuration copies the small address arrays;
it never derives physical file offsets and never asks Task 2.2 for a view.

## Address and state behavior

Every token emits 16 immutable-by-contract values in this order:

1. bigram local heads 0–7 (global heads 0–7);
2. trigram local heads 0–7 (global heads 8–15).

Each `kq_ple_address_intent` records the stream position, current token,
order/head identity, global concatenated row, head offset/modulus and the
canonical logical member plus member-local row. The member/row pair is the
future storage-facing semantic address. It is not a GGUF byte offset.

The public stream state is exactly 32 bytes: absolute position, two raw prior
token IDs, a version and an integrity value bound to the configuration. Reset
installs two EOS sentinels. The state contains only address-generation history;
the future PLE value path's nine-position dilation-convolution state,
projections and layer residuals are not present.

Prefill is transactional. Required intent count and context bounds are checked
before token access, all token IDs are validated before output/state mutation,
and insufficient output capacity leaves state unchanged. One-token decode uses
the same implementation and is exactly equal to full recomputation.

## Oracle and differential evidence

The unchanged Task 1.4 oracle
`canonical/ple-address-vectors.json` retains SHA-256
`495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`.
Native comparison passes all seven sequence vectors and four decode steps.
Offline regeneration from the pinned Transformers source and official config
is byte-identical; native Kestrel-Q output is never used to create expected
values.

The expanded pinned-oracle evidence is:

`research/ple/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json`

SHA-256:
`b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`.

It adds 12 sequence cases, three prefill/decode streams and one native
tokenizer-to-PLE case. Coverage includes empty-history transitions, repeated
and alternating tokens, low/high IDs, history lengths 1/2/3, lengths 9/10 at
the future value-state dilation boundary, multiple EOS segments, a 64-token
prefill followed by decode, and reset/replay equivalence.

## Validation and characterization

Synthetic tests cover exact addresses and ordering, full/incremental equality,
EOS segmentation, capacity transactionality, state corruption, count/context
overflow and incompatible descriptor mutations. Real integration constructs
the model, tokenizer and PLE config from `KQ_GGUF_PATH`, verifies independent
addresses and the tokenization of `Hello, Kestrel-Q.`, and reports:

```text
PLE_payload_views_opened = 0
model_tensor_payload_bytes_touched = 0
```

The final clean KQ-01 CPU Release characterization run measured 1,400 ns config
construction, 376 config-owned bytes, 32 bytes per stream, 16 addresses/token,
600 ns for a three-token prefill and 100 ns for one decode step. These timer
samples characterize the implementation and are not performance guarantees or
storage-throughput measurements.

## Deliberate exclusions

Task 2.4 does not implement PLE lookup/value math, tensor math, inference,
dequantization, disk or mmap access, async I/O, cache, prefetch, eviction,
batching, scheduling, threads or CUDA kernels. `PLE_DISK_BACKED_CANDIDATE`
remains an annotation and `KQ-BACKLOG-BENCH-002` remains deferred and mandatory
before final storage/residency policy.
