# Qwen3.8-Flash-Next GGUF quantization footprint

Status: **TASK 1.3 COMPLETE / PASS**

These are observed packed bytes in `KQ-MODEL-ARTIFACT-001`, not estimates of a
generic Q4 model and not a claim that the artifact is runnable by Kestrel-Q yet.
File overhead is kept separate from tensor storage.

## Container and type accounting

- total file: **111,334,654,400 bytes** (103.688 GiB);
- packed tensors: **111,323,630,080 bytes** (103.678 GiB);
- header/directory/alignment overhead: **11,024,320 bytes** (0.009902%);
- GGUF tensor parameters: **176,943,899,520**;
- weighted packed footprint: **5.033172 bits/parameter**.

| GGML type | Block geometry | Tensors | Parameters | Packed bytes | Effective bpp |
|---|---:|---:|---:|---:|---:|
| `BF16` | 1 / 2 bytes | 24 | 19,660,800 | 39,321,600 | 16.0 |
| `F32` | 1 / 4 bytes | 557 | 78,373,760 | 313,495,040 | 32.0 |
| `IQ4_NL` | 32 / 18 bytes | 1 | 51,200,245,760 | 28,800,138,240 | 4.5 |
| `Q4_K` | 256 / 144 bytes | 94 | 78,852,915,200 | 44,354,764,800 | 4.5 |
| `Q5_1` | 32 / 24 bytes | 43 | 36,071,014,400 | 27,053,260,800 | 6.0 |
| `Q5_K` | 256 / 176 bytes | 2 | 1,677,721,600 | 1,153,433,600 | 5.5 |
| `Q8_0` | 32 / 34 bytes | 503 | 9,043,968,000 | 9,609,216,000 | 8.5 |

Block geometry is pinned to the MIT-licensed llama.cpp revision recorded in
`mapping-evidence.json` and is validated against every observed tensor span.

## Actual packed footprint by canonical family

The idealized Q4 column is the Task-1.2 parameter-count floor. It excludes block
metadata, mixed precision and exceptions; it is neither a GGUF size nor a
runnable-runtime size.

| Family | Canonical parameters | Canonical stored bytes | Ideal Q4 floor | Actual packed bytes | Effective bpp | Packed share |
|---|---:|---:|---:|---:|---:|---:|
| PLE/N-gram | 51,233,085,475 | 102,466,171,160 | 25,616,542,738 | 28,835,240,960 | 4.502597 | 25.902% |
| Routed experts | 120,795,955,200 | 241,591,910,400 | 60,397,977,600 | 77,017,907,200 | 5.100694 | 69.184% |
| Shared experts/gates | 236,052,480 | 472,104,960 | 118,026,240 | 251,166,720 | 8.512233 | 0.226% |
| Dense/non-routed excluding shared | 4,678,806,400 | 9,357,612,800 | 2,339,403,200 | 5,219,315,200 | 8.924182 | 4.688% |
| Dense + shared/non-routed | 4,914,858,880 | 9,829,717,760 | 2,457,429,440 | 5,470,481,920 | 8.904397 | 4.914% |
| GDN | 2,086,510,464 | 4,173,020,928 | 1,043,255,232 | 2,247,261,696 | 8.616345 | 2.019% |
| QSA attention | 597,694,464 | 1,195,388,928 | 298,847,232 | 635,068,416 | 8.500242 | 0.570% |
| QSA indexer | 19,663,872 | 39,327,744 | 9,831,936 | 39,333,888 | 16.002500 | 0.035% |
| Gated residual / hyper-connections | 640,624,640 | 1,281,249,280 | 320,312,320 | 695,132,160 | 8.680680 | 0.624% |
| Routers | 62,914,560 | 125,829,120 | 31,457,280 | 251,658,240 | 32.0 | 0.226% |
| Token embedding | 635,699,200 | 1,271,398,400 | 317,849,600 | 675,430,400 | 8.5 | 0.607% |
| LM head | 635,699,200 | 1,271,398,400 | 317,849,600 | 675,430,400 | 8.5 | 0.607% |
| Vision omitted by ADR 0005 | 448,931,056 | 897,862,112 | 224,465,528 | 0 | n/a | 0% |
| MTP omitted by ADR 0005 | 2,607,150,848 | 5,214,301,696 | 1,303,575,424 | 0 | n/a | 0% |

The PLE row includes 35 exact int64 address parameters represented by metadata;
its effective bpp uses the 51,233,085,440 parameters that remain GGUF tensors.
The actual PLE table itself is one 28,800,138,240-byte `IQ4_NL` tensor. PLE
dense weights add 35,102,720 bytes. Vision and MTP are valid initial-scope
omissions, not quantized zero-byte implementations.

## Precision-policy observations

- PLE is mixed: the large table is `IQ4_NL`, two dense matrices are `Q8_0`
  and four small tensors are `F32`.
- Routed experts are mixed across `Q4_K`, `Q5_1`, `Q5_K` and `Q8_0`.
- Routers remain `F32`; QSA indexer tensors are `BF16` or `F32`.
- Shared experts, GDN, GR, QSA attention, embedding and head retain a mixture
  dominated by `Q8_0` plus small `F32`/`BF16` tensors.

These choices are observed precision placement. They do not prove that a
family is more or less sensitive, nor that the same placement is optimal for a
Kestrel-Q-native quantizer.

## Per-expert and selected top-10 footprint

Each routed expert contains **4,915,200 parameters per layer**. Packed bytes
vary with Unsloth's layer precision policy:

| Layer set | Per-expert packed bytes | Effective bpp | Top-10 packed bytes/layer |
|---|---:|---:|---:|
| 43 ordinary layers | 3,072,000 | 5.0 | 30,720,000 |
| layer 2 | 3,993,600 | 6.5 | 39,936,000 |
| layers 4, 30, 46, 47 | 3,584,000 | 5.833333 | 35,840,000 |

Across all 48 layers, selecting ten experts per layer names **2,359,296,000
parameters** and **1,504,256,000 packed bytes** (1.401 GiB). This is a selected
parameter payload only. It is not measured physical I/O, cache traffic,
residency, latency or throughput; experts may already be resident or cached and
the runtime may access them in a different physical pattern.

## Imatrix and provenance

The artifact records:

- `general.quantized_by = Unsloth`;
- quantization version 2 and `general.file_type = 15`;
- imatrix file `Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf`;
- calibration dataset `unsloth_calibration_Qwen3.8-Flash-Next.txt`;
- 926 imatrix entries and 45 chunks.

The scores themselves are not embedded in a recoverable per-tensor form in the
local model metadata, and the 580,038,720-byte upstream imatrix artifact was not
downloaded. The metadata proves calibration provenance, not the score-to-type
decision for any individual tensor and not quality/sensitivity claims.

## KQ-01 feasibility implications

The whole text-only GGUF is 103.688 GiB, far beyond both the planned ~8 GiB VRAM
working budget and ~20 GiB managed host model/cache budget. PLE alone is 26.855
GiB and routed experts are 71.729 GiB, so neither class can be wholly resident
in those budgets. Dense + shared/non-routed text weights are 5.095 GiB, which is
a plausible placement candidate within the nominal VRAM budget, but leaves only
about 2.9 GiB for expert cache, runtime state, activations, workspaces and CUDA
allocation headroom; this is not yet an accepted placement policy.

The KQ-01 measurements (~25.63 GB/s RAM and ~26 GB/s large pinned PCIe transfer)
show why cache hit rate and transfer overlap will matter, while the ~0.44 GB/s
SATA baseline makes cold storage traffic especially expensive. These bandwidths
do not convert the selected top-10 footprint into a tokens/s prediction. Final
placement, prefetch, eviction and scheduling policy remain future work.

### Preliminary tiering hypothesis to validate

For KQ-01, PLE is now labeled `PLE_DISK_BACKED_CANDIDATE`: its 26.855 GiB
packed footprint makes full residency in the approximately 20 GiB managed host
budget undesirable, so the candidate design uses disk-backed/mapped storage, a
bounded explicit RAM page/row cache, deterministic predictive/asynchronous
prefetch and only materialized lookup results or required working data in VRAM.
The 71.729 GiB routed-expert family is separately a candidate for an active RAM
cache, hot VRAM subset and cold disk backing. Uncontrolled Windows paging is not
the proposed mechanism.

This is an architectural hypothesis, not a performance result or final policy.
`KQ-BACKLOG-BENCH-002` must distinguish OS page-cache hits, explicit Kestrel-Q
RAM-cache hits and cold physical reads and must validate the approach before a
final scheduler/residency decision.
