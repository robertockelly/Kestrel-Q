# Qwen3.8-Flash-Next footprint baseline

Status: **COMPLETE / PASS** for Task 1.2

This document derives static stored-weight and batch-1 persistent-state sizes
from the pinned canonical Safetensors headers and Task 1.1 runtime-state model.
It does not describe the registered GGUF, implement a quantization, select a
scheduler or predict token throughput.

## Static canonical footprint

| Scope/family | Parameters/elements | Stored bytes | Stored GiB |
|---|---:|---:|---:|
| Complete canonical checkpoint | 179,999,981,459 | 359,999,963,128 | 335.276093 |
| ADR-0005 initial text-only scope | 176,943,899,555 | 353,887,799,320 | 329.583696 |
| Excluded vision | 448,931,056 | 897,862,112 | 0.836199 |
| Excluded MTP | 2,607,150,848 | 5,214,301,696 | 4.856197 |
| PLE/N-gram | 51,233,085,475 | 102,466,171,160 | 95.429058 |
| Routed experts | 120,795,955,200 | 241,591,910,400 | 225.000000 |
| Shared experts and gates | 236,052,480 | 472,104,960 | 0.439682 |
| Non-PLE/non-routed text, including shared | 4,914,858,880 | 9,829,717,760 | 9.154638 |
| Non-PLE/non-routed text, excluding shared | 4,678,806,400 | 9,357,612,800 | 8.714956 |

The complete count includes 35 I64 PLE address elements; the remaining
179,999,981,424 elements are BF16. Consequently a generic `parameters × 2`
shortcut is 210 bytes smaller than actual canonical payload. The exact stored
bytes above always come from tensor offsets, not that shortcut.

The initial text-only scope excludes vision and MTP but must retain PLE. It is
not a small subset of the released checkpoint: it is 98.302177% of stored
checkpoint bytes. Routed experts are 68.267940% and PLE is 28.954423% of the
initial text-only static footprint.

## Idealized quantization payload floors

The following values apply a uniform `parameter_count × bits / 8` lower bound.
They are theoretical payload floors only, not implemented quantization sizes.
They exclude scales, zero points, block metadata, padding, alignment, format
headers and mixed-precision exceptions. They are not GGUF sizes, do not describe
the registered Unsloth artifact and do not establish runnable support.

| Family | Canonical stored GiB | Q8 GiB | Q6 GiB | Q5 GiB | Q4 GiB | Q3 GiB |
|---|---:|---:|---:|---:|---:|---:|
| Full checkpoint | 335.276093 | 167.638046 | 125.728535 | 104.773779 | 83.819023 | 62.864267 |
| Initial text-only | 329.583696 | 164.791848 | 123.593886 | 102.994905 | 82.395924 | 61.796943 |
| PLE/N-gram | 95.429058 | 47.714529 | 35.785897 | 29.821581 | 23.857265 | 17.892948 |
| Routed experts | 225.000000 | 112.500000 | 84.375000 | 70.312500 | 56.250000 | 42.187500 |
| Shared experts/gates | 0.439682 | 0.219841 | 0.164881 | 0.137401 | 0.109921 | 0.082440 |
| Dense/shared non-routed text | 9.154638 | 4.577319 | 3.432989 | 2.860824 | 2.288660 | 1.716495 |
| Dense non-routed, excluding shared | 8.714956 | 4.357478 | 3.268109 | 2.723424 | 2.178739 | 1.634054 |

Exact byte ceilings are retained in `tensor-summary.json`; GiB here is only a
human-readable rendering. For example, initial-text lower bounds are exactly:

| Floor | Bytes |
|---|---:|
| Q8 | 176,943,899,555 |
| Q6 | 132,707,924,667 |
| Q5 | 110,589,937,222 |
| Q4 | 88,471,949,778 |
| Q3 | 66,353,962,334 |

## Batch-1 persistent runtime state

Source-backed precision assumptions are FP32 for GDN recurrent matrices and
BF16 for GDN convolution, QSA K/V/raw-index keys and PLE convolution state.
Token history and position bookkeeping are I64.

Fixed state, independent of context length:

| Component | Bytes |
|---|---:|
| 36 GDN recurrent matrices `[48,128,128]` FP32 | 113,246,208 |
| 36 GDN convolution tails `[10240,4]` BF16 | 2,949,120 |
| PLE dilated-convolution tail `[10240,9]` BF16 | 184,320 |
| PLE two-token history I64 | 16 |
| Scalar position bookkeeping I64 | 8 |
| **Fixed total** | **116,379,672** |

Each context token adds, across the 12 QSA layers:

- K cache: 12,288 bytes;
- V cache: 12,288 bytes;
- raw index keys: 3,072 bytes;
- semantic QSA total: 27,648 bytes.

The pinned reference additionally caches three equivalent text position-ID
axes, 24 I64 bytes per token. A Kestrel-Q-native representation may reconstruct
these from scalar position/cache length only after proving numerical
equivalence. Both views are therefore reported:

| Context | Semantic persistent bytes | Semantic GiB | Pinned-reference-container bytes | Reference GiB |
|---:|---:|---:|---:|---:|
| 1 | 116,407,320 | 0.108413 | 116,407,344 | 0.108413 |
| 4,096 | 229,625,880 | 0.213856 | 229,724,184 | 0.213947 |
| 16,384 | 569,364,504 | 0.530262 | 569,757,720 | 0.530628 |
| 65,536 | 1,928,319,000 | 1.795887 | 1,929,891,864 | 1.797352 |
| 262,144 | 7,364,136,984 | 6.858387 | 7,370,428,440 | 6.864246 |

No second pooled QSA block-key cache is canonical: the pinned implementation
derives it. GR four-branch activations, QSA selection buffers and MoE routing
data are current-forward temporaries, not context-persistent cache. Optional
vision and MTP state is outside the initial scope and is not hidden in these
totals.

## KQ-01 feasibility implications

Reference planning constraints are approximately 8 GiB of VRAM working budget,
20 GiB of managed host model/cache budget, 25.63 GB/s measured RAM bandwidth,
about 26 GB/s measured large pinned PCIe H2D/D2H and 0.44 GB/s measured SATA
sequential read.

1. Neither full nor text-only weights fit the combined planning budgets at any
   idealized Q8–Q3 floor. Initial-text Q3 alone is 61.796943 GiB before all real
   quantization overhead and exceptions.
2. PLE alone is 95.429058 GiB canonical BF16. Its idealized Q3 floor is
   17.892948 GiB, nominally below the 20 GiB host budget but leaving insufficient
   room for the rest of the model/cache and omitting real overhead. PLE's early,
   deterministic addresses justify a prefetch-candidate label, not a residency
   decision.
3. All routed experts occupy 225 GiB BF16 and still 42.1875 GiB at the
   idealized Q3 floor. A bounded expert cache/backing design is therefore an
   architectural requirement for KQ-01, but its capacity and eviction are not
   selected here.
4. Dense/shared non-routed text is 9.154638 GiB BF16; its theoretical Q8 floor
   is 4.577319 GiB. This identifies an always-needed placement candidate, not a
   proven VRAM layout or supported quantization.
5. The exact BF16 parameter set selected by top-10 routing across all 48 layers
   is 4.394531 GiB. It is a selection footprint only; it is neither simultaneous
   residency nor measured transfer volume.
6. At maximum configured context, persistent text state is about 6.86 GiB even
   before forward-call temporaries. An 8 GiB VRAM working budget cannot be
   assumed to hold maximum-context state plus kernels, activations and hot
   weights.
7. SATA is orders slower than RAM/PCIe in the measured baseline. Reactive
   per-token reads of multi-gigabyte expert or PLE sets are not plausible, but
   Task 1.2 makes no token-rate claim and selects no prefetch depth.

Preliminary labels in the inventory (`ALWAYS_NEEDED_CANDIDATE`,
`ROUTED_EXPERT_CACHE_CANDIDATE`, `PLE_PREFETCH_CANDIDATE` and
`EXCLUDED_INITIAL_SCOPE`) are analysis aids only. Final placement, cache,
eviction and synchronization policy remains later work.
