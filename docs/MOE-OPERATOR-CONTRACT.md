# MoE operator contract

Status: **TASK 2.8 CHARACTERIZATION COMPLETE — PINNED CONTRACT**

This document fixes the Qwen3.8-Flash-Next mixture-of-experts contract before
native implementation. Generic MoE descriptions, older Qwen releases and GGUF
storage conventions are not semantic authorities for this operator.

## Provenance

Canonical model artifacts are pinned to
`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Executable Class-C behavior is pinned to the Apache-2.0 Transformers revision
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`.

| Pinned source | SHA-256 |
|---|---|
| `modeling_qwen4_exp.py` | `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f` |
| `configuration_qwen4_exp.py` | `26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254` |
| `integrations/moe.py` | `2c8894f6d1392980a61ff265f90f7a99fa90678e0eec461d958b4d32fda9628c` |
| `activations.py` | `5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665` |
| official pinned `config.json` | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |

The canonical modules/functions are `Qwen4ExpTextSparseMoeBlock`,
`Qwen4ExpTextTopKRouter`, `Qwen4ExpTextExperts`, `Qwen4ExpTextMLP`,
`torch.nn.functional.linear`, `torch.nn.functional.softmax`, `torch.topk`,
the `silu` entry in `ACT2FN`, and `torch.sigmoid`. The training-only auxiliary
path is `load_balancing_loss_func`; it is outside inference execution.

## Target topology and bindings

Every one of the 48 text layers contains one MoE block. The target geometry is:

| Property | Target value |
|---|---:|
| input/output hidden width | 2,560 |
| routed experts | 512 |
| selected routed experts/token | 10 |
| routed intermediate width | 640 |
| shared experts/layer | 1 |
| shared intermediate width | 640 |
| activation | SiLU |
| router/shared/routed biases | none |
| normalization of selected router weights | enabled |
| persistent token-to-token MoE state | none |

Each layer requires seven semantic entries:

| Stable suffix | Canonical role/shape |
|---|---|
| `moe.router` | router `[512,2560]` |
| `moe.routed.gate_up` | canonical fused stack `[512,1280,2560]`, physical ordered gate then up |
| `moe.routed.down` | stack `[512,2560,640]` |
| `moe.shared.gate` | shared gate projection `[640,2560]` |
| `moe.shared.up` | shared up projection `[640,2560]` |
| `moe.shared.down` | shared down projection `[2560,640]` |
| `moe.shared.gate_weight` | shared-output gate `[1,2560]` |

The routed expert axis is canonical axis zero and physical GGUF axis two. The
two physical `gate_up` bindings are ordered `GATE` then `UP`; they are not one
concatenated payload view. A right-looking name with the wrong shape, relation,
part order, expert axis, type or block geometry is incompatible.

## Router and exact selection

For each input token `x`:

```text
router_logits = linear(x, router_weight)
router_probabilities = softmax(router_logits, dtype=FP32)
(selected_probabilities, selected_ids) = topk(router_probabilities, 10)
selected_weights = selected_probabilities / sum(selected_probabilities)
selected_weights = cast(selected_weights, router_logits.dtype)
```

The target checkpoint uses BF16 activation semantics; the scalar executable
reference uses F32 throughout. Router logits inherit the activation/weight
dtype, while softmax is explicitly FP32. The selected values are normalized
only after top-k and then cast back to the router-logit dtype.

Selected IDs, membership, order and count are `EXACT_DISCRETE`. The pinned
CPU oracle (`torch 2.11.0+cpu`, deterministic algorithms, one thread) exhibits
the following deterministic partial-selection behavior: the first `k`
candidates seed the set; a later candidate replaces the lowest selected value
only when strictly greater; among an exactly tied lowest set, the lowest
selected ID is evicted; an equal later candidate does not replace; and the
final result is ordered by descending probability then ascending ID. Thus a
512-way tie returns IDs 0–9, while one dominant late ID 511 plus 511 tied
non-dominant values returns `511,1..9`. Kestrel-Q pins these observed results
rather than claiming that upstream `torch.topk` promises a portable stable
tie rule. Near ties are resolved by their actual FP32 values. Floating
closeness never excuses a different route.

## Routed experts

For selected expert `e`:

```text
gate_e = linear(x, routed_gate[e])
up_e = linear(x, routed_up[e])
activated_e = silu(gate_e) * up_e
expert_output_e = linear(activated_e, routed_down[e])
weighted_e = expert_output_e * selected_weight[e]
```

The eager canonical implementation visits hit experts by increasing expert ID
and accumulates their weighted outputs in that order. This accumulation order
is part of the scalar reference contract. The top-k order remains separately
observable and is not rewritten to match execution order. Only selected
experts execute; there is no all-expert materialization.

## Shared path and final combination

The shared expert consumes the same MoE input and is not part of top-k:

```text
shared_gate = linear(x, shared_gate_projection)
shared_up = linear(x, shared_up_projection)
shared_activated = silu(shared_gate) * shared_up
shared_output = linear(shared_activated, shared_down_projection)
shared_scale_logit = linear(x, shared_expert_gate_weight)
shared_scale = sigmoid(shared_scale_logit)
gated_shared = shared_scale * shared_output
output = routed_weighted_sum + gated_shared
```

The shared scale is one scalar per token and is applied after the shared down
projection. The final output has the input activation dtype. MoE owns no
surrounding Gated Residual read/write, residual addition or branch mixing.

## Dtype, accumulation and failure policy

The released execution path uses the activation/checkpoint dtype for linear
outputs, expert products, routed accumulation and final output; router softmax
alone is explicitly FP32. The Task 2.8 executable oracle is F32 and preserves
source order under MSVC `/fp:strict` with no contraction, reassociation, SIMD
or fast math. Matrix rows use left-to-right F32 dot products. Routed experts
accumulate by ascending expert ID. Shared and routed results are added once at
the end.

All input, weights and intermediate results must remain finite. Public calls
validate complete dimensions, capacities and non-aliasing before execution.
MoE has no mutable stream state. Output is unspecified on arithmetic failure,
but invalid arguments, dimensions, capacity and aliasing fail before execution.

## Oracle and comparison classes

Task 2.8 uses two independent configurations from the pinned module:

- Tier A: reduced dimensions/expert count with unchanged router, expert,
  shared and combination equations; it supplies calibration and disjoint
  holdout vectors.
- Tier B: 512 experts and top-k 10 with bounded hidden width; it exercises
  equal logits, kth-boundary ties, near ties, IDs 0/511, one and exactly ten
  dominant experts, more than ten plausible experts and repeated inputs.

Routing fields use `EXACT_DISCRETE`. Floating checkpoint classes are calibrated
independently for router logits/probabilities, selected weights, routed expert
outputs, routed accumulation, shared stages and final output. No generic Task
2.5/GDN/QSA tolerance is inherited. Expected values are generated before and
independently from Kestrel-Q.

## Training and scope boundaries

`output_router_logits`, the load-balancing auxiliary loss and coefficient
`0.001` are diagnostics/training behavior. They do not alter ordinary MoE
inference output and are not implemented by the native operator.

This contract does not authorize PLE value lookup, Gated Residual composition,
a complete transformer layer, multi-layer/full-model execution, LM head,
sampling, expert cache/prefetch/residency, a scheduler, SIMD or CUDA MoE.
`KQ-BACKLOG-BENCH-002` remains deferred.
