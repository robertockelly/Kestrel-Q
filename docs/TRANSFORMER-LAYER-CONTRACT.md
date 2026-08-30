# Qwen3.8-Flash-Next transformer-layer contract

Status: **TASK 2.10 CHARACTERIZED / PINNED**

This contract is derived from the pinned canonical model revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and Apache-2.0 Transformers
revision `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. The executable source is
`Qwen4ExpTextDecoderLayer.forward` and `Qwen4ExpTextGatedResidual.forward` in
`modeling_qwen4_exp.py`, SHA-256
`91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f`.
Kestrel-Q output is not an oracle.

## Families and placement

There are 48 zero-based text layers. Layers `3,7,...,47` are QSA; the other
36 are GDN. Zero-based layer 1 is the sole PLE-enabled layer and is a GDN
layer. Therefore complete-layer families are 35 ordinary GDN, 12 QSA and one
PLE-GDN. Every layer owns a MoE and two independent Gated Residual (GR)
modules, one around the mixer and one around the MoE.

PLE is not a generic pre-layer callback. At layer 1 only, its four-branch
output is added elementwise to the four input branches immediately before the
attention/GDN GR read:

```text
H = H + PLE(H, token_ids, PLE state)
```

## Gated Residual

For target branch count `C=4`, branch width `D=2560`, flattened width
`W=10240`, rank `R=320`, and epsilon `1e-6`, each GR owns:

| Parameter | Shape |
|---|---:|
| zero-centered Group-RMSNorm delta | `[W]` |
| read down projection | `[R,W]` |
| read up projection | `[W,R]` |
| block-injection projection | `[C,W]` |

Group-RMSNorm normalizes each 2,560-wide branch independently in FP32 and
multiplies by `1 + delta`. With normalized flattened branches `N`:

```text
low       = silu(linear(N, down) / C)
read_gate = sigmoid(linear(low, up)).reshape(C,D)
block_in  = mean_branch(read_gate * N.reshape(C,D))
write     = 2 * sigmoid(linear(N, inject) / C)       # C scalars
H_out[b]  = H_in[b] + block_output * write[b]
```

The read gate is elementwise; the write gate is one scalar per branch. There
is no branch-mixing `H_res` matrix. Division occurs before SiLU/sigmoid exactly
as shown. Biases are absent. GR values and branches are forward activations,
not persistent context state.

## Exact layer order

The canonical layer executes:

1. optional PLE value computation and branchwise addition at layer 1;
2. attention GR read over the current four branches;
3. GDN or QSA on the resulting one-hidden-vector stream;
4. attention GR write back to each original branch;
5. MoE GR read over those updated branches;
6. sparse MoE, including routed and separate gated shared paths;
7. MoE GR write back, producing the four-branch layer output.

No standalone input/output RMSNorm exists outside the two GR group norms.
GDN receives the convolution padding mask. QSA receives canonical positions
and causal selection semantics through its own reference API. PLE consumes
token IDs and its address/value states. The initial API is batch 1.

## State and transactions

Persistent state is suboperator-only:

| Family | Persistent semantic state |
|---|---:|
| ordinary GDN | recurrent FP32 `786432` + conv BF16 `40960` = `3,227,648` bytes |
| QSA | K/V/raw-index = `2,304` bytes per retained token |
| PLE-GDN | GDN + 32-byte address state + 184,320-byte BF16 value history = `3,412,000` bytes |

MoE and both GRs are stateless. The native reference keeps an inactive staging
copy of every persistent substate. A call copies active to staging, executes
all components there, and swaps slots only after the final MoE GR write.
Failure therefore leaves active GDN/QSA/PLE state and layer position unchanged.

Prefill applies the same causal transition in sequence order. One-token decode
continues from the committed suboperator states. Split prefill/decode and
reset/replay must match one-shot execution under the layer-specific calibrated
floating contract.

## Dtype and scope

Canonical low-level operator dtype transitions remain those pinned by Tasks
2.6–2.9. The Task 2.10 executable oracle/native reference uses scalar F32,
MSVC `/fp:strict`, no fast math and no contraction assumption. Complete-layer
composition boundaries use per-family `CALIBRATED_FLOAT`; family IDs, PLE
placement, state counts and layer classification are `EXACT_DISCRETE`.

This contract does not include embeddings, a 48-layer executor, final model
normalization, LM head, logits, sampling, scheduler/cache policy, SIMD or CUDA
model kernels.
