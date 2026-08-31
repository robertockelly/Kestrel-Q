# Qwen3.8-Flash-Next model-execution contract

Status: **TASK 2.12 CHARACTERIZED / PINNED**

This contract is derived from canonical model revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` and Apache-2.0 Transformers
revision `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. The executable references are
`Qwen4ExpTextModel.forward`, `Qwen4ExpTextGatedResidual.forward` and
`Qwen4ExpForCausalLM.forward` in `modeling_qwen4_exp.py` (repository-pinned
SHA-256 `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f`).
Kestrel-Q output is not an oracle.

## Entry

The text model owns a distinct embedding
`model.language_model.embed_tokens.weight`, registered as
`text.token_embedding`. Its canonical shape is `[248320,2560]`; the physical
GGUF tensor `token_embd.weight` has GGUF dimensions `[2560,248320]` and Q8_0
rows. A canonical token ID selects exactly one 2,560-element row. There is no
embedding scale. Canonical storage is BF16; the Task 2.12 scalar target path
decodes the quantized row to F32.

Input IDs for the initial text path are canonical tokenizer IDs `0..248076`.
The 243 capacity-padding IDs `248077..248319` are not accepted as prompt input.
The tokenizer adds neither BOS nor EOS. The selected M1 raw prompt is the
shortest governed raw-text vector, `KQ-PROMPT-001` (`Hello, Kestrel-Q.`, UTF-8
SHA-256 `6a013508a5747146c4c37c1cdc104d49d2a75dbaa2950b23e2d72eb8ac2b4787`):

```text
9419, 11, 710, 467, 3621, 27325, 13
```

The embedding stream is repeated four times to form contiguous canonical
`[token,4,2560]` hyper-connection branches. Position IDs start at zero after
reset and advance by one per prompt token.

## Decoder stack and state

Exactly 48 zero-based layers execute in ascending order. The already-pinned
family schedule is GDN/GDN/GDN/QSA repeated twelve times, with layer 1 using
the PLE-GDN family. Each layer owns its existing Task 2.10 persistent state:
36 GDN-family fixed states, 12 explicitly capacity-bounded QSA states, and the
single PLE address/value state inside layer 1. Zero/reset semantics are those
of the accepted layer APIs.

M1 uses context capacity 8: seven prompt tokens plus one possible continuation
position. Each layer retains its Task 2.10 committed/staging slots. A failed
model call leaves the public model position at zero, marks the private layer
set for reset and writes neither logits, decode bytes nor result. The next call
resets all 48 layer states before execution. Early-, middle- and late-layer
faults followed by complete token-271 control runs verify that externally
observable behavior is transactional.

## Final hyper-connection mix

There is no separate legacy Qwen final RMSNorm. After layer 47,
`Qwen4ExpTextModel.forward` applies
`Qwen4ExpTextGatedResidual(use_combine=False)`, registered as:

| Semantic | Canonical shape | Physical type |
|---|---:|---|
| `text.final_gr.norm` | `[10240]` | F32 |
| `text.final_gr.down` | `[320,10240]` | Q8_0 |
| `text.final_gr.up` | `[10240,320]` | Q8_0 |

For each token, with four branches `H`, branch width `D=2560`, flattened width
`W=10240`, low rank `R=320`, and epsilon `1e-6`:

```text
N[b] = rms_norm_f32(H[b], delta_norm[b], epsilon)
L    = silu(linear(N.flatten, down) / 4)
G    = sigmoid(linear(L, up)).reshape(4,2560)
Y    = mean_b(G[b] * N[b])
```

The norm uses the model's zero-centered delta convention, multiplying by
`1 + delta`. The mean divides the four-branch sum by exactly `4`. There is no
final injection projection. Intermediate scalar execution is F32 under the
Task 2.5 `/fp:strict` contract; the canonical BF16 module casts output back to
its input dtype.

## LM head and logits

`lm_head.weight` is distinct because `tie_word_embeddings=false`. It is
registered as `text.lm_head`, canonical shape `[248320,2560]`, physical GGUF
tensor `output.weight` with dimensions `[2560,248320]` and Q8_0 rows. The head
has no bias. One row-dot per vocabulary ID produces all 248,320 F32 scalar-path
logits; no full F32 head is materialized. For next-token generation only the
last prompt position's hidden vector is passed to the head, matching
`logits_to_keep=1` semantics.

## Greedy selection and decode

M1 explicitly overrides the repository generation defaults with
`do_sample=false`. Transformers selects `torch.argmax(next_token_scores,
dim=-1)`; PyTorch returns the first index of a maximum. Native selection
therefore scans IDs in ascending order and replaces the winner only on a
strictly greater finite logit. The lower ID wins exact ties. All 248,320 logits
participate; no pre-argmax truncation is allowed.

The selected ID must be a decodable canonical tokenizer ID. Native decode uses
`KQ_TOKENIZER_DECODE_KEEP_SPECIAL`, so a selected special token remains
observable. Generation EOG identity for the pinned model records IDs 248044
and 248046; canonical tokenizer EOS is 248046. Task 2.12 emits exactly one
selected token and records the EOG flag; it does not append or execute it.

## Numerical and scope boundary

Entry row decode, final mix and LM-head row dots reuse the Task 2.5 numeric
contract and Task 2.11 semantic provider. Lower-level operator/provider
contracts remain frozen. Full-model greedy-token equality is exact-discrete.
Top-N logits and hidden summaries are characterization diagnostics only; M1
does not invent a full-model floating tolerance from one prompt.

This contract excludes vision, MTP/speculation, batching, stochastic sampling,
multi-token generation, cache/prefetch/scheduler policy, SIMD and CUDA model
kernels.
