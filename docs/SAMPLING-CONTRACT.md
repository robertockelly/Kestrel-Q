# Qwen3.8 native sampling contract

Status: **TASK 3.0 CHARACTERIZED / PINNED BEFORE NATIVE ALGORITHM CODE**

This contract characterizes the first Kestrel-Q stochastic-selection boundary.
It is model-specific, batch-1 and independent of model execution. It consumes
one complete finite F32 score vector and produces one token ID from an immutable
policy plus caller-owned random state.

## Pinned semantic sources

| Source | Identity | SHA-256 |
|---|---|---|
| official model | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | revision pin |
| `generation_config.json` | 202 bytes | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| Transformers | `huggingface/transformers@805a9e939fa8c1bff8d8ffdf041c051b71a914aa` | revision pin |
| `generation/logits_process.py` | executed processor definitions | `c5b5d5666576b19e19a09b99b55806ea009dd8e999892f7a8d2d7876dbfc2002` |
| `generation/utils.py` | processor construction and `_sample` | `c45e19eb4534a2478f8e4825dc35a28fb5efe5a2f8d09695e4b2ea532b4dbd82` |
| `generation/configuration_utils.py` | defaults and validation | `427f7e57922fb58c3b321f5e5837ee782bd640cd5b22833029b16f9d4b9c4cdb` |

The executable Class-C environment is CPython 3.13.12, CPU PyTorch
2.11.0+cpu, NumPy 1.26.4, pinned Transformers 5.16.0.dev0 and tokenizers
0.23.1. PyTorch is BSD-3-Clause and is an offline oracle dependency only.
Production Kestrel-Q links none of these components.

The relevant pinned functions are
`GenerationMixin._get_logits_processor`, `GenerationMixin._sample`,
`TemperatureLogitsWarper.__call__`, `TopKLogitsWarper.__call__` and
`TopPLogitsWarper.__call__`.

## Official profile and token domain

The required official configuration is:

```text
do_sample   = true
temperature = 1.0
top_k       = 20
top_p       = 0.95
bos_id      = 248044
eog_ids     = [248046, 248044]
pad_id      = 248044
```

The sampler requires exactly 248,320 F32 logits with IDs `0..248319`.
Canonical tokenizer IDs are `0..248076`; IDs `248077..248319` are 243 model
capacity-padding IDs. The official configuration has no `suppress_tokens`
entry and the pinned processor path does not mask those IDs. They therefore
participate in top-k, top-p and probability normalization. Kestrel-Q does not
silently alter that distribution: it validates the sampled ID after
categorical selection and fails with `INVALID_TOKEN_ID`, without consuming RNG
state or publishing a result, if a padded ID wins. Task 3.1 must preserve this
fail-closed rule.

IDs 248044 and 248046 are ordinary eligible sampling outcomes. EOG stopping is
owned by the later generation loop, not by this sampler. BOS is not inserted
by Task 3.0, and the accepted tokenizer contract still inserts no automatic
BOS or EOS.

## Score and processor contract

`GenerationMixin._sample` copies the final-position model logits to F32 before
running the processor list. The native boundary accepts a one-dimensional,
complete, contiguous batch-1 F32 vector. Every input score must be finite.
NaN, positive infinity and negative infinity are rejected rather than repaired;
the official `remove_invalid_values` default is false.

Only temperature, top-k and top-p are supported. No repetition, frequency,
presence or length penalty, n-gram blocking, bad-word list, forced token,
minimum-p, top-h, typical-p, epsilon/eta cutoff, grammar, watermark, stop
string, custom processor, beam, MTP/speculation or vision option is accepted.

Processor construction order is:

1. temperature division only when `temperature != 1.0`;
2. top-k only when `top_k != 0`;
3. top-p only when `top_p < 1.0`;
4. final F32 softmax;
5. one categorical draw.

The supported override domain remains deliberately narrow to these three
characterized fields:

- temperature is a finite F32 value strictly greater than zero;
- top-k is `0..248320`, where zero disables top-k;
- top-p is a finite F32 value in `[0,1]`, where one disables top-p.

Values outside those domains fail. A top-k larger than the target vocabulary
is rejected instead of silently clamped. No additional option is inferred
from a near-match configuration.

### Temperature

Each F32 score is divided by the F32 temperature. A non-finite result fails.
At the official value 1.0 the processor is absent, preserving the input score
bits.

### Top-k

For nonzero `k`, the threshold is the kth largest score. A score is removed
only when it is strictly less than that threshold. Exact ties at the threshold
are all retained, so retained count can exceed `k`. The released single-beam
path uses `min_tokens_to_keep=1`. The native deterministic ordering for
diagnostics is descending score then ascending token ID; membership, rather
than an unspecified partial-sort order, is the canonical processor result.

### Top-p

After top-k, the pinned implementation sorts scores in ascending order,
computes F32 softmax and cumulative probability in that order, and removes a
token when cumulative probability is `<= (1 - top_p)`. The highest-scoring
token is always retained. This is equivalent to retaining the smallest
high-probability suffix whose mass is at least `top_p`, with the equality
boundary on the removed side.

Pinned CPU PyTorch 2.11.0 preserves ascending input ID for the characterized
equal-score sort cases. Kestrel-Q makes that observed tie rule explicit:
ascending score, then ascending token ID. The independent vectors freeze
below/on/above-cutoff and equal-logit behavior so a future oracle/toolchain
change cannot silently alter it.

### Final softmax

After filtering, the pinned path applies F32 `softmax(dim=-1)` over original
token-ID order. The native scalar reference subtracts the maximum retained
score, applies `expf`, accumulates in F32 in ascending token-ID order and
normalizes to F32. It is compiled with MSVC `/fp:strict` (or
`-fno-fast-math -ffp-contract=off`) and requires `FE_TONEAREST`. Intermediate
floating fields are `CALIBRATED_FLOAT` against CPU PyTorch; retained IDs and
the selected ID remain exact. A non-positive or non-finite mass fails closed.

## Portable RNG and categorical selection

Transformers delegates sampling to `torch.multinomial`, whose RNG algorithm,
device behavior and global generator state are not a portable model semantic.
Kestrel-Q therefore governs a separate RNG while retaining the canonical
processor/probability path.

The selected generator is PCG-XSH-RR 64/32 (`pcg32`), pinned to the published
minimal PCG C specification at
`imneme/pcg-c-basic@bc39cd76ac3d541e618606bcc6e1e5ba5e5e6aa3`.
That source is Apache-2.0. Reference hashes are:

- `pcg_basic.c`:
  `b3a77c42f9e7b57a095aa0c5f684c42a2fdade5cb823cdb483bb09a88ffc6fe0`;
- `pcg_basic.h`:
  `18b8752b39fe07d527b4afe7756d1349fd44871f218811ec4d57c1ea7fc1e9f5`;
- `LICENSE.txt`:
  `e03ba41d7fab20700769fe4118bab50d800cb74f990353a05d2f5fff1c228363`.

Kestrel-Q implements the published arithmetic independently; it does not add
or link the PCG package. The 64-bit LCG multiplier is
`6364136223846793005`. Seeding is the published set-sequence procedure:
state zero, odd increment `(stream << 1) | 1`, one transition, add the 64-bit
seed, then one transition. Seed accepts every `uint64_t`; stream accepts
`0..2^63-1` so its identity is not truncated. All unsigned operations wrap
modulo their width.

State format version 1 records seed, stream, internal 64-bit state, successful
draw count and an integrity word. Reset repeats seed expansion. Snapshot/import
copies semantic fields, verifies version/reserved bits, the stream-derived odd
increment and integrity, and introduces no global state. State is caller-owned
and not safe for concurrent mutation without caller synchronization.

Each successful selection consumes exactly one 32-bit PCG output. Uniform
mapping is exact binary64 `u = word / 2^32`, giving `[0,1]`'s half-open subset
`[0,1)`. Categorical probabilities are traversed in ascending token-ID order
with binary64 cumulative mass. Word `w` selects the unique interval
`[previous_mass,total_mass)` containing `u * total_mass`; exact internal
boundaries belong to the following token. The maximum word `2^32-1` remains
strictly below one. A final-rounding guard may select the last positive-mass
token only when the threshold remains below the validated total.

The sampler works on a private copy of RNG state. It publishes the selected
canonical ID, diagnostics and one-draw state only after every processor,
probability and ID check succeeds. Any failure leaves caller RNG and result
unchanged.

## Ownership, buffers and limits

The immutable config owns only its validated profile. The caller owns logits,
scratch, RNG state and result. Scratch capacity is queried deterministically
for the target vocabulary and holds only bounded score/probability/order work
arrays. Input logits are never modified, buffers may not overlap, arithmetic is
checked and no global mutable sampler state exists.

Task 3.0 does not load a model, access GGUF payload, call the executor, expose a
sampling CLI or alter the accepted greedy argmax. Sampled model-session and
whole-token RNG rollback integration belong exclusively to Task 3.1.
