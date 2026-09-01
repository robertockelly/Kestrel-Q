# Native sampled generation

Status: **TASK 3.1 COMPLETE / PASS**

Task 3.1 composes the accepted Task 3.0 sampler with the accepted incremental
model executor. The production sequence is:

```text
canonical prompt IDs -> prefill once -> sample first token
accepted sampled ID  -> decode once  -> sample next token
```

The integration does not duplicate temperature, top-k, top-p, softmax,
categorical or PCG behavior. `kq_sampling` remains the sole owner of those
semantics. The model executor owns model-state staging and calls the sampler
with a private snapshot of the caller's RNG state.

## Combined transaction

`kq_model_exec_sampled_prefill_f32` and
`kq_model_exec_sampled_decode_one_f32` accept an immutable sampling config,
explicit caller-owned RNG state and separate model/sampling scratch buffers.
The model result, sampling result, logits, decoded fragment, model state and
RNG state become visible only after model execution, sampling, canonical-token
validation and native token decode all succeed.

A later-step error rolls back every model layer through the existing Task 2.13
transaction slots and discards the private RNG snapshot. Caller outputs remain
unchanged. This was exercised after successful output at early, middle and
late layers, after decode during sampler validation, and on selection of a
model-capacity padding ID. Retry proceeds from the exact pre-step state.

EOG IDs 248044 and 248046 remain eligible. A successful EOG selection commits
the current draw and any decode of the preceding token, then the caller stops.
The sampled decode API rejects an EOG input explicitly, preventing a stopped
generation from being advanced accidentally. There is no hidden global or
wall-clock-derived RNG state.

## Governed real traces

Both traces use the registered GGUF, `KQ-PROMPT-001`, canonical IDs
`[9419, 11, 710, 467, 3621, 27325, 13]`, context capacity 16 and the official
profile temperature 1.0 / top-k 20 / top-p 0.95.

| Trace | Seed | Stream | Sampled IDs | UTF-8 fragments |
|---|---:|---:|---|---|
| primary | 0 | 5427223837140668492 | `[271, 248068, 198, 760]` | `"\n\n"`, `"<think>"`, `"\n"`, `"The"` |
| holdout | 18446744073709551615 | 9223372036854775807 | `[271, 248068, 198, 760]` | same |

The equal token sequences are an observed consequence of sharply concentrated
per-step survivor distributions, not an assumption that seeds are
interchangeable. PCG words and states differ between traces. Primary replay
matches every token, fragment, retained-order hash, RNG state, model-state
hash and lower-level request-order hash exactly.

Temporary full F32 logits are captured only under the ignored research cache.
The independent Python evidence tool applies the frozen Task 3.0 policy and
PCG transition to those arrays without importing Kestrel-Q sampling code. It
requires exact retained count/order hash, RNG word/post-state/integrity and
selected ID for every primary and holdout step. Full logits are not governed
or committed.

The bounded evidence is stored beside the M1/Task 2.13 milestone artifacts at
`research/milestones/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
Its manifest SHA-256 is
`2506073d9e9c4d0733cd109429e74b420e755a161193f815ea3bb6bbb951097a`;
all six asset hashes are listed in `GOLDEN-VECTORS.md`.

## Logical payload and invariants

| Operation | Logical packed bytes | Blocks | Routed selections | Expert matrices | PLE rows |
|---|---:|---:|---:|---:|---:|
| seven-token prefill | 40,208,768,960 | 1,693,222,880 | 3,360 | 10,080 | 112 |
| each one-token decode | 6,334,414,400 | 260,441,120 | 480 | 1,440 | 16 |
| four-token successful trace | 59,212,012,160 | 2,474,546,240 | 4,800 | 14,400 | 160 |

The 55.14 GiB successful total remains under the predeclared 96 GiB
correctness ceiling and exactly matches the accepted greedy path: one prefill
and three decodes, with no prompt replay. Counters are logical semantic-view
requests, not physical I/O or performance measurements.

Every token retains exact Task 2 invariants: ten unique routed experts per
layer and zero unselected member requests; 16 PLE intents and row requests;
and QSA candidate/selected block counts 48/48 for prefill and 24/24 per decode.
Selected-token counts are 336, 96, 108 and 120 as model positions advance
from 7 through 10.

## Memory and CLI

The 361,208,872-byte context-16 model state remains caller-owned. Sampling adds
a 56-byte RNG state, a 56-byte immutable config and 3,973,127 bytes of caller
sampling scratch. Model scratch, logits and existing double-slot state remain
separate. No full target F32 matrix is materialized; the maximum simultaneous
provider F32 result remains 993,280 bytes.

`kq-run` preserves `--greedy` and adds:

```text
kq-run <model.gguf> --prompt <text> --max-new-tokens N \
  --sample --seed <u64> [--stream <u63>]
```

The stream defaults to the governed Task 3.0 stream when omitted. A seed is
mandatory, the stream is limited to 63 bits and selection modes cannot be
combined. Generation supports one through four new tokens in this bounded CLI
and stops on EOG without another decode or draw.

## Dirty-journey findings

The first CLI build produced MSVC C4701 because conservative analysis could
not infer that the sampled-only reporting branch followed RNG initialization;
explicit initialization fixed the warning without changing RNG semantics. A
review of the new context-exhaustion harness then found a latent false-PASS
path when scratch sizing failed while an earlier status remained `OK`; an
explicit success flag now closes that path.

One non-governed real capture printed the output sentinel as the later EOG
token and repeated position 11, even though the test had already validated the
correct 248046 selection and 10→11 transition. The stopped-state rejection
probe reused the result/summary objects before printing. Accepted fields are
now snapshotted first, and the evidence generator independently requires EOG
248046, positions 10→11 and draws 4→5. That capture remains ignored and is not
evidence.

## Scope

This is scalar CPU correctness work. It adds no new sampling processor, model
operator, batching, persisted session, server, scheduler, cache, prefetch,
SIMD or CUDA model kernel. `KQ-BACKLOG-BENCH-002` remains deferred.
