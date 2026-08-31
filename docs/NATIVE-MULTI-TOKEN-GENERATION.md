# Native multi-token generation

Status: **TASK 2.13 COMPLETE / PASS**

Task 2.13 turns the accepted M1 executor into a reusable incremental lifecycle:

```text
canonical prompt IDs -> prefill once -> token 271
token 271            -> decode_one   -> token 248068
token 248068         -> decode_one   -> token 198
token 198            -> decode_one   -> token 760
```

The exact independent and native sequences are both
`[271, 248068, 198, 760]`, decoded as `"\n\n<think>\nThe"`. The first-token
API and `--max-new-tokens 1` continue to return token 271 and bytes `\n\n`.

## Implementation

`kq_model_exec_decode_one_f32` consumes one canonical ID using the existing
embedding provider, Task 2.10 layer decode path, final mixer, LM head, stable
argmax and Task 2.3 tokenizer decode. It never replays the prompt and never
materializes a complete target matrix in F32.

The model executor coordinates the 48 existing layer transactions. Completed
layers are recorded during a token; any later failure rolls those layers back
to their previous active slots. Caller outputs use scratch/local staging and
are copied only after all validation succeeds. State summaries expose separate
content hashes for GDN, QSA, PLE-address and PLE-value state plus positions and
QSA length/block/tail bounds.

The minimal `kq-run` CLI accepts `--max-new-tokens 1..4 --greedy`, uses context
capacity 16, tokenizes once, reports bounded phase progress and prints per-token
IDs, fragments, positions, state hashes and logical counters. Sampling and
unsupported generation options fail before opening a model.

## Logical payload

| Operation | Logical packed bytes | Blocks | Routed selections | Expert matrices | PLE rows |
|---|---:|---:|---:|---:|---:|
| seven-token prefill | 40,208,768,960 | 1,693,222,880 | 3,360 | 10,080 | 112 |
| decode token 271 | 6,334,414,400 | 260,441,120 | 480 | 1,440 | 16 |
| decode token 248068 | 6,334,414,400 | 260,441,120 | 480 | 1,440 | 16 |
| decode token 198 | 6,334,414,400 | 260,441,120 | 480 | 1,440 | 16 |
| **successful total** | **59,212,012,160** | **2,474,546,240** | **4,800** | **14,400** | **160** |

The 55.14 GiB successful total passes the frozen 96 GiB correctness ceiling.
This is logical payload requested through bounded semantic views, not physical
I/O, page residency, storage traffic or a throughput benchmark. The provider
touches the same 1,239 execution semantics; its cumulative unique-set counter
does not pretend that a later decode introduced new tensor identities.

Every per-token MoE route contains exactly ten unique ordered expert IDs. Only
those members are opened, once logically and through three gate/up/down matrix
requests. Every decode produces exactly 16 PLE intents/row requests. QSA state
advances from length 7 through 10 with the exact four-token block/tail sequence
defined by the contract. Aggregate QSA candidate/selected-block counts are
48/48 for prefill and 24/24 for every decode; selected-token counts are 336,
96, 108 and 120 respectively.

## Memory and characterization

With context capacity 16, the observed owned model/layer state is 361,208,872
bytes. Released semantic capacity is 116,822,048 bytes: 116,195,328 fixed GDN,
442,368 QSA capacity, 32 PLE-address and 184,320 PLE-value bytes. The remaining
244,386,824 bytes are the scalar-correctness container delta comprising F32
reference representation, duplicate transaction slots, copy workspace and C
object overhead; it is not claimed as a pure semantic cache requirement.

Peak scratch is 8,693,472 bytes, logits are 993,280 bytes and the largest
simultaneous provider F32 result/staging allocation is also 993,280 bytes.
Complete target F32 weight matrices materialized: zero.

The real successful integration took about 201–212 seconds in development
runs. This is characterization, not a performance guarantee or accepted
optimization baseline.

## Dirty-journey findings

The first two real runs failed in the new trace validator after correct native
prefill. The first validator assumed token-major MoE trace order, while the
executor is layer-major with tokens inside each layer. The second conflated the
three matrix-request metric with the one-entry-per-selected-member expert
trace. The corrected regressions validate layer-major route groups, one member
trace per selected expert and three matrix requests separately. Both failed raw
runs remain preserved under the ignored Task 2.13 research cache.

The production incremental path itself then matched all four independent
oracle tokens on its first full execution and passed the injected rollback and
retry gate.
