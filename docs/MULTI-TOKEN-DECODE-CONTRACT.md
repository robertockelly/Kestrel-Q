# Multi-token decode contract

Status: **TASK 2.13 COMPLETE / PASS**

This contract extends the accepted M1 one-token path without changing its
model arithmetic. It governs batch-1, text-only, scalar CPU greedy generation
for the pinned Qwen3.8-Flash-Next GGUF.

## Governed sequence

`KQ-PROMPT-001` is UTF-8 `Hello, Kestrel-Q.` and its canonical token IDs are:

```text
[9419, 11, 710, 467, 3621, 27325, 13]
```

The independent `llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`
oracle, using those explicit IDs and the verified GGUF, produces:

| Step | Model input | Selected ID | Decoded fragment | EOG |
|---:|---|---:|---|---|
| 1 | prompt positions 0..6 | 271 | `\n\n` | no |
| 2 | token 271 at position 7 | 248068 | `<think>` | no |
| 3 | token 248068 at position 8 | 198 | `\n` | no |
| 4 | token 198 at position 9 | 760 | `The` | no |

The complete ID sequence is `EXACT_DISCRETE`. Kestrel-Q output is compared to
this frozen sequence and never defines it.

## Prefill and decode progression

The prompt is tokenized and prefilled exactly once. Prefill consumes all seven
canonical IDs and commits model position 7. Its logits select generated token
1, but that selected token is not yet part of persistent model state.

Every following operation accepts the preceding selected ID directly. It does
not decode and re-tokenize accumulated text, replay the prompt, or rerun an
earlier generated token. A successful `decode_one` consumes one ID at the
current position, advances every layer and the public model position by one,
then selects the following ID.

Consequently four returned tokens require one prefill plus three decode calls.
The fourth selected token is returned but is not fed back unless the caller asks
for a fifth token. Context capacity 16 is explicit; no 262,144-token QSA cache
is allocated. A request needs capacity for `prompt_tokens + max_new_tokens - 1`
consumed positions.

## Persistent state

Each successful consumed position advances, as one unit:

- all 36 GDN recurrent/convolution states;
- all 12 QSA K/V/raw-index-key caches and lengths;
- PLE address history on layer 1;
- PLE value convolution history on layer 1;
- every layer position and the public model position.

GR branches, MoE routes, sparse selections and the current logits are
current-forward temporaries, not persistent session state. State evidence hashes
explicit semantic bytes/fields; it does not hash C padding or commit raw state
blobs.

At position `p`, every QSA layer has length `p`, `floor(p / 4)` complete
four-token blocks and tail `p mod 4`. The governed positions 7, 8, 9 and 10
therefore have block/tail pairs `(1,3)`, `(2,0)`, `(2,1)` and `(2,2)`.

## Greedy and stop semantics

Greedy selection examines all 248,320 finite F32 logits. A strictly greater
value replaces the winner, so the lower token ID wins an exact tie. Selected
IDs `248077..248319` are padded capacity rather than canonical tokenizer IDs and
fail closed.

Generation EOG IDs remain 248044 and 248046. An EOG token is returned and
decoded with `KQ_TOKENIZER_DECODE_KEEP_SPECIAL`, then the caller stops before
feeding it. `max_new_tokens` counts returned IDs; zero is invalid in the Task
2.13 CLI, one preserves M1 exactly, and the governed CLI limit is four.
Decoded fragments are produced one selected ID at a time and concatenated by
the caller; they are never used to drive model state.

## Per-token transaction

Every consumed token is a complete transaction. Caller logits, decoded bytes,
result, public position and all layer active slots commit only after the final
mix, LM head, finite argmax and native token decode succeed.

Each layer already maintains committed and staging slots. During model decode,
successfully completed layers flip once. If a later layer or final stage fails,
the executor flips those completed layers back in reverse order. The failing
layer never flipped, and its staging data is overwritten from the committed
slot on retry. A rollback invariant failure marks the model state as requiring
reset and fails closed.

Weight-provider byte counters and diagnostic traces are monotonic audit data,
not persistent model context, and are not rolled back. Evidence therefore
separates bytes spent by an injected failed attempt from the successful-path
payload gate.

The real regression injects a provider failure at layer 24 while consuming the
third-step input after two successful outputs. Pre/post-failure state summaries
match exactly, the earlier prefix remains valid, and retry selects token 198.

## Scope boundary

This contract adds no stochastic sampling, penalties, batching, conversation
object, MTP/speculation, vision, scheduler, cache/prefetch policy, SIMD or CUDA
model kernel. Timings and logical payload counters are characterization only.
