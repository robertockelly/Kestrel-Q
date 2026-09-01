# Kestrel-Q model-execution API

The public C17 declarations are in `include/kq_model_exec.h`.

## Lifetime

`kq_model_exec_config_open` borrows its GGUF, semantic model, tokenizer and
weight provider. They must outlive the config and every state created from it.
A state must not outlive its config. Config and state are not thread-safe for
concurrent mutation; there is no global model-execution state.

## Construction and queries

- `kq_model_exec_config_open` validates the exact target entry, 48 layers,
  final mixer and LM head for an explicit context capacity.
- `kq_model_exec_state_create` allocates bounded per-layer state.
- `kq_model_exec_state_reset` deterministically returns all layers and the
  model position to zero.
- `kq_model_exec_state_get_summary` reports positions, QSA block/tail bounds
  and deterministic semantic-state hashes without exporting state blobs.
- config/state owned-byte and context/position queries return zero for invalid
  objects.
- `kq_model_exec_required_scratch_bytes` is checked and requires a reset state,
  non-empty prompt and room for a continuation position.
- `kq_model_exec_required_decode_scratch_bytes` sizes one incremental token
  against the current bounded QSA length and context capacity.

## Execution

`kq_model_exec_prefill_first_token_f32` accepts explicit canonical token IDs.
`kq_model_exec_generate_first_token_f32` first invokes the native tokenizer on
explicit pointer+length UTF-8 and then calls the same executor. Both require
caller-owned full-logits, decoded-output and scratch buffers.

On success the call commits:

- all 248,320 F32 logits;
- one decoded selected token;
- the result/metrics structure;
- public model position equal to the prompt length.

`kq_model_exec_decode_one_f32` consumes one previously selected canonical ID
directly and returns the following greedy ID/logits/decoded fragment. It
requires a successful prefill or decode state, room in the configured context
and caller-owned buffers under the same lifetime and aliasing rules.

`kq_model_exec_sampled_prefill_f32` and
`kq_model_exec_sampled_decode_one_f32` compose the same execution path with an
immutable Task 3.0 `kq_sampling_config`, caller-owned
`kq_sampling_rng_state`, caller-owned sampling scratch and a separate
`kq_sampling_result`. They do not contain sampling mathematics. The prefill
call stages the first draw after one prompt prefill. The decode call consumes
one preceding accepted sampled ID, stages one incremental model step and then
draws its successor.

For sampled calls, model state, RNG state, model/sampling results, logits and
decoded output commit together. Failed model execution, sampling, padded-ID
validation or output decode leaves all caller-visible state/output unchanged.
The sampled decode entry rejects EOG input because a successful EOG selection
is a stopped generation and must never be fed back. The caller remains
responsible for max-token and EOG loop termination; there is no hidden session
or RNG state.

On prefill failure caller logits, decoded output, result and public position
remain unchanged; legacy M1 recovery resets private staging before retry. On
incremental failure, completed layer commits are rolled back in reverse order,
so all persistent state and caller outputs remain at the immediately preceding
successful token. Provider accounting is monotonic audit data and is not part
of the rollback state.

Input IDs `248077..248319` and all IDs outside model capacity fail explicitly.
Prompts must contain fewer tokens than context capacity. Decode fails closed
at capacity and on a reset/invalid continuation state. Buffers must not
overlap, arithmetic is checked and non-finite hidden/logit results fail closed.

`kq_model_exec_greedy_argmax_f32` validates finite logits, examines the complete
array and retains the first/lower ID on exact ties. It does not apply a
tokenizer/model ID policy by itself; the model executor validates the selected
ID before native decode.

`kq_model_exec_token_is_eog` recognizes the pinned generation EOG IDs 248044
and 248046. The caller returns/decodes an EOG selection and stops before
feeding it. Greedy selection remains unchanged. Sampled execution accepts the
Task 3.0 policy/RNG surface explicitly and never seeds from wall-clock state.

## Metrics

Metrics distinguish semantic logical touches from physical I/O and expose
embedding/head bytes, blocks, unique semantics, routed selections, selected
expert matrix requests, PLE rows, QSA selection counts, prefill/decode counts,
state/scratch/logits bytes and elapsed host time. They are characterization
counters, not performance guarantees.
