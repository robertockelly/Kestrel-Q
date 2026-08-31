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
- config/state owned-byte and context/position queries return zero for invalid
  objects.
- `kq_model_exec_required_scratch_bytes` is checked and requires a reset state,
  non-empty prompt and room for a continuation position.

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

On failure caller logits, decoded output, result and public position remain
unchanged. Private layer staging may have been written, so the next execution
resets all layers automatically. A successful prefill state cannot be reused
as an incremental generator in this M1 API.

Input IDs `248077..248319` and all IDs outside model capacity fail explicitly.
Prompts must contain fewer tokens than context capacity. Buffers must not
overlap, arithmetic is checked and non-finite hidden/logit results fail closed.

`kq_model_exec_greedy_argmax_f32` validates finite logits, examines the complete
array and retains the first/lower ID on exact ties. It does not apply a
tokenizer/model ID policy by itself; the model executor validates the selected
ID before native decode.

## Metrics

Metrics distinguish semantic logical touches from physical I/O and expose
embedding/head bytes, blocks, unique semantics, routed selections, selected
expert matrix requests, PLE rows, state/scratch/logits bytes and elapsed host
time. They are characterization counters, not performance guarantees.
