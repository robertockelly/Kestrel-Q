# Kestrel-Q QSA API

The public C17 API is declared in `include/kq_qsa.h`.

## Configuration

`kq_qsa_config_create` creates an immutable target BF16 descriptor from a
borrowed `kq_model` and QSA layer ID.  It validates the model-specific topology
and all semantic/physical bindings.  `kq_qsa_config_create_reference_f32`
performs the same target validation but produces an F32 scalar-reference
descriptor.  A GDN layer returns `KQ_STATUS_INCOMPATIBLE_QSA`.

The config borrows the model and its semantic descriptors; therefore it must
not outlive the model.  A state borrows its config and must be destroyed before
the config.  No global mutable mapping or execution state exists.

Geometry getters expose layer ID, hidden/head/index widths, block size,
selection limit, context limit, activation dtype and semantic state growth.
They do not expose GGUF tensor names or physical offsets.

## Stream state

`kq_qsa_state_create(config, capacity, ...)` allocates bounded K, V and raw
index-key storage.  Capacity must be positive and no larger than the validated
context limit.  `kq_qsa_state_reset` sets length to zero.  Length, capacity and
owned-byte queries are explicit.

F32 state import/export is intended for correctness tests, snapshots and
controlled continuation.  Counts must exactly match `length` and config
geometry.  Non-finite state, foreign config/state pairs, and shape mismatch
fail explicitly.

## Execution

`kq_qsa_prefill_f32` appends one or more tokens.  `kq_qsa_decode_f32` is the
one-token form.  Both require:

- a validated F32 reference config and compatible state;
- complete row-major F32 reference weights;
- explicit input/output lengths and output capacity;
- aligned caller workspace of the exact size returned by
  `kq_qsa_required_scratch_bytes`;
- non-overlapping input, output, state and workspace.

The F32 weight structure separates the canonical index-Q and index-K parts in
their validated binding order.  The query projection retains canonical
per-head `[query, gate]` row ordering.

Calls are transactional for visible state and output.  Observer callbacks are
diagnostic and receive borrowed pointers valid only for the duration of the
callback.

## Selection and checkpoints

The selection observer receives:

- local token index and absolute position;
- candidate block IDs and FP32 scores;
- exact selected block IDs;
- exact selected token positions;
- incomplete-tail count.

`kq_qsa_select_blocks_f32` is the checked deterministic selector used by the
operator and Tier-B threshold validation.  It sorts by descending score and
ascending block ID for equal scores.  Non-finite scores, an invalid limit,
overflow, or insufficient output capacity fail closed.

The checkpoint observer can inspect index query, raw index key, normalized
block keys, candidate scores, core Q/K/V, attention logits/probabilities,
head context, gated context and final operator output.  Shapes and counts are
included in every callback.

## Error boundary

Important statuses are:

- `KQ_STATUS_INCOMPATIBLE_QSA` for wrong model/layer/topology/dtype;
- `KQ_STATUS_INVALID_QSA_STATE` for invalid selection/state invariants;
- `KQ_STATUS_SEMANTIC_MAPPING_FAILED` for a missing required semantic;
- `KQ_STATUS_TENSOR_LAYOUT_MISMATCH` for wrong split/type/layout;
- `KQ_STATUS_LIMIT_EXCEEDED` for state-capacity/context overflow;
- `KQ_STATUS_BUFFER_TOO_SMALL`, `KQ_STATUS_DIMENSION_MISMATCH`,
  `KQ_STATUS_ALIASING_VIOLATION`, and `KQ_STATUS_NUMERIC_DOMAIN` for their
  explicit conditions.

The API never opens GGUF payload views by itself and has no scheduler, cache
placement, threading, Python, Transformers, llama.cpp or CUDA dependency.
