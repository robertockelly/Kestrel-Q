# Kestrel-Q GDN API

Status: **TASK 2.6 BASELINE**

The public C17 declarations are in `include/kq_gdn.h`.

## Ownership and lifetime

- A `kq_gdn_config` created from a model borrows `kq_model`, its `kq_gguf` and
  the underlying file. Those owners must outlive the config.
- A `kq_gdn_state` owns its state arrays but borrows its config. The config must
  outlive the state.
- `kq_gdn_weights_f32` borrows every caller array for the duration of one call.
- Input, output, state, weights and scratch must obey the no-overlap checks.
- Checkpoint values borrow scratch and are valid only during the callback.
- No API uses global mutable GDN state.

Close state before config, then close the model/GGUF/file owners.

## Configuration

```c
kq_status kq_gdn_config_create(
    const kq_model *model,
    uint32_t layer_id,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic);

kq_status kq_gdn_config_create_reference_f32(
    const kq_model *model,
    uint32_t layer_id,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic);
```

The first function preserves the released checkpoint's BF16 activation/state
descriptor. The second validates the same semantic registry but constructs an
F32 scalar-reference descriptor that can be passed to the Task 2.6 execution
functions. Both reject QSA IDs, missing roles, wrong shapes/dtypes, unsupported
physical types and binding/layout mismatches.

Geometry, immutable owned bytes, state-element counts, scratch bytes and
dequant scratch can be queried without payload access.

## Weights

`kq_gdn_weights_f32` supplies nine finite arrays in canonical row-major order:

```text
A_log[value_heads]
conv[conv_channels][4]
dt_bias[value_heads]
alpha[value_heads][hidden]
beta[value_heads][hidden]
qkv[conv_channels][hidden]
gate[value_width][hidden]
norm[value_head_dimension]
output[hidden][value_width]
```

Counts are mandatory and exact. These are canonical semantic layouts. The
GGUF's transformed QKV/gate/alpha/beta/output layouts and transformed
`-exp(A_log)` tensor are not accepted as if canonical. A future bounded loader
must perform the explicit transformations before constructing this view.

## State

`kq_gdn_state_create` allocates zero state and marks it uninitialized.
`kq_gdn_state_reset` returns to the same semantic state. F32 import/export
requires exact convolution and recurrent element counts, finite values and an
explicit initialization flag. BF16 descriptors round imported convolution
state to BF16 ties-to-even and expand it on export; recurrent state remains
F32.

Import/export arrays are semantic element sequences. Callers must not inspect
or compare opaque object bytes.

## Prefill and decode

`kq_gdn_prefill_f32` accepts batch-1 contiguous `[sequence,hidden]` input and
optional zero/one padding bytes. Sequence length must be non-zero. Output has
the same shape. `kq_gdn_decode_f32` is the one-token form and never accepts a
padding token.

The caller allocates at least `kq_gdn_config_scratch_bytes(config)` aligned
bytes. Scratch does not grow with sequence length. No implicit output
truncation, full-tensor materialization or hidden heap allocation occurs in the
operator call.

All preconditions are checked before computation. Numeric failure leaves the
state unchanged. Successful prefill/decode commits the final convolution and
recurrent state and marks it initialized.

## Checkpoints

The optional observer receives masked input, four projection families,
convolution input/output and state, query/key/value before and after
normalization, beta/log decay, recurrent read/delta/output/state, gated norm
and final operator output. Every event includes token index, logical dimensions
and F32 count. The observer is diagnostics-only and cannot alter execution.

## Stable failures

Important statuses include:

- `KQ_STATUS_INCOMPATIBLE_GDN` for wrong layer/topology/reference dtype;
- `KQ_STATUS_SEMANTIC_MAPPING_FAILED` for a missing required meaning;
- `KQ_STATUS_TENSOR_LAYOUT_MISMATCH` for wrong binding/type/layout;
- `KQ_STATUS_INVALID_GDN_STATE` for invalid state operations;
- `KQ_STATUS_DIMENSION_MISMATCH`, `KQ_STATUS_BUFFER_TOO_SMALL`,
  `KQ_STATUS_ARITHMETIC_OVERFLOW`, `KQ_STATUS_ALIASING_VIOLATION` and
  `KQ_STATUS_NUMERIC_DOMAIN` for execution-contract failures.

Unsupported input never falls through to an approximate path.
