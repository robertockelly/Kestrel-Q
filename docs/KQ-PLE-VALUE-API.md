# Kestrel-Q PLE value API

The public C17 API is declared in `include/kq_ple_value.h`. It is a batch-1,
scalar reference API for the characterized Qwen3.8 PLE value path.

## Lifetime and ownership

- `kq_ple_value_config` is immutable and owned by the caller after successful
  creation. It borrows its `kq_model` and `kq_ple_config`; close it first.
- `kq_ple_value_state` is mutable per stream, owned by the caller and tied to
  the exact configuration that created it.
- weights, hidden inputs, intents, output and scratch are caller-owned and must
  remain valid for the call.
- checkpoint values are borrowed transient views into call storage and are
  valid only during the observer callback.
- a GGUF provider borrows the GGUF, model and value configuration and must be
  closed before any of them.

There is no global mutable PLE state.

## Configuration and state

`kq_ple_value_config_create` validates the target semantic configuration.
`kq_ple_value_config_create_reference_f32` creates the executable F32 scalar
variant after the same validation. Query functions expose layer, geometry,
owned bytes, released semantic-state bytes, F32 state bytes and scratch bytes.

`kq_ple_value_state_create`, `reset`, `import_f32` and `export_f32` manage the
explicit nine-position value history. Import rejects non-finite elements.
State position is checked for overflow during execution.

## Prefill and decode

`kq_ple_value_prefill_f32` requires:

- a valid F32 reference configuration and matching state;
- finite hidden input and canonical F32 weights with exact counts;
- exactly 16 Task 2.4 intents per token, in canonical order;
- a compatible provider with 128 members, 2,500,012 rows/member and width 160
  for the real target;
- caller output and scratch capacities reported by the configuration.

`kq_ple_value_decode_f32` applies the same operation for one token. There is no
silent truncation. Writable output must not alias hidden input, value state or
scratch. Failure leaves the externally visible state unchanged.

The optional observer reports canonical checkpoints by token. Its callback
must not retain checkpoint pointers. Timing and scratch fields in
`kq_ple_value_run_metrics` are characterization measurements, not performance
guarantees.

## Providers

`kq_ple_value_lookup_provider` is the minimal synchronous semantic interface.
The callback must fill exactly the requested F32 row or fail. It must not
return partial success.

`kq_ple_value_gguf_provider_open` creates the bounded test/integration provider.
The provider resolves semantic member views and records logical packed bytes,
blocks and rows. A request exceeding its immutable byte budget fails before
the next row is accessed.

## Stable failures

Task 2.9 adds `KQ_STATUS_INCOMPATIBLE_PLE_VALUE`,
`KQ_STATUS_INVALID_PLE_VALUE_STATE` and `KQ_STATUS_PLE_LOOKUP_FAILED`.
Existing invalid-argument, buffer, overflow, numeric-domain, aliasing and limit
statuses remain applicable. Unsupported batching, masking and alternate model
geometries fail closed; they are not approximated.
