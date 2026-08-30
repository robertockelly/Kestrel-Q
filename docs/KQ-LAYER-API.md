# Kestrel-Q transformer-layer API

The public C17 declarations are in `include/kq_layer.h`.

## Ownership

`kq_layer_config_create` returns an immutable target descriptor borrowing its
model/GGUF/file. `kq_layer_config_create_reference_f32` validates the same
bindings for scalar F32 execution. Close layer state before config and config
before model owners. Weight and provider structures are borrowed for one call.
No global mutable layer state exists.

`kq_layer_state_create` owns two bounded state slots: one committed and one
transaction staging slot. QSA capacity is explicit. GR activations, MoE routes
and selected experts are scratch, never stream state. State reset resets both
slots and position.

## Execution

Input/output layout is contiguous `[token,4,hidden]`. `kq_layer_prefill_f32`
accepts one or more tokens; `kq_layer_decode_f32` accepts exactly one. Token IDs
and a PLE row provider are mandatory only for the PLE-GDN family. Input,
output, scratch and borrowed arrays must not alias. Required scratch is queried
for the current state and token count; no silent truncation occurs.

`kq_layer_weights_f32` contains separate attention/MoE GR weights plus the
already-defined family suboperator weight structures. Physical GGUF layouts
are not accepted as canonical F32 arrays. No tensor view is opened implicitly.

Every call is transactional for externally visible persistent state. GDN/QSA
and PLE execute on the inactive slot. Only a completely successful final GR
write swaps it active and advances position. Argument, shape, capacity,
non-finite, provider and downstream operator failures leave the active slot
unchanged.

## Diagnostics and limits

The optional checkpoint observer exposes borrowed layer-boundary values only
during the callback: input, PLE output/enhanced input, both GR read/write
boundaries, mixer input/output, MoE input/output and final four branches.
Suboperator-internal observers remain governed by their own APIs.

`KQ_STATUS_INCOMPATIBLE_LAYER` reports wrong family/ID/GR/subconfig/PLE
semantics. `KQ_STATUS_INVALID_LAYER_STATE` reports foreign or corrupt layer
state. Existing suboperator, buffer, overflow, alias and numeric statuses are
preserved rather than hidden.

This is one-layer scalar reference execution only. It is not a full model
executor and carries no cache placement, prefetch, scheduler or CUDA behavior.
