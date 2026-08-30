# Kestrel-Q MoE API

The public scalar reference interface is declared in `include/kq_moe.h`.

## Lifetime and ownership

`kq_moe_config_create` validates a target model layer and returns an immutable
configuration. The configuration borrows semantic/model descriptors and must
not outlive the model/GGUF/file objects. Close it with
`kq_moe_config_close`. There is no global mutable MoE state and no persistent
per-stream state.

`kq_moe_config_create_reference_f32` validates the same real bindings but
selects the executable F32 scalar-reference contract. The ordinary constructor
describes the released BF16 activation contract and cannot be passed to F32
execution.

## Routing

`kq_moe_route_f32` accepts one hidden vector and caller-owned output arrays for
all router logits/probabilities and the top-k IDs/weights. Capacities are in
elements. Expert IDs/order/count are exact-discrete outputs. No automatic
fallback or approximate routing is permitted.

## Execution

`kq_moe_execute_f32` accepts one or more contiguous hidden vectors, complete
synthetic/reference F32 weights, caller output and reusable scratch. The
operator executes selected routed experts, shared expert/gate and the final
sum. Route and checkpoint observers are synchronous; their pointers are valid
only during the callback.

`kq_moe_execute_routed_expert_f32` is a bounded one-member reference path. The
caller supplies an explicit expert ID, only that member's gate/up/down arrays,
and one-expert workspace. It does not route, accept a complete 512-expert stack
or materialize unselected experts.

Weight fields use canonical row-major mathematical layout:

- router `[experts, hidden]`;
- routed gate/up `[experts, intermediate, hidden]`;
- routed down `[experts, hidden, intermediate]`;
- shared gate/up `[shared_intermediate, hidden]`;
- shared down `[hidden, shared_intermediate]`;
- shared gate weight `[hidden]`.

## Failure behavior

The API checks nulls, model topology, semantic roles, physical ranks/shapes/
types, expert axes, split order, finite inputs/weights, capacities, counts,
checked arithmetic, expert IDs, scratch alignment and writable-buffer aliasing.
Unsupported or near-match structures fail explicitly. Output is not promised
transactional after a floating numeric-domain failure; argument, capacity and
binding validation occurs before execution.

No API opens model payload views automatically. Real expert views remain an
explicit Task 2.2 operation and their lifetime is caller-managed.
