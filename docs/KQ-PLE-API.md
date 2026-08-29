# Kestrel-Q PLE address API

Status: **TASK 2.4 COMPLETE / PASS**

The public C17 interface is declared in `include/kq_ple.h`. It is a
model-specific discrete address API, not a PLE tensor or inference API.

## Ownership and lifetime

- `kq_ple_config_open_from_model` allocates an immutable PLE config owned by
  the caller; release it with `kq_ple_config_close`.
- Construction copies all address constants needed at runtime. The config does
  not retain the model or GGUF, so it may outlive them.
- `kq_ple_stream_state` is caller-owned, explicit and resettable. Its integrity
  is bound to the complete canonical config values, so incompatible-config and
  corrupted state is detected. Separately constructed byte-identical canonical
  configs are intentionally state-compatible.
- Address intents are caller-owned output values and borrow no pointers.
- Configs have no mutation API and there is no global mutable PLE state.

## Configuration

```c
kq_status kq_ple_config_open_from_model(
    const kq_model *model,
    kq_ple_config **out_config,
    kq_diagnostic *diagnostic);
```

The model must remain valid during the call. Exact semantic-registry topology,
fused-member geometry and metadata-derived arrays are checked before success.
`kq_ple_config_get_info` exposes canonical geometry;
`kq_ple_config_get_metrics` exposes construction time, owned bytes and state
size as characterization values.

## State

```c
kq_status kq_ple_state_reset(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    kq_diagnostic *diagnostic);

kq_status kq_ple_state_validate(
    const kq_ple_config *config,
    const kq_ple_stream_state *state,
    kq_diagnostic *diagnostic);
```

Reset establishes position zero and two EOS history sentinels. Callers must
not edit state fields. State is rejected for version, reserved-field, token,
position, beginning-of-stream or integrity mismatch.

## Prefill and decode

```c
kq_status kq_ple_generate_prefill(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    const uint32_t *token_ids,
    uint64_t token_count,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic);

kq_status kq_ple_generate_decode_step(
    const kq_ple_config *config,
    kq_ple_stream_state *state,
    uint32_t token_id,
    kq_ple_address_intent *intents,
    uint64_t intent_capacity,
    uint64_t *required_intents,
    kq_ple_run_metrics *metrics,
    kq_diagnostic *diagnostic);
```

The exact requirement is `token_count * 16`, checked for overflow. A
required-size query uses `intents == NULL` and capacity zero; non-empty input
returns `KQ_STATUS_BUFFER_TOO_SMALL` plus the exact count without advancing
state. Any insufficient-capacity, invalid-token, invalid-state or limit error
is transactional. Empty `(NULL, 0)` input is a valid no-op.

Tokens must be below the model vocabulary capacity, including canonical model
IDs that are not emitted by the tokenizer. Prefill may advance up to the
262,144-token model context. Decode is the one-token form of the same canonical
transition.

## Intent fields

`kq_ple_address_intent` contains:

- absolute sequence `position` and current `token_id`;
- `ngram_order` 2 or 3;
- `local_head` 0–7 and `global_head` 0–15;
- `head_offset`, `head_vocab_size` and `global_address`;
- canonical `logical_member` 0–127 and bounded `member_row` 0–2,500,011.

Consumers must preserve the emitted bigram-then-trigram/head order unless a
later separately governed layer explicitly transforms it. No intent contains
a physical tensor pointer, GGUF offset, cache key or scheduler decision.

## Stable failure modes

Invalid arguments, output capacity, token ID, arithmetic/context limits,
corrupted state and incompatible model configuration have distinct existing
`kq_status` values. Unsupported/contradictory PLE semantics return
`KQ_STATUS_INCOMPATIBLE_PLE`; state validation failures return
`KQ_STATUS_INVALID_PLE_STATE`. No fallback or reinterpretation occurs.

Passing a state or config after its caller-managed lifetime ends is ordinary C
undefined behavior and cannot be detected safely; callers must close configs
only after all concurrent/synchronous API use has ended.
