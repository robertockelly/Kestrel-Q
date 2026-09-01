# Kestrel-Q sampling API

The public C17 API is declared in `include/kq_sampling.h`.

## Configuration

`kq_sampling_policy_qwen38_default` initializes the exact official policy.
`kq_sampling_config_open_qwen38` validates and copies that policy into an opaque
immutable config. The caller closes it with `kq_sampling_config_close`.

The config has no model/GGUF owner and may be shared for concurrent read-only
use. Its policy surface is only temperature, top-k and top-p. Version, flags
and reserved fields fail closed when unsupported. Query functions report zero
for an invalid object.

`kq_sampling_required_scratch_bytes` returns the complete caller-owned scratch
requirement independent of the scratch pointer's alignment. No selection
allocates heap memory or truncates output.

## RNG state

`kq_sampling_rng_seed` creates state-format v1 from a 64-bit seed and a stream
in `0..2^63-1`. `KQ_SAMPLING_DEFAULT_STREAM` is the governed default.

- `kq_sampling_rng_reset` repeats the original seed expansion;
- `kq_sampling_rng_snapshot` exports semantic state;
- `kq_sampling_rng_import` validates and imports a snapshot;
- `kq_sampling_rng_next_u32` exposes the exact PCG word path for deterministic
  tooling and advances only on success.

The state is caller-owned mutable data. It must not be mutated concurrently.
Version, reserved fields, stream-derived increment and integrity are validated
before use. The integrity word detects accidental field corruption; it is not
a security authentication mechanism.

## Selection

`kq_sampling_select_f32` requires:

- one valid immutable config;
- one valid caller-owned RNG state;
- exactly 248,320 finite F32 logits;
- the queried scratch capacity; and
- one caller result object.

Logits, scratch, state, result and config storage may not overlap. The input
logits remain unchanged. On success, the result reports selected ID, top-k and
final retained counts, selected/maximum probability, normalized F32 sum,
uniform value, PCG word and draw counts. These diagnostics are bounded and do
not expose the full logit vector.

On every failure, caller RNG and result remain unchanged. Scratch contents are
unspecified. A selected model-capacity padding ID returns
`KQ_STATUS_INVALID_TOKEN_ID`; EOG IDs are returned normally.

## Lifetime and scope

The config must outlive calls that borrow it. RNG, logits, scratch and result
need only remain valid for the call. There is no hidden global sampler state.

The API does not represent greedy mode and does not duplicate
`kq_model_exec_greedy_argmax_f32`. It does not provide tokenizer, stopping,
model-session or CLI behavior. Task 3.1 must combine model and RNG rollback as
one whole-token transaction.
