# Qwen3.8-Flash-Next PLE address contract

Status: **CHARACTERIZED FOR TASK 2.4**

This contract fixes the discrete address semantics used by the native PLE
engine. It is derived from the canonical model revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, the official research revision
`69885871a64393807d988b27b1b5e380e8f28526`, and the pinned Apache-2.0
Transformers implementation at
`805a9e939fa8c1bff8d8ffdf041c051b71a914aa`. The executable source file has
SHA-256 `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f`.
Kestrel-Q output is not an oracle.

## Placement and geometry

The model contains one PLE at one-based decoder layer 2, zero-based layer 1,
immediately before that layer's GDN read. Its maximum n-gram order is 3. Orders
2 and 3 each own eight heads, producing 16 address intents per token in this
canonical order:

```text
bigram local heads 0..7, then trigram local heads 0..7
```

Global heads 0..7 are bigrams and 8..15 are trigrams. Each head addresses one
contiguous range in a single logical concatenated table. The 16 active ranges
contain 320,001,446 rows. The embedding is padded to 320,001,536 rows and split
into 128 canonical logical members of 2,500,012 rows each; the final 90 rows are
padding and are never emitted by canonical addressing. A global address maps
to `logical_member = global_address / 2500012` and
`member_row = global_address % 2500012`. Logical member identity is independent
of the fused GGUF tensor and of physical file offsets.

## Canonical constants

- model/token-ID capacity: 248,320 (`0..248319`);
- EOS/history sentinel: 248,044;
- PLE layer index: 1;
- seed: 1,234;
- layer multipliers, in current/one-back/two-back order:
  `23703573157769`, `20109073645365`, `8052911324071`;
- heads per order: 8;
- member count: 128;
- member rows: 2,500,012.

| Global head | Order | Local head | Offset | Prime modulus |
|---:|---:|---:|---:|---:|
| 0 | 2 | 0 | 0 | 20000003 |
| 1 | 2 | 1 | 20000003 | 20000023 |
| 2 | 2 | 2 | 40000026 | 20000033 |
| 3 | 2 | 3 | 60000059 | 20000047 |
| 4 | 2 | 4 | 80000106 | 20000059 |
| 5 | 2 | 5 | 100000165 | 20000063 |
| 6 | 2 | 6 | 120000228 | 20000069 |
| 7 | 2 | 7 | 140000297 | 20000077 |
| 8 | 3 | 0 | 160000374 | 20000081 |
| 9 | 3 | 1 | 180000455 | 20000093 |
| 10 | 3 | 2 | 200000548 | 20000107 |
| 11 | 3 | 3 | 220000655 | 20000147 |
| 12 | 3 | 4 | 240000802 | 20000153 |
| 13 | 3 | 5 | 260000955 | 20000159 |
| 14 | 3 | 6 | 280001114 | 20000161 |
| 15 | 3 | 7 | 300001275 | 20000171 |

The registered semantic model must expose these three metadata-derived arrays
exactly: multipliers, head offsets, and head vocabulary sizes. A near match is
not compatible.

## History and segment semantics

Address state stores exactly the two most recent raw token IDs plus the next
absolute sequence position. Reset initializes both history slots to EOS and
position to zero. For a current token `t`:

- `x0 = t`;
- `x1` is the immediately preceding token unless it is absent or EOS, in which
  case `x1 = EOS`;
- `x2` is the token two positions back only when neither intervening history
  token crosses an EOS boundary; otherwise `x2 = EOS`.

The current EOS token still uses the preceding segment's history. After it is
committed, the next token observes an EOS boundary. Beginning of sequence and
missing history use the same EOS sentinel rule. State then shifts the raw
history to `[old newest, current]` and advances position by one.

## Exact integer arithmetic

For order `n` in `{2,3}`:

```text
mixed = uint64(x0) * multiplier[0]
for shift = 1 .. n-1:
    mixed = mixed XOR (uint64(x_shift) * multiplier[shift])

head_local = 0 .. 7
head_global = (n - 2) * 8 + head_local
head_row = mixed modulo head_vocab_size[head_global]
global_address = head_offset[head_global] + head_row
```

The canonical implementation performs signed Torch `int64` products and XOR.
The multiplier construction bounds every valid product below `INT64_MAX`; the
largest supported products are respectively 5,886,047,582,964,040,311,
4,993,465,058,543,391,435, and 1,999,690,887,081,986,649. Consequently every
operand and XOR result is non-negative and fits signed 64 bits. Kestrel-Q uses
checked `uint64_t` multiplication, verifies the signed bound, applies XOR in the
same order, and then uses ordinary positive unsigned remainder. It never relies
on signed overflow, implementation-defined casts, or wraparound.

Every derived head row must be below its prime, every global address below
320,001,446, every logical member below 128, and every member row below
2,500,012.

## Prefill and incremental transition

Prefill applies the same one-token transition in original sequence order and
emits 16 intents for each token without I/O-oriented reordering. Chunked prefill
continues from the supplied state and must equal one-shot processing.

Incremental decode consumes the saved two-token history and the newly selected
input token, emits that token's 16 intents at the current position, and commits
the same shift/position transition. Failed capacity, input, state, or arithmetic
validation exposes no partial state advance.

## Address state versus later value state

Address generation requires only two token IDs and position bookkeeping. The
nine-position dilation-3, width-4 convolution tail, retrieved 16-by-160 values,
projection/gating activations, and four residual branches belong to later PLE
value computation. They are deliberately absent from Task 2.4 state.

Task 2.4 opens no PLE tensor view and performs no lookup, disk access, cache,
prefetch, batching, eviction, scheduling, tensor math, or inference.
`KQ-BACKLOG-BENCH-002` remains deferred before any final storage policy.
