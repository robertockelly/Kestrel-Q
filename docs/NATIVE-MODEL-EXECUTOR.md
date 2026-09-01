# Native model executor

Status: **TASK 3.1 COMPLETE / PASS**

The Task 2.12 executor is the first production C17 path that composes all
accepted target components into one real-model operation:

```text
UTF-8 -> native tokenizer -> bounded embedding rows
       -> canonical layers 0..47 -> final hyper-connection mixer
       -> bounded LM-head row dots -> 248320 F32 logits
       -> stable greedy argmax -> native decode
```

It is target-specific, batch 1 and text-only. Greedy remains accepted and Task
3.1 adds the separate accepted Task 3.0 sampled-selection boundary. It contains no new
GDN, QSA, MoE, PLE or GR equations; those remain owned by the Task 2.6–2.10
modules. The GGUF is accessed only through the semantic provider and bounded
views. No complete target matrix is materialized as F32.

## Configuration and ownership

`kq_model_exec_config` borrows one verified GGUF, semantic model, tokenizer and
weight provider. Construction requires the exact 2,560 hidden width, 248,320
model vocabulary, 48-layer 36-GDN/12-QSA topology, Q8_0 embedding/head and
the three final-mixer tensors. It constructs every immutable layer config in
ascending order and rejects any missing, reordered or incompatible binding.
All borrowed owners outlive config, state and calls.

`kq_model_exec_state` owns one Task 2.10 state per layer and a public position.
M1 explicitly uses context capacity 8; Task 2.13 uses capacity 16 for a
seven-token prompt plus up to four returned tokens. No 262,144-token cache is allocated.
A failed operation leaves the public position and caller outputs unchanged and
marks private state for reset. The next attempt resets all layers before work.
Real injected failures at layers 0, 24 and 47 each recover to token 271. The
incremental API uses a stronger whole-model per-token rollback: earlier layer
commits flip back immediately if a later layer/final stage fails, preserving a
valid generated prefix without resetting it.

## Bounded execution

One embedding row is decoded per input token. The layer executor receives the
four repeated residual branches and runs layer IDs `0..47` without selection
or reordering. The final mixer uses epsilon `1e-6`; its converter-folded F32
gamma is restored to canonical zero-centred delta once in the provider. The
LM head consumes only the final prompt position and computes every vocabulary
logit without holding an F32 copy of its matrix.

The successful M1 run used 8,693,472 scratch bytes, a 993,280-byte logits
buffer and 359,881,768 bytes of owned model/layer state. Task 2.10 layer state
includes two transactional suboperator slots and bounded copy workspace; this
container allocation is distinct from semantic state. The target semantic
baseline remains 35 ordinary GDN states, one PLE-GDN state (including 32-byte
address history and 184,320-byte released BF16 value state) and 12 QSA states
at 2,304 bytes per retained token per layer.

## Payload accounting

The seven-token scalar run recorded 40,208,768,960 logical packed bytes,
1,693,222,880 quantized blocks, 1,239 unique semantics, 19,040 embedding
bytes, 3,360 routed selections / 10,080 selected-expert matrix requests, 112
PLE rows and 675,430,400 LM-head bytes. This is below the governed 64 GiB
correctness ceiling. The 24 GiB planning suggestion was raised only after the
dirty run proved that the correctness provider rereads selected weights for
each of seven prompt tokens.

Logical touches are not physical disk I/O, page-cache traffic or storage
throughput. No performance conclusion follows. Cache, prefetch, SIMD, CUDA
model kernels and scheduler policy remain deferred.

## Incremental extension

Task 2.13 preserves the M1 path and adds direct one-token decode. The governed
four-token result is `[271, 248068, 198, 760]`; prefill runs once and three
decode calls advance position 7 -> 8 -> 9 -> 10. See
`MULTI-TOKEN-DECODE-CONTRACT.md` and `NATIVE-MULTI-TOKEN-GENERATION.md` for the
state, transaction, payload and evidence contracts.

Task 3.1 adds sampled prefill/decode entry points without changing the model
arithmetic or greedy entry points. Each sampled call snapshots the explicit
caller-owned PCG state, runs the existing model transaction, delegates the
complete logits to `kq_sampling`, validates/decodes the selected canonical ID
and commits model state, RNG and outputs together. Sampling or token-domain
failure rolls model and RNG back together. See `SAMPLED-GENERATION-CONTRACT.md`
and `NATIVE-SAMPLED-GENERATION.md`.

## CLI

The strict milestone command is:

```powershell
kq-run $env:KQ_GGUF_PATH --prompt "Hello, Kestrel-Q." --max-new-tokens 1 --greedy
```

The CLI accepts `--max-new-tokens 2..4` with context 16. It reports
bounded phase/layer and per-token progress, not activations or weights. Zero,
larger counts, chat options, vision and MTP are rejected by the command shape.
`--greedy` is unchanged. Task 3.1 additionally accepts `--sample --seed <u64>`
and an optional governed `--stream <u63>`; it never derives a seed from wall
clock or process-global state.
