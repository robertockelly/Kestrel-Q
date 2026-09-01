# PLAN-TASK-3.1-SAMPLED-INCREMENTAL-GENERATION.md

Status: **COMPLETE / PASS**

## Objective

Integrate the accepted Task 3.0 native sampling policy with the accepted Epic 2 incremental model executor and prove sampled autoregressive generation over the real Qwen3.8-Flash-Next GGUF.

Task 3.1 is the final governed task of Epic 3.

Required flow:

```text
prompt IDs
-> native prefill exactly once
-> logits
-> Task 3.0 sampler + caller-owned RNG
-> sampled token
-> incremental native decode of that token
-> next logits
-> next sample
```

After prefill, every generation step transactionally combines model state and RNG state. On failure neither may advance.

## Entry gate

Published baseline:
`9986b934d639b61575018653dfc8a4eabe2965ea`

Require:
- Task 3.0 COMPLETE/PASS
- ADR 0022 ACCEPTED
- Task 3.1 NOT STARTED
- Epic 3 IN PROGRESS
- working tree clean
- backlog benchmarks remain DEFERRED

## Frozen Task 3.0 contract

Do not reopen unless an evidence-backed defect is found:

- temperature 1.0
- top-k 20
- top-p 0.95
- processor order: temperature -> top-k -> top-p -> F32 softmax -> categorical draw
- PCG-XSH-RR 64/32
- caller-owned 56-byte RNG state
- uniform = uint32 / 2^32
- EOG IDs 248044/248046 eligible
- padded IDs 248077..248319 participate but selection fails transactionally

## Sampled-generation contract

Create `docs/SAMPLED-GENERATION-CONTRACT.md` and pin:
- prefill/first-draw lifecycle
- sampled-token/decode lifecycle
- position progression
- RNG draw progression
- combined model/RNG transaction
- EOG stop point
- max-token/context stop
- invalid sampled-token behavior
- output visibility vs commit
- retry/replay
- caller ownership/lifetime

## State machine

### Prefill
1. prefill prompt once
2. produce final-prompt logits
3. sample first token
4. commit RNG state only on successful selection
5. return first sampled token

Model state after successful prefill represents the prompt-prefilled state.

### Incremental sampled step
1. stage model state
2. stage RNG state
3. decode previous accepted sampled token
4. produce logits
5. sample next token with Task 3.0
6. validate token
7. commit model + RNG state together
8. return next token

On failure: model state, RNG state and output are unchanged; retry is deterministic.

## EOG

- EOG remains selectable.
- If first sampled token is EOG, stop without decoding it.
- If later sampled token is EOG, commit successful decode of the previous token plus successful RNG draw, return EOG, then stop.
- No decode of EOG and no extra RNG draw after stop.
- Add synthetic/control coverage if real traces do not hit EOG.

## Padded IDs

If Task 3.0 selects 248077..248319:
- sampled step returns INVALID_TOKEN_ID
- model state rolls back
- RNG state rolls back
- output/prefix unchanged

Do not mask padded IDs in Task 3.1.

## Real traces

Use real GGUF and:
`KQ-PROMPT-001 — Hello, Kestrel-Q.`

Canonical IDs:
`[9419, 11, 710, 467, 3621, 27325, 13]`

Official profile only for governed real traces.

Use:
- one primary explicit seed/stream trace
- one disjoint holdout seed/stream trace
- deterministic replay of at least the primary trace

Generate up to 4 sampled tokens or stop on EOG.

Prefer context capacity 16 unless evidence requires another bounded value.

## Independent integration oracle

Do not require sampled-token parity with llama.cpp unless RNG/processor/FP semantics are proven identical.

For every real sampled step:
1. capture exact native full logits temporarily
2. process those logits with a separate Task 3.0 reference implementation and exact pre-step RNG state
3. compare survivor/order hash, RNG word, selected token and post-step RNG state
4. do not commit full logits

llama.cpp remains optional/diagnostic for forced-prefix model/logit comparison.

This is not a self-oracle: model logits are integration inputs; expected sampling is computed independently.

## Exact gates

Per step require exact:
- pre/post RNG state
- consumed RNG word
- survivor count/hash/order
- selected token
- model position

Preserve existing exact QSA/MoE/PLE invariants.

## Transaction injection

Inject failures:
- early decode layer
- middle layer
- late layer
- after decode / before sampler commit
- padded-ID sampler failure
- sampler failure where injectable

Require combined rollback and deterministic retry. At least one failure occurs after successful sampled tokens.

## Prompt replay

For N generated tokens without EOG:
- prefill count = 1
- sample draws = N
- incremental decodes = N-1
- prompt replay = 0

## State evidence

Per successful sampled token record compact hashes/summaries:
- structural/model state
- GDN
- QSA + sequence length
- PLE address
- PLE value
- RNG state + draw count
- model position
- stopped flag

## Greedy preservation

Historical evidence remains untouched:
- M1 token 271
- Task 2.13 sequence `[271, 248068, 198, 760]`

Greedy API/CLI semantics remain unchanged.

## Native API and CLI

Extend model executor with a narrow sampled-generation integration using Task 3.0 directly.

Suggested concepts:
- `kq_model_sampled_prefill`
- `kq_model_sampled_decode_one`

or equivalent explicit stateful API.

Minimal CLI:
```text
kq-run <model.gguf> --prompt "Hello, Kestrel-Q." --max-new-tokens 4 --sample --seed <u64> [--stream <u63>]
```

No wall-clock seeding. `--greedy` remains unchanged. Overrides are optional and only within Task 3.0's validated domain.

## Evidence

Create bounded governed evidence, e.g.:
- sampled-generation-contract.json
- sampled-generation-oracle.json
- sampled-generation-native.json
- sampled-generation-state.json
- sampled-generation-validation.json
- sampled-generation-manifest.json

Record baseline, GGUF, prompt, profile, seed/stream, tokens/fragments, per-step independent result, state summaries, lower-level invariants, stop reason, payload/memory/timing and hashes.

Do not commit temporary full logits.

## Payload/memory

Record prefill/per-decode/total logical payload and blocks, semantics, expert requests, PLE rows, LM-head bytes.

Set an evidence-backed correctness ceiling before final acceptance. Do not hide prompt replay/redundancy with a loose ceiling.

Record:
- model state
- RNG/generation state
- transaction staging
- QSA capacity
- peak model scratch
- sampling scratch
- logits
- max simultaneous F32 weights

No complete F32 target weight matrix.

## Fail closed

Cover:
- missing sampler config
- corrupt RNG
- invalid continuation
- padded-ID selection
- processor/scratch failure
- early/mid/late provider failure
- combined rollback
- context/position overflow
- EOG stopped-state misuse
- invalid CLI args / missing seed / mode conflict

## Hard boundaries

Do not implement:
- new RNG or processors
- repetition/frequency penalties
- beam/Mirostat
- MTP/speculation
- vision
- batching
- long-context production
- session serialization
- server/UI
- scheduler/cache/prefetch
- memory tiering
- SIMD/CUDA model kernels

Backlog benchmarks remain deferred.

## Documentation

Create:
- `docs/SAMPLED-GENERATION-CONTRACT.md`
- `docs/NATIVE-SAMPLED-GENERATION.md`
- `docs/adr/0023-sampled-generation-state-transaction.md`

Update model-exec API/reference, architecture, runtime state, goldens, Epic 3 plan/tasks, TASKS, ROADMAP, tooling provenance and CHANGELOG.

## ADR 0023

Decision should freeze:
1. Task 3.0 owns sampler math/RNG
2. executor consumes sampler API
3. prefill once
4. later sampled step stages model+RNG together
5. failure rolls both back and preserves prefix
6. EOG is selectable; integration stops
7. padded-ID selection fails transactionally
8. explicit seed/stream replay is deterministic
9. independent oracle processes identical recorded logits
10. performance/CUDA/scheduler remain out of scope

Accept only after real primary/holdout traces, replay, EOG and rollback gates pass.

## Validation

Run clean CPU and CUDA Release suites.
Require all Task 2.x, Task 3.0 and Task 3.1 tests PASS, frozen greedy evidence unchanged, no new /W4 warnings, repository safety PASS, `git diff --check` PASS.

## Epic 3 closure

Only if every Task 3.1 gate passes:
- `TASK_3_1 = COMPLETE/PASS`
- `EPIC_3_CPU_CORRECTNESS_ENGINE = COMPLETE/PASS`
- ADR 0023 ACCEPTED

Do not start Epic 4.

Do not commit or push automatically.
