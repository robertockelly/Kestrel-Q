# PLAN-TASK-2.13-MULTI-TOKEN-GREEDY-DECODE.md

Status: **COMPLETE / PASS**

## Objective

Extend the published M1 one-token native execution into a correctness-first,
multi-token incremental greedy generation path over the real
Qwen3.8-Flash-Next quantized GGUF.

Baseline:
`1f2b967d6bf5ec7aaa263192462ede1eb5c9bd28`

Required path after one prompt prefill:

```text
selected token
-> bounded real embedding row
-> incremental layers 0..47
-> final norm
-> real LM head/logits
-> greedy argmax
-> native decode
-> persistent state advance
```

Task 2.13 remains scalar CPU correctness work.

Out of scope:
- stochastic sampling;
- speculative/MTP;
- vision;
- batching;
- scheduler/cache/prefetch optimization;
- SIMD/CUDA model kernels;
- server/UI.

`KQ-BACKLOG-BENCH-002` remains DEFERRED.

## Mandatory contract

Create `docs/MULTI-TOKEN-DECODE-CONTRACT.md`.

Pin:
- prompt prefill occurs exactly once;
- generated IDs feed incremental decode directly;
- no full prompt replay;
- exact position progression;
- GDN/QSA/PLE state carry-forward;
- EOG stop semantics;
- max-new-token stop semantics;
- greedy argmax semantics;
- incremental tokenizer decode behavior;
- one transaction boundary per generated token;
- rollback behavior after a later decode failure.

## Governed oracle

Use the existing pinned full-model Class-Q oracle:

`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`

Use the same GGUF and explicit canonical IDs.

Prompt:
`KQ-PROMPT-001` — `Hello, Kestrel-Q.`

Canonical prompt IDs:
`[9419, 11, 710, 467, 3621, 27325, 13]`

Generate up to **4 greedy tokens total**, or stop earlier if the canonical oracle
returns EOG.

Freeze the independent oracle token sequence before native validation.

The first generated token must remain M1 token `271`.

## Context

Use an explicit small context capacity sufficient for 7 prompt tokens plus up
to 4 generated tokens.

Preferred:
`context_capacity = 16`

Do not allocate 262144-token QSA state.

## Native API

Extend the native model executor with explicit prefill and repeated one-token
decode semantics, or an equivalent narrow greedy-generation API.

Requirements:
- prompt prefill exactly once;
- `max_new_tokens`;
- deterministic greedy only;
- EOG stop;
- explicit model position;
- no hidden global session state.

## Per-token transaction

If generated token step N fails:
- state after successful token N-1 remains valid;
- the failed step leaves no partial state advance;
- retry from the pre-step state reproduces the independent oracle token.

Inject failures during later decode steps.

## No prompt replay

Instrument and test that:
- prompt embedding/layer prefill occurs once;
- subsequent tokens use incremental decode;
- prompt tokens are not rerun through 48 layers.

Prompt replay is a hard failure.

## State continuity

Record compact summaries/hashes per decode step for:
- model position;
- GDN states;
- QSA sequence length/state;
- PLE address state;
- PLE value state.

Do not commit huge raw state blobs.

## Exact lower-level invariants

QSA:
- exact complete-block/tail behavior;
- exact selected blocks/tokens where applicable.

MoE:
- exact top-10 expert IDs/order;
- no unselected expert access.

PLE:
- exactly 16 logical intents/token;
- no extra/speculative rows.

## Primary acceptance gate

`native generated token ID sequence == independent oracle token ID sequence`

This is `EXACT_DISCRETE`.

A later mismatch means Task 2.13 is not PASS even if M1 token 271 still matches.

## Stronger diagnostics

Where available per generated token record:
- oracle/native greedy-token logit;
- runner-up ID/logit;
- top-N diagnostic IDs.

Do not require identical full-logit ranking unless a governed contract supports
it.

## Payload accounting

Record separately:
- prompt prefill logical payload;
- each decode-token logical payload;
- total logical payload;
- blocks;
- unique semantics;
- expert requests;
- PLE rows.

These remain logical counters, not physical I/O or throughput.

Freeze a task-specific correctness ceiling before final run. Suggested absolute
ceiling: `<= 96 GiB logical packed bytes`, preferably tighter if the M1-derived
estimate supports it.

## Memory

With small context capacity record:
- owned model state;
- semantic state;
- transaction overhead;
- QSA capacity bytes;
- peak scratch;
- logits;
- max simultaneous F32 weight materialization.

No complete target matrix may be materialized in F32.

## CLI

Extend the minimal CLI with `--max-new-tokens N`.

Example:
`kq-run <model.gguf> --prompt "Hello, Kestrel-Q." --max-new-tokens 4 --greedy`

Requirements:
- prompt tokenized once;
- per-token progress;
- greedy only;
- EOG stop;
- no massive model/activation dumps.

## Evidence

Create a governed evidence set, e.g.:
- `multi-token-contract.json`
- `multi-token-oracle.json`
- `multi-token-native.json`
- `multi-token-state.json`
- `multi-token-validation.json`
- `multi-token-manifest.json`

Record prompt identity, canonical input IDs, context capacity, oracle/native
token sequences, decoded fragments, per-step equality, state summaries,
QSA/MoE/PLE invariants, payload/memory/timing and deterministic hashes.

Do not alter historical M1 evidence.

## Fail-closed coverage

Cover:
- `max_new_tokens=0`;
- `max_new_tokens=1` preserves M1 token 271 / `\n\n`;
- multi-token continuation;
- EOG early stop;
- context exhaustion;
- invalid continuation state;
- position/QSA-capacity overflow;
- later-step provider failure rollback;
- retry after failure;
- invalid selected token/decode failure;
- prompt-replay regression;
- unexpected state reset.

## Documentation

Create:
- `docs/MULTI-TOKEN-DECODE-CONTRACT.md`
- `docs/NATIVE-MULTI-TOKEN-GENERATION.md`
- `docs/adr/0021-native-multi-token-greedy-generation.md`

Update:
- `docs/KQ-MODEL-EXEC-API.md`
- `docs/NATIVE-MODEL-EXECUTOR.md`
- `docs/ARCHITECTURE.md`
- `docs/MODEL-RUNTIME-STATE.md`
- `docs/GOLDEN-VECTORS.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- Epic 2 plan/status
- Task 2.13 checklist
- tools provenance
- `CHANGELOG.md`

Do not invent an Epic 2 closure rule.

## Acceptance

Task 2.13 COMPLETE/PASS requires:
- prefill exactly once;
- true incremental native decode;
- independent oracle continuation generated first;
- entire native token sequence exactly equals oracle;
- M1 one-token backward compatibility;
- model/GDN/QSA/PLE state continuity PASS;
- QSA boundary semantics PASS;
- MoE/PLE exact access invariants PASS;
- failed-step rollback/retry PASS;
- EOG/max-token/context semantics PASS;
- payload/memory accounting complete;
- no full F32 target matrix;
- CPU/CUDA clean regressions PASS;
- ADR 0021 ACCEPTED;
- Epic 2 status follows existing roadmap;
- KQ-BACKLOG-BENCH-002 DEFERRED;
- repository safety PASS;
- `git diff --check` PASS.

Do not commit or push automatically.
