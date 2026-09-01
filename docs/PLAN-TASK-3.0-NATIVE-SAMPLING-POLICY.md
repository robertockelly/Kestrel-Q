# PLAN-TASK-3.0-NATIVE-SAMPLING-POLICY.md

Status: **NOT STARTED**

## Objective

Implement the first native C17 stochastic selection boundary for
Qwen3.8-Flash-Next. The task consumes a bounded complete F32 logit vector and
produces one token ID from an immutable validated policy plus explicit
caller-owned RNG state.

Task 3.0 does not invoke the model executor or load model weights. It isolates
sampling transformations, probability construction, random-number generation
and categorical selection so each can be independently validated before
sampled generation changes a live model session in Task 3.1.

Implementation may start only from the reviewed Epic 3 inception checkpoint;
the implementer must establish and record the actual clean baseline rather
than substituting this planning iteration's uncommitted state.

## Mandatory characterization before code

Create `docs/SAMPLING-CONTRACT.md` before production algorithm code.

Re-open and pin:

- official Qwen model revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- official `generation_config.json`, 202 bytes, SHA-256
  `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`;
- Transformers revision
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`; and
- the exact executed generation configuration, logits-processor and sampling
  source files/functions with independently verified hashes.

Record exactly:

- score input dtype/shape and complete-vocabulary requirement;
- supported token domain and padded/unused-ID behavior;
- EOG, EOS and pad identities;
- all active and inactive generation-config fields;
- processor construction and execution order;
- temperature division and accepted domain;
- top-k threshold, ties, minimum-token rule and output order;
- top-p sorting, cumulative boundary, ties and minimum-token rule;
- softmax dtype, maximum subtraction, accumulation and normalization;
- non-finite/zero-mass behavior;
- categorical interval convention and boundary behavior;
- selected-ID validation;
- RNG algorithm, state width, seed expansion, output mapping, draw count,
  reset/snapshot/import semantics and integer wrap rules; and
- every unsupported processor or generation option.

The official profile is mandatory:

```text
do_sample  = true
temperature = 1.0
top_k       = 20
top_p       = 0.95
eog_ids     = [248046, 248044]
pad_id      = 248044
```

The characterization must determine the exact supported parameter domain. It
must not silently broaden this model-specific requirement into a generic
Transformers-compatible sampler.

## Source hierarchy

Canonical semantic authority is the pinned official configuration plus the
pinned Transformers execution path. The registered GGUF currently records
matching `general.sampling.temp`, `top_k` and `top_p` values, but that metadata
is derived-artifact compatibility evidence, not authority for processor or RNG
semantics.

Pinned llama.cpp remains the Class-Q model/logit oracle. Its built-in sampler
may be used as a diagnostic, but its selected tokens are exact acceptance
evidence only if Task 3.0 first proves identical processors, floating
arithmetic, RNG algorithm and draw consumption.

## Durable architecture decision

Finalize `docs/adr/0022-native-sampling-policy-boundary.md` only after evidence
supports the choice.

The accepted decision must specify:

- sampler separation from model-logit production;
- immutable config and caller-owned mutable state;
- ownership/lifetime/thread-safety rules;
- exact portable RNG algorithm and state representation;
- reset, clone/snapshot and failure semantics;
- supported policy/profile surface;
- production/oracle dependency boundary; and
- why the selected design remains model-specific and removable.

## Independent oracle first

Generate expected vectors before comparing native output.

Use a pinned offline Class-C tool based on the exact Transformers revision for
logit transforms and categorical probability construction. Use an independent
implementation of the selected published RNG specification for RNG/state and
uniform/categorical mapping vectors. The independent implementation must not
invoke Kestrel-Q or transcribe Kestrel-Q output as expected data.

If the chosen RNG is implemented from a third-party specification or source,
record its license, provenance, version/revision, attribution obligations and
clean-room implementation boundary before production code. Do not copy an
incompatible implementation.

## Evidence namespace

Create deterministic evidence under:

```text
research/sampling/Qwen3.8-Flash-Next/
  de4b8e4d43b917e7706784d8bb445c9af86a3540/
```

Expected files:

- `sampling-contract.json`
- `sampling-calibration.json`
- `sampling-holdout.json`
- `sampling-rng-vectors.json`
- `sampling-statistical.json`
- `sampling-native-validation.json`
- `sampling-manifest.json`

The manifest records all source revisions/hashes, dependency versions/licenses,
generation commands, deterministic asset hashes and comparison classes. Raw
real-model logits are not required and model payload bytes must not be added.

## Comparison contracts

Use separate contracts:

- processor list/order, retained IDs, top-k/top-p membership, RNG state/words
  and final selected ID from a fixed logit vector/state: `EXACT_DISCRETE` or
  `EXACT_BITS`;
- transformed logits, probabilities and cumulative masses:
  per-field `CALIBRATED_FLOAT` with recorded dtype, absolute/relative/ULP
  observations and NaN/Inf policy;
- distributional behavior across repeated draws: a predeclared statistical
  test with fixed corpus, sample count, alpha/error bounds and multiple-test
  policy.

Do not use token-by-token equality as a distributional test. Do not relax a
threshold after observing holdout failure. Keep calibration and holdout
inputs/seeds disjoint.

## Required cases

At minimum include deterministic synthetic cases for:

- official default profile;
- one-token and two-token support;
- top-k threshold and exact ties;
- top-p boundary immediately below, on and above the cutoff;
- equal logits;
- dominant logit;
- negative and large-magnitude finite logits;
- token IDs 0, 248044, 248046, 248076, 248077 and 248319;
- padded/unused IDs according to the characterized contract;
- random variates at interval start, internal boundaries and the largest
  representable value below one;
- repeat/reset/snapshot/import equivalence;
- multiple seeds including zero and maximum accepted seed; and
- official profile plus non-unit-temperature edge cases if the supported API
  exposes temperature overrides.

## Production API

Suggested files:

- `include/kq_sampling.h`
- `src/kq_sampling.c`
- `tests/sampling_test.c`
- `tests/sampling_oracle.c`

Exact layout may differ. The public C17 API must provide the equivalent of:

```text
validated immutable sampling config
+ explicit mutable RNG state
+ F32 logits with explicit count
-> one selected canonical token ID
+ bounded diagnostics
+ updated RNG state on success only
```

Requirements:

- explicit pointer+length/capacity arguments;
- checked arithmetic and bounded allocation/scratch;
- no implicit NUL assumptions;
- no global mutable state;
- deterministic required-size semantics;
- no silent truncation or option fallback;
- caller-visible state unchanged on any failure;
- stable status/diagnostic behavior;
- explicit ownership and lifetime; and
- no model/GGUF owner required by the sampler core.

The sampler must not modify the caller's logits. Diagnostics may expose bounded
retained-count, selected ID, RNG-before/after summary and normalization facts;
they must not dump full production logits by default.

## Greedy compatibility

Task 2.12's `kq_model_exec_greedy_argmax_f32` remains the accepted greedy
contract: complete finite vector, strictly greater replacement and lower ID on
ties. Task 3.0 must not change its results, EOG identities or historical
evidence.

Whether the new API represents greedy as a sampling-policy variant or leaves
it solely in the executor is decided explicitly in ADR 0022. No duplicate
greedy rule may drift from the accepted implementation.

## Fail-closed coverage

Cover at least:

- null/invalid arguments;
- zero or wrong vocabulary size where model-specific validation applies;
- unsupported policy or processor;
- invalid temperature/top-k/top-p;
- non-finite input scores;
- no retained token or non-positive/non-finite probability mass;
- padded/unused or otherwise invalid selected ID according to the contract;
- invalid/corrupted RNG state or unsupported state version;
- count/byte/allocation overflow;
- insufficient output/scratch capacity;
- forbidden aliasing or input mutation;
- RNG-state partial mutation after any downstream failure; and
- host floating-point environment incompatible with the pinned contract.

## Metrics

Record characterization only:

- immutable config bytes;
- RNG state bytes;
- scratch bytes for the full target vocabulary;
- retained candidate count;
- RNG draws per selection; and
- synthetic selection latency.

No performance target or optimization claim is part of Task 3.0.

## Hard boundaries

Task 3.0 must not add:

- model prefill/decode integration or CLI sampling;
- model payload access or weight evidence;
- repetition/frequency/presence penalties unless the pinned official profile
  proves one mandatory (the current source baseline does not);
- stop strings, grammar constraints or arbitrary custom processors;
- batching;
- long-context/session serialization;
- MTP/speculation or vision;
- SIMD or CUDA sampling/model kernels;
- scheduler, cache, prefetch, memory tiering or benchmark execution; or
- a production Python/Transformers/llama.cpp dependency.

`KQ-BACKLOG-BENCH-001` and `KQ-BACKLOG-BENCH-002` remain deferred.

## Documentation and governance

Create during implementation:

- `docs/SAMPLING-CONTRACT.md`
- `docs/NATIVE-SAMPLING.md`
- `docs/KQ-SAMPLING-API.md`

Finalize ADR 0022 and update:

- `docs/ARCHITECTURE.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- `docs/PLAN-EPIC-3-CPU-CORRECTNESS-ENGINE.md`
- `docs/TASKS-EPIC-3-CPU-CORRECTNESS-ENGINE.md`
- this task checklist;
- oracle/provenance documentation; and
- `CHANGELOG.md` in the same iteration.

## Final acceptance

Task 3.0 is COMPLETE/PASS only when:

- mandatory characterization is complete before native algorithm code;
- official default profile is represented exactly;
- portable RNG algorithm/state is pinned and independently vectored;
- native immutable config and explicit state are implemented;
- exact processor/RNG/fixed-input selection vectors pass;
- floating calibration and disjoint holdout pass unchanged;
- predeclared statistical corpus passes;
- all fail-closed and transactional-state tests pass;
- greedy Task 2.12/2.13 regressions remain exact;
- no model payload, executor sampling, self-oracle or production external
  runtime dependency is introduced;
- clean CPU and CUDA Release suites pass with no new Kestrel-Q warning;
- ADR 0022 is ACCEPTED;
- Task 3.1 remains NOT STARTED;
- both backlog benchmarks remain deferred;
- repository safety and `git diff --check` pass; and
- the complete delta is left review-ready without automatic commit/push.
