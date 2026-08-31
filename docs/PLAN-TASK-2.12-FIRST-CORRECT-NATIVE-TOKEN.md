# PLAN-TASK-2.12-FIRST-CORRECT-NATIVE-TOKEN.md

## Objective

Produce the first independently verified next token from the real
Qwen3.8-Flash-Next GGUF using only the native Kestrel-Q production runtime.

Milestone:

> **M1 — First Correct Native Token**

Required path:

```text
UTF-8 prompt
-> native tokenizer
-> real quantized token embedding
-> native 48-layer target executor
-> final model norm
-> real quantized LM head
-> logits
-> greedy argmax
-> native tokenizer decode
```

The first run is correctness-first, batch 1, short context and one generated
token.

No performance target is attached to M1.

---

## Entry gate

Task 2.12 starts only after Task 2.11 has been checkpointed and pushed.

Expected prior state:

- Task 2.0–2.11 COMPLETE/PASS;
- ADR 0019 ACCEPTED;
- 48/48 target layers provider-compatible;
- representative real GDN/QSA/PLE-GDN layers execute against the quantized
  GGUF and match the independent target oracle;
- `FIRST-TOKEN-READINESS.md` contains no unexplained blocker;
- Task 2.12 NOT STARTED;
- `KQ-BACKLOG-BENCH-002` DEFERRED.

---

# 1. Mandatory model entry/output characterization

Before model-executor implementation, create:

```text
docs/MODEL-EXECUTION-CONTRACT.md
```

Re-open the pinned canonical Qwen/Transformers implementation and pin exactly:

- token embedding semantic tensor and physical binding;
- embedding lookup orientation;
- embedding output dtype;
- any embedding scaling;
- whether input/output embeddings are tied;
- final normalization class/equation/epsilon;
- final norm semantic tensor/shape;
- LM-head semantic tensor/shape/orientation;
- LM-head dtype/accumulation contract;
- logits dtype;
- whether logits use the final token only;
- causal/prefill position progression;
- exact 48-layer iteration order;
- layer-state initialization;
- per-layer QSA state capacity semantics;
- GDN fixed-state initialization;
- PLE address/value state initialization;
- final greedy argmax/tie behavior;
- token decode options after selection;
- BOS/EOS behavior relevant to raw prompt generation;
- end-of-generation behavior if the selected token is special/EOS.

Do not infer entry/output semantics from older Qwen models.

---

# 2. Canonical target facts to revalidate

Previously established leads:

- hidden size = 2560;
- vocabulary/model capacity = 248320;
- 48 text layers;
- 35 ordinary GDN + 12 QSA + 1 PLE-GDN;
- token embedding exists as a distinct target tensor;
- LM head exists as a distinct target tensor;
- both embedding and LM-head packed footprint leads are ~675,430,400 bytes;
- initial text runtime is not using vision or MTP.

All facts used by production execution must be revalidated by the model
execution contract and semantic registry.

---

# 3. Native model-executor architecture

Suggested production modules:

```text
include/
  kq_model_exec.h

src/
  kq_model_exec.c
  kq_model_exec_qwen38.c

src/
  run_main.c

tests/
  model_exec_test.c
  first_token_integration.c
```

Exact names may differ.

The executor composes existing APIs; it must not copy operator equations.

---

# 4. Immutable model execution config

Create an immutable target-execution configuration containing/referencing:

- tokenizer;
- target weight provider;
- embedding descriptor;
- 48 immutable layer configs;
- final norm descriptor;
- LM-head descriptor;
- model dimensions;
- configured runtime context capacity.

Construction must:

- validate all 48 layers;
- validate embedding/final norm/LM head;
- validate no vision/MTP dependency;
- fail closed on missing/near-match semantics;
- copy no full model weights.

---

# 5. Global runtime state

Create explicit batch-1 model state composed from the already defined
per-layer persistent state.

Expected categories:

- 36 GDN-family fixed states, including the PLE layer's GDN state;
- 12 QSA states with an explicitly bounded token capacity;
- one PLE address state;
- one PLE value state;
- current sequence/token position.

Do not create duplicate copies of state already owned by layer state.

Requirements:

- deterministic zero/reset;
- explicit context capacity;
- state-size query;
- no allocation for the full 262144 context during M1;
- transactional prefill/model-step behavior;
- no hidden global state.

---

# 6. M1 context capacity

The First Correct Native Token run must use a deliberately small configured
context capacity sufficient only for the chosen test prompt plus generated
token(s).

Recommended M1 capacity:

```text
8 or 16 tokens
```

Choose the smallest safe capacity supported by the selected prompt/test.

Do not allocate QSA cache for 262144 tokens merely because the model supports
that context length.

The public API should remain capable of accepting a larger explicit capacity
later, subject to checked allocation.

---

# 7. Native token embedding lookup

Implement the actual target embedding lookup through the Task 2.11 weight
provider / Task 2.2 bounded view / Task 2.5 decode path.

Required:

```text
token_id
-> one embedding row
-> F32 hidden[2560]
```

No whole embedding matrix materialization.

Fail closed on:

- token ID outside canonical tokenizer/model input range;
- padded/unused tokenizer ID when not valid canonical input;
- wrong embedding shape/type/layout;
- provider failure;
- insufficient output capacity.

---

# 8. 48-layer execution loop

Implement exact sequential execution:

```text
layer 0
layer 1
...
layer 47
```

using Task 2.10 layer APIs and Task 2.11 real quantized provider.

Requirements:

- no layer skipped/reordered;
- canonical layer family already validated;
- exact token position passed to stateful suboperators;
- explicit prefill semantics;
- transactional state handling across the complete model call;
- no whole-model weight materialization.

Task 2.12 must not reimplement GDN/QSA/MoE/PLE equations.

---

# 9. Model-level transactional state

A failure in any of 48 layers must not leave the externally visible model state
partially advanced.

Define and test a model-level transactional strategy.

For M1's small context, staging/copying state is acceptable if bounded and
documented.

Inject at least:

- an early-layer provider failure;
- a middle-layer failure;
- a late-layer failure.

Require reset/control-stream equivalence after failure.

---

# 10. Final model normalization

Implement the exact canonical final norm using Task 2.5 scalar numerics where
possible.

Validate:

- weight shape;
- epsilon;
- input/output dtype;
- finite-input policy;
- aliasing contract.

No duplicate generic RMSNorm implementation if Task 2.5 already provides the
required primitive.

---

# 11. LM head / logits

Compute all `248320` target logits from the final hidden vector through the
real quantized LM-head tensor.

Requirements:

- bounded row/chunk access;
- no full LM-head F32 materialization;
- full logits buffer is allowed (~1 MiB F32);
- checked vocabulary dimension;
- deterministic accumulation;
- Task 2.5 numeric path reused;
- no output-logit truncation before argmax.

Record LM-head logical packed bytes touched separately.

---

# 12. Greedy argmax

Characterize and implement exact greedy selection:

```text
argmax(logits)
```

Tie behavior must match the canonical reference, expected to be deterministic
lower/first index only if the pinned source/API proves this.

Greedy selection is EXACT_DISCRETE.

No temperature, top-p, top-k sampling or RNG in M1.

---

# 13. Native token decode

Decode the selected ID through Task 2.3 native tokenizer.

Report:

- selected token ID;
- decoded UTF-8 bytes/text;
- whether token is canonical special/EOS.

Do not use external tokenizer code in production.

---

# 14. M1 prompt selection

Use an existing committed Task 1.4 synthetic prompt/tokenizer golden wherever
possible.

Select deterministically the **shortest suitable raw-text golden prompt** to
minimize scalar reference cost.

Requirements:

- native tokenization exact-match with the committed canonical tokenizer
  golden;
- no chat-template dependency for the first raw-text M1 unless the canonical
  test deliberately chooses chat;
- prompt token count recorded;
- prompt must fit the small configured M1 context capacity.

Prefer a one-token prompt if one exists in the governed suite. Otherwise use
the shortest available prompt and record the cost implication.

Do not invent expected token IDs from Kestrel-Q.

---

# 15. Independent full-model Class-Q oracle

Task 2.12 requires an independent end-to-end next-token oracle.

Preferred oracle:

> a test-only helper linked to the pinned llama.cpp revision, loading the same
> verified GGUF and accepting **explicit canonical token IDs** rather than
> relying on the GGUF tokenizer.

Pinned:

`llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`

The helper should:

1. mmap/load the verified model in a constrained reference configuration;
2. accept committed canonical input token IDs;
3. run one bounded prompt prefill;
4. emit final-token logits or at minimum deterministic top-N logits/IDs and
   greedy argmax;
5. never modify the GGUF.

This bypasses known GGUF tokenizer-semantic differences.

---

# 16. Oracle feasibility gate

Before relying on the full llama.cpp oracle:

- prove the pinned revision can load/execute this exact `qwen4exp` artifact;
- use a deliberately small context;
- avoid full-model preload when mmap is supported;
- CPU-only is preferred for semantic independence from Kestrel-Q CUDA
  development;
- record peak committed/private memory if practical;
- do not disable system safeguards or pagefile;
- do not copy the 111 GB GGUF.

If the pinned oracle cannot safely execute the artifact on KQ-01:

- preserve the exact blocker/evidence;
- do NOT mark M1 complete based only on Kestrel-Q output;
- use another independently governed full-model Class-Q oracle strategy or
  return BLOCKED/FAIL-CLOSED for maintainer decision.

No self-oracle exception.

---

# 17. Oracle input identity

The oracle and Kestrel-Q must receive the exact same canonical token IDs.

Preferred source for M1 expected input IDs:

- committed Task 1.4 tokenizer golden evidence.

Flow:

```text
raw prompt
-> native tokenizer
-> exact comparison to committed canonical input IDs

committed canonical input IDs
-> independent full-model oracle

native IDs
-> Kestrel-Q full-model executor
```

This isolates tokenizer correctness from model correctness.

---

# 18. End-to-end correctness evidence

At minimum compare:

## Exact discrete

- input token IDs;
- 48 layer order/family sequence;
- QSA selection IDs where recorded;
- MoE expert IDs/order where recorded;
- PLE intents where recorded;
- final greedy token ID.

## Floating

Prefer independent comparison of:

- final hidden vector;
- final normalized hidden vector;
- top-N logits and IDs;
- greedy-token logit.

If the independent oracle exposes the full logits vector, compare it under a
calibrated contract and store only derived safe evidence/hashes.

Do not commit enormous raw logit arrays unnecessarily.

---

# 19. Full-model numerical contract

Do not invent a generic tolerance.

Use the lower-level frozen operator/provider contracts plus at least one
independent full-model comparison.

If the oracle exposes floats:

- measure max abs/relative/ULP where meaningful;
- freeze an M1-specific contract based on evidence;
- do not widen lower-level contracts.

Greedy token ID must be exact.

If top-N order differs despite same argmax, document/root-cause before PASS
unless the contract explicitly limits M1 to argmax and lower-level checks prove
the remainder.

Prefer stronger evidence where practical.

---

# 20. M1 payload accounting

Instrument the complete native run.

Record:

```text
prompt_tokens
generated_tokens = 1

logical_payload_bytes_touched
payload_blocks_touched
unique_semantic_tensors_touched
routed_expert_member_requests
PLE_row_requests
LM_head_logical_bytes_touched
```

These are semantic/logical counters, not physical disk I/O.

No storage-performance conclusions.

---

# 21. M1 payload ceiling

Because the scalar provider may reread weights across prompt tokens, choose the
shortest suitable prompt and enforce a generous correctness-only ceiling.

Suggested hard ceiling:

```text
KQ_TASK212_REAL_PAYLOAD_BUDGET <= 24 GiB logical packed bytes
```

Before setting the final constant, estimate expected bytes from the selected
prompt using Task 2.11 counters and document the margin.

If the shortest governed prompt would exceed the budget, root-cause before
raising it.

Do not turn the ceiling into an optimization target.

---

# 22. State-footprint accounting

Record actual M1 batch-1 state allocation:

- total GDN fixed state;
- PLE address/value state;
- QSA state for chosen context capacity;
- model wrapper/object overhead;
- total persistent state;
- peak scratch;
- logits buffer.

Reconcile with `MODEL-RUNTIME-STATE.md`.

Do not allocate the model's 262k QSA state for M1.

---

# 23. Minimal user-facing CLI

Add a minimal native executable, recommended:

```text
kq-run <model.gguf> --prompt "<text>" --max-new-tokens 1 --greedy
```

Exact CLI may differ.

M1 requirements:

- raw text prompt;
- native tokenizer;
- one generated token;
- greedy only;
- deterministic;
- useful progress/diagnostics without dumping secrets/model bytes.

Do not add a general production inference server.

---

# 24. Long-run safety / progress

A scalar full-model run may be slow.

Provide bounded progress diagnostics suitable for a correctness run:

- prompt tokenization complete;
- embedding complete;
- current layer `n/48`;
- final norm;
- LM head/logits;
- selected token.

Do not log huge activations or model bytes.

A normal slow run is not a failure.

Do not add a performance timeout that invalidates a correct scalar run unless
the execution is clearly hung.

---

# 25. First-token evidence namespace

Create a governed milestone evidence set, e.g.:

```text
research/milestones/Qwen3.8-Flash-Next/
  <canonical-revision>/
    first-token-contract.json
    first-token-oracle.json
    first-token-native.json
    first-token-validation.json
    first-token-manifest.json
```

Record:

- GGUF identity;
- canonical/oracle revisions;
- prompt ID/text hash;
- canonical input IDs;
- context capacity;
- selected token ID;
- decoded output;
- oracle selected token ID;
- top-N/logit evidence if available;
- payload counters;
- state/scratch memory;
- elapsed characterization timings;
- deterministic hashes.

Never commit raw model weights.

---

# 26. Regression / fail-closed tests

Add synthetic/model-level tests for:

- missing embedding;
- wrong embedding shape;
- invalid input token;
- wrong layer count/order;
- missing layer config;
- state-capacity overflow;
- mid-model provider failure rollback;
- missing/wrong final norm;
- missing/wrong LM head;
- logits capacity;
- non-finite final hidden/logits where forbidden;
- argmax tie behavior;
- invalid selected token ID;
- tokenizer decode failure;
- unsupported vision/MTP input path.

---

# 27. Hard boundaries

Task 2.12 does NOT implement:

- more than the minimal greedy generation loop needed for one token unless
  trivial reuse naturally supports it;
- temperature/top-p/random sampling;
- speculative decoding/MTP;
- vision;
- batching;
- production scheduler/cache/prefetch optimization;
- SIMD/CUDA model kernels;
- HTTP/server/UI.

M1 is correctness, not usability/performance.

---

# 28. Documentation

Create:

```text
docs/MODEL-EXECUTION-CONTRACT.md
docs/NATIVE-MODEL-EXECUTOR.md
docs/KQ-MODEL-EXEC-API.md
docs/FIRST-CORRECT-NATIVE-TOKEN.md
```

Create/finalize:

```text
docs/adr/0020-first-correct-native-token.md
```

Update:

- `docs/ARCHITECTURE.md`
- `docs/MODEL-RUNTIME-STATE.md`
- `docs/GOLDEN-VECTORS.md`
- `docs/TASKS.md`
- `docs/ROADMAP.md`
- Epic 2 plan/status;
- Task 2.12 checklist;
- tools/oracle provenance;
- `CHANGELOG.md`.

If the governed project plan explicitly defines M1 as the end of Epic 2, mark
Epic 2 COMPLETE/PASS only after all Task 2.12 gates pass. Otherwise leave Epic
2 status according to the existing roadmap; do not invent a closure rule.

---

# 29. ADR 0020 decision

ADR 0020 should capture:

1. M1 is one independently verified native greedy next token from the real
   quantized model;
2. model execution composes the existing semantic provider and 48 canonical
   layer references;
3. model state uses explicitly bounded short-context capacity for M1;
4. embedding and LM-head access remain row/chunk based with no full F32 matrix;
5. final token selection is deterministic greedy argmax;
6. independent full-model Class-Q oracle remains mandatory;
7. raw prompt token IDs are independently pinned to canonical tokenizer
   evidence;
8. M1 has no performance guarantee;
9. scheduler/cache/SIMD/CUDA optimization follows after correctness;
10. vision/MTP/sampling/batching remain outside M1.

Accept only after the independent oracle and native token ID agree and all
model-level regression gates pass.

---

# 30. Acceptance gates

## Entry/output contract
- [x] embedding exact semantics;
- [x] final norm exact semantics;
- [x] LM-head exact semantics;
- [x] greedy tie semantics.

## Native model
- [x] immutable model exec config;
- [x] bounded global state;
- [x] embedding lookup;
- [x] 48-layer prefill;
- [x] final norm;
- [x] full logits;
- [x] greedy argmax;
- [x] native decode.

## Independent oracle
- [x] pinned full-model Class-Q oracle safely executable;
- [x] exact canonical input IDs;
- [x] oracle next token recorded;
- [x] no self-oracle.

## M1
- [x] shortest governed prompt selected;
- [x] native tokenizer IDs exact;
- [x] all 48 layers execute;
- [x] native next token ID == oracle;
- [x] decoded token recorded;
- [x] stronger final-hidden/logit comparison PASS where available.

## Safety
- [x] no full F32 model/matrix materialization;
- [x] logical payload budget respected;
- [x] small explicit context capacity;
- [x] model-level transactional failures PASS;
- [x] no production Python/llama dependency;
- [x] no tracked model weights/secrets/local paths.

## Regression/governance
- [x] CPU clean PASS;
- [x] CUDA clean PASS;
- [x] Task 2.0–2.11 PASS;
- [x] no new warnings;
- [x] ADR 0020 ACCEPTED;
- [x] CHANGELOG/TASKS/ROADMAP updated;
- [x] KQ-BACKLOG-BENCH-002 still DEFERRED;
- [x] `git diff --check` PASS.

---

## Definition of done

Task 2.12 is COMPLETE/PASS when the native Kestrel-Q runtime takes a governed
UTF-8 prompt, reproduces its canonical token IDs, executes all 48 layers of the
verified real quantized Qwen3.8-Flash-Next model, computes final logits, selects
the exact same greedy next token as an independent full-model Class-Q oracle,
decodes that token natively, and records reproducible evidence without relying
on Python/llama.cpp in production or on any optimized scheduler/kernel path.

Do not commit or push automatically.
