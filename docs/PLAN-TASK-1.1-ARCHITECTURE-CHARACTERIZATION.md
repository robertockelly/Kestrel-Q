# PLAN-TASK-1.1-ARCHITECTURE-CHARACTERIZATION.md

## Objective

Produce an implementation-grade, evidence-backed characterization of the Qwen3.8-Flash-Next architecture before Kestrel-Q implements any model loader, tensor runtime, CPU forward pass, or CUDA model kernel.

Task 1.1 must describe the model well enough that a future implementer can follow a text token from input embedding to output logits and identify all persistent runtime state required by prefill and decode.

This is a research/documentation task, not an inference implementation task.

## Baseline

- Kestrel-Q checkpoint: `aaecee5b38269166f478023430a1208f2a49478f`
- Canonical HF revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Canonical Qwen research revision: `69885871a64393807d988b27b1b5e380e8f28526`
- `KQ-MODEL-ARTIFACT-001 = REGISTERED_VERIFIED`
- GGUF SHA-256: `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`

Task 1.0 established that pinned official Qwen artifacts/specifications define model semantics. The GGUF remains a derived representation.

## Mandatory governance prelude

Before architecture work, make the previously agreed project-history rule normative.

Update `AGENTS.md` so that `CHANGELOG.md` is the canonical chronological record of project changes. Every mandate or iteration that modifies source code, build/runtime configuration, benchmarks, architecture/assumptions, governed documentation, supported behavior/interfaces, model/runtime capabilities, bug fixes/root causes, or governance must update `CHANGELOG.md` in that same iteration before completion and before commit.

Also record the responsibility split:

- `CHANGELOG.md` — chronological record of what changed;
- `docs/TASKS.md` — current Epic/Task status;
- task-specific `TASKS-*.md` — execution checklist/evidence;
- `docs/adr/` — durable architectural decisions;
- Git history — exact implementation history.

Do not create a parallel worklog unless an ADR explicitly introduces one.

Update `docs/PRINCIPLES.md` with the principle that every material change must be reconstructible. Record this governance change in `CHANGELOG.md` in this same iteration.

## Evidence hierarchy

### Tier A — canonical

1. Pinned Hugging Face repository `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
2. Pinned Qwen architecture repository `QwenLM/Qwen3.8-Flash-Next` at revision `69885871a64393807d988b27b1b5e380e8f28526`.
3. Technical report from that pinned Qwen revision.
4. Official Qwen release article, clearly labeled as unversioned supporting evidence.

### Tier B — implementation reference

If Tier A does not fully specify exact execution semantics, inspect an exactly pinned implementation such as Hugging Face Transformers Qwen4-Exp. Record exact revision/release, license, files/classes/functions used, and treat it as implementation evidence rather than canonical checkpoint semantics.

### Tier C — cross-check

SGLang, vLLM, llama.cpp or other runtimes may be inspected only as secondary cross-checks. Their naming, cache formats, kernels and optimizations are not canonical model semantics. Community issues are operational evidence only.

## No-weight-download rule

Task 1.1 must not download any Safetensors weight shard.

Use the Task 1.0 pinned cache, config, index, technical report, and pinned source implementations when needed. The external GGUF may be inspected read-only only for limited structural cross-checks; Task 1.1 must not begin Task 1.3 mapping.

## Required outputs

Create:

```text
docs/MODEL-ARCHITECTURE.md
docs/MODEL-RUNTIME-STATE.md
research/model-architecture/Qwen3.8-Flash-Next/<HF_REVISION>/evidence.json
docs/adr/0005-initial-text-only-runtime-scope.md
```

Update:

```text
AGENTS.md
docs/PRINCIPLES.md
docs/TASKS.md
docs/ROADMAP.md
CHANGELOG.md
docs/TASKS-TASK-1.1-ARCHITECTURE-CHARACTERIZATION.md
```

## Workstream A — top-level multimodal wrapper

Document the relationship among `Qwen4ExpForConditionalGeneration`, language model, vision subsystem, multimodal merge path, LM head, embeddings, special image/video tokens, and generation outputs.

Trace precisely what executes for a text-only request with no image/video input. Do not assume the vision path is optional without tracing the conditional execution path in pinned evidence.

## Workstream B — text backbone

Verify and document:

- hidden size and vocabulary;
- 48 text layers;
- exact `layer_types` sequence;
- 36 linear-attention layers;
- 12 full/sparse-attention layers;
- full-attention interval;
- attention/KV head counts and head dimension;
- positional/rotary configuration;
- normalization;
- activation;
- output gating;
- embedding → layer stack → final norm/head flow.

Produce an explicit layer-number/type table.

## Workstream C — Gated DeltaNet

Characterize exact linear-attention forward semantics:

- symbolic input/output shapes;
- projections;
- gating/decay/update terms;
- normalization;
- local convolution/state if applicable;
- fixed recurrent state;
- prefill state update;
- single-token decode update;
- source-backed dtypes;
- relationship to hybrid cache.

Separate mathematical semantics from implementation fusions.

## Workstream D — Qwen Sparse Attention

Characterize both indexer and selected attention:

- Q/K/V dimensions;
- query/KV head counts;
- micro-block construction;
- indexer compression ratio;
- indexer heads, KV heads and head dimension;
- indexer budget;
- selection semantics;
- incomplete trailing-block handling;
- causality;
- selected-context representation;
- RoPE/position handling;
- prefill and decode behavior;
- persistent KV/index state.

Separate canonical algorithm from external optimized kernels.

## Workstream E — Gated Residual

Characterize:

- `hc_count = 4`;
- low-rank gating configuration;
- four-branch residual state;
- read operation;
- elementwise dynamic gate;
- write-back;
- branch mixing presence/absence;
- placement around attention and MoE;
- source-backed FP8 residual-state capability.

Provide diagrams or pseudocode sufficient for future implementation.

## Workstream F — ultra-sparse MoE

Characterize:

- router input/scoring;
- 512 routed experts;
- top-10 experts/token;
- shared expert;
- expert/shared intermediate size;
- activation;
- selected-output weighting/combination;
- shared-expert combination;
- inference routing outputs;
- separation of training-only load balancing.

Do not infer expert storage layout from GGUF.

## Workstream G — N-gram embedding / PLE

This is especially important for Kestrel-Q.

Characterize:

- `ngram_size = 3`;
- `ngram_vocab_size_base = 20,000,000`;
- PLE layer placement;
- embedding dimension;
- `heads_per_ngram`;
- split/partition configuration;
- convolution setting;
- deterministic lookup address computation;
- how token context forms indices;
- how retrieved values enter the hidden/residual path;
- prefill lookahead opportunity;
- decode lookup behavior;
- what can be known early enough for asynchronous prefetch.

Separate canonical algorithm from future Kestrel-Q cache design.

## Workstream H — Multi-Token Prediction

Characterize:

- MTP layer count/topology;
- hybrid configuration;
- attention type;
- embedding sharing;
- main hidden-state input;
- training role;
- inference/speculative-decoding role.

Answer with evidence:

> Is MTP required for canonical ordinary autoregressive next-token logits, or is it optional speculative drafting?

## Workstream I — runtime state

Create `docs/MODEL-RUNTIME-STATE.md` and inventory:

- GDN recurrent state;
- QSA KV cache;
- QSA index/compressed state;
- GR residual branches;
- N-gram token-history/lookup state;
- position/RoPE state;
- optional MTP state;
- optional multimodal state.

For each state record owner, lifetime, fixed/context-growing class, symbolic shape, known dtype, prefill/decode mutation semantics, and whether text-only inference needs it.

Detailed byte accounting remains Task 1.2.

## Workstream J — prefill and decode flow

Document explicit end-to-end flows for:

### Prefill
Token IDs → embeddings → N-gram/PLE → GR → GDN/QSA + MoE → state/cache construction → final norm/logits.

### Decode
One new token step, identifying state reads/writes, context-dependent vs fixed-size operations, and future prefetch candidates.

## ADR 0005 — initial text-only scope

Task 1.1 must finalize the proposed ADR based on evidence.

Evaluate whether first Kestrel-Q inference may support:

- text-token input only;
- ordinary autoregressive next-token generation;
- no image/video input;
- no vision execution;
- no MTP speculative acceleration initially.

Accept only if pinned evidence proves the scope preserves canonical ordinary text-generation semantics. Otherwise keep Proposed with an exact blocker.

## Evidence file

Create machine-readable evidence under:

```text
research/model-architecture/Qwen3.8-Flash-Next/<HF_REVISION>/evidence.json
```

Each major architectural fact must have a stable `KQ-ARCH-*` claim ID and include source tier, exact revision when versioned, artifact/file, section/page/function/symbol/config path, confidence, and notes.

If sources disagree, record the discrepancy explicitly; do not silently reconcile it.

## Acceptance gates

### Governance
- [ ] `AGENTS.md` has mandatory CHANGELOG rule.
- [ ] `docs/PRINCIPLES.md` has reconstructibility principle.
- [ ] `CHANGELOG.md` records governance change.

### Evidence
- [ ] canonical revisions remain pinned;
- [ ] any Tier B source is pinned/licensed;
- [ ] evidence JSON validates;
- [ ] no major architecture claim lacks evidence.

### Architecture
- [ ] multimodal wrapper characterized;
- [ ] complete 48-layer sequence characterized;
- [ ] GDN characterized;
- [ ] QSA/indexer characterized;
- [ ] Gated Residual characterized;
- [ ] MoE characterized;
- [ ] N-gram/PLE characterized;
- [ ] MTP characterized;
- [ ] prefill/decode documented;
- [ ] runtime-state inventory documented.

### Scope
- [ ] ADR 0005 accepted or explicitly blocked with evidence;
- [ ] vision boundary determined;
- [ ] MTP dependency determined.

### Safety
- [ ] zero Safetensors shards downloaded;
- [ ] zero model weights tracked;
- [ ] zero GGUF tracked;
- [ ] no incompatible source code copied;
- [ ] no model/runtime implementation added;
- [ ] `git diff --check` PASS;
- [ ] no secrets.

### Project records
- [ ] `docs/TASKS.md` updated;
- [ ] `docs/ROADMAP.md` updated;
- [ ] `CHANGELOG.md` updated for every material Task 1.1 change;
- [ ] Task 1.2 and Task 1.3 remain NOT STARTED.

## Definition of done

Task 1.1 is COMPLETE/PASS only when the architecture documents are implementation-grade, every major claim is traceable to pinned evidence, the runtime-state model is explicit, and ADR 0005 has an evidence-backed outcome.

Do not commit/push automatically; leave the intentional Task 1.1 delta ready for review.
