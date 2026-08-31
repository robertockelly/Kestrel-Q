# Changelog

All notable changes to this project will be documented here.

The project is currently pre-alpha.

## Unreleased

### Changed

- Completed Task 2.13 and accepted ADR 0021: the native C17 model executor now
  prefills the prompt exactly once, consumes each selected canonical token ID
  through a true one-token incremental decode and stops on EOG or the caller's
  bounded greedy `max_new_tokens` limit without replay or retokenization.
- Added an independent multi-token Class-Q oracle at pinned
  `llama.cpp@90c26fcd4b2114b4aa39d09d69318cb8f438d27a`. Given the exact canonical
  `KQ-PROMPT-001` IDs and context 16 it freezes `[271, 248068, 198, 760]` with
  decoded fragments `["\n\n", "<think>", "\n", "The"]`; native Kestrel-Q
  matches the complete sequence exactly and preserves the one-token M1 gate.
- Added explicit model-state summaries for position, QSA block/tail geometry
  and compact active GDN/QSA/PLE-address/PLE-value hashes. Positions advance
  7, 8, 9 and 10; all 12 QSA caches follow them with complete-block/tail pairs
  `(1,3)`, `(2,0)`, `(2,1)` and `(2,2)`.
- Made one-token model decode transactional across all 48 layers. A forced
  provider failure at layer 24 after two prior generated tokens rolls completed
  layers back in reverse order, leaves state and caller outputs unchanged, and
  retry produces oracle token 198. Provider metrics/traces remain monotonic
  audit evidence rather than rollback-managed semantic state.
- Added exact per-step access invariants: prompt prefill is one, incremental
  decode is three, each decode has 480 ordered top-10 selections, zero
  unselected expert member access and 16 exact PLE rows. Successful logical
  payload is 40,208,768,960 bytes for prefill plus 6,334,414,400 per decode,
  totaling 59,212,012,160 bytes under the governed 96 GiB ceiling; these are
  not physical-I/O measurements.
- Strengthened the final QSA evidence to freeze exact aggregate candidate /
  selected-block counts of 48/48 for prefill and 24/24 for each decode, with
  selected-token counts 336, 96, 108 and 120. Complete-block/tail state and
  sparse-selection work are therefore checked independently.
- Extended `kq-run` with bounded `--max-new-tokens N` greedy continuation and
  per-token progress while retaining `--max-new-tokens 1` token 271 / `\n\n`
  compatibility. Zero generation, padded IDs, invalid continuation, prompt
  replay, context exhaustion and failed-step output publication fail closed.
- Added deterministic six-file Task 2.13 milestone evidence and research-only
  oracle/evidence tooling. Historical Task 2.12 evidence remains unchanged;
  no model payload or local model path is stored.
- Preserved two internal test findings before repair. The first invariant
  validator assumed token-major MoE trace order, while the executor correctly
  emits layer-major traces with tokens inside each layer; its expected layer
  calculation and regression were corrected. The second validator conflated
  three physical gate/up/down matrix requests with one logical selected-expert
  member trace; the two independently checked counters are now distinct.
- The first final clean build also caught a CLI-reporting-only identifier typo
  in the new final-token logit label. The generation loop stores its bounded
  prefix in `results`/`generated`; using those actual locals fixes the compile
  failure, and both clean CPU/CUDA builds cover the regression.
- The first clean CPU CTest then caught a status-code compatibility regression
  before any model payload work: the refactored scratch query had grouped an
  overlong prompt with invalid continuation state and returned
  `INVALID_MODEL_STATE` instead of M1's governed `INVALID_ARGUMENT`. Input
  geometry and state validity now fail through separate branches, with the
  existing first-token regression enforcing the stable result.
- The same clean suite preserved a Task 2.11 determinism failure after the
  initial Task 2.13 test enlarged fixed provider trace arrays: although numeric
  output was unchanged, `provider_owned_bytes` changed from 18,576 to 151,696
  and therefore invalidated accepted evidence. Fixed trace capacities and the
  historical owned-byte result are restored. The multi-token test now opts
  into a separately allocated, bounded extended trace through an internal
  test-only hook; ordinary production providers and prior evidence remain
  byte-identical.
- Final clean Release validation passes 40/40 CPU tests and 42/42 CUDA tests,
  including all Task 2.0–2.12 regressions, the real multi-token transaction
  gate, the restored Task 2.11 deterministic oracle and CUDA smoke/self-test.
  No new Kestrel-Q `/W4` warning appears; only the previously governed
  NVCC-generated external C4211 remains.

- Completed Task 2.12 / M1 and accepted ADR 0020: the native C17 scalar path
  tokenizes governed raw UTF-8, reads bounded real embedding rows, executes
  target layers 0 through 47, applies the final qwen4exp hyper-connection
  mixer, computes all 248,320 LM-head logits, selects a stable greedy argmax
  and decodes the selected token without a production llama.cpp/Python
  dependency or a complete F32 target matrix.
- Added the independently governed full-model Class-Q oracle and milestone
  evidence. Pinned llama.cpp receives the explicit Task 1.4 canonical IDs for
  `KQ-PROMPT-001`, bypasses the divergent GGUF tokenizer, and selects token 271
  (`\n\n`), exactly matching native Kestrel-Q. Native/oracle top-two IDs also
  agree; floating logits/checkpoint summaries remain characterization only.
- Added `kq-run` and the immutable config / explicit bounded-state
  `kq_model_exec` API. M1 uses context capacity 8 and records 359,881,768 state
  bytes, 8,693,472 scratch bytes, a 993,280-byte logits buffer and useful
  phase/layer progress without logging raw weights or full activations.
- Recorded the M1 logical payload: 40,208,768,960 bytes, 1,693,222,880 blocks,
  1,239 unique semantics, 19,040 embedding bytes, 3,360 routed selections /
  10,080 selected-expert matrix requests, 112 PLE rows and 675,430,400 LM-head
  bytes. The governed ceiling is 64 GiB because the scalar correctness path
  rereads weights per prompt token; these counters are not physical I/O,
  page-cache traffic or throughput.
- Preserved and fixed three whole-model findings. The former 256-entry
  semantic trace was too small for 1,239 full-run semantics and is now bounded
  at 2,048. The qwen4exp converter stores zero-centred RMS gamma as `1+delta`,
  so the provider now restores canonical deltas exactly once. Per-layer oracle
  checkpoints then exposed an accidental second PLE conversion at layer 1;
  removing it changed the remaining wrong token 20 to the oracle token 271.
- Added exact-token regression coverage, invalid/padded ID and bounded-capacity
  failures, and early/middle/late provider faults. Each real failed run keeps
  caller outputs/position unchanged and is followed by a complete
  control-equivalent token-271 recovery.
- Fixed the rollback recovery order exposed by that strengthened regression.
  A failed QSA path could leave private staging advanced, and the following
  call queried scratch geometry before resetting it, producing an invalid-state
  failure despite an unchanged public position. The executor now resets all
  dirty private layer state before the scratch query; faults at layers 0, 24
  and 47 each recover through a complete token-271 control run.
- Corrected the independent Task 2.11 target-layer oracle after the complete
  model exposed its incomplete converter inverse. The generator had restored
  PLE zero-centered gammas but not HC or QSA/indexer gammas, so it now applies
  the pinned converter rule to every affected canonical role and explicitly
  excludes the direct GDN linear-attention gamma. Four independent calibration
  profiles now bound the unchanged disjoint holdout (30 floating comparisons),
  while all route, selected-expert and PLE-request comparisons remain exact.

- Completed Task 2.11 and accepted ADR 0019: a bounded semantic target-weight
  provider now executes the accepted scalar GDN, QSA, MoE, PLE-value and
  complete-layer equations directly from the registered quantized GGUF without
  materializing a complete target matrix in F32.
- Added independent exact-GGUF target-layer evidence. Pinned llama.cpp Class-Q
  decoding plus pinned Transformers equations validate calibration and
  disjoint holdout for ordinary GDN layer 0, QSA layer 3 and PLE-GDN layer 1;
  all 18 floating comparisons and exact MoE/QSA/PLE decisions pass without a
  Kestrel-Q self-oracle.
- Validated 48/48 target layer configurations with zero preflight payload
  dereference. The accepted correctness run accounts 772,826,304 logical
  packed bytes including rollback injections, below 768 MiB, while only
  selected routed experts and the 32 requested PLE rows are accessed.
- Fixed a Task 2.11 semantic-trace capacity failure exposed by the real
  PLE-GDN run: the initial 96-entry bound could not represent its 109 unique
  touched semantics. The bounded capacity is now 256 and the complete real
  integration covers the regression.
- Removed redundant gated-residual weight reads discovered by the real payload
  budget: the complete layer previously recomputed identical attention/MoE GR
  reads solely to recover write gates. Per-token write gates are now retained
  and reused, preserving Task 2.10 exact synthetic output while keeping the
  target correctness run under its hard payload ceiling.
- Hardened provider failure behavior: capacity/alias/budget failures are
  detected before payload dereference or visible output mutation, QPC timing
  conversion uses checked 64-bit arithmetic, and injected failures before the
  first read and after 7,332,736 bytes preserve complete layer state.
- At the Task 2.11 checkpoint, documented the then-unstarted First Correct
  Native Token boundary; Task 2.12 now completes it. Cache/prefetch policy
  remains deferred with `KQ-BACKLOG-BENCH-002`.

- Completed Task 2.10 and accepted ADR 0018: a scalar C17 one-layer reference
  now composes exact Qwen3.8 PLE placement, four-branch Gated Residual reads and
  writes, GDN/QSA and MoE for all three canonical layer families.
- Added independent pinned-Transformers complete-layer calibration and disjoint
  holdout evidence. Ordinary GDN, QSA and PLE-GDN native results pass their
  family-specific `2e-6` maximum-absolute contracts without using Kestrel-Q as
  an oracle.
- Made complete-layer oracle regeneration fail closed on the pinned
  `tokenizers==0.23.1` dependency after a global-environment version mismatch
  was detected during final reproducibility validation.
- Added committed/staging persistent layer state. Synthetic prefill/decode,
  two-step decode, reset/replay and a forced downstream MoE failure prove no partial visible
  GDN/QSA/PLE state advance; all 48 target layer configs validate with zero
  model-payload bytes touched.
- Fixed a Task 2.10 internal state-clone bug found by the PLE split-decode
  regression: the first implementation returned immediately after cloning GDN
  state and therefore skipped PLE address/value history. Cloning now continues
  through every family-owned substate before execution.
- Fixed the initial Task 2.10 GR activation path after the numeric API correctly
  rejected an in-place input/output alias. The layer implementation now uses
  explicit scalar sigmoid/SiLU evaluation with the governed strict-FP contract;
  the independent layer oracle and alias/failure regressions cover the fix.

- Replaced the provisional MIT license with Apache License 2.0.
- Added contribution and dependency licensing/provenance rules.
- Documented the validated KQ-01 Windows, MSVC, CMake, CUDA and GPU baseline.
- Ignored disposable root `build-*` CMake build trees.
- Recorded deferred backlog item `KQ-BACKLOG-CUDA-001` to select and document a
  portable CUDA binary strategy before the first public binary release.
- Replaced KQ-01's theoretical-only PCIe assumption with measured pageable and
  pinned H2D/D2H results across 1–256 MiB, plus under-load Gen4 x16 evidence.
- Consolidated Foundation Tasks 0.1–0.3 as complete/pass, including the
  previously measured Task 0.2D clean-idle RAM/VRAM baseline.
- Deferred WDDM CUDA allocation-headroom measurement to separate backlog item
  `KQ-BACKLOG-BENCH-001`; it is not part of Task 0.2D.
- Pinned the official Qwen3.8-Flash-Next artifact and research revisions and
  accepted ADR 0004: official Qwen artifacts/specifications are canonical,
  while GGUF remains a derived representation.
- Documented the Apache-2.0 Kestrel-Q source boundary separately from the pinned
  Qwen Community License 1.0 model-artifact terms, including commercial MaaS
  and AI Work Assistant review triggers.
- Made `CHANGELOG.md` the canonical chronological project-change record and
  required every material mandate/iteration to update it before completion and
  commit, with task status, task evidence, ADRs and Git history retaining their
  separate responsibilities.
- Completed Task 1.1 architecture characterization against pinned official
  model/research sources and a pinned Apache-2.0 Transformers implementation
  reference; Tasks 1.2 and 1.3 remain not started.
- Accepted ADR 0005: the initial Kestrel-Q model path may support ordinary
  text-only autoregressive logits while explicitly rejecting multimodal inputs
  and deferring optional MTP speculative acceleration.
- Completed Task 1.2 canonical tensor inventory and footprint analysis: all 131
  Safetensors headers and 1,658 tensors reconcile exactly, with zero weight
  payload bytes fetched and zero unexplained classifications; Task 1.3 remains
  not started.
- Completed Task 1.3 canonical-to-GGUF mapping: all 1,658 canonical tensors and
  all 1,224 derived GGUF tensors reconcile with zero unresolved or unexplained
  entries, including exact PLE fusion/metadata, MoE/QSA splits, vision/MTP scope
  omissions and converter transforms, closing the exact 434-tensor difference.
- Accepted ADR 0006's staged container strategy: begin future correctness work
  from the verified GGUF while preserving canonical internal semantics, and
  require profiling evidence before introducing a Kestrel-Q-native container.
- Proved the local merged GGUF's 384-byte reduction from the published four
  shards is header/directory format overhead only; bounded upstream Range audit
  fetched 342 metadata bytes and zero tensor-payload bytes.
- Completed Task 1.4 and Epic 1 with separate canonical and exact-GGUF
  correctness classes, pinned Apache-2.0 Transformers and MIT llama.cpp
  oracles, and accepted ADR 0007's independence/resource-deferral policy.
- Preserved canonical BF16 and exact-GGUF full-model vectors as explicit
  capable-reference-environment gates after KQ-01 resource checks; no
  weight-dependent output was fabricated or executed.
- Recorded `PLE_DISK_BACKED_CANDIDATE` as a preliminary KQ-01 placement
  hypothesis: disk-backed/mapped PLE with a bounded explicit RAM cache,
  deterministic predictive prefetch and materialized lookup data only in VRAM;
  routed experts remain candidates for RAM-active/VRAM-hot/disk-cold tiering.
  This is not a validated performance result or final scheduler policy, and
  uncontrolled Windows paging is explicitly not the implementation mechanism.
- Added `KQ-BACKLOG-BENCH-002`, the deferred PLE disk-backed access benchmark
  required before final scheduler/residency design. It must separate OS page
  cache, explicit Kestrel-Q RAM cache and cold physical reads; it does not block
  Task 2 loader/correctness work.
- Completed Task 2.0 and accepted ADR 0008: the first production C17 runtime
  component is a Kestrel-Q-owned, Windows-native, read-only GGUF v3 layer with
  bounded logical mappings, immutable physical descriptors and no llama.cpp or
  GGML runtime dependency. Task 2.1 remains not started and
  `KQ-BACKLOG-BENCH-002` remains deferred.
- Reproduced the complete Epic 1 structural oracle through `kq-inspect` on the
  registered 111 GB artifact without a full-file hash or intentional payload
  access. File size/last-write time remained unchanged and the parser reported
  `payload_bytes_accessed = 0`.
- Fixed two intermediate MSVC C4701 warnings caused by conservative data-flow
  analysis of parser locals populated through status-returning helpers. Explicit
  initialization restored a warning-free `/W4` build; the clean CPU/CUDA build
  gate protects against recurrence.
- Completed Task 2.1 and accepted ADR 0009: a target-specific immutable
  Qwen3.8-Flash-Next semantic registry now validates the complete `qwen4exp`
  identity/topology and reconciles 1,294 initial-text meanings with all 1,224
  registered physical GGUF tensors and three metadata-derived PLE meanings.
- Preserved the exact non-one-to-one representation boundary: 48 MoE gate/up
  and 12 QSA index query/key splits, 128 logical PLE table members fused into
  one physical tensor, and explicit PLE address metadata. Routed expert stacks
  retain their 512-expert axis without fabricating per-expert tensors.
- Confirmed the complete deterministic native registry against every
  initial-text row and physical name in the pinned Epic 1 mapping oracle, with
  zero unknown physical tensors, zero unbound required semantics and zero
  payload bytes accessed. `KQ-BACKLOG-BENCH-002` remains deferred and placement
  values remain annotations rather than scheduler policy.
- Fixed two Task 2.1 implementation findings before the clean gate: explicit
  pointer initialization resolved conservative MSVC C4701/C4703 diagnostics,
  and semantic CLI dispatch was moved after successful GGUF parsing after an
  initial null-GGUF ordering failure. Synthetic and CLI-oracle regressions cover
  both corrected paths.
- Completed Task 2.2 and accepted ADR 0010: the C17 runtime now exposes
  immutable bounded read-only quantized tensor views with checked F32, BF16,
  Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL block geometry, packed-size and logical
  element/range-to-block helpers.
- Preserved semantic/physical layout boundaries through exact whole-tensor
  views, ordered non-concatenated MoE/QSA split parts, provably contiguous
  routed-expert members and one-member-only views over the fused 128-member PLE
  tensor. Metadata-derived requests and transformed-canonical misuse fail
  explicitly; mapping adds no allocator, cache, prefetch or scheduler policy.
- Added deterministic guarded payload fixtures and real-artifact view coverage.
  Synthetic tests dereference only their 1,192 known fixture bytes; dense,
  all-seven-type, layer-2 expert, split and PLE 0/64/127 real mappings open and
  close with zero model-payload bytes touched by the test.
- Added a deterministic physical-geometry inspector dump and research-only
  Task 1.3 validator. All 1,224 physical descriptors and 111,323,630,080 packed
  bytes match the pinned inventory; production has no CSV/JSON or Python
  dependency.
- Corrected an intermediate Task 2.2 descriptor-contract omission found during
  implementation review: the first draft separated requested elements from
  physical block bytes but omitted canonical unpacked bytes. The final view
  info records `logical_unpacked_bytes` and a copied stable semantic ID, with
  synthetic regression assertions. `KQ-BACKLOG-BENCH-002` remains deferred.
- Fixed an intermediate Task 2.2 MSVC C4701 diagnostic caused by conservative
  data-flow analysis of a canonical element count returned through a
  status-reporting helper. Explicit initialization preserves fail-closed
  control flow and restores the warning-free project build.
- Corrected the synthetic non-contiguous-expert mutation so it preserves the
  canonical element total; the original mutation was validly rejected by the
  earlier shape gate and therefore did not exercise the intended contiguity
  branch. The corrected regression now isolates the slowest-axis requirement.
- Completed Task 2.3's mandatory tokenizer characterization and stopped
  fail-closed before production code. The registered GGUF's 248,044 base
  vocabulary entries, 33 added-token strings and 247,587 ordered merges match
  the pinned official assets exactly, with 243 additional unused padded IDs.
- Recorded the Task 2.3 production-source blocker: GGUF metadata omits the
  canonical NFC normalizer, selects a marks-inclusive `qwen35` pre-tokenizer
  instead of the executed Class-C `Qwen2Tokenizer` expression, classifies six
  FIM/repository tokens contrary to canonical skip-special decode, and embeds a
  modified Unsloth chat template. The modified template matches all four
  committed initial-subset chat cases, but its other branches are not
  canonical. ADR 0011 remains proposed; no tokenizer/chat runtime or sidecar
  was introduced without maintainer approval.
- Revalidated all eight independent Task 1.4 golden assets and regenerated them
  byte-identically offline. Clean CPU Release tests remain 7/7 and clean CUDA
  Release tests remain 9/9; no Kestrel-Q warning or runtime regression was
  introduced by the documentation-only fail-closed result.
- Continued the preserved Task 2.3 blocker delta after the maintainer selected
  a governed model-specific canonical tokenizer override. The immutable C17
  tokenizer now validates the GGUF's exact 248,320 token/ID stream, 247,587
  ordered merge/rank stream, token types, special tokens and 243 padded IDs,
  then applies canonical Qwen3.8 NFC, marks-excluding byte-level BPE,
  special/BOS/EOS and skip-special semantics without a sidecar or GGUF change.
- Added the separate native text-only Qwen3.8 chat formatter for the supported
  optional-system and alternating user/assistant subset, with exact generation
  prompt off/on behavior. Developer/tool, multimodal, reasoning/tool options
  and malformed role sequences fail explicitly rather than falling through to
  the divergent embedded Unsloth template.
- Vendored utf8proc 2.10.0 at commit
  `a1b99daa2a3393884220264c927a48ba1251a9c6` under its MIT/Unicode-data
  notices. Source inspection found that the pinned tokenizer oracle splits
  Unicode semantics: Unicode 9.0.0 NFC and Unicode 16.0.0 regex properties.
  A deterministic table generated from the pinned Unicode-9 `DerivedAge.txt`
  gates utf8proc composition to reproduce that split without host-locale
  dependence. Added `NOTICE`, hashes and removal/provenance guidance;
  Python/Transformers/tokenizers remain test-only.
- Added independent canonical tokenizer divergence evidence (SHA-256
  `f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`)
  for 22 encode, five decode, two supported-chat and three rejected-chat cases,
  including a post-Unicode-9 normalization boundary.
  The native oracle comparison also passes all unchanged original 10 prompts,
  14 segments and four chat vectors; model tensor payload touched remains zero.
- Accepted ADR 0011. The decision preserves the original GGUF-only
  insufficiency finding, uses exact GGUF vocabulary/ID/merge data as physical
  substrate, compiles only the pinned canonical semantic overrides, rejects
  near-match artifacts and introduces neither a sidecar nor a generic override
  framework. Task 2.3 is COMPLETE/PASS; Task 2.4 remains NOT STARTED and
  `KQ-BACKLOG-BENCH-002` remains DEFERRED.
- Corrected two bugs in the independent differential generator before evidence
  capture. It initially discarded valid merge entries whose first token began
  with `#` by treating every such line as a version banner; it now skips only
  the literal `#version:` form. It also tried to serialize a Transformers
  `BatchEncoding` directly; the fixed path extracts `input_ids`. Deterministic
  regeneration and hash comparison cover both corrections.
- Corrected implementation-review findings before the clean Task 2.3 gate:
  empty `(NULL,0)` text could form an invalid pointer before a zero-length
  append, special-token decode could reorder pending ordinary bytes, and an
  encode-time characterization counter made the tokenizer object mutable.
  Zero-length, interleaved special decode and const API/oracle regressions cover
  the fixes; encode/decode now accept an immutable tokenizer.
- Final clean Release validation passes CPU 11/11 and CUDA 13/13, including all
  Task 2.0–2.2 regressions, the real tokenizer substrate gate and independent
  canonical oracle. Kestrel-Q emits no new `/W4` warning; only the already
  documented NVCC-generated external C4211 remains.
- The final dependency-provenance audit caught four incorrectly transcribed
  utf8proc file hashes in the first draft of its local README. The vendored
  bytes already matched the pinned upstream checkout; the documentation now
  records the independently recomputed full SHA-256 values, and the final
  vendor-hash gate verifies all four files. A path-specific Git whitespace
  attribute preserves the verified upstream-generated data bytes while keeping
  the repository-wide staged `git diff --check` gate green.
- The first Unicode-9 assigned-range generator preserved the source file's
  age-group order even though the runtime binary search requires code-point
  order. The expanded NFC oracle immediately caught the resulting mismatch on
  an existing Angstrom case. The generator now sorts and merges all ranges
  deterministically before emission; byte-identical regeneration plus the full
  native canonical corpus guard the corrected table.
- Completed Task 2.4 and accepted ADR 0012. The production C17 PLE address
  engine validates the exact Qwen3.8 semantic-registry topology, all 128 fused
  table members and three metadata-derived address arrays, then emits 16
  ordered logical member/row intents per canonical token from an immutable
  configuration and explicit bounded 32-byte stream state.
- Matched all seven unchanged Task 1.4 PLE sequences and four incremental
  decode steps exactly; the original SHA-256 remains
  `495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6`
  after independent byte-identical regeneration. Added independent expanded
  evidence SHA-256
  `b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`
  for 12 sequence cases, three decode streams and tokenizer-to-PLE integration.
- Kept the PLE address boundary storage-neutral: real integration opens zero
  PLE payload views and touches zero model tensor payload bytes. No lookup,
  PLE value math, disk access, cache, prefetch, scheduler, inference or CUDA
  model kernel was added; `KQ-BACKLOG-BENCH-002` remains deferred and required
  before final disk-backed policy. Task 2.5 remains not started.
- Corrected two Task 2.4 validation findings before the clean gate. A first
  incremental-equivalence test used `memcmp` across compiler padding in the
  intent struct and could report a false mismatch despite the independent
  oracle passing; the regression now compares every semantic field. MSVC also
  conservatively reported C4701 for run metrics populated through successful
  status-returning calls; explicit initialization restores a warning-free
  `/W4` path without weakening error handling.
- Completed Task 2.5 and accepted ADR 0013. The production C17 runtime now has
  an MSVC `/fp:strict` scalar reference layer for bit-preserving F32/BF16 and
  independently validated Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL block decode, plus
  a block-by-block quantized-row dot with fixed 1,024-byte scratch.
- Added only the characterized generic F32 primitives: add/multiply/scale/dot,
  sigmoid, SiLU, SwiGLU combine, Qwen weight-delta RMSNorm, softmax, stable
  lower-index-tie top-k and selected-weight renormalization. No SIMD, CUDA
  numeric kernel, model operator, full forward path, sampler or scheduler was
  introduced.
- Added pinned independent Task 2.5 evidence: 39 llama.cpp Class-Q synthetic
  decode cases, seven row-dot cases, 31 NumPy calibration cases and 21 disjoint
  holdout cases. Bit-exact/discrete operations remain exact; sigmoid, SiLU,
  softmax and SwiGLU use separately measured ULP contracts rather than a
  blanket tolerance.
- Performed the first deliberately bounded real-model payload validation via
  semantic descriptor and Task 2.2 views. Nine blocks cover all seven formats,
  routed experts, the layer-2 Q8_0/Q5_K mix and PLE IQ4_NL; exactly 612 packed
  bytes were touched under the 1 MiB guard and every decoded hash matched the
  independent pinned llama.cpp helper. No raw sampled block was committed and
  no storage-throughput/residency claim is made.
- Corrected four Task 2.5 development findings before the clean gate: a dot
  test expected-value transcription (`1` versus the recomputed `-5`), the
  research helper's incorrect assumption that GGML F32 exposes a `to_float`
  callback, an 11-versus-12-field real-evidence parser count, and a dot-alias
  guard initially inserted in the binary-vector helper. Exact regression and
  deterministic regeneration cover each corrected path; no invalid real raw
  evidence was written.
- Completed Task 2.6 and accepted ADR 0014. The first model-specific production
  operator is an immutable C17 Qwen3.8 GDN scalar reference with explicit
  convolution/recurrent stream state, transactional batch-1 prefill and
  one-token decode, bounded caller scratch and synchronous intermediate
  checkpoint observation. It adds no GR composition, QSA, MoE, PLE value,
  complete-layer/full-forward, SIMD or CUDA model execution.
- Characterized the exact pinned `Qwen4ExpTextGatedDeltaNet` contract before
  native code and generated an independent Class-C evidence namespace from the
  pinned offline Transformers module. Five calibration, five disjoint holdout
  and five state/continuation cases cover sequence/history boundaries,
  repeated/alternating/masked input, non-zero state, reset/replay and two decode
  steps. All final output/state and 21 emitted intermediate checkpoint classes
  pass their per-class calibrated contracts; native split-prefill/decode and
  reset/replay are bit-identical. Kestrel-Q never supplies expected values.
- Added target semantic integration for all 36 real GDN layers and fail-closed
  rejection for all 12 QSA IDs, missing/wrong semantics, shape/dtype/type/
  transformed-layout mismatches, invalid state/count/capacity/finiteness and
  forbidden aliasing. The released BF16 state descriptor and executable F32
  scalar-reference descriptor validate the same nine canonical bindings. No
  real payload view was opened and real model payload bytes touched remain
  zero; the full BF16 checkpoint was not downloaded.
- Fixed two Task 2.6 build findings before the clean gate. MSVC C4090 arose from
  an aggregate `memcpy` of pointer-to-const semantic descriptors; explicit
  element assignment preserves qualifiers. Adding probe timing then exposed
  conservative C4701 analysis across status/timer checks; explicit
  `LARGE_INTEGER` initialization preserves fail-closed flow. Focused rebuilds
  are warning-free and regression tests cover the affected construction and
  oracle paths.
- Corrected Task 2.6 characterization timing before publication: the first
  draft included registry construction and checkpoint-observer/output work.
  Real-config timing now starts after registry construction, while reduced
  execution timing uses a second no-observer run; timings remain explicitly
  descriptive rather than performance guarantees.
- Completed Task 2.7 and accepted ADR 0015. The production runtime now has an
  immutable C17 Qwen3.8 QSA scalar-reference config, explicit bounded
  K/V/raw-index stream state, transactional batch-1 prefill and decode, exact
  sparse-selection observation and synchronous intermediate checkpoints. It
  adds no GR composition, MoE, PLE value, complete-layer/full-forward, SIMD or
  CUDA model execution and no final cache/scheduler policy.
- Characterized the pinned `Qwen4ExpTextAttention` and
  `Qwen4ExpTextQSAIndexer` contract before native code and generated independent
  Class-C calibration, disjoint holdout, cache-transition and exact sparse
  selection evidence. A second bounded oracle configuration crosses the
  512-block limit at 512 and 513 complete blocks and covers deterministic ties
  and positive near-ties. Native floating checkpoints pass their separately
  calibrated contracts; block IDs, token positions, counts and ordering pass
  exact comparison. Kestrel-Q never supplies expected values.
- Added target QSA semantic integration for all 12 real QSA layers and
  fail-closed rejection of all 36 GDN IDs, missing/wrong semantics, malformed
  ordered `index_qk` splits, physical type/shape errors, invalid state/count/
  position/finiteness, selection-capacity errors and forbidden aliasing. The
  target BF16 descriptor reconciles exactly to 2,304 semantic bytes per token
  per layer and 27,648 across all 12 layers. Real payload bytes touched remain
  zero and no hard-coded artifact offset is used.
- Fixed two Task 2.7 research-validation findings before the clean gate. The
  canonical boolean selection mask preserves membership rather than top-k
  gather order, so the generator now verifies sorted membership while capturing
  `aten::topk` order independently. Initial calibration under-covered longer
  K/V and raw-index state arithmetic; two disjoint longer random calibration
  cases and value-by-value holdout checks now protect those classes. Checked
  64-bit arithmetic and a malformed physical split-shape regression also close
  scratch/state-span overflow and near-match binding paths.
- The first clean Task 2.7 CPU gate exposed an inconsistent state-construction
  status: capacity beyond the validated context limit was grouped with malformed
  arguments despite the public contract reserving `LIMIT_EXCEEDED`. State
  construction now clears a supplied output before validation, distinguishes
  invalid/zero capacity from an exceeded context limit, and the exact regression
  remains in the fail-closed suite before the clean rerun.
- The next clean rerun rejected a newly added test assumption rather than the
  selector: canonical QSA deliberately applies `min(limit, candidates)`, so a
  nominal limit above the available candidates is valid. The regression now
  exercises the actual invalid ranges—zero selection limit and a candidate
  count outside the 32-bit block-ID domain—without changing canonical behavior.
- Completed Task 2.8 and accepted ADR 0016. The production runtime now has an
  immutable, stateless C17 Qwen3.8 MoE scalar reference with FP32 router
  softmax, exact top-10 observation, selected routed-expert SwiGLU execution,
  a separate sigmoid-gated shared expert and final routed-plus-shared sum. It
  adds no PLE-value path, complete layer/full forward, SIMD/CUDA model kernel,
  expert cache, prefetch, residency policy or scheduler.
- Generated independent Class-C evidence before native comparison from the
  pinned Apache-2.0 Transformers `Qwen4ExpTextSparseMoeBlock`. Five reduced
  calibration cases, four disjoint holdout cases and seven 512-expert/top-10
  routing cases pass. Expert membership/order/count are exact-discrete; routed,
  shared and final checkpoints pass separate calibration-only contracts.
  Kestrel-Q never supplies expected values and no BF16 checkpoint was loaded.
- Added real semantic integration for all 48 target MoE configs, including the
  512-expert axes, ordered gate/up splits, shared bindings and physical types.
  Three bounded member views per layer re-derive 43 ordinary 3,072,000-byte
  experts, the 3,993,600-byte layer-2 mix and four 3,584,000-byte special-layer
  experts. The test touches zero real payload bytes; these sizes remain selected
  parameter footprints, not measured I/O/token.
- Characterization found that pinned CPU `torch.topk` tie behavior is a partial
  selection rather than a generic globally stable sort. Tier-B vectors now lock
  equal, kth-boundary, near-tie, expert-0/511 and dominant-set behavior exactly.
  Oracle generation initially failed closed because the global environment had
  `tokenizers` 0.22.2 instead of the pinned 0.23.1; the governed Task 1.4
  environment resolves the dependency without a production runtime dependency.
- Fixed a Task 2.8 MSVC C4701 warning caused by conservative data-flow analysis
  of shared-gate locals assigned through status-returning numeric helpers.
  Explicit initialization preserves fail-closed status propagation. Writable
  output/scratch alias checks also cover every F32 weight span before native
  execution.
- A final API-boundary review found that the first one-expert helper still
  accepted complete routed stacks even though it indexed only one expert. The
  corrected API accepts only the selected member's gate/up/down arrays, with
  exact shape and writable-alias validation; a regression prevents accidental
  reintroduction of all-512 materialization at this boundary.
- Final clean Release validation passes CPU 26/26 and CUDA 28/28, including all
  Task 2.0–2.7 regressions, the real 48-layer MoE structural gate and the
  deterministic independent oracle. Kestrel-Q emits no new `/W4` warnings;
  only the previously documented NVCC-generated external C4211 remains.
- Completed Task 2.9 and accepted ADR 0017. The production runtime now has a
  scalar C17 Qwen3.8 PLE value reference that consumes exact Task 2.4 logical
  intents, projects and gates table rows, applies the canonical nine-position
  dilation-3 depthwise convolution, and maintains an explicit transactional
  value state. Address state and value state remain separate.
- Generated pinned independent Class-C evidence before native comparison:
  four reduced calibration cases, three disjoint holdout cases, value-state
  transitions and three real-address-stream cases pass separate per-checkpoint
  contracts. Lookup/embedding identity and intent order are exact; Kestrel-Q
  never defines expected values and no BF16 checkpoint was downloaded.
- Added strict real target validation for all 128 logical PLE members, the
  fused IQ4_NL binding and six dense semantics. The table reconciles to
  28,800,138,240 packed bytes and the complete PLE family to 28,835,240,960.
  Sixteen bounded rows touch 1,440 logical packed bytes in 80 blocks under the
  8 MiB guard. No raw payload is committed and no throughput/residency claim is
  inferred.
- Corrected Task 2.9 development findings before the clean gate. The oracle
  generator initially shadowed its output-directory variable with a decode
  tensor; the native convolution draft tried to source dilated history from a
  current-token buffer; and the native validator attempted set subtraction
  against a dictionary. Root-cause fixes plus deterministic regeneration,
  prefix-prefill/decode equivalence and state regressions cover all three.
- `/W4` also caught reversed scalar numeric-helper arguments in the first PLE
  value draft. Correct signatures, explicit position/block arithmetic checks
  and output/scratch alias rejection now preserve fail-closed transactional
  behavior. The synchronous GGUF row provider remains correctness plumbing,
  not a disk cache, prefetcher, scheduler or full-layer implementation;
  `KQ-BACKLOG-BENCH-002` remains deferred.
- Final clean Task 2.9 Release validation passes CPU 29/29 and CUDA 31/31,
  including all Task 2.0–2.8 regressions, the bounded real PLE-row gate and the
  deterministic independent oracle. Kestrel-Q emits no new `/W4` warning;
  only the previously documented NVCC-generated external C4211 remains.

### Added

- Initial repository structure.
- Project vision and roadmap.
- High-level project plan and task backlog.
- AI-agent operating contract.
- Initial architecture direction.
- Benchmark policy.
- Initial architecture decision records.
- Minimal C/CMake build skeleton.
- Optional `KQ_ENABLE_CUDA` smoke backend with a C ABI, checked CUDA operations,
  device diagnostics and host-validated kernel result.
- CPU-only and CUDA-enabled CTest coverage for the smoke paths.
- CUDA-only `kq_cuda_bandwidth` benchmark with host and CUDA-event timing,
  byte-complete transfer validation, setup metrics and stable CSV output.
- PowerShell evidence harness with robust `nvidia-smi` resolution and PCIe link
  sampling, plus immutable Task 0.2E KQ-01 raw CSV/text evidence.
- Reproducible Task 1.0 model-source manifest with a fail-closed, explicit
  non-weight metadata allowlist and hashes/sizes for the complete upstream file
  inventory.
- Model source baseline, model-license boundary and completed artifact register;
  `KQ-MODEL-ARTIFACT-001` records the exact merged Unsloth UD-Q4_K_XL GGUF
  size, SHA-256, GGUF v3 metadata, tensor-type evidence and pinned provenance.
- Completed Task 1.0 after read-only registration of the maintainer's local
  derived GGUF; no model artifact was copied into or tracked by the repository.
- Implementation-grade Qwen3.8-Flash-Next architecture and runtime-state
  baselines covering the exact 48-layer GDN/QSA schedule, Gated Residual, MoE,
  deterministic n-gram/PLE addressing, MTP boundaries, and prefill/decode state
  transitions.
- Machine-readable Task 1.1 evidence with stable `KQ-ARCH-*` claim IDs, exact
  source revisions, source locations, licenses and implementation file hashes.
- Fail-closed, standard-library Task 1.2 research tooling for bounded
  Safetensors-header Range capture and deterministic offline tensor analysis,
  with exact static, idealized-quantization, per-layer/expert and persistent
  runtime-state evidence; failure audits are preserved against later overwrite
  or cleanup, and each inventory row's `classification_rule` records its exact
  versioned component rule rather than only the global rule version.
- Canonical tensor-inventory and KQ-01 footprint documents, including exact
  text/vision/MTP/PLE/expert families and explicit placement-candidate limits.
- Fail-closed, standard-library Task 1.3 tooling for read-only GGUF structure,
  bounded upstream split-header audit and deterministic canonical mapping, plus
  five machine-readable evidence artifacts and exact packed quantization/type,
  family, per-layer and per-expert footprints and embedded imatrix/calibration
  provenance findings without unsupported sensitivity claims.
- A deterministic original/synthetic ten-case reference prompt suite and exact
  weight-independent tokenizer, chat-template and PLE address goldens, plus
  machine-readable GR/GDN/QSA/MoE/final checkpoint and full-model plans.
- Fail-closed Task 1.4 generation/validation tooling with pinned metadata,
  source-revision/hash and dependency checks. The checks caught a one-nibble
  source-hash transcription error and a Transformers `BatchEncoding` handling
  error before a valid manifest; a final scan also removed 19 zero-tensor
  llama.cpp vocabulary GGUF fixtures from ignored cache and added source-tree
  fixture rejection. Corrected regeneration is byte-identical.
- Production `kq_status`, Win32 `kq_file`/bounded-view and target-first `kq_gguf`
  APIs with checked 64-bit arithmetic, bounded file-backed strings, seven
  target tensor block geometries, duplicate/alignment/span validation and
  deterministic cleanup on every tested failure path.
- `kq-inspect` structural summary CLI plus deterministic runtime-created GGUF
  fixtures covering one valid seven-type container and 21 malformed cases,
  including truncation, strings/arrays, overflow, rank/dimensions, type,
  duplicate name, alignment/span and quantized-geometry failures.
- Opt-in `KQ_GGUF_PATH` integration coverage asserting the exact artifact size,
  version, architecture, metadata/tensor counts, packed bytes, overhead and all
  tensor-type counts without committing or modifying model payload.
- Production `kq_model` C17 API with stable semantic IDs, canonical
  component/role/layer descriptors, explicit binding cardinality, expert/fused
  geometry, runtime scope and preliminary placement annotations. Construction
  reads only Task 2.0 metadata/descriptors and has no global mutable state.
- `kq-inspect --semantic-summary`, exact semantic lookup and deterministic TSV
  dump modes, all retaining `payload_bytes_accessed = 0`.
- A complete in-memory Task 2.1 target fixture with 20 fail-closed mutations,
  real-artifact semantic integration, and a standard-library Python validator
  that uses the Epic 1 CSV only as a test/research oracle; production has no
  research-file or Python dependency.
