# TASKS-TASK-2.12-FIRST-CORRECT-NATIVE-TOKEN.md

## Baseline
- [x] Record HEAD/worktree after Task 2.11 checkpoint.
- [x] Confirm Task 2.11 COMPLETE/PASS.
- [x] Confirm Task 2.12 NOT STARTED.
- [x] Keep KQ-BACKLOG-BENCH-002 DEFERRED.

## Characterization
- [x] Create MODEL-EXECUTION-CONTRACT.md.
- [x] Pin embedding semantics.
- [x] Pin final norm.
- [x] Pin LM-head shape/orientation/dtype.
- [x] Pin logits dtype.
- [x] Pin 48-layer order.
- [x] Pin state initialization.
- [x] Pin greedy argmax/tie semantics.
- [x] Pin token decode behavior.

## Model config/state
- [x] Immutable model-exec config.
- [x] Validate embedding.
- [x] Validate 48 layers.
- [x] Validate final norm.
- [x] Validate LM head.
- [x] Explicit small context capacity.
- [x] Global persistent state composition.
- [x] Reset/state-size.
- [x] Transactional model-call policy.

## Embedding
- [x] Native canonical token IDs.
- [x] Bounded embedding-row lookup.
- [x] No full embedding matrix materialization.
- [x] Invalid/padded token fail closed.

## 48-layer executor
- [x] Layer 0..47 exact order.
- [x] No skipped/reordered layer.
- [x] Correct positions/masks.
- [x] Correct per-layer state.
- [x] Prefill path.
- [x] Failure rollback early/mid/late.

## Output path
- [x] Final RMSNorm.
- [x] Full 248320 logits.
- [x] No full LM-head F32 materialization.
- [x] Greedy argmax exact.
- [x] Native selected-token decode.

## M1 prompt
- [x] Select shortest suitable governed raw prompt.
- [x] Record prompt ID/hash/text.
- [x] Native tokenizer exact vs committed golden IDs.
- [x] Smallest safe context capacity.

## Independent full-model oracle
- [x] Pinned llama.cpp helper.
- [x] Explicit canonical token-ID input.
- [x] Same verified GGUF.
- [x] Small context.
- [x] Safe mmap/load.
- [x] Oracle argmax/token.
- [x] Top-N/logit evidence where practical.
- [x] No self-oracle.
- [x] Fail closed if oracle not safely executable.

## M1 correctness
- [x] Native full 48-layer run completes.
- [x] Final hidden finite.
- [x] Final norm finite.
- [x] Logits finite.
- [x] Native argmax == oracle argmax.
- [x] Native decoded token recorded.
- [x] Exact QSA/MoE/PLE discrete invariants preserved.
- [x] Stronger float/logit comparison where available.

## Payload/memory
- [x] Logical payload bytes recorded.
- [x] Blocks recorded.
- [x] Unique semantic tensors recorded.
- [x] Expert requests recorded.
- [x] PLE row requests recorded.
- [x] LM-head bytes recorded.
- [x] M1 payload ceiling documented/respected.
- [x] No whole target matrix/model F32 materialization.
- [x] Total persistent state recorded.
- [x] Peak scratch/logits recorded.

## Fail closed
- [x] Missing/wrong embedding.
- [x] Invalid token ID.
- [x] Wrong layer count/order.
- [x] State-capacity overflow.
- [x] Early/mid/late provider failure rollback.
- [x] Missing/wrong final norm.
- [x] Missing/wrong LM head.
- [x] Logits capacity.
- [x] Non-finite output handling.
- [x] Argmax tie behavior.
- [x] Invalid selected token/decode failure.
- [x] Vision/MTP unsupported fail closed.

## CLI
- [x] Minimal kq-run or equivalent.
- [x] Raw prompt.
- [x] max-new-tokens 1.
- [x] greedy only.
- [x] Progress by phase/layer.
- [x] No huge activation/model dumps.

## Evidence
- [x] first-token contract.
- [x] oracle result.
- [x] native result.
- [x] validation result.
- [x] manifest/SHA.
- [x] No raw model weights/logit megadumps.

## Regression/governance
- [x] CPU clean PASS.
- [x] CUDA clean PASS.
- [x] Task 2.0-2.11 PASS.
- [x] No new warnings.
- [x] ADR 0020 ACCEPTED.
- [x] ARCHITECTURE/TASKS/ROADMAP/Epic 2 updated.
- [x] CHANGELOG updated.
- [x] KQ-BACKLOG-BENCH-002 DEFERRED.
- [x] git diff --check PASS.

## Safety
- [x] No sampling beyond greedy.
- [x] No vision/MTP.
- [x] No batching.
- [x] No final cache/scheduler optimization.
- [x] No SIMD/CUDA model kernels.
- [x] No production Python/llama dependency.
- [x] No tracked model weights/secrets/local paths.
