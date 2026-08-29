# Qwen3.8-Flash-Next reference-oracle strategy

Status: **PINNED / VERIFIED FOR TASK 1.4**

Kestrel-Q maintains two independent correctness classes. Canonical behavior and
behavior of one mixed-quantized derived artifact are related, but they are not
interchangeable and must not share an unexplained tolerance.

Both model-artifact classes remain subject to the pinned Qwen Community License
1.0 boundary documented in Task 1.0. Apache-2.0 (Transformers) and MIT
(llama.cpp) describe the oracle implementations only and do not relicense the
model or the derived GGUF.

## Class C — canonical semantics

Class C is anchored in:

- `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- official Qwen research revision
  `69885871a64393807d988b27b1b5e380e8f28526`; and
- `huggingface/transformers` revision
  `805a9e939fa8c1bff8d8ffdf041c051b71a914aa`, under Apache-2.0,
  when an executable implementation is required.

The Transformers revision registers `qwen4_exp`, `qwen4_exp_text` and
`Qwen4ExpForConditionalGeneration`, contains the architecture implementation
characterized in Task 1.1, and executes the pinned official `Qwen2Tokenizer`
and chat template offline without remote code. Task 1.4 executed only those
weight-independent paths. The canonical BF16 model path is source-verified and
contracted, but was not loaded on KQ-01.

Relevant pinned Transformers source SHA-256 values are recorded in
`research/goldens/Qwen3.8-Flash-Next/manifest.json`. The license remains with
Transformers; no implementation source was copied into Kestrel-Q.

Status:

- tokenizer and chat-template oracle: **PINNED / EXECUTED / VERIFIED**;
- PLE address semantics: **PINNED / SOURCE-DERIVED / VERIFIED** against the
  official config and the independently registered GGUF metadata constants;
- full BF16 forward oracle: **DEFERRED_CAPABLE_REFERENCE_ENV**.

## Class Q — exact GGUF semantics

Class Q applies only to `KQ-MODEL-ARTIFACT-001`:

- `Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf`;
- 111,334,654,400 bytes;
- SHA-256
  `8003d03db822cb089ab55caa5fff0f82f6f4199fe6ac58368bc9989365b6f8c2`;
- Unsloth revision
  `c8b5954a88c2775c546b92593eda40ea041d3176`.

The independent runtime is `ggml-org/llama.cpp` revision
`90c26fcd4b2114b4aa39d09d69318cb8f438d27a`, under MIT. It was selected only
after source and build verification established:

- `qwen4exp` architecture registration and a complete model graph;
- CPU traits for every artifact type: `F32`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`,
  `IQ4_NL` and `BF16`;
- ordinary `llama-cli` text generation controls;
- explicit greedy settings and `--spec-type none`; and
- explicit model, context, mmap/load-mode and layer-offload controls.

An isolated CPU-only Release build completed on KQ-01 and reported
`0.3.0-dev (build 1, commit 90c26fc)`, built with MSVC 19.44.35228 x64. The
first verification configuration set `LLAMA_BUILD_SERVER=OFF`, but at this
revision `llama-cli` is nested under the server-enabled CMake branch, so the
target did not exist. Reconfiguration with `LLAMA_BUILD_SERVER=ON` produced the
binary. This was a verification-build configuration issue; no model was loaded.

The 103.688 GiB artifact exceeds KQ-01's 32 GiB RAM and 10 GiB VRAM. With SATA
backing, a local full run would invite uncontrolled paging. Class-Q model output
therefore remains **DEFERRED_CAPABLE_REFERENCE_ENV**.

## Comparison and independence rules

- Exact equality applies to rendered UTF-8, token IDs, token counts, PLE
  addresses, deterministic selected IDs and greedy IDs.
- Every floating checkpoint records exact dtype and shape, `atol`, `rtol`,
  NaN/Inf policy and calibration status.
- Until independent runs exist, floating tolerances remain
  `TO_BE_CALIBRATED_FROM_REFERENCE_RUNS`; null tolerances are not a wildcard.
- Quantized Class-Q drift may not weaken a Class-C gate.
- Kestrel-Q may consume these vectors, but its own output may never be the only
  oracle or generate a replacement golden.
- Future CPU and CUDA paths must satisfy the applicable class gate before they
  are called correct.

## Resource-deferred gates

The exact capable-machine contracts are machine-readable in:

- `canonical/full-model-vector-plan.json` for Class C; and
- `quantized/gguf-vector-plan.json` for Class Q.

The Class-C gate requires the exact pinned BF16 checkpoint on a machine with
aggregate accelerator and host/storage headroom. The Class-Q gate uses a fixed
CPU-only llama.cpp invocation, explicit mmap, zero GPU layers, greedy sampling
and no speculation on a machine with at least 256 GiB RAM and NVMe storage.
Neither gate was executed or fabricated during Task 1.4.
