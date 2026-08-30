# PLAN-EPIC-2-NATIVE-RUNTIME-FOUNDATIONS.md

Status: **IN PROGRESS — TASKS 2.0–2.6 COMPLETE / PASS**

## Epic 2 — Native Runtime Foundations

Entry checkpoint: `4f9f1dc35420d7987d75b0524cc204eb993d7618`.

Epic 1 is COMPLETE/PASS. Epic 2 starts the production C runtime.

Initial task order:

1. Task 2.0 — Native GGUF introspection & Windows memory-mapped container layer.
   **COMPLETE / PASS**
2. Task 2.1 — Canonical tensor registry & semantic descriptors.
   **COMPLETE / PASS**
3. Task 2.2 — Quantized tensor views & block geometry.
   **COMPLETE / PASS**
4. Task 2.3 — Native tokenizer and chat formatting.
   **COMPLETE / PASS**
5. Task 2.4 — Native PLE address engine. **COMPLETE / PASS**
6. Task 2.5 — CPU reference numeric primitives and quantized dequantization.
   **COMPLETE / PASS**
7. Task 2.6 — GDN reference operator and independent golden vectors.
   **COMPLETE / PASS**

Constraints:

- C17 runtime core.
- Windows first.
- CUDA behind C ABI.
- No llama.cpp/GGML runtime dependency.
- No Python production dependency.
- Fail closed on malformed data.
- Checked 64-bit file arithmetic.
- No eager full-model allocation.
- `CHANGELOG.md` updated for every material iteration.
- `KQ-BACKLOG-BENCH-002` stays deferred until scheduler design.

Task 2.6 builds the first model-specific operator above the Task 2.5 scalar
boundary. The pinned Class-C Qwen GDN module provides reduced-shape calibration,
disjoint holdout and state-transition expectations before native comparison.
Scalar prefill/decode and explicit convolution/recurrent state pass; 36 real
GDN bindings validate and 12 QSA layers reject without payload access. ADR 0014
is ACCEPTED. No GR composition, QSA, MoE, PLE value execution, complete layer,
full forward, SIMD/CUDA model kernel, cache, prefetch or scheduler policy is
introduced; the next model-operator task is **NOT STARTED**.
