# PLAN-EPIC-2-NATIVE-RUNTIME-FOUNDATIONS.md

Status: **IN PROGRESS — TASKS 2.0–2.9 COMPLETE / PASS**

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
8. Task 2.7 — QSA reference operator and independent golden vectors.
   **COMPLETE / PASS**
9. Task 2.8 — MoE reference operator and independent golden vectors.
   **COMPLETE / PASS**
10. Task 2.9 — PLE value reference operator and independent golden vectors.
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

Task 2.7 adds the second model-specific scalar reference boundary. The pinned
Class-C Qwen QSA module provides reduced-shape calibration, disjoint holdout,
cache-state and exact threshold-crossing sparse-selection expectations before
native comparison. Scalar prefill/decode and explicit K/V/raw-index state pass;
12 real QSA bindings validate and 36 GDN layers reject without payload access.
ADR 0015 is ACCEPTED.

Task 2.8 characterizes the exact Qwen3.8 MoE router, selected routed experts,
separate shared expert/gate and final combination before implementing the
scalar CPU reference. Independent Tier-A calibration/holdout and Tier-B
512-expert/top-10 routing evidence pass; all 48 target bindings validate with
zero model-payload bytes touched. ADR 0016 is ACCEPTED. No GR composition, PLE
value execution, complete layer, full forward, SIMD/CUDA model kernel,
cache/prefetch policy or scheduler is introduced; the next model-operator task
is **NOT STARTED**.

Task 2.9 characterizes and implements the scalar PLE value path over exact Task
2.4 intents. Independent calibration, disjoint holdout, state-transition and
address-integration evidence pass. The real target validates all 128 members,
the fused IQ4_NL table and six dense bindings; a bounded synchronous provider
touches 1,440 logical packed bytes. ADR 0017 is ACCEPTED. The next integration
task is **NOT STARTED**, and no final PLE cache/prefetch/residency policy,
complete layer or full forward is introduced.
