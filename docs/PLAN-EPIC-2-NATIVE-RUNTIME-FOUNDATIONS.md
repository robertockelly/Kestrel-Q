# PLAN-EPIC-2-NATIVE-RUNTIME-FOUNDATIONS.md

Status: **IN PROGRESS — TASKS 2.0–2.2 COMPLETE / PASS**

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
4. Task 2.3 — Native tokenizer.
5. Task 2.4 — Native PLE address engine.
6. Task 2.5 — CPU reference tensor primitives.

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

Current next gate: Task 2.3 native tokenizer.
Task 2.3 is **NOT STARTED**.
