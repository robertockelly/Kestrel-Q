# PLAN-EPIC-2-NATIVE-RUNTIME-FOUNDATIONS.md

Status: **COMPLETE / PASS — TASKS 2.0–2.13 COMPLETE / PASS**

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
11. Task 2.10 — Canonical transformer-layer composition and GR/residual
    reference. **COMPLETE / PASS**
12. Task 2.11 — Target quantized layer execution through bounded semantic
    weight access. **COMPLETE / PASS**
13. Task 2.12 — First Correct Native Token. **COMPLETE / PASS**
14. Task 2.13 — Native Multi-Token Greedy Decode & Session Continuity.
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

Task 2.10 pins the complete decoder-layer/GR contract and composes the existing
operators for 35 ordinary-GDN, 12 QSA and one PLE-GDN family. Independent
complete-layer calibration/holdout, prefill/decode/reset and transactional
rollback pass; 48/48 target descriptors validate with zero payload access.
ADR 0018 is ACCEPTED. Embedding, a 48-layer executor, final norm, LM head,
logits and the final scheduler remain **NOT STARTED**.

Task 2.11 connects those equations to the exact registered GGUF through a
semantic weight provider backed only by Task 2.2 views and Task 2.5 numerics.
Independent pinned llama.cpp decode plus pinned Transformers equations validates
four calibration profiles and a disjoint holdout for ordinary GDN, QSA and
PLE-GDN prefill/decode. Exact route/access/PLE decisions, 48/48 provider
preflight and transactional failures pass under the 768 MiB logical payload
ceiling. ADR 0019 is ACCEPTED. At that checkpoint, embedding, full 48-layer
orchestration, final mixing, LM head/logits/argmax and native token decode were
not started; Task 2.12 is the separately governed completion of that path.

Task 2.12 composes the accepted components into M1: native tokenizer IDs,
bounded embedding rows, all 48 target layers in canonical order, the final
qwen4exp hyper-connection mixer, all 248,320 logits, stable greedy argmax and
native decode. Pinned llama.cpp independently selects the same token 271 from
the explicit committed IDs. Three real provider faults recover to the same
control token. ADR 0020 is ACCEPTED. Epic 2 remains **IN PROGRESS** because
this plan does not define M1 as its final acceptance gate; no post-M1 task is
started or invented here.

Task 2.13 extends M1 into a bounded true incremental continuation. The prompt
is prefilled exactly once and selected IDs feed one-token decode directly.
Pinned llama.cpp independently produces `[271, 248068, 198, 760]`; native
Kestrel-Q matches the full sequence exactly, preserves the token-271 M1 gate
and proves later-step rollback/retry without resetting the successful prefix.
ADR 0021 is ACCEPTED.

## Closure review — 2026-08-31

`EPIC_2_NATIVE_RUNTIME_FOUNDATIONS = COMPLETE / PASS`.

The governance-only closure review reached conclusion A: the existing Epic 2
objectives are fully satisfied and the status had not yet been advanced. The
decision is based on the committed governance history rather than a new
technical objective:

- the original Project Plan Phase 1 objective was safe target-model loading and
  introspection without full inference, with exit gates for complete expected
  tensor enumeration/validation, fail-closed malformed-artifact handling and no
  full-model RAM-residency requirement;
- the first committed Epic 2 plan started the production C runtime and listed
  Tasks 2.0–2.5 as its initial task order under the constraints above; and
- later governed revisions extended that order through Task 2.13. Every listed
  task and its ADR/evidence gate is COMPLETE/PASS, and no later Epic 2 task or
  unsatisfied closure criterion is defined.

The first correct token and bounded multi-token continuation are additional
governed correctness milestones completed within Epic 2; they are not used to
retrofit scheduler, optimization, product or long-context requirements into
the original runtime-foundation objective.

### Objective traceability

| Objective / acceptance criterion | Satisfying task(s) | Governing evidence | Status |
| --- | --- | --- | --- |
| Native read-only GGUF access, mapping, metadata and diagnostics | 2.0 | ADR 0008; real/synthetic GGUF gates | PASS |
| Canonical semantic tensor identity and full physical reconciliation | 2.1 | ADR 0009; 1,294 semantic / 1,224 physical coverage | PASS |
| Bounded quantized tensor/member views | 2.2 | ADR 0010; seven-type geometry and bounded-view gates | PASS |
| Native canonical tokenizer and supported chat subset | 2.3 / 2.3A | ADR 0011; original and differential oracle vectors | PASS |
| Deterministic PLE address intents and bounded state | 2.4 | ADR 0012; exact sequence/decode vectors | PASS |
| Scalar CPU numerics and seven-format dequantization | 2.5 | ADR 0013; independent calibration/holdout and real samples | PASS |
| GDN scalar reference operator | 2.6 | ADR 0014; independent operator/state vectors | PASS |
| QSA scalar reference operator and exact sparse selection | 2.7 | ADR 0015; selection/state and 12-layer gates | PASS |
| MoE routed/shared scalar reference | 2.8 | ADR 0016; exact 512/top-10 routing evidence | PASS |
| PLE value scalar reference | 2.9 | ADR 0017; value/state/address-integration evidence | PASS |
| Canonical transformer-layer and GR composition | 2.10 | ADR 0018; three-family layer evidence | PASS |
| Real quantized target-layer execution | 2.11 | ADR 0019; provider/oracle and 48-layer preflight | PASS |
| First correct native token | 2.12 | ADR 0020; independent token-271 Class-Q oracle | PASS |
| True incremental multi-token greedy continuation | 2.13 | ADR 0021; exact four-token sequence and state continuity | PASS |
| Independent oracle validation | 2.3–2.13 | Pinned Class C/Class Q evidence and deterministic manifests | PASS |
| Fail-closed and transactional failure behavior | 2.0–2.13 | Task checklists, malformed/near-match and rollback gates | PASS |
| No committed model weights, secrets or local artifact paths | 2.0–2.13 | Per-task repository-safety gates | PASS |
| CPU/CUDA regression preservation | 2.0–2.13 | Latest published clean Release suites: CPU 40/40, CUDA 42/42 | PASS |

### Deferred and future work classification

None of the following is an Epic 2 closure blocker:

| Item | Classification from the existing roadmap |
| --- | --- |
| `KQ-BACKLOG-BENCH-001` | D — deferred benchmark gate before production WDDM VRAM capacity policy |
| `KQ-BACKLOG-BENCH-002` | D — deferred benchmark gate required before final scheduler/residency design |
| CUDA optimized model kernels | B — later Epic 4 / R3 work |
| SIMD optimization | B — later performance work under R5; no task is currently defined |
| Expert cache/residency | B — later Epic 5 / R4 memory-tiering work |
| PLE disk cache/prefetch | B, with D gate — later Epic 5 / R4 work gated by `KQ-BACKLOG-BENCH-002` |
| Scheduler | B, with D gate — later Epic 5 / R4 and R5 work; final design requires `KQ-BACKLOG-BENCH-002` |
| Memory tiering | B — later Epic 5 / R4 work |
| Sampling | B — remaining Epic 3 / R2 work |
| Batching | B — future work outside Epic 2; no task is currently defined |
| Long-context production execution | B — later Epic 7 / R6 work |
| Session serialization | B — later Epic 7 / R6 work |
| MTP/speculative decoding | B — future capability outside the accepted initial text-only path; no task is defined |
| Vision | B — future capability outside the accepted initial text-only path; no task is defined |

`KQ-BACKLOG-BENCH-002` therefore remains **DEFERRED / REQUIRED BEFORE FINAL
SCHEDULER DESIGN** and is not required for Epic 2 closure.

The next existing roadmap phase is **Epic 3 — CPU correctness engine / R2 —
Compute correctly**. No numbered next task or task-specific plan is defined, so
`NEXT_GOVERNED_TASK = NOT_DEFINED`. This closure review starts no subsequent
implementation.
