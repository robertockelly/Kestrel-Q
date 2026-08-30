# Native MoE reference

Task 2.8 adds the scalar C17 reference implementation of the
Qwen3.8-Flash-Next sparse MoE block. It is a correctness boundary, not a
performance implementation.

## Scope

The operator owns router projection, FP32 softmax, exact top-10 selection and
normalization, selected routed-expert SwiGLU execution, the separate shared
expert and sigmoid gate, and the final routed-plus-shared sum. It is stateless
across tokens. Residual addition belongs to a future complete-layer operator.

The production implementation is in `src/kq_moe.c` and
`src/kq_moe_qwen38.c`. Generic arithmetic remains in the Task 2.5 numeric
layer. Production code does not read research JSON and does not link Python,
Transformers, PyTorch, llama.cpp or GGML.

## Independent evidence

The offline generator instantiates the pinned Apache-2.0 Transformers
`Qwen4ExpTextSparseMoeBlock` before native validation. Tier A uses hidden size
6, four routed experts, top-2 and intermediate width 4. Five calibration and
four disjoint holdout cases cover lengths 1 through 5, zero, random, repeated
and alternating inputs. Tier B retains the released 512-expert/top-10 router
while reducing hidden/intermediate dimensions and covers equal logits, the
kth boundary, near ties, expert 0/511 and dominant-set boundaries.

Routing IDs, membership, order and count compare exactly. Each floating
checkpoint has a calibration-only limit and the disjoint holdout must pass it.
Native results never define expected values.

## Target structure

All 48 text layers expose the same semantic MoE roles:

- router `[512,2560]`;
- routed gate/up/down stacks with expert axis 512 and width 640;
- ordered physical gate/up split;
- separate shared gate/up/down width 640;
- scalar shared-expert gate from the hidden input.

The real integration test constructs BF16-storage and F32-reference
descriptors for all 48 layers. It opens one bounded down, gate and up expert
member view per layer and never calls the payload accessor. The independently
derived packed member groups are 3,072,000 bytes in 43 layers, 3,993,600 bytes
in layer 2, and 3,584,000 bytes in layers 4, 30, 46 and 47. These are selected
parameter footprints, not bytes read per token and not an I/O measurement.

## Memory and execution boundary

The immutable config is 152 bytes in the observed Release build. The reduced
reference workspace is 264 bytes. The released topology describes 70,736
bytes of reusable caller-owned scratch: 4,096 router, 80 top-k, 28,160 one
expert, 10,240 routed accumulation and 28,160 shared-path bytes.

One descriptive KQ-01 Release run of the length-1 Tier-A case was repeated 21
times through the no-output timing observer. Median first-token phase times
were 1,200 ns router, 1,200 ns selected-routed path, 700 ns shared-plus-final,
and 3,100 ns total (observed ranges 1,100–1,400; 1,100–1,500; 600–800; and
3,000–3,800 ns respectively). The real structural run constructed all 48
configs in approximately 3.0 ms total. These are characterization values, not
performance guarantees or optimization baselines.

Only selected experts are evaluated. The real structural test touches zero
model-payload bytes. Task 2.8 adds no cache, residency, prefetch, scheduler,
PLE-value path, complete layer, SIMD or CUDA model kernel.

The public one-expert path accepts member-local gate/up/down arrays rather than
a full 512-expert F32 stack. This keeps the execution API compatible with the
bounded semantic-ID to Task 2.2 member-view to Task 2.5 decode pipeline; the
current real integration stops before payload decode by design.

## Findings

Pinned CPU `torch.topk` ties are not represented accurately by a generic
global stable sort. The evidence-backed native selector seeds the first k
entries, admits later entries only on a strictly greater value, evicts the
lowest ID among tied current minima, then orders the retained set by descending
value and ascending ID. Tier B locks that exact pinned behavior.

The first MSVC build conservatively reported C4701 for shared-gate locals set
through status-returning numeric helpers. Explicit initialization resolved the
warning without changing status propagation; clean builds guard the fix.

`KQ-BACKLOG-BENCH-002` remains deferred. Expert-cache and storage policy require
separate measured work.
