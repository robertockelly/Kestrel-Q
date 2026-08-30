# First Correct Native Token readiness

Status after Task 2.11: **QUANTIZED SINGLE-LAYER BRIDGE READY; MODEL LOOP NOT STARTED**

The exact target artifact can now supply every weight needed by an ordinary
GDN, QSA or PLE-GDN decoder layer through bounded semantic operations. All 48
layers preflight, the three representative families execute prefill+decode
against an independent target oracle, and persistent state rolls back on
provider failure. There is no unexplained model-layer blocker.

The next first-token milestone still requires these explicit model-level
pieces:

1. bounded token-embedding lookup from `token_embd.weight` and construction of
   the four initial hyper-connection branches;
2. ownership/orchestration of 48 independent layer states and the ordered
   48-layer prefill/decode loop;
3. final hyper-connection mixer semantics and target weights;
4. final model RMS norm;
5. LM-head row-dot/logit production from `output.weight`;
6. exact greedy argmax with deterministic tie policy;
7. native token decode through the accepted Task 2.3 tokenizer.

Each addition must retain the semantic-provider boundary, bounded scratch,
transactional visible state and independent-oracle discipline. A first-token
correctness task must not silently become a final cache/scheduler or optimized
kernel task.

No final placement policy is implied by Task 2.11. The correctness provider is
synchronous and repeatedly reads requested logical weights. Expert residency,
PLE row caching, transfer overlap and eviction require separate measurement and
design. `KQ-BACKLOG-BENCH-002` remains **DEFERRED / REQUIRED BEFORE FINAL
SCHEDULER DESIGN**.
