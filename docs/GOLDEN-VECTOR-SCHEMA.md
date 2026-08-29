# GOLDEN-VECTOR-SCHEMA.md

Status: **ACCEPTED / TASK 1.4**

## Golden classes

- `C` — canonical Qwen semantics.
- `Q` — exact verified GGUF artifact semantics.

## Status values

- `GENERATED_VERIFIED`
- `PLANNED_REFERENCE_REQUIRED`
- `DEFERRED_CAPABLE_REFERENCE_ENV`
- `NOT_APPLICABLE`

Never mark an asset generated without a physical artifact and SHA-256.

## Comparison modes

### EXACT

Use for token IDs, rendered UTF-8, PLE addresses, deterministic selected IDs and greedy token IDs.

### NUMERIC_TOLERANCE

Floating vectors require:

```json
{
  "dtype": "BF16",
  "shape": [1, 2560],
  "atol": null,
  "rtol": null,
  "tolerance_status": "TO_BE_CALIBRATED_FROM_REFERENCE_RUNS",
  "nan_policy": "FORBID_UNLESS_REFERENCE_HAS_NAN",
  "inf_policy": "MATCH_REFERENCE"
}
```

Do not invent permissive tolerances before independent reference runs exist.

## Suggested IDs

```text
KQ-GOLD-TOK-*
KQ-GOLD-CHAT-*
KQ-GOLD-PLE-*
KQ-GOLD-GR-*
KQ-GOLD-GDN-*
KQ-GOLD-QSA-*
KQ-GOLD-MOE-*
KQ-GOLD-LOGIT-*
KQ-GOLD-GEN-*
```

## Prompt identity

Every prompt records stable ID, canonical UTF-8 or structured messages, SHA-256, mode and notes.

## Oracle identity

Every vector records model/revision, optional GGUF SHA, oracle implementation/revision/license, backend/device, dependency versions and generation configuration.

## Full-model checkpoint contract

Where exposed, future captures should cover:

- token IDs;
- PLE addresses;
- early hidden checkpoints;
- MoE top-10 IDs/weights;
- GDN states;
- QSA selected blocks/index state;
- middle/late hidden checkpoints;
- final hidden state;
- logits/top-k;
- greedy token and short generation.

If Class Q cannot expose internal checkpoints without invasive patches, final-output Class Q goldens remain valid while internal semantic checkpoints come from Class C.

## Canonical serialization

JSON assets use UTF-8, deterministic key/list ordering where controlled, exact large integers, no user-specific local paths and no volatile timestamps in byte-stable outputs unless semantically required.

Use SHA-256 over exact committed bytes.
