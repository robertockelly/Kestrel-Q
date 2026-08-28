# MODEL-ARCHITECTURE-EVIDENCE-GUIDE.md

Task 1.1 must create a machine-readable evidence file under:

```text
research/model-architecture/Qwen3.8-Flash-Next/<HF_REVISION>/evidence.json
```

Suggested schema:

```json
{
  "schema_version": 1,
  "model": "Qwen/Qwen3.8-Flash-Next",
  "model_revision": "<sha>",
  "research_revision": "<sha>",
  "secondary_sources": [],
  "claims": [
    {
      "id": "KQ-ARCH-TEXT-001",
      "topic": "text",
      "claim": "short factual statement",
      "source_tier": "A",
      "source": "Qwen/Qwen3.8-Flash-Next",
      "revision": "<sha>",
      "artifact": "config.json",
      "location": "text_config.num_hidden_layers",
      "confidence": "verified",
      "notes": ""
    }
  ]
}
```

## Rules

- One stable claim ID per material architectural fact.
- Exact revision for every versioned source.
- Prefer section/page/function/symbol/config paths over long excerpts.
- Tier A wins when the evidence hierarchy resolves a conflict.
- Record disagreements explicitly; do not silently reconcile inconsistent sources.
- Do not copy large passages from source code, papers or documentation.
- Implementation-derived claims must identify implementation source, revision and license.
- Major claims in `docs/MODEL-ARCHITECTURE.md` should be traceable to one or more claim IDs.
