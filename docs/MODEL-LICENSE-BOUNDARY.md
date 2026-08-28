# Qwen3.8-Flash-Next model-license boundary

Status: **VERIFIED FOR TASK 1.0**

This document records provenance and license text facts. It is not legal advice.
Commercial distribution, hosting or product integration should receive review
from qualified counsel based on the intended facts of use.

## 1. Separate licensed works

Kestrel-Q source code remains licensed under the repository's **Apache License
2.0**. That license applies to Kestrel-Q code and documentation contributed
under the repository's provenance rules.

The official Qwen model repository identifies its model artifacts, weights,
parameters, configuration, inference code and associated documentation under
**Qwen Community License 1.0**. Apache-2.0 does not relicense those artifacts.

Kestrel-Q therefore treats model files as external artifacts with their own
license and provenance. Task 1.0 does not redistribute upstream model files or
license text in the Kestrel-Q source tree.

## 2. Pinned license evidence

- repository: `Qwen/Qwen3.8-Flash-Next`;
- revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
- path: `LICENSE`;
- upstream Git blob: `9557a8961782792814714b7488346732c043793d`;
- size: **3,235 bytes**;
- downloaded SHA-256:
  `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`;
- displayed identifier: **Qwen Community License 1.0**;
- copyright notice: **Copyright (c) 2026 Qwen**.

Canonical URL:

`https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE`

## 3. Factual terms relevant to Kestrel-Q

The pinned text grants broad permissions subject to stated conditions. Terms
material to this project's roadmap include:

1. Copies or substantial portions must retain the Qwen copyright and permission
   notice.
2. If a licensee uses the software or derivatives in a commercial product or
   service exceeding either 100,000,000 monthly active users or USD 20,000,000
   equivalent monthly revenue, the respective model name must be prominently
   displayed in that product or service's user interface.
3. A licensee or affiliate conducting a commercial **Model as a Service** or
   **AI Work Assistant** business must obtain a separate license from Qwen
   before commercial use of the covered software or derivatives.
4. The text excludes internal use from that separate-license requirement when
   the software, outputs and underlying model capabilities are not made
   available to a third party.
5. The license defines Model as a Service around third-party access to inference
   or fine-tuning with meaningful control over inputs, parameters or training
   data. It separately defines an AI Work Assistant as an independent product
   primarily designed for AI-assisted coding or office productivity, with
   specified exclusions.
6. The text includes an as-is/no-warranty provision and requires compliance with
   applicable law and third-party intellectual-property rights.

These are summaries for engineering governance, not conclusions about whether
a particular deployment falls inside or outside a definition.

## 4. Project implications

- Kestrel-Q's planned native coding-agent work makes the AI Work Assistant term
  relevant before any commercial coding-agent distribution or hosting.
- A future local server or hosted endpoint must not be assumed license-neutral;
  commercial MaaS facts require review before release or operation.
- Model weights, derived GGUF files and model outputs are not made Apache-2.0 by
  being used with Kestrel-Q.
- Any future redistribution package containing model artifacts must separately
  satisfy the pinned model license, attribution and other applicable terms.
- Changes in upstream license text require a new pinned review rather than
  silently replacing this baseline.

## 5. Provenance result

Task 1.0 read official metadata and documentation only. It copied no upstream
implementation code and committed no model artifact or license copy.
