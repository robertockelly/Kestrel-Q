# Qwen3.8-Flash-Next golden-vector baseline

Status: **TASK 1.4 COMPLETE / PASS**

## Generated safe vectors

All generated vectors are weight-independent and use exact comparison.

| Asset | Coverage | SHA-256 |
|---|---|---|
| `prompt-suite.json` | 10 original/synthetic prompt cases | `ffee472cac6e57f85df5f50104535b6e4e2d801c4ae6cac1f775840a29b7ed15` |
| `canonical/tokenizer-vectors.json` | 10 prompts, 14 text/message segments | `cbe1290d84a7a61113cf201edaf5034893eb1e70ba5a3c4bc9f4ea50bdcaf153` |
| `canonical/chat-template-vectors.json` | 2 message prompts × generation prompt off/on | `72858a105a0f009b94eaea5dec24ae9f45ec0a400b9ce7325efc9341f0fbd1d6` |
| `canonical/ple-address-vectors.json` | 7 sequence cases plus 4 incremental decode steps | `495ef70f091e8d61caac99bb14ad8cea0fdb77940ec4dc6e8ce9811a144da3b6` |

Tokenizer vectors record exact IDs with `add_special_tokens` both false and
true, token counts, decoded strings/hashes and round-trip results. All 14
segments round-trip exactly, and the Qwen tokenizer injects no automatic BOS or
EOS token. Special-token literal, preserved decode and skipped decode behavior
are also pinned.

Chat vectors use the exact official `chat_template.jinja` SHA-256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
They record messages, options, exact rendered strings and hashes, exact token
IDs and all special-token positions. `enable_thinking=false` is explicit. The
verified `add_generation_prompt=true` delta is exactly:

```text
<|im_start|>assistant
<think>

</think>

```

PLE vectors cover lengths 1, 2, 3 and greater than 3, repeated tokens, large
valid vocabulary IDs, an EOS segment boundary and incremental decode. Every
bigram/trigram row records the logical head, prime vocabulary, head offset,
global address and canonical 128-way table partition/row. Source-derived
multipliers, primes and offsets must match the pinned registered-artifact
metadata before generation succeeds.

## Plans and deferred vectors

| Asset | Status | SHA-256 |
|---|---|---|
| `canonical/operator-vector-plan.json` | `PLANNED_REFERENCE_REQUIRED` | `b2fc0c6bfe2401d2199cabe80ef0e8627fb0260ffbf7450e4a4f57c01c4a35c8` |
| `canonical/full-model-vector-plan.json` | `DEFERRED_CAPABLE_REFERENCE_ENV` | `852c326c6c1ba89fd87c7717c5297eb6fd3ae7b087ee1ede5169473d4d87a7a9` |
| `quantized/gguf-vector-plan.json` | `DEFERRED_CAPABLE_REFERENCE_ENV` | `b51e4e25cda36dcf2fdcc14bbd2394b3a03a8e9ec74b45a3497f50a9a5d83c07` |

Task 1.4 did not mark standalone GR, GDN, QSA or MoE floating vectors generated.
Task 2.6 satisfied the GDN portion: a pinned offline Transformers
`Qwen4ExpTextGatedDeltaNet` produces reduced-shape Class-C calibration,
disjoint holdout and state-transition evidence before native execution. The
native scalar path is compared to those expectations across 24 output/state/
checkpoint classes. Task 2.7 independently satisfies the QSA portion with
reduced Class-C floating calibration/holdout/cache-state evidence and exact
sparse block/token selection evidence. Task 2.8 independently satisfies the
MoE portion with reduced routed/shared/final vectors and 512/top-10 exact
routing vectors. GR remains planned and is not inferred from these operators.

The Task 2.6 namespace is
`research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
Its manifest pins the exact source/config hashes, all oracle assets and native
validation. No real model weight bytes or self-oracle values are present.

| Task 2.6 asset | SHA-256 |
|---|---|
| `gdn-contract.json` | `1b51bced8c0c6cb74e7cef1ce506339b2b6ccdc214c17cd3afdb0a343226fe93` |
| `gdn-calibration.json` | `ea13e2d8a307a41cea964015436fd895a5f4e285437d2fb9007448baf85e3a05` |
| `gdn-holdout.json` | `eb33540fa2ccce2237981b62c336dbe3ec0c38101675a21e5a0defb38facd8f7` |
| `gdn-state-vectors.json` | `fed6f9a2f5db8a3b669776a54213202ddb15ba3b19856853b9458cc45eb4680a` |
| `gdn-native-validation.json` | `6d8ee4499e811d59158dabefce51382142e152663c115889c2f2e9ef910a2eeb` |
| `gdn-manifest.json` | `b0c5e6de7c76972b876c48e31f913fb57c84c51877a64c9ca404f73f82e35964` |

Task 2.7 uses the same governed operator namespace. The QSA generator imports
only the pinned offline Transformers source and creates expectations before
native execution. A separate validator compares native scalar results to those
files; Kestrel-Q never writes expected values.

| Task 2.7 asset | SHA-256 |
|---|---|
| `qsa-contract.json` | `043e30c051436a6789257538323f9685bd79330dfe259a8bc78037f5a3aa3878` |
| `qsa-calibration.json` | `3ec63b771300d47eccc11e027b5fc7637f99983f51373aa7dbbb4784c4987d2f` |
| `qsa-holdout.json` | `34e57d551e5aefd4aa040b92ed3564bd6b00dda310f4742a258e69d7f15cbc71` |
| `qsa-selection-vectors.json` | `65c59c290210fef8851d867b6d7f6d19f864fd5d84584ca1a82048b20f571c33` |
| `qsa-state-vectors.json` | `76ff49375e63d6dcbb485c8cd41e9f3fd219c4d11d478601e30629c2689c68e1` |
| `qsa-native-validation.json` | `80123f3a88e9373dffadc16ba8fd09337d984881328db0d1caceb9bb469e26ee` |
| `qsa-manifest.json` | `15a1100fb764c01a0aed91e3e8edc156b995003688ab63665156696ad425cf3b` |

Task 2.8 uses the same governed namespace. The independent generator imports
the pinned `Qwen4ExpTextSparseMoeBlock` and captures expected values before any
native execution. Native validation calibrates each floating checkpoint from
five cases, applies unchanged contracts to four disjoint holdout cases, and
requires exact expert selection across seven Tier-B cases.

| Task 2.8 asset | SHA-256 |
|---|---|
| `moe-contract.json` | `646044a9b187c20503d81b08d294be2df4925e89ddc84e9e9fd375e3efd977bf` |
| `moe-calibration.json` | `c9d9cde90b3d93c07cfdebab281105eb720c97cd04f94254e2c41d835798f938` |
| `moe-holdout.json` | `517af0bb054c7aaa3576e0d6b1d3e0e59801eb5aba97fee97f7550250fb34d32` |
| `moe-routing-vectors.json` | `d148c2e12d93f086f342c75fb997bcc810af0226f5276958147f258d2056f3b2` |
| `moe-native-validation.json` | `6920c7750c95d52c1299d167a7a3b053c12d7f418626127a87aba09c438dae62` |
| `moe-manifest.json` | `3b730e0fbb47940c57358b73aaa4db60b9c03c007450b20ecfb3a76842deb1c8` |

Task 2.9 uses the same governed namespace. The independent generator imports
the pinned `Qwen4ExpTextPLELayer`, produces reduced calibration, disjoint
holdout and value-state expectations before native execution, and separately
feeds real Task 2.4 address streams into deterministic synthetic tables. The
native validator applies calibration-derived contracts unchanged to holdout;
intent fields and lookup/embedding identity compare exactly.

| Task 2.9 asset | SHA-256 |
|---|---|
| `ple-value-contract.json` | `9e61eb76d9c96c4535eb50436b1d68f23a900e1be02fede15ef38b1edb098acb` |
| `ple-value-calibration.json` | `bc4b9d51ba37e1f5b3630bab587637b047b54069297d7a34aa4209e8a3dcd389` |
| `ple-value-holdout.json` | `17eb9bc2c5ab63e6dc54d059f2c5643deeff56785631851dda4f3fb345b3c786` |
| `ple-value-state-vectors.json` | `d35b375ba7133af587591cc40becc445bd5ec2a6488f385964743a9cfcf5f9a2` |
| `ple-value-address-integration.json` | `05d8a20270b3a904f408d1f473050299442ee0dade009ef1fe51edc919931b09` |
| `ple-value-native-validation.json` | `472b24242c22b2896ce0daa1891bbe0cd99c1b6c14f2107dacd58c5780e7d0f4` |
| `ple-value-manifest.json` | `dc4f0ca4a0303d37c49aec9549f54cc84bbd437a73044679e430ac5599d56589` |

Task 2.10 adds independent complete-layer evidence from the pinned
`Qwen4ExpTextDecoderLayer` and `Qwen4ExpTextGatedResidual`. Reduced ordinary
GDN, QSA and PLE-GDN families each have calibration and disjoint holdout cases;
native output is compared to those expectations and never generates them.

| Task 2.10 asset | SHA-256 |
|---|---|
| `layer-contract.json` | `c79f460ac7d68d4b15b0aec8a19d110f38a5bfe6b507e92e3f431b96877140a6` |
| `layer-calibration.json` | `6bc9c9aeab0a952e4c090be62f3630ee33502ae3be55006b3b0f0d16b914e40b` |
| `layer-holdout.json` | `9d2d5a74b05558b7328dc9133a8ea25b0f71bf44ba6c43b94fe9fe5496b7bb93` |
| `layer-state-vectors.json` | `0ee833074cdca8558f31d955b978809743a41b7bbcbac079b9fbd38705b9929b` |
| `layer-family-vectors.json` | `87d15c93931792abfd6dd853c0dc661a02cebf8a8d2e0d5fe8c2a6e8353fddcd` |
| `layer-native-validation.json` | `ab5688724722c8f22c0970b34c70325f7422f42c6beba564900e15216b509b0f` |
| `layer-manifest.json` | `75f58a60366bcb7f47934d8d4e14e4ceb76b636716c23928520e4a9355ee291f` |

The Class-C full-model plan fixes three prompt IDs, batch 1, BF16, text-only,
greedy generation, no vision/MTP/speculation, exact dependencies and hooks
spanning embedding/GR, early GDN, PLE, first QSA, routers, middle/late layers,
persistent state, final hidden/logits and eight greedy tokens.

The Class-Q plan fixes the exact GGUF SHA, llama.cpp revision/build, CPU-only
backend, mmap enabled, zero GPU layers, context/batch sizes, three prompt IDs,
temperature zero and `--spec-type none`. Both plans are future acceptance gates,
not generated model results.

## Numeric policy

Discrete/text/address/token outputs compare with exact equality. Future floating
fields carry shape, dtype, `atol`, `rtol`, NaN and Inf rules. All unexecuted
floating checkpoints currently use null tolerances with status
`TO_BE_CALIBRATED_FROM_REFERENCE_RUNS`; null does not mean unlimited tolerance.

## Manifest and regeneration

Manifest:

`research/goldens/Qwen3.8-Flash-Next/manifest.json`

Manifest SHA-256:

`aa572756672f288957d429a60d7180650ffb2d603a792b21cd72def0a14ec0c4`

The manifest records both oracle identities/licenses, every asset class,
status, hash, comparison mode, dependencies and normalized generation command.
`tools/generate-reference-goldens.py` is offline/fail-closed and
`tools/validate-reference-goldens.py` revalidates all references and hashes.

Two development failures were caught before a valid manifest existed: one
manually transcribed llama.cpp source hash contained a one-nibble error, and the
pinned Transformers API returned a `BatchEncoding` from
`apply_chat_template(tokenize=true)` where the first comparison expected a raw
ID list. Exact source-hash checking and explicit `input_ids` extraction now
guard both paths. A later safety scan also found 19 llama.cpp vocabulary-only
GGUF fixtures (all had zero tensors) in the ignored full source checkout; they
were removed, and generation now rejects `.gguf` or `.safetensors` fixtures in
either source checkout. The corrected output regenerated byte-identically.

Task 2.3 revalidated the prompt, tokenizer, chat and manifest hashes unchanged
during pre-code characterization, then compared the native implementation to
them after the governed source decision. Native output did not replace or
regenerate these expectations. All 10 prompts, 14 segments and four chat
combinations match exactly. The separate canonical differential corpus adds 22
encode, five decode, two supported-chat and three rejected-chat cases at
`research/tokenizer/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json`
(SHA-256 `f383e213ccc4cde06b47a9855ca1eabb54d0b8911acbd43b2078be5fc546b463`).

Task 2.4 independently regenerated `canonical/ple-address-vectors.json`
byte-identically and compared the native C17 engine to all 7 sequence cases
and 4 decode steps. The original PLE SHA-256 remains unchanged. The expanded
canonical differential evidence adds 12 sequence cases, three incremental
streams and one tokenizer-to-PLE integration case at
`research/ple/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/canonical-differential.json`
(SHA-256 `b9c9be4d927d59c9ac12ba2313034cda5a1857d5484fca479327c9b771cb9671`).
Expected values come only from the pinned canonical oracle; the native runtime
is never its own oracle.

Task 2.5 does not modify or replace any Task 1.4 golden. Its low-level floating
and quantized-storage evidence lives separately under `research/numerics/`:
synthetic dequant expected values come from the pinned Class-Q llama.cpp helper,
and primitive calibration/holdout expected values come from an explicitly
ordered pinned NumPy oracle. Model-operator vectors above remain planned and
must not be inferred from the low-level primitive evidence.

Task 2.11 adds a separate exact-GGUF operator namespace under
`research/operators/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
The pinned llama.cpp helper independently decodes requested target rows and
pinned Transformers components execute canonical layer equations. Four
calibration profiles and a disjoint holdout cover GDN layer 0, QSA layer 3 and
PLE-GDN layer 1, each with prefill plus decode. Native output is only the
comparison subject. Floating outputs use recorded per-family/per-phase
calibration limits; route order, selected expert membership/access, QSA local
selection and PLE member/row requests remain exact discrete results.

| Task 2.11 evidence | SHA-256 |
|---|---|
| `target-layer-contract.json` | `84c6043cd4bb6c56037bd7747766feebdd8289b4c46af3a0b795fdb3f521ef70` |
| `target-layer-calibration.json` | `584bd247f7d578ce6a2f997b19e7f06934f11df491ea3182240badc387688936` |
| `target-layer-holdout.json` | `85d67f9fa9f4ba19f7219dbb46c74ffe09c881779355e0f5b451b7bb114adef7` |
| `target-layer-state-vectors.json` | `30ce3f4ce11174fd2cacb239b6c5823cf5d2bd1179e4197b6dc23156a15986a0` |
| `target-layer-native-validation.json` | `c7575a9e98cb03cbeeb2845068b616e025e51f6705e2dc1f22e97345966a769d` |
| `target-layer-manifest.json` | `44a12bdd9477a8493cd47a0517f396a8f5991cc42c31766df42a4463611e64a0` |

Task 2.12 adds an independently executed full-model Class-Q milestone under
`research/milestones/Qwen3.8-Flash-Next/de4b8e4d43b917e7706784d8bb445c9af86a3540/`.
Pinned llama.cpp receives the committed canonical IDs directly; Kestrel-Q is
only the comparison subject. The exact next-token contract is token ID 271.
Top-N logits and hidden summaries are characterization-only and do not replace
the lower-level calibrated operator contracts.

| Task 2.12 evidence | SHA-256 |
|---|---|
| `first-token-contract.json` | `37c614713fa940f623a918b9d0a10242b7f0cba9d174364377d763869f05f2ae` |
| `first-token-oracle.json` | `02bca9f4c5b90a1e153a35133d10fc9aaa0c1adcf894083ce70c99bfc57add63` |
| `first-token-native.json` | `27dc45eeb4002125010e3e54eec67aca7881b43648eca0a0b752a2aaee7ea546` |
| `first-token-validation.json` | `632f12319b817fe9e043e44df05e00f5c9e20ba3c245292ae45ecd36ec8d1d06` |
| `first-token-manifest.json` | `1f993d201edcd10864a52d99ca3cdd8bef3c9ec5bcd910cd99627e5edaa7fce9` |

Task 2.13 preserves the M1 evidence unchanged and adds an independent
four-token greedy continuation in the same milestone namespace. The oracle is
again pinned llama.cpp over explicit canonical prompt IDs; native results are
comparison subjects only. Exact discrete equality is required for the full
sequence `[271, 248068, 198, 760]`. Floating selected/runner-up logits are
diagnostic characterization, not a widened correctness substitute.

| Task 2.13 evidence | SHA-256 |
|---|---|
| `multi-token-contract.json` | `2ae8808f68bfe10b658f32cae11d56d49aaed7cfb419f6aec93b64c9b0b24f6e` |
| `multi-token-oracle.json` | `9204ea21bac16f288bca389153bd919cb5d629cf6ce5818e8bcb0af85b4294b8` |
| `multi-token-native.json` | `5d2292995bccc3012aa2e0ebbd2a26eafdd1b73560acc407ecf45845040abfaf` |
| `multi-token-state.json` | `677bc8a06e06dca3591070ec030d2b73de195fc137785caf36bce2f9b01a5aac` |
| `multi-token-validation.json` | `df1bf3aa3810a4cad0492d8d62747e7c6a5d54a70a3a01f960b212349c19c4b0` |
| `multi-token-manifest.json` | `b7e2a3fa1ca098f0b7a8caa3b31f964aa5d140cfafaeb072ff5ad9d1e9ec1efe` |
