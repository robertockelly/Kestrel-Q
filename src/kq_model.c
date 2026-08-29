#include "kq_model.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_model_internal.h"

#define KQ_QWEN_HIDDEN_SIZE 2560U
#define KQ_QWEN_VOCABULARY_SIZE 248320U
#define KQ_QWEN_CONTEXT_LENGTH 262144U
#define KQ_QWEN_LAYER_COUNT 48U
#define KQ_QWEN_GDN_LAYER_COUNT 36U
#define KQ_QWEN_QSA_LAYER_COUNT 12U
#define KQ_QWEN_EXPERT_COUNT 512U
#define KQ_QWEN_EXPERT_TOP_K 10U
#define KQ_QWEN_EXPERT_INTERMEDIATE 640U
#define KQ_QWEN_GR_BRANCHES 4U
#define KQ_QWEN_GR_WIDTH 10240U
#define KQ_QWEN_GR_LOW_RANK 320U
#define KQ_QWEN_INITIAL_SEMANTICS 1294U
#define KQ_QWEN_PLE_LAYER 1U
#define KQ_QWEN_PLE_TABLE_MEMBERS 128U
#define KQ_QWEN_PLE_SHARD_ROWS 2500012U
#define KQ_QWEN_PLE_TABLE_WIDTH 160U
#define KQ_QWEN_PLE_FUSED_ROWS 320001536U
#define KQ_QWEN_PLE_NGRAM_SIZE 3U
#define KQ_QWEN_PLE_HEADS_PER_NGRAM 8U
#define KQ_QWEN_PLE_HEAD_COUNT 16U

#define KQ_TYPE_MASK(type_id) (UINT64_C(1) << (type_id))
#define KQ_MASK_F32 KQ_TYPE_MASK(KQ_GGUF_TYPE_F32)
#define KQ_MASK_BF16 KQ_TYPE_MASK(KQ_GGUF_TYPE_BF16)
#define KQ_MASK_Q8_0 KQ_TYPE_MASK(KQ_GGUF_TYPE_Q8_0)
#define KQ_MASK_Q5_1 KQ_TYPE_MASK(KQ_GGUF_TYPE_Q5_1)
#define KQ_MASK_Q4_K KQ_TYPE_MASK(KQ_GGUF_TYPE_Q4_K)
#define KQ_MASK_Q5_K KQ_TYPE_MASK(KQ_GGUF_TYPE_Q5_K)
#define KQ_MASK_IQ4_NL KQ_TYPE_MASK(KQ_GGUF_TYPE_IQ4_NL)

struct kq_model {
    const kq_gguf *gguf;
    kq_semantic_tensor *semantics;
    unsigned char *physical_coverage;
    uint64_t semantic_count;
    uint64_t semantic_capacity;
    uint64_t physical_count;
    uint64_t physical_coverage_count;
    uint64_t metadata_derived_count;
    uint64_t unknown_physical_count;
    uint64_t unbound_required_count;
    uint64_t relation_counts[KQ_BINDING_RELATION_COUNT];
    uint64_t component_counts[KQ_COMPONENT_COUNT];
    uint64_t placement_counts[KQ_PLACEMENT_HINT_COUNT];
    uint32_t hidden_size;
    uint32_t vocabulary_size;
    uint32_t context_length;
    uint32_t layer_count;
    uint32_t gdn_layer_count;
    uint32_t qsa_layer_count;
    uint32_t expert_count;
    uint32_t expert_top_k;
    kq_model_layer_type layer_types[KQ_QWEN_LAYER_COUNT];
};

static kq_status kq_model_fail(kq_diagnostic *diagnostic,
                               kq_status status,
                               const char *format,
                               ...) {
    va_list arguments;
    char message[KQ_DIAGNOSTIC_CAPACITY];

    message[0] = '\0';
    if (format != NULL) {
        va_start(arguments, format);
        (void)vsnprintf(message, sizeof(message), format, arguments);
        va_end(arguments);
        message[sizeof(message) - 1U] = '\0';
    }
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static kq_status kq_format_checked(char *destination,
                                   size_t capacity,
                                   kq_diagnostic *diagnostic,
                                   const char *format,
                                   ...) {
    va_list arguments;
    int written;

    va_start(arguments, format);
    written = vsnprintf(destination, capacity, format, arguments);
    va_end(arguments);
    if (written < 0 || (size_t)written >= capacity) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic identifier formatting exceeded its bound");
    }
    return KQ_STATUS_OK;
}

static const kq_gguf_metadata *kq_source_metadata(
    const kq_model_source *source,
    const char *key) {
    if (source == NULL || source->find_metadata == NULL) {
        return NULL;
    }
    return source->find_metadata(source->context, key);
}

static const kq_gguf_tensor *kq_source_tensor_at(
    const kq_model_source *source,
    uint64_t index) {
    if (source == NULL || source->tensor_at == NULL ||
        index >= source->tensor_count) {
        return NULL;
    }
    return source->tensor_at(source->context, index);
}

static kq_status kq_source_find_tensor(const kq_model_source *source,
                                       const char *name,
                                       const kq_gguf_tensor **out_tensor,
                                       uint64_t *out_index,
                                       kq_diagnostic *diagnostic) {
    const kq_gguf_tensor *candidate;
    const kq_gguf_tensor *found = NULL;
    uint64_t found_index = 0U;
    uint64_t index;

    for (index = 0U; index < source->tensor_count; ++index) {
        candidate = kq_source_tensor_at(source, index);
        if (candidate == NULL) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "physical tensor %llu is unavailable",
                                 (unsigned long long)index);
        }
        if (kq_string_view_equal_cstr(&candidate->name, name)) {
            if (found != NULL) {
                return kq_model_fail(diagnostic,
                                     KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                     "physical tensor name %s is ambiguous",
                                     name);
            }
            found = candidate;
            found_index = index;
        }
    }
    if (found == NULL) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "required physical tensor %s is absent",
                             name);
    }
    *out_tensor = found;
    *out_index = found_index;
    return KQ_STATUS_OK;
}

static kq_status kq_require_u32(const kq_model_source *source,
                                const char *key,
                                uint32_t expected,
                                kq_diagnostic *diagnostic) {
    const kq_gguf_metadata *metadata = kq_source_metadata(source, key);
    uint32_t value = 0U;

    if (!kq_gguf_metadata_u32(metadata, &value)) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "required uint32 model metadata %s is absent or mistyped",
                             key);
    }
    if (value != expected) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "model metadata %s is %u, expected %u",
                             key,
                             (unsigned int)value,
                             (unsigned int)expected);
    }
    return KQ_STATUS_OK;
}

static kq_status kq_require_i32_array(const kq_model_source *source,
                                      const char *key,
                                      const int32_t *expected,
                                      uint64_t expected_count,
                                      kq_diagnostic *diagnostic) {
    const kq_gguf_metadata *metadata = kq_source_metadata(source, key);
    int32_t value = 0;
    uint64_t index;

    if (metadata == NULL || metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_INT32 ||
        metadata->array_length != expected_count) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "required int32 array metadata %s has incompatible type or length",
                             key);
    }
    for (index = 0U; index < expected_count; ++index) {
        if (!kq_gguf_metadata_array_i32_at(metadata, index, &value) ||
            value != expected[(size_t)index]) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                                 "model metadata %s differs at element %llu",
                                 key,
                                 (unsigned long long)index);
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_require_u64_array(const kq_model_source *source,
                                      const char *key,
                                      const uint64_t *expected,
                                      uint64_t expected_count,
                                      kq_diagnostic *diagnostic) {
    const kq_gguf_metadata *metadata = kq_source_metadata(source, key);
    uint64_t value = 0U;
    uint64_t index;

    if (metadata == NULL || metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_UINT64 ||
        metadata->array_length != expected_count) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "required uint64 array metadata %s has incompatible type or length",
                             key);
    }
    for (index = 0U; index < expected_count; ++index) {
        if (!kq_gguf_metadata_array_u64_at(metadata, index, &value) ||
            value != expected[(size_t)index]) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                                 "model metadata %s differs at element %llu",
                                 key,
                                 (unsigned long long)index);
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_validate_model_identity(const kq_model_source *source,
                                            kq_model *model,
                                            kq_diagnostic *diagnostic) {
    static const int32_t ple_layers[] = {1};
    static const uint64_t ple_multipliers[] = {
        UINT64_C(23703573157769),
        UINT64_C(20109073645365),
        UINT64_C(8052911324071)
    };
    static const uint64_t ple_offsets[] = {
        0U, 20000003U, 40000026U, 60000059U,
        80000106U, 100000165U, 120000228U, 140000297U,
        160000374U, 180000455U, 200000548U, 220000655U,
        240000802U, 260000955U, 280001114U, 300001275U
    };
    static const uint64_t ple_vocab_sizes[] = {
        20000003U, 20000023U, 20000033U, 20000047U,
        20000059U, 20000063U, 20000069U, 20000077U,
        20000081U, 20000093U, 20000107U, 20000147U,
        20000153U, 20000159U, 20000161U, 20000171U
    };
    int32_t layer_pattern[KQ_QWEN_LAYER_COUNT];
    const kq_gguf_metadata *tokens;
    kq_status status;
    uint32_t gdn_count = 0U;
    uint32_t layer;
    uint32_t qsa_count = 0U;

    if (!kq_string_view_equal_cstr(&source->architecture, "qwen4exp")) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_UNSUPPORTED_MODEL,
                             "model architecture is not qwen4exp");
    }
    if (source->payload_bytes_accessed != 0U) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic registry source reports payload access");
    }

#define KQ_REQUIRE_U32(key, value)                                            \
    do {                                                                      \
        status = kq_require_u32(source, key, value, diagnostic);              \
        if (status != KQ_STATUS_OK) {                                         \
            return status;                                                    \
        }                                                                     \
    } while (0)

    KQ_REQUIRE_U32("qwen4exp.block_count", KQ_QWEN_LAYER_COUNT);
    KQ_REQUIRE_U32("qwen4exp.context_length", KQ_QWEN_CONTEXT_LENGTH);
    KQ_REQUIRE_U32("qwen4exp.embedding_length", KQ_QWEN_HIDDEN_SIZE);
    KQ_REQUIRE_U32("qwen4exp.attention.head_count", 24U);
    KQ_REQUIRE_U32("qwen4exp.attention.head_count_kv", 2U);
    KQ_REQUIRE_U32("qwen4exp.attention.key_length", 256U);
    KQ_REQUIRE_U32("qwen4exp.attention.value_length", 256U);
    KQ_REQUIRE_U32("qwen4exp.expert_count", KQ_QWEN_EXPERT_COUNT);
    KQ_REQUIRE_U32("qwen4exp.expert_used_count", KQ_QWEN_EXPERT_TOP_K);
    KQ_REQUIRE_U32("qwen4exp.expert_feed_forward_length",
                   KQ_QWEN_EXPERT_INTERMEDIATE);
    KQ_REQUIRE_U32("qwen4exp.expert_shared_feed_forward_length",
                   KQ_QWEN_EXPERT_INTERMEDIATE);
    KQ_REQUIRE_U32("qwen4exp.ssm.conv_kernel", 4U);
    KQ_REQUIRE_U32("qwen4exp.ssm.state_size", 128U);
    KQ_REQUIRE_U32("qwen4exp.ssm.group_count", 16U);
    KQ_REQUIRE_U32("qwen4exp.ssm.time_step_rank", 48U);
    KQ_REQUIRE_U32("qwen4exp.ssm.inner_size", 6144U);
    KQ_REQUIRE_U32("qwen4exp.full_attention_interval", 4U);
    KQ_REQUIRE_U32("qwen4exp.rope.dimension_count", 64U);
    KQ_REQUIRE_U32("qwen4exp.hyper_connection.count", KQ_QWEN_GR_BRANCHES);
    KQ_REQUIRE_U32("qwen4exp.hyper_connection.low_rank", KQ_QWEN_GR_LOW_RANK);
    KQ_REQUIRE_U32("qwen4exp.attention.indexer.head_count", 4U);
    KQ_REQUIRE_U32("qwen4exp.attention.indexer.key_length", 128U);
    KQ_REQUIRE_U32("qwen4exp.attention.indexer.top_k", 2048U);
    KQ_REQUIRE_U32("qwen4exp.ple.ngram_size", KQ_QWEN_PLE_NGRAM_SIZE);
    KQ_REQUIRE_U32("qwen4exp.ple.heads_per_ngram",
                   KQ_QWEN_PLE_HEADS_PER_NGRAM);
    KQ_REQUIRE_U32("qwen4exp.ple.conv_kernel", 4U);
    KQ_REQUIRE_U32("qwen4exp.ple.eos_token_id", 248044U);
    KQ_REQUIRE_U32("qwen4exp.embedding_length_per_layer_input",
                   KQ_QWEN_PLE_TABLE_WIDTH);
#undef KQ_REQUIRE_U32

    tokens = kq_source_metadata(source, "tokenizer.ggml.tokens");
    if (tokens == NULL || tokens->value_type != KQ_GGUF_VALUE_ARRAY ||
        tokens->array_element_type != KQ_GGUF_VALUE_STRING ||
        tokens->array_length != KQ_QWEN_VOCABULARY_SIZE) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "tokenizer vocabulary metadata is not 248320 strings");
    }

    for (layer = 0U; layer < KQ_QWEN_LAYER_COUNT; ++layer) {
        layer_pattern[layer] = ((layer + 1U) % 4U) == 0U ? 4 : 0;
        model->layer_types[layer] = layer_pattern[layer] == 4
                                        ? KQ_MODEL_LAYER_QSA
                                        : KQ_MODEL_LAYER_GDN;
        if (model->layer_types[layer] == KQ_MODEL_LAYER_QSA) {
            qsa_count += 1U;
        } else {
            gdn_count += 1U;
        }
    }
    status = kq_require_i32_array(source,
                                  "qwen4exp.attention.compress_ratios",
                                  layer_pattern,
                                  KQ_QWEN_LAYER_COUNT,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (gdn_count != KQ_QWEN_GDN_LAYER_COUNT ||
        qsa_count != KQ_QWEN_QSA_LAYER_COUNT) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "model layer schedule is not 36 GDN / 12 QSA");
    }
    status = kq_require_i32_array(source,
                                  "qwen4exp.ple.layers",
                                  ple_layers,
                                  1U,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_require_u64_array(source,
                                  "qwen4exp.ple.layer_multipliers",
                                  ple_multipliers,
                                  3U,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_require_u64_array(source,
                                  "qwen4exp.ple.head_offsets",
                                  ple_offsets,
                                  KQ_QWEN_PLE_HEAD_COUNT,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_require_u64_array(source,
                                  "qwen4exp.ple.head_vocab_sizes",
                                  ple_vocab_sizes,
                                  KQ_QWEN_PLE_HEAD_COUNT,
                                  diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }

    model->hidden_size = KQ_QWEN_HIDDEN_SIZE;
    model->vocabulary_size = KQ_QWEN_VOCABULARY_SIZE;
    model->context_length = KQ_QWEN_CONTEXT_LENGTH;
    model->layer_count = KQ_QWEN_LAYER_COUNT;
    model->gdn_layer_count = gdn_count;
    model->qsa_layer_count = qsa_count;
    model->expert_count = KQ_QWEN_EXPERT_COUNT;
    model->expert_top_k = KQ_QWEN_EXPERT_TOP_K;
    return KQ_STATUS_OK;
}

typedef struct kq_single_spec {
    const char *semantic_leaf;
    const char *canonical_leaf;
    const char *physical_leaf;
    kq_semantic_component component;
    kq_semantic_role role;
    kq_binding_relation relation;
    uint32_t canonical_rank;
    uint64_t canonical_dimensions[KQ_SEMANTIC_MAX_DIMS];
    uint32_t physical_rank;
    uint64_t physical_dimensions[KQ_GGUF_MAX_DIMS];
    uint64_t allowed_type_mask;
} kq_single_spec;

static kq_placement_hint kq_component_placement(
    kq_semantic_component component) {
    if (component == KQ_COMPONENT_ROUTED_EXPERT_STACK) {
        return KQ_PLACEMENT_ROUTED_EXPERT_CACHE_CANDIDATE;
    }
    if (component == KQ_COMPONENT_PLE_TABLE ||
        component == KQ_COMPONENT_PLE_DENSE ||
        component == KQ_COMPONENT_PLE_ADDRESS_METADATA) {
        return KQ_PLACEMENT_PLE_DISK_BACKED_CANDIDATE;
    }
    return KQ_PLACEMENT_ALWAYS_NEEDED_CANDIDATE;
}

static kq_status kq_begin_semantic(kq_model *model,
                                   const char *semantic_id,
                                   const char *canonical_name,
                                   kq_semantic_component component,
                                   kq_semantic_role role,
                                   kq_binding_relation relation,
                                   uint32_t layer_id,
                                   kq_model_layer_type layer_type,
                                   kq_canonical_dtype dtype,
                                   uint32_t canonical_rank,
                                   const uint64_t *canonical_dimensions,
                                   uint32_t canonical_expert_axis,
                                   uint32_t expert_count,
                                   kq_semantic_tensor **out_semantic,
                                   kq_diagnostic *diagnostic) {
    kq_semantic_tensor *semantic = NULL;
    uint64_t index;
    uint32_t dimension;
    int written;

    if (model->semantic_count >= model->semantic_capacity ||
        component >= KQ_COMPONENT_COUNT ||
        relation >= KQ_BINDING_RELATION_COUNT) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic registry capacity or enum invariant failed");
    }
    for (index = 0U; index < model->semantic_count; ++index) {
        if (strcmp(model->semantics[(size_t)index].semantic_id,
                   semantic_id) == 0) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "duplicate semantic identifier %s",
                                 semantic_id);
        }
    }
    if (canonical_rank == 0U || canonical_rank > KQ_SEMANTIC_MAX_DIMS) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic %s has invalid canonical rank",
                             semantic_id);
    }
    semantic = &model->semantics[(size_t)model->semantic_count];
    written = snprintf(semantic->semantic_id,
                       sizeof(semantic->semantic_id),
                       "%s",
                       semantic_id);
    if (written < 0 || (size_t)written >= sizeof(semantic->semantic_id)) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic identifier exceeds fixed bound");
    }
    written = snprintf(semantic->canonical_name,
                       sizeof(semantic->canonical_name),
                       "%s",
                       canonical_name);
    if (written < 0 || (size_t)written >= sizeof(semantic->canonical_name)) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "canonical tensor name exceeds fixed bound");
    }
    semantic->component = component;
    semantic->role = role;
    semantic->relation = relation;
    semantic->runtime_scope = KQ_SCOPE_REQUIRED_INITIAL_TEXT;
    semantic->placement_hint = kq_component_placement(component);
    semantic->layer_id = layer_id;
    semantic->layer_type = layer_type;
    semantic->canonical_dtype = dtype;
    semantic->canonical_rank = canonical_rank;
    semantic->canonical_expert_axis = canonical_expert_axis;
    semantic->expert_count = expert_count;
    for (dimension = 0U; dimension < canonical_rank; ++dimension) {
        if (canonical_dimensions[dimension] == 0U) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "semantic %s has a zero canonical dimension",
                                 semantic_id);
        }
        semantic->canonical_dimensions[dimension] =
            canonical_dimensions[dimension];
    }
    model->semantic_count += 1U;
    model->relation_counts[relation] += 1U;
    model->component_counts[component] += 1U;
    model->placement_counts[semantic->placement_hint] += 1U;
    *out_semantic = semantic;
    return KQ_STATUS_OK;
}

static kq_status kq_bind_physical(const kq_model_source *source,
                                  kq_model *model,
                                  kq_semantic_tensor *semantic,
                                  const char *physical_name,
                                  uint32_t expected_rank,
                                  const uint64_t *expected_dimensions,
                                  uint64_t allowed_type_mask,
                                  kq_binding_part_role part_role,
                                  uint32_t part_index,
                                  uint32_t part_count,
                                  uint32_t fused_member_index,
                                  uint32_t fused_member_count,
                                  uint32_t physical_expert_axis,
                                  int allow_shared_physical,
                                  kq_diagnostic *diagnostic) {
    const kq_gguf_tensor *tensor;
    kq_tensor_binding *binding;
    uint64_t physical_index;
    uint32_t dimension;
    kq_status status;

    if (semantic->binding_count >= KQ_SEMANTIC_MAX_BINDINGS) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "semantic %s has too many physical parts",
                             semantic->semantic_id);
    }
    status = kq_source_find_tensor(source,
                                   physical_name,
                                   &tensor,
                                   &physical_index,
                                   diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (tensor->rank != expected_rank) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "physical tensor %s rank is %u, expected %u",
                             physical_name,
                             (unsigned int)tensor->rank,
                             (unsigned int)expected_rank);
    }
    for (dimension = 0U; dimension < expected_rank; ++dimension) {
        if (tensor->dimensions[dimension] != expected_dimensions[dimension]) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                                 "physical tensor %s dimension %u is incompatible",
                                 physical_name,
                                 (unsigned int)dimension);
        }
    }
    if (tensor->type_id >= 64U ||
        (allowed_type_mask & KQ_TYPE_MASK(tensor->type_id)) == 0U) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "physical tensor %s type %u is incompatible",
                             physical_name,
                             (unsigned int)tensor->type_id);
    }
    if (physical_expert_axis != KQ_MODEL_NO_AXIS &&
        (physical_expert_axis >= tensor->rank ||
         tensor->dimensions[physical_expert_axis] != KQ_QWEN_EXPERT_COUNT)) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "physical tensor %s has an invalid expert axis",
                             physical_name);
    }
    if (model->physical_coverage[(size_t)physical_index] != 0U &&
        !allow_shared_physical) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "physical tensor %s is bound ambiguously",
                             physical_name);
    }
    if (model->physical_coverage[(size_t)physical_index] == 0U) {
        model->physical_coverage[(size_t)physical_index] = 1U;
        model->physical_coverage_count += 1U;
    }
    binding = &semantic->bindings[semantic->binding_count];
    binding->physical = tensor;
    binding->physical_index = physical_index;
    binding->part_role = part_role;
    binding->part_index = part_index;
    binding->part_count = part_count;
    binding->fused_member_index = fused_member_index;
    binding->fused_member_count = fused_member_count;
    binding->physical_expert_axis = physical_expert_axis;
    semantic->binding_count += 1U;
    return KQ_STATUS_OK;
}

static kq_status kq_add_single_entry(const kq_model_source *source,
                                     kq_model *model,
                                     const char *semantic_id,
                                     const char *canonical_name,
                                     const char *physical_name,
                                     const kq_single_spec *spec,
                                     uint32_t layer_id,
                                     kq_model_layer_type layer_type,
                                     uint32_t canonical_expert_axis,
                                     uint32_t expert_count,
                                     uint32_t physical_expert_axis,
                                     kq_diagnostic *diagnostic) {
    kq_semantic_tensor *semantic = NULL;
    kq_status status;

    status = kq_begin_semantic(model,
                               semantic_id,
                               canonical_name,
                               spec->component,
                               spec->role,
                               spec->relation,
                               layer_id,
                               layer_type,
                               KQ_CANONICAL_DTYPE_BF16,
                               spec->canonical_rank,
                               spec->canonical_dimensions,
                               canonical_expert_axis,
                               expert_count,
                               &semantic,
                               diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    return kq_bind_physical(source,
                            model,
                            semantic,
                            physical_name,
                            spec->physical_rank,
                            spec->physical_dimensions,
                            spec->allowed_type_mask,
                            KQ_BINDING_PART_WHOLE,
                            0U,
                            1U,
                            0U,
                            0U,
                            physical_expert_axis,
                            0,
                            diagnostic);
}

static kq_status kq_add_layer_single(const kq_model_source *source,
                                     kq_model *model,
                                     uint32_t layer,
                                     const kq_single_spec *spec,
                                     uint32_t canonical_expert_axis,
                                     uint32_t expert_count,
                                     uint32_t physical_expert_axis,
                                     kq_diagnostic *diagnostic) {
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    char physical_name[KQ_SEMANTIC_ID_CAPACITY];
    kq_status status;

    status = kq_format_checked(semantic_id,
                               sizeof(semantic_id),
                               diagnostic,
                               "layer.%02u.%s",
                               (unsigned int)layer,
                               spec->semantic_leaf);
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(canonical_name,
                                   sizeof(canonical_name),
                                   diagnostic,
                                   "model.language_model.layers.%u.%s",
                                   (unsigned int)layer,
                                   spec->canonical_leaf);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(physical_name,
                                   sizeof(physical_name),
                                   diagnostic,
                                   "blk.%u.%s",
                                   (unsigned int)layer,
                                   spec->physical_leaf);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    return kq_add_single_entry(source,
                               model,
                               semantic_id,
                               canonical_name,
                               physical_name,
                               spec,
                               layer,
                               model->layer_types[layer],
                               canonical_expert_axis,
                               expert_count,
                               physical_expert_axis,
                               diagnostic);
}

static kq_status kq_build_top_level(const kq_model_source *source,
                                    kq_model *model,
                                    kq_diagnostic *diagnostic) {
    static const kq_single_spec specs[] = {
        {"text.token_embedding", "model.language_model.embed_tokens.weight",
         "token_embd.weight", KQ_COMPONENT_TOKEN_EMBEDDING,
         KQ_ROLE_TOKEN_EMBEDDING, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {248320U, 2560U}, 2U, {2560U, 248320U}, KQ_MASK_Q8_0},
        {"text.lm_head", "lm_head.weight", "output.weight",
         KQ_COMPONENT_LM_HEAD, KQ_ROLE_LM_HEAD,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {248320U, 2560U}, 2U, {2560U, 248320U}, KQ_MASK_Q8_0},
        {"text.final_gr.norm",
         "model.language_model.hyper_connection_mixer.hc_norm.weight",
         "output_hc_norm.weight", KQ_COMPONENT_FINAL_GATED_RESIDUAL,
         KQ_ROLE_HC_NORM, KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"text.final_gr.down",
         "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
         "output_hc_down.weight", KQ_COMPONENT_FINAL_GATED_RESIDUAL,
         KQ_ROLE_HC_DOWN, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {320U, 10240U}, 2U, {10240U, 320U}, KQ_MASK_Q8_0},
        {"text.final_gr.up",
         "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
         "output_hc_up.weight", KQ_COMPONENT_FINAL_GATED_RESIDUAL,
         KQ_ROLE_HC_UP, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {10240U, 320U}, 2U, {320U, 10240U}, KQ_MASK_Q8_0}
    };
    size_t index;
    kq_status status;

    for (index = 0U; index < sizeof(specs) / sizeof(specs[0]); ++index) {
        status = kq_add_single_entry(source,
                                     model,
                                     specs[index].semantic_leaf,
                                     specs[index].canonical_leaf,
                                     specs[index].physical_leaf,
                                     &specs[index],
                                     KQ_MODEL_NO_LAYER,
                                     KQ_MODEL_LAYER_INVALID,
                                     KQ_MODEL_NO_AXIS,
                                     0U,
                                     KQ_MODEL_NO_AXIS,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_build_gated_residual(const kq_model_source *source,
                                         kq_model *model,
                                         uint32_t layer,
                                         kq_diagnostic *diagnostic) {
    static const kq_single_spec specs[] = {
        {"gr.attn.inject", "attn_hyper_connection.block_inject_weight.weight",
         "hc_attn_inject.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_INJECT, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {4U, 10240U}, 2U, {10240U, 4U}, KQ_MASK_F32},
        {"gr.attn.norm", "attn_hyper_connection.hc_norm.weight",
         "hc_attn_norm.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_NORM, KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"gr.attn.down", "attn_hyper_connection.input_mix_weight_down.weight",
         "hc_attn_down.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_DOWN, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {320U, 10240U}, 2U, {10240U, 320U}, KQ_MASK_Q8_0},
        {"gr.attn.up", "attn_hyper_connection.input_mix_weight_up.weight",
         "hc_attn_up.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_UP, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {10240U, 320U}, 2U, {320U, 10240U}, KQ_MASK_Q8_0},
        {"gr.moe.inject", "mlp_hyper_connection.block_inject_weight.weight",
         "hc_ffn_inject.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_INJECT, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {4U, 10240U}, 2U, {10240U, 4U}, KQ_MASK_F32},
        {"gr.moe.norm", "mlp_hyper_connection.hc_norm.weight",
         "hc_ffn_norm.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_NORM, KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"gr.moe.down", "mlp_hyper_connection.input_mix_weight_down.weight",
         "hc_ffn_down.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_DOWN, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {320U, 10240U}, 2U, {10240U, 320U}, KQ_MASK_Q8_0},
        {"gr.moe.up", "mlp_hyper_connection.input_mix_weight_up.weight",
         "hc_ffn_up.weight", KQ_COMPONENT_GATED_RESIDUAL,
         KQ_ROLE_HC_UP, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {10240U, 320U}, 2U, {320U, 10240U}, KQ_MASK_Q8_0}
    };
    size_t index;
    kq_status status;

    for (index = 0U; index < sizeof(specs) / sizeof(specs[0]); ++index) {
        status = kq_add_layer_single(source,
                                     model,
                                     layer,
                                     &specs[index],
                                     KQ_MODEL_NO_AXIS,
                                     0U,
                                     KQ_MODEL_NO_AXIS,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_build_gdn(const kq_model_source *source,
                              kq_model *model,
                              uint32_t layer,
                              kq_diagnostic *diagnostic) {
    static const kq_single_spec specs[] = {
        {"gdn.a_log", "linear_attn.A_log", "ssm_a",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_A_LOG,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         1U, {48U}, 1U, {48U}, KQ_MASK_F32},
        {"gdn.conv", "linear_attn.conv1d.weight", "ssm_conv1d.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_CONV,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         3U, {10240U, 1U, 4U}, 2U, {4U, 10240U}, KQ_MASK_F32},
        {"gdn.dt_bias", "linear_attn.dt_bias", "ssm_dt.bias",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_DT_BIAS,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {48U}, 1U, {48U}, KQ_MASK_F32},
        {"gdn.alpha", "linear_attn.in_proj_a.weight", "ssm_alpha.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_ALPHA,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         2U, {48U, 2560U}, 2U, {2560U, 48U}, KQ_MASK_F32},
        {"gdn.beta", "linear_attn.in_proj_b.weight", "ssm_beta.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_BETA,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         2U, {48U, 2560U}, 2U, {2560U, 48U}, KQ_MASK_F32},
        {"gdn.qkv", "linear_attn.in_proj_qkv.weight", "attn_qkv.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_QKV,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         2U, {10240U, 2560U}, 2U, {2560U, 10240U}, KQ_MASK_Q8_0},
        {"gdn.gate", "linear_attn.in_proj_z.weight", "attn_gate.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_GATE,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         2U, {6144U, 2560U}, 2U, {2560U, 6144U}, KQ_MASK_Q8_0},
        {"gdn.norm", "linear_attn.norm.weight", "ssm_norm.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_NORM,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {128U}, 1U, {128U}, KQ_MASK_F32},
        {"gdn.out", "linear_attn.out_proj.weight", "ssm_out.weight",
         KQ_COMPONENT_GDN, KQ_ROLE_GDN_OUT,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         2U, {2560U, 6144U}, 2U, {6144U, 2560U}, KQ_MASK_Q8_0}
    };
    size_t index;
    kq_status status;

    for (index = 0U; index < sizeof(specs) / sizeof(specs[0]); ++index) {
        status = kq_add_layer_single(source,
                                     model,
                                     layer,
                                     &specs[index],
                                     KQ_MODEL_NO_AXIS,
                                     0U,
                                     KQ_MODEL_NO_AXIS,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_build_qsa(const kq_model_source *source,
                              kq_model *model,
                              uint32_t layer,
                              kq_diagnostic *diagnostic) {
    static const kq_single_spec specs[] = {
        {"qsa.k_norm", "self_attn.k_norm.weight", "attn_k_norm.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_K_NORM,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {256U}, 1U, {256U}, KQ_MASK_F32},
        {"qsa.k", "self_attn.k_proj.weight", "attn_k.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_K,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {512U, 2560U}, 2U, {2560U, 512U}, KQ_MASK_Q8_0},
        {"qsa.output", "self_attn.o_proj.weight", "attn_output.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_OUTPUT,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {2560U, 6144U}, 2U, {6144U, 2560U}, KQ_MASK_Q8_0},
        {"qsa.q_norm", "self_attn.q_norm.weight", "attn_q_norm.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_Q_NORM,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {256U}, 1U, {256U}, KQ_MASK_F32},
        {"qsa.q", "self_attn.q_proj.weight", "attn_q.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_Q,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {12288U, 2560U}, 2U, {2560U, 12288U}, KQ_MASK_Q8_0},
        {"qsa.v", "self_attn.v_proj.weight", "attn_v.weight",
         KQ_COMPONENT_QSA_ATTENTION, KQ_ROLE_QSA_V,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {512U, 2560U}, 2U, {2560U, 512U}, KQ_MASK_Q8_0},
        {"qsa.indexer.k_norm", "self_attn.indexer.k_layernorm.weight",
         "indexer.k_norm.weight", KQ_COMPONENT_QSA_INDEXER,
         KQ_ROLE_QSA_INDEX_K_NORM, KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {128U}, 1U, {128U}, KQ_MASK_F32},
        {"qsa.indexer.q_norm", "self_attn.indexer.q_layernorm.weight",
         "indexer.q_norm.weight", KQ_COMPONENT_QSA_INDEXER,
         KQ_ROLE_QSA_INDEX_Q_NORM, KQ_BINDING_RENAMED_ONE_TO_ONE,
         1U, {128U}, 1U, {128U}, KQ_MASK_F32}
    };
    static const uint64_t canonical_dimensions[] = {640U, 2560U};
    static const uint64_t query_dimensions[] = {2560U, 512U};
    static const uint64_t key_dimensions[] = {2560U, 128U};
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    char query_name[KQ_SEMANTIC_ID_CAPACITY];
    char key_name[KQ_SEMANTIC_ID_CAPACITY];
    kq_semantic_tensor *semantic = NULL;
    size_t index;
    kq_status status;

    for (index = 0U; index < sizeof(specs) / sizeof(specs[0]); ++index) {
        status = kq_add_layer_single(source,
                                     model,
                                     layer,
                                     &specs[index],
                                     KQ_MODEL_NO_AXIS,
                                     0U,
                                     KQ_MODEL_NO_AXIS,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    status = kq_format_checked(semantic_id, sizeof(semantic_id), diagnostic,
                               "layer.%02u.qsa.indexer.qk",
                               (unsigned int)layer);
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(
            canonical_name, sizeof(canonical_name), diagnostic,
            "model.language_model.layers.%u.self_attn.indexer.index_qk_proj.weight",
            (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(query_name, sizeof(query_name), diagnostic,
                                   "blk.%u.indexer.q_proj.weight",
                                   (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(key_name, sizeof(key_name), diagnostic,
                                   "blk.%u.indexer.k_proj.weight",
                                   (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_begin_semantic(
            model, semantic_id, canonical_name, KQ_COMPONENT_QSA_INDEXER,
            KQ_ROLE_QSA_INDEX_QK,
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL,
            layer, KQ_MODEL_LAYER_QSA, KQ_CANONICAL_DTYPE_BF16,
            2U, canonical_dimensions, KQ_MODEL_NO_AXIS, 0U,
            &semantic, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_bind_physical(source, model, semantic, query_name,
                                  2U, query_dimensions, KQ_MASK_BF16,
                                  KQ_BINDING_PART_INDEX_QUERY, 0U, 2U,
                                  0U, 0U, KQ_MODEL_NO_AXIS, 0, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_bind_physical(source, model, semantic, key_name,
                                  2U, key_dimensions, KQ_MASK_BF16,
                                  KQ_BINDING_PART_INDEX_KEY, 1U, 2U,
                                  0U, 0U, KQ_MODEL_NO_AXIS, 0, diagnostic);
    }
    return status;
}

static kq_status kq_build_moe(const kq_model_source *source,
                              kq_model *model,
                              uint32_t layer,
                              kq_diagnostic *diagnostic) {
    static const kq_single_spec specs[] = {
        {"moe.router", "mlp.gate.weight", "ffn_gate_inp.weight",
         KQ_COMPONENT_MOE_ROUTER, KQ_ROLE_MOE_ROUTER,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {512U, 2560U}, 2U, {2560U, 512U}, KQ_MASK_F32},
        {"moe.routed.down", "mlp.experts.down_proj", "ffn_down_exps.weight",
         KQ_COMPONENT_ROUTED_EXPERT_STACK, KQ_ROLE_ROUTED_DOWN,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         3U, {512U, 2560U, 640U}, 3U, {640U, 2560U, 512U},
         KQ_MASK_Q5_1 | KQ_MASK_Q8_0},
        {"moe.shared.down", "mlp.shared_expert.down_proj.weight",
         "ffn_down_shexp.weight", KQ_COMPONENT_SHARED_EXPERT,
         KQ_ROLE_SHARED_DOWN, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {2560U, 640U}, 2U, {640U, 2560U}, KQ_MASK_Q8_0},
        {"moe.shared.gate", "mlp.shared_expert.gate_proj.weight",
         "ffn_gate_shexp.weight", KQ_COMPONENT_SHARED_EXPERT,
         KQ_ROLE_SHARED_GATE, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {640U, 2560U}, 2U, {2560U, 640U}, KQ_MASK_Q8_0},
        {"moe.shared.up", "mlp.shared_expert.up_proj.weight",
         "ffn_up_shexp.weight", KQ_COMPONENT_SHARED_EXPERT,
         KQ_ROLE_SHARED_UP, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {640U, 2560U}, 2U, {2560U, 640U}, KQ_MASK_Q8_0},
        {"moe.shared.gate_weight", "mlp.shared_expert_gate.weight",
         "ffn_gate_inp_shexp.weight", KQ_COMPONENT_SHARED_EXPERT,
         KQ_ROLE_SHARED_GATE_WEIGHT, KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {1U, 2560U}, 1U, {2560U}, KQ_MASK_F32}
    };
    static const uint64_t canonical_dimensions[] = {512U, 1280U, 2560U};
    static const uint64_t part_dimensions[] = {2560U, 640U, 512U};
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    char gate_name[KQ_SEMANTIC_ID_CAPACITY];
    char up_name[KQ_SEMANTIC_ID_CAPACITY];
    kq_semantic_tensor *semantic = NULL;
    size_t index;
    kq_status status;

    for (index = 0U; index < sizeof(specs) / sizeof(specs[0]); ++index) {
        uint32_t canonical_expert_axis = KQ_MODEL_NO_AXIS;
        uint32_t expert_count = 0U;
        uint32_t physical_expert_axis = KQ_MODEL_NO_AXIS;
        if (specs[index].component == KQ_COMPONENT_ROUTED_EXPERT_STACK) {
            canonical_expert_axis = 0U;
            expert_count = KQ_QWEN_EXPERT_COUNT;
            physical_expert_axis = 2U;
        }
        status = kq_add_layer_single(source,
                                     model,
                                     layer,
                                     &specs[index],
                                     canonical_expert_axis,
                                     expert_count,
                                     physical_expert_axis,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }

    status = kq_format_checked(semantic_id, sizeof(semantic_id), diagnostic,
                               "layer.%02u.moe.routed.gate_up",
                               (unsigned int)layer);
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(
            canonical_name, sizeof(canonical_name), diagnostic,
            "model.language_model.layers.%u.mlp.experts.gate_up_proj",
            (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(gate_name, sizeof(gate_name), diagnostic,
                                   "blk.%u.ffn_gate_exps.weight",
                                   (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(up_name, sizeof(up_name), diagnostic,
                                   "blk.%u.ffn_up_exps.weight",
                                   (unsigned int)layer);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_begin_semantic(
            model, semantic_id, canonical_name,
            KQ_COMPONENT_ROUTED_EXPERT_STACK, KQ_ROLE_ROUTED_GATE_UP,
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL,
            layer, model->layer_types[layer], KQ_CANONICAL_DTYPE_BF16,
            3U, canonical_dimensions, 0U, KQ_QWEN_EXPERT_COUNT,
            &semantic, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_bind_physical(
            source, model, semantic, gate_name, 3U, part_dimensions,
            KQ_MASK_Q4_K | KQ_MASK_Q5_K, KQ_BINDING_PART_GATE,
            0U, 2U, 0U, 0U, 2U, 0, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_bind_physical(
            source, model, semantic, up_name, 3U, part_dimensions,
            KQ_MASK_Q4_K | KQ_MASK_Q5_K, KQ_BINDING_PART_UP,
            1U, 2U, 0U, 0U, 2U, 0, diagnostic);
    }
    return status;
}

static kq_status kq_add_ple_metadata(const kq_model_source *source,
                                     kq_model *model,
                                     const char *semantic_leaf,
                                     const char *canonical_leaf,
                                     const char *metadata_key,
                                     kq_semantic_role role,
                                     uint64_t length,
                                     kq_diagnostic *diagnostic) {
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    const kq_gguf_metadata *metadata;
    const uint64_t dimensions[] = {length};
    kq_semantic_tensor *semantic = NULL;
    kq_status status;

    metadata = kq_source_metadata(source, metadata_key);
    if (metadata == NULL || metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_UINT64 ||
        metadata->array_length != length) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                             "required PLE metadata %s is absent or incompatible",
                             metadata_key);
    }
    status = kq_format_checked(semantic_id, sizeof(semantic_id), diagnostic,
                               "layer.01.ple.address.%s", semantic_leaf);
    if (status == KQ_STATUS_OK) {
        status = kq_format_checked(
            canonical_name, sizeof(canonical_name), diagnostic,
            "model.language_model.layers.1.ple.ple_embedding.%s",
            canonical_leaf);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_begin_semantic(
            model, semantic_id, canonical_name,
            KQ_COMPONENT_PLE_ADDRESS_METADATA, role,
            KQ_BINDING_METADATA_DERIVED, KQ_QWEN_PLE_LAYER,
            KQ_MODEL_LAYER_GDN, KQ_CANONICAL_DTYPE_I64,
            1U, dimensions, KQ_MODEL_NO_AXIS, 0U, &semantic, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        return status;
    }
    semantic->binding_count = 1U;
    semantic->bindings[0].metadata = metadata;
    semantic->bindings[0].physical_index = UINT64_MAX;
    semantic->bindings[0].part_role = KQ_BINDING_PART_METADATA;
    semantic->bindings[0].part_count = 1U;
    model->metadata_derived_count += 1U;
    return KQ_STATUS_OK;
}

static kq_status kq_build_ple(const kq_model_source *source,
                              kq_model *model,
                              kq_diagnostic *diagnostic) {
    static const kq_single_spec dense_specs[] = {
        {"ple.conv", "ple.conv1d.weight", "ple_conv1d.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_CONV,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         3U, {10240U, 1U, 4U}, 2U, {4U, 10240U}, KQ_MASK_F32},
        {"ple.key", "ple.key_proj.weight", "ple_key.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_KEY,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {10240U, 2560U}, 2U, {2560U, 10240U}, KQ_MASK_Q8_0},
        {"ple.norm_conv", "ple.norm_conv.weight", "ple_norm_conv.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_NORM_CONV,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"ple.norm_key", "ple.norm_key.weight", "ple_norm_key.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_NORM_KEY,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"ple.norm_query", "ple.norm_query.weight", "ple_norm_query.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_NORM_QUERY,
         KQ_BINDING_TRANSFORMED_LAYOUT,
         1U, {10240U}, 1U, {10240U}, KQ_MASK_F32},
        {"ple.value", "ple.value_proj.weight", "ple_value.weight",
         KQ_COMPONENT_PLE_DENSE, KQ_ROLE_PLE_VALUE,
         KQ_BINDING_RENAMED_ONE_TO_ONE,
         2U, {2560U, 2560U}, 2U, {2560U, 2560U}, KQ_MASK_Q8_0}
    };
    static const uint64_t canonical_dimensions[] = {
        KQ_QWEN_PLE_SHARD_ROWS, KQ_QWEN_PLE_TABLE_WIDTH
    };
    static const uint64_t physical_dimensions[] = {
        KQ_QWEN_PLE_TABLE_WIDTH, KQ_QWEN_PLE_FUSED_ROWS
    };
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    char canonical_name[KQ_CANONICAL_NAME_CAPACITY];
    kq_semantic_tensor *semantic = NULL;
    uint32_t member;
    size_t index;
    kq_status status;

    for (index = 0U;
         index < sizeof(dense_specs) / sizeof(dense_specs[0]);
         ++index) {
        status = kq_add_layer_single(source,
                                     model,
                                     KQ_QWEN_PLE_LAYER,
                                     &dense_specs[index],
                                     KQ_MODEL_NO_AXIS,
                                     0U,
                                     KQ_MODEL_NO_AXIS,
                                     diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }

    for (member = 0U; member < KQ_QWEN_PLE_TABLE_MEMBERS; ++member) {
        status = kq_format_checked(semantic_id, sizeof(semantic_id), diagnostic,
                                   "layer.01.ple.table.%03u",
                                   (unsigned int)member);
        if (status == KQ_STATUS_OK) {
            status = kq_format_checked(
                canonical_name, sizeof(canonical_name), diagnostic,
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%u.weight",
                (unsigned int)member);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_begin_semantic(
                model, semantic_id, canonical_name, KQ_COMPONENT_PLE_TABLE,
                KQ_ROLE_PLE_TABLE,
                KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL,
                KQ_QWEN_PLE_LAYER, KQ_MODEL_LAYER_GDN,
                KQ_CANONICAL_DTYPE_BF16, 2U, canonical_dimensions,
                KQ_MODEL_NO_AXIS, 0U, &semantic, diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_bind_physical(
                source, model, semantic, "per_layer_token_embd.weight",
                2U, physical_dimensions, KQ_MASK_IQ4_NL,
                KQ_BINDING_PART_FUSED_MEMBER, 0U, 1U, member,
                KQ_QWEN_PLE_TABLE_MEMBERS, KQ_MODEL_NO_AXIS,
                member != 0U, diagnostic);
        }
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }

    status = kq_add_ple_metadata(
        source, model, "layer_multipliers", "layer_multipliers",
        "qwen4exp.ple.layer_multipliers", KQ_ROLE_PLE_LAYER_MULTIPLIERS,
        3U, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_add_ple_metadata(
            source, model, "head_offsets", "ngram_heads_offsets",
            "qwen4exp.ple.head_offsets", KQ_ROLE_PLE_HEAD_OFFSETS,
            KQ_QWEN_PLE_HEAD_COUNT, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_add_ple_metadata(
            source, model, "head_vocab_sizes", "ngram_heads_vocab_sizes",
            "qwen4exp.ple.head_vocab_sizes",
            KQ_ROLE_PLE_HEAD_VOCAB_SIZES,
            KQ_QWEN_PLE_HEAD_COUNT, diagnostic);
    }
    return status;
}

static kq_status kq_validate_registry_complete(const kq_model_source *source,
                                               kq_model *model,
                                               kq_diagnostic *diagnostic) {
    const kq_semantic_tensor *semantic;
    uint64_t index;

    model->unbound_required_count = 0U;
    for (index = 0U; index < model->semantic_count; ++index) {
        semantic = &model->semantics[(size_t)index];
        if (semantic->runtime_scope == KQ_SCOPE_REQUIRED_INITIAL_TEXT &&
            semantic->binding_count == 0U) {
            model->unbound_required_count += 1U;
        }
        if (semantic->relation ==
                KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL &&
            semantic->binding_count != 2U) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "split semantic %s does not have two parts",
                                 semantic->semantic_id);
        }
        if (semantic->relation ==
                KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL &&
            (semantic->binding_count != 1U ||
             semantic->bindings[0].fused_member_count !=
                 KQ_QWEN_PLE_TABLE_MEMBERS)) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "fused semantic %s has invalid member geometry",
                                 semantic->semantic_id);
        }
        if (semantic->relation == KQ_BINDING_METADATA_DERIVED &&
            (semantic->binding_count != 1U ||
             semantic->bindings[0].metadata == NULL ||
             semantic->bindings[0].physical != NULL)) {
            return kq_model_fail(diagnostic,
                                 KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                                 "metadata-derived semantic %s is malformed",
                                 semantic->semantic_id);
        }
    }
    model->unknown_physical_count = 0U;
    for (index = 0U; index < source->tensor_count; ++index) {
        if (model->physical_coverage[(size_t)index] == 0U) {
            model->unknown_physical_count += 1U;
        }
    }
    if (model->semantic_count != KQ_QWEN_INITIAL_SEMANTICS) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "registry produced %llu semantics instead of %u",
                             (unsigned long long)model->semantic_count,
                             (unsigned int)KQ_QWEN_INITIAL_SEMANTICS);
    }
    if (model->unknown_physical_count != 0U) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_UNKNOWN_PHYSICAL_TENSOR,
                             "%llu physical tensors are not explained by target rules",
                             (unsigned long long)model->unknown_physical_count);
    }
    if (model->unbound_required_count != 0U) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "%llu required semantics are unbound",
                             (unsigned long long)model->unbound_required_count);
    }
    if (model->metadata_derived_count != 3U ||
        model->gdn_layer_count != KQ_QWEN_GDN_LAYER_COUNT ||
        model->qsa_layer_count != KQ_QWEN_QSA_LAYER_COUNT) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "registry summary invariants are inconsistent");
    }
    return KQ_STATUS_OK;
}

kq_status kq_model_open_from_source(const kq_model_source *source,
                                    kq_model **out_model,
                                    kq_diagnostic *diagnostic) {
    kq_model *model = NULL;
    kq_status status;
    uint32_t layer;

    kq_diagnostic_clear(diagnostic);
    if (source == NULL || out_model == NULL ||
        source->find_metadata == NULL || source->tensor_at == NULL) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_INVALID_ARGUMENT,
                             "model source and output registry are required");
    }
    *out_model = NULL;
    if (source->tensor_count == 0U ||
        source->tensor_count > (uint64_t)SIZE_MAX) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                             "model source tensor count is invalid");
    }
    model = (kq_model *)calloc(1U, sizeof(*model));
    if (model == NULL) {
        return kq_model_fail(diagnostic,
                             KQ_STATUS_OUT_OF_MEMORY,
                             "could not allocate semantic model registry");
    }
    model->semantic_capacity = KQ_QWEN_INITIAL_SEMANTICS;
    model->physical_count = source->tensor_count;
    model->semantics = (kq_semantic_tensor *)calloc(
        (size_t)model->semantic_capacity,
        sizeof(*model->semantics));
    model->physical_coverage = (unsigned char *)calloc(
        (size_t)model->physical_count,
        sizeof(*model->physical_coverage));
    if (model->semantics == NULL || model->physical_coverage == NULL) {
        kq_model_close(model);
        return kq_model_fail(diagnostic,
                             KQ_STATUS_OUT_OF_MEMORY,
                             "could not allocate semantic registry entries");
    }

    status = kq_validate_model_identity(source, model, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_build_top_level(source, model, diagnostic);
    }
    for (layer = 0U;
         status == KQ_STATUS_OK && layer < KQ_QWEN_LAYER_COUNT;
         ++layer) {
        status = kq_build_gated_residual(source, model, layer, diagnostic);
        if (status == KQ_STATUS_OK) {
            status = model->layer_types[layer] == KQ_MODEL_LAYER_GDN
                         ? kq_build_gdn(source, model, layer, diagnostic)
                         : kq_build_qsa(source, model, layer, diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_build_moe(source, model, layer, diagnostic);
        }
    }
    if (status == KQ_STATUS_OK) {
        status = kq_build_ple(source, model, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_validate_registry_complete(source, model, diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        kq_model_close(model);
        return status;
    }
    *out_model = model;
    return KQ_STATUS_OK;
}

static const kq_gguf_metadata *kq_gguf_source_find_metadata(void *context,
                                                            const char *key) {
    return kq_gguf_find_metadata((const kq_gguf *)context, key);
}

static const kq_gguf_tensor *kq_gguf_source_tensor_at(void *context,
                                                      uint64_t index) {
    return kq_gguf_tensor_at((const kq_gguf *)context, index);
}

kq_status kq_model_open_from_gguf(const kq_gguf *gguf,
                                  kq_model **out_model,
                                  kq_diagnostic *diagnostic) {
    kq_model_source source;
    kq_status status;

    if (gguf == NULL || out_model == NULL) {
        kq_diagnostic_clear(diagnostic);
        return kq_model_fail(diagnostic,
                             KQ_STATUS_INVALID_ARGUMENT,
                             "GGUF and output registry are required");
    }
    source.context = (void *)gguf;
    source.architecture = kq_gguf_architecture(gguf);
    source.metadata_count = kq_gguf_metadata_count(gguf);
    source.find_metadata = kq_gguf_source_find_metadata;
    source.tensor_count = kq_gguf_tensor_count(gguf);
    source.tensor_at = kq_gguf_source_tensor_at;
    source.payload_bytes_accessed = kq_gguf_payload_bytes_accessed(gguf);
    status = kq_model_open_from_source(&source,
                                       out_model,
                                       diagnostic);
    if (status == KQ_STATUS_OK) {
        (*out_model)->gguf = gguf;
    }
    return status;
}

void kq_model_close(kq_model *model) {
    if (model == NULL) {
        return;
    }
    free(model->physical_coverage);
    free(model->semantics);
    free(model);
}

uint32_t kq_model_hidden_size(const kq_model *model) {
    return model == NULL ? 0U : model->hidden_size;
}

uint32_t kq_model_vocabulary_size(const kq_model *model) {
    return model == NULL ? 0U : model->vocabulary_size;
}

uint32_t kq_model_context_length(const kq_model *model) {
    return model == NULL ? 0U : model->context_length;
}

uint32_t kq_model_layer_count(const kq_model *model) {
    return model == NULL ? 0U : model->layer_count;
}

uint32_t kq_model_gdn_layer_count(const kq_model *model) {
    return model == NULL ? 0U : model->gdn_layer_count;
}

uint32_t kq_model_qsa_layer_count(const kq_model *model) {
    return model == NULL ? 0U : model->qsa_layer_count;
}

uint32_t kq_model_expert_count(const kq_model *model) {
    return model == NULL ? 0U : model->expert_count;
}

uint32_t kq_model_expert_top_k(const kq_model *model) {
    return model == NULL ? 0U : model->expert_top_k;
}

kq_model_layer_type kq_model_layer_type_at(const kq_model *model,
                                           uint32_t layer_id) {
    if (model == NULL || layer_id >= model->layer_count) {
        return KQ_MODEL_LAYER_INVALID;
    }
    return model->layer_types[layer_id];
}

uint64_t kq_model_semantic_tensor_count(const kq_model *model) {
    return model == NULL ? 0U : model->semantic_count;
}

uint64_t kq_model_physical_tensor_count(const kq_model *model) {
    return model == NULL ? 0U : model->physical_count;
}

uint64_t kq_model_physical_coverage_count(const kq_model *model) {
    return model == NULL ? 0U : model->physical_coverage_count;
}

uint64_t kq_model_metadata_derived_count(const kq_model *model) {
    return model == NULL ? 0U : model->metadata_derived_count;
}

uint64_t kq_model_unknown_physical_count(const kq_model *model) {
    return model == NULL ? 0U : model->unknown_physical_count;
}

uint64_t kq_model_unbound_required_count(const kq_model *model) {
    return model == NULL ? 0U : model->unbound_required_count;
}

uint64_t kq_model_relation_count(const kq_model *model,
                                 kq_binding_relation relation) {
    if (model == NULL || relation >= KQ_BINDING_RELATION_COUNT) {
        return 0U;
    }
    return model->relation_counts[relation];
}

uint64_t kq_model_component_count(const kq_model *model,
                                  kq_semantic_component component) {
    if (model == NULL || component >= KQ_COMPONENT_COUNT) {
        return 0U;
    }
    return model->component_counts[component];
}

uint64_t kq_model_placement_count(const kq_model *model,
                                  kq_placement_hint placement) {
    if (model == NULL || placement >= KQ_PLACEMENT_HINT_COUNT) {
        return 0U;
    }
    return model->placement_counts[placement];
}

const kq_semantic_tensor *kq_model_semantic_tensor_at(const kq_model *model,
                                                      uint64_t index) {
    if (model == NULL || index >= model->semantic_count) {
        return NULL;
    }
    return &model->semantics[(size_t)index];
}

const kq_semantic_tensor *kq_model_find_semantic_tensor(
    const kq_model *model,
    const char *semantic_id) {
    uint64_t index;
    if (model == NULL || semantic_id == NULL) {
        return NULL;
    }
    for (index = 0U; index < model->semantic_count; ++index) {
        if (strcmp(model->semantics[(size_t)index].semantic_id,
                   semantic_id) == 0) {
            return &model->semantics[(size_t)index];
        }
    }
    return NULL;
}

const char *kq_model_layer_type_name(kq_model_layer_type type) {
    switch (type) {
        case KQ_MODEL_LAYER_GDN:
            return "GDN";
        case KQ_MODEL_LAYER_QSA:
            return "QSA";
        default:
            return "INVALID";
    }
}

const char *kq_semantic_component_name(kq_semantic_component component) {
    static const char *const names[KQ_COMPONENT_COUNT] = {
        "TOKEN_EMBEDDING",
        "LM_HEAD",
        "FINAL_GATED_RESIDUAL",
        "GATED_RESIDUAL",
        "GDN",
        "QSA_ATTENTION",
        "QSA_INDEXER",
        "MOE_ROUTER",
        "ROUTED_EXPERT_STACK",
        "SHARED_EXPERT",
        "PLE_TABLE",
        "PLE_DENSE",
        "PLE_ADDRESS_METADATA"
    };
    return component < KQ_COMPONENT_COUNT ? names[component] : "INVALID";
}

const char *kq_semantic_role_name(kq_semantic_role role) {
    static const char *const names[] = {
        "TOKEN_EMBEDDING", "LM_HEAD", "HC_NORM", "HC_DOWN", "HC_UP",
        "HC_INJECT", "GDN_A_LOG", "GDN_CONV", "GDN_DT_BIAS",
        "GDN_ALPHA", "GDN_BETA", "GDN_QKV", "GDN_GATE", "GDN_NORM",
        "GDN_OUT", "QSA_K_NORM", "QSA_K", "QSA_OUTPUT", "QSA_Q_NORM",
        "QSA_Q", "QSA_V", "QSA_INDEX_QK", "QSA_INDEX_K_NORM",
        "QSA_INDEX_Q_NORM", "MOE_ROUTER", "ROUTED_DOWN",
        "ROUTED_GATE_UP", "SHARED_DOWN", "SHARED_GATE", "SHARED_UP",
        "SHARED_GATE_WEIGHT", "PLE_TABLE", "PLE_CONV", "PLE_KEY",
        "PLE_NORM_CONV", "PLE_NORM_KEY", "PLE_NORM_QUERY", "PLE_VALUE",
        "PLE_LAYER_MULTIPLIERS", "PLE_HEAD_OFFSETS",
        "PLE_HEAD_VOCAB_SIZES"
    };
    return (size_t)role < sizeof(names) / sizeof(names[0])
               ? names[(size_t)role]
               : "INVALID";
}

const char *kq_binding_relation_name(kq_binding_relation relation) {
    static const char *const names[KQ_BINDING_RELATION_COUNT] = {
        "DIRECT_ONE_TO_ONE",
        "RENAMED_ONE_TO_ONE",
        "TRANSFORMED_LAYOUT",
        "ONE_CANONICAL_TO_MULTIPLE_PHYSICAL",
        "MULTIPLE_CANONICAL_TO_ONE_PHYSICAL",
        "METADATA_DERIVED",
        "ABSENT_INITIAL_SCOPE"
    };
    return relation < KQ_BINDING_RELATION_COUNT ? names[relation] : "INVALID";
}

const char *kq_binding_part_role_name(kq_binding_part_role role) {
    static const char *const names[] = {
        "WHOLE", "GATE", "UP", "INDEX_QUERY", "INDEX_KEY",
        "FUSED_MEMBER", "METADATA"
    };
    return (size_t)role < sizeof(names) / sizeof(names[0])
               ? names[(size_t)role]
               : "INVALID";
}

const char *kq_runtime_scope_name(kq_runtime_scope scope) {
    switch (scope) {
        case KQ_SCOPE_REQUIRED_INITIAL_TEXT:
            return "REQUIRED_INITIAL_TEXT";
        case KQ_SCOPE_EXCLUDED_INITIAL_VISION:
            return "EXCLUDED_INITIAL_VISION";
        case KQ_SCOPE_EXCLUDED_INITIAL_MTP:
            return "EXCLUDED_INITIAL_MTP";
        default:
            return "INVALID";
    }
}

const char *kq_placement_hint_name(kq_placement_hint placement) {
    static const char *const names[KQ_PLACEMENT_HINT_COUNT] = {
        "ALWAYS_NEEDED_CANDIDATE",
        "ROUTED_EXPERT_CACHE_CANDIDATE",
        "PLE_DISK_BACKED_CANDIDATE",
        "EXCLUDED_INITIAL_SCOPE",
        "NEUTRAL"
    };
    return placement < KQ_PLACEMENT_HINT_COUNT ? names[placement] : "INVALID";
}
