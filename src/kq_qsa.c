#include "kq_qsa_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_tensor_view.h"

typedef struct kq_qsa_role_spec {
    kq_semantic_role role;
    const char *suffix;
    kq_binding_relation relation;
    uint32_t rank;
    uint64_t dimensions[2];
    uint32_t physical_type;
} kq_qsa_role_spec;

static const kq_qsa_role_spec kq_qsa_specs[KQ_QSA_WEIGHT_ROLE_COUNT] = {
    {KQ_ROLE_QSA_K_NORM, "k_norm", KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {256U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_QSA_K, "k", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {512U, 2560U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_QSA_OUTPUT, "output", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {2560U, 6144U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_QSA_Q_NORM, "q_norm", KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {256U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_QSA_Q, "q", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {12288U, 2560U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_QSA_V, "v", KQ_BINDING_RENAMED_ONE_TO_ONE,
     2U, {512U, 2560U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_QSA_INDEX_QK, "indexer.qk",
     KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL,
     2U, {640U, 2560U}, KQ_GGUF_TYPE_BF16},
    {KQ_ROLE_QSA_INDEX_K_NORM, "indexer.k_norm",
     KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {128U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_QSA_INDEX_Q_NORM, "indexer.q_norm",
     KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {128U, 0U}, KQ_GGUF_TYPE_F32}
};

static kq_status kq_qsa_fail(kq_diagnostic *diagnostic,
                             kq_status status,
                             const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

int kq_qsa_u64_add(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || UINT64_MAX - left < right) return 0;
    *output = left + right;
    return 1;
}

int kq_qsa_u64_mul(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || (left != 0U && right > UINT64_MAX / left)) return 0;
    *output = left * right;
    return 1;
}

static int kq_qsa_accumulate(uint64_t *total, uint64_t value) {
    return kq_qsa_u64_add(*total, value, total);
}

int kq_qsa_config_valid(const kq_qsa_config *config) {
    return config != NULL && config->magic == KQ_QSA_CONFIG_MAGIC;
}

int kq_qsa_state_valid(const kq_qsa_state *state) {
    return state != NULL && state->magic == KQ_QSA_STATE_MAGIC &&
           kq_qsa_config_valid(state->config) && state->key_cache != NULL &&
           state->value_cache != NULL && state->raw_index_key_cache != NULL &&
           state->length <= state->capacity;
}

static uint16_t kq_qsa_f32_to_bf16(float value) {
    uint32_t bits;
    uint32_t rounded;
    memcpy(&bits, &value, sizeof(bits));
    rounded = bits + UINT32_C(0x00007fff) + ((bits >> 16U) & 1U);
    return (uint16_t)(rounded >> 16U);
}

static float kq_qsa_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16U;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static kq_status kq_qsa_validate_dimensions(
    const kq_qsa_dimensions *d,
    uint32_t *repeat,
    uint32_t *block_topk,
    uint32_t *selected_capacity,
    uint64_t *kv_per_token,
    uint64_t *raw_per_token,
    uint64_t *bytes_per_token,
    kq_diagnostic *diagnostic) {
    uint64_t kv;
    uint64_t total;
    uint64_t element_bytes;
    if (d == NULL || repeat == NULL || block_topk == NULL ||
        selected_capacity == NULL || kv_per_token == NULL ||
        raw_per_token == NULL || bytes_per_token == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "complete QSA dimensions and outputs are required");
    }
    if (d->hidden_size == 0U || d->query_head_count == 0U ||
        d->key_value_head_count == 0U || d->head_dimension == 0U ||
        d->query_head_count % d->key_value_head_count != 0U ||
        d->index_query_head_count == 0U || d->index_head_dimension == 0U ||
        d->block_size == 0U || d->token_budget == 0U ||
        d->token_budget % d->block_size != 0U || d->max_context == 0U ||
        d->rotary_dimension == 0U ||
        (d->rotary_dimension & 1U) != 0U ||
        d->rotary_dimension > d->head_dimension ||
        d->rotary_dimension > d->index_head_dimension ||
        !(d->rms_norm_epsilon > 0.0f) || !isfinite(d->rms_norm_epsilon) ||
        !(d->rope_theta > 1.0f) || !isfinite(d->rope_theta) ||
        (d->activation_dtype != KQ_QSA_ACTIVATION_F32 &&
         d->activation_dtype != KQ_QSA_ACTIVATION_BF16)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                           "invalid QSA dimensions, block geometry, RoPE, epsilon, or dtype");
    }
    if (!kq_qsa_u64_mul(d->key_value_head_count, d->head_dimension, &kv) ||
        !kq_qsa_u64_add(kv, kv, &total) ||
        !kq_qsa_u64_add(total, d->index_head_dimension, &total)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "QSA state geometry overflows");
    }
    element_bytes = d->activation_dtype == KQ_QSA_ACTIVATION_BF16 ?
        sizeof(uint16_t) : sizeof(float);
    if (!kq_qsa_u64_mul(total, element_bytes, bytes_per_token)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "QSA state byte count overflows");
    }
    *repeat = d->query_head_count / d->key_value_head_count;
    *block_topk = d->token_budget / d->block_size;
    if (d->token_budget > UINT32_MAX - (d->block_size - 1U)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "QSA selected-token capacity overflows");
    }
    *selected_capacity = d->token_budget + d->block_size - 1U;
    *kv_per_token = kv;
    *raw_per_token = d->index_head_dimension;
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_validate_target_tensor(
    const kq_semantic_tensor *semantic,
    const kq_qsa_role_spec *spec,
    uint32_t layer_id,
    kq_diagnostic *diagnostic) {
    uint32_t dim;
    uint32_t binding;
    kq_quant_geometry geometry;
    if (semantic == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                           "required QSA semantic tensor is missing");
    }
    if ((semantic->component != KQ_COMPONENT_QSA_ATTENTION &&
         semantic->component != KQ_COMPONENT_QSA_INDEXER) ||
        semantic->role != spec->role ||
        semantic->layer_type != KQ_MODEL_LAYER_QSA ||
        semantic->layer_id != layer_id ||
        semantic->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
        semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 ||
        semantic->relation != spec->relation ||
        semantic->canonical_rank != spec->rank) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                           "QSA semantic identity, dtype, relation, or rank mismatch");
    }
    for (dim = 0U; dim < spec->rank; ++dim) {
        if (semantic->canonical_dimensions[dim] != spec->dimensions[dim]) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                               "QSA canonical tensor shape mismatch");
        }
    }
    if (spec->role == KQ_ROLE_QSA_INDEX_QK) {
        if (semantic->binding_count != 2U ||
            semantic->bindings[0].part_role != KQ_BINDING_PART_INDEX_QUERY ||
            semantic->bindings[0].part_index != 0U ||
            semantic->bindings[0].part_count != 2U ||
            semantic->bindings[1].part_role != KQ_BINDING_PART_INDEX_KEY ||
            semantic->bindings[1].part_index != 1U ||
            semantic->bindings[1].part_count != 2U) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "QSA index_qk split binding is missing or out of order");
        }
    } else if (semantic->binding_count != 1U ||
               semantic->bindings[0].part_role != KQ_BINDING_PART_WHOLE ||
               semantic->bindings[0].part_index != 0U ||
               semantic->bindings[0].part_count != 1U) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                           "QSA whole-tensor binding is malformed");
    }
    for (binding = 0U; binding < semantic->binding_count; ++binding) {
        const kq_gguf_tensor *physical =
            semantic->bindings[binding].physical;
        uint64_t expected_first;
        uint64_t expected_second = 0U;
        if (spec->role == KQ_ROLE_QSA_INDEX_QK) {
            expected_first = spec->dimensions[1];
            expected_second = binding == 0U ? 512U : 128U;
        } else if (spec->rank == 2U) {
            expected_first = spec->dimensions[1];
            expected_second = spec->dimensions[0];
        } else {
            expected_first = spec->dimensions[0];
        }
        if (physical == NULL || physical->type_id != spec->physical_type ||
            physical->rank != spec->rank ||
            physical->dimensions[0] != expected_first ||
            (spec->rank == 2U &&
             physical->dimensions[1] != expected_second) ||
            kq_quant_geometry_for_type(
                physical->type_id,
                &geometry, diagnostic) != KQ_STATUS_OK) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                               "QSA physical binding type, shape, or geometry mismatch");
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_qsa_config_create_from_source(
    const kq_qsa_semantic_source *source,
    int require_target_bindings,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_qsa_config *config;
    kq_status status;
    uint32_t index;
    kq_diagnostic_clear(diagnostic);
    if (source == NULL || out_config == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "QSA semantic source and output are required");
    }
    *out_config = NULL;
    if (source->layer_type != KQ_MODEL_LAYER_QSA) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                           "QSA configuration requested for a non-QSA layer");
    }
    if (require_target_bindings) {
        const kq_qsa_dimensions *d = &source->dimensions;
        if (source->model == NULL || source->layer_id >= 48U ||
            source->layer_id % 4U != 3U || d->hidden_size != 2560U ||
            d->query_head_count != 24U || d->key_value_head_count != 2U ||
            d->head_dimension != 256U || d->index_query_head_count != 4U ||
            d->index_head_dimension != 128U || d->block_size != 4U ||
            d->token_budget != 2048U || d->max_context != 262144U ||
            d->rotary_dimension != 64U ||
            (d->activation_dtype != KQ_QSA_ACTIVATION_F32 &&
             d->activation_dtype != KQ_QSA_ACTIVATION_BF16)) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                               "target Qwen3.8 QSA topology mismatch");
        }
        for (index = 0U; index < KQ_QSA_WEIGHT_ROLE_COUNT; ++index) {
            status = kq_qsa_validate_target_tensor(
                source->tensors[index], &kq_qsa_specs[index],
                source->layer_id, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
    }
    config = (kq_qsa_config *)calloc(1U, sizeof(*config));
    if (config == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate immutable QSA config");
    }
    status = kq_qsa_validate_dimensions(
        &source->dimensions, &config->key_value_repeat,
        &config->block_topk, &config->selected_token_capacity,
        &config->key_values_per_token, &config->raw_index_values_per_token,
        &config->semantic_state_bytes_per_token, diagnostic);
    if (status != KQ_STATUS_OK) {
        free(config);
        return status;
    }
    config->magic = KQ_QSA_CONFIG_MAGIC;
    config->model = source->model;
    config->layer_id = source->layer_id;
    config->dimensions = source->dimensions;
    for (index = 0U; index < KQ_QSA_WEIGHT_ROLE_COUNT; ++index) {
        config->tensors[index] = source->tensors[index];
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

kq_status kq_qsa_test_config_create(
    const kq_qsa_dimensions *dimensions,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_qsa_semantic_source source;
    if (dimensions == NULL) {
        kq_diagnostic_clear(diagnostic);
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "test QSA dimensions are required");
    }
    memset(&source, 0, sizeof(source));
    source.layer_type = KQ_MODEL_LAYER_QSA;
    source.dimensions = *dimensions;
    return kq_qsa_config_create_from_source(&source, 0, out_config, diagnostic);
}

static kq_status kq_qsa_config_create_for_model(
    const kq_model *model,
    uint32_t layer_id,
    kq_qsa_activation_dtype dtype,
    kq_qsa_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_qsa_semantic_source source;
    char id[KQ_SEMANTIC_ID_CAPACITY];
    uint32_t index;
    int length;
    kq_diagnostic_clear(diagnostic);
    if (model == NULL || out_config == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "model and QSA config output are required");
    }
    *out_config = NULL;
    if (layer_id >= kq_model_layer_count(model) ||
        kq_model_layer_type_at(model, layer_id) != KQ_MODEL_LAYER_QSA) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                           "requested layer is not a canonical QSA layer");
    }
    memset(&source, 0, sizeof(source));
    source.model = model;
    source.layer_id = layer_id;
    source.layer_type = KQ_MODEL_LAYER_QSA;
    source.dimensions.hidden_size = kq_model_hidden_size(model);
    source.dimensions.query_head_count = 24U;
    source.dimensions.key_value_head_count = 2U;
    source.dimensions.head_dimension = 256U;
    source.dimensions.index_query_head_count = 4U;
    source.dimensions.index_head_dimension = 128U;
    source.dimensions.block_size = 4U;
    source.dimensions.token_budget = 2048U;
    source.dimensions.max_context = kq_model_context_length(model);
    source.dimensions.rotary_dimension = 64U;
    source.dimensions.rms_norm_epsilon = 1.0e-6f;
    source.dimensions.rope_theta = 10000000.0f;
    source.dimensions.activation_dtype = dtype;
    for (index = 0U; index < KQ_QSA_WEIGHT_ROLE_COUNT; ++index) {
        length = snprintf(id, sizeof(id), "layer.%02u.qsa.%s",
                          layer_id, kq_qsa_specs[index].suffix);
        if (length < 0 || (size_t)length >= sizeof(id)) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "QSA semantic ID formatting overflow");
        }
        source.tensors[index] = kq_model_find_semantic_tensor(model, id);
    }
    return kq_qsa_config_create_from_source(&source, 1, out_config, diagnostic);
}

kq_status kq_qsa_config_create(const kq_model *model, uint32_t layer_id,
                               kq_qsa_config **out_config,
                               kq_diagnostic *diagnostic) {
    return kq_qsa_config_create_for_model(
        model, layer_id, KQ_QSA_ACTIVATION_BF16, out_config, diagnostic);
}

kq_status kq_qsa_config_create_reference_f32(
    const kq_model *model, uint32_t layer_id,
    kq_qsa_config **out_config, kq_diagnostic *diagnostic) {
    return kq_qsa_config_create_for_model(
        model, layer_id, KQ_QSA_ACTIVATION_F32, out_config, diagnostic);
}

void kq_qsa_config_close(kq_qsa_config *config) {
    if (config == NULL) return;
    config->magic = 0U;
    free(config);
}

uint32_t kq_qsa_config_layer_id(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->layer_id : UINT32_MAX;
}
uint32_t kq_qsa_config_hidden_size(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.hidden_size : 0U;
}
uint32_t kq_qsa_config_query_head_count(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.query_head_count : 0U;
}
uint32_t kq_qsa_config_key_value_head_count(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.key_value_head_count : 0U;
}
uint32_t kq_qsa_config_head_dimension(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.head_dimension : 0U;
}
uint32_t kq_qsa_config_index_query_head_count(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.index_query_head_count : 0U;
}
uint32_t kq_qsa_config_index_head_dimension(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.index_head_dimension : 0U;
}
uint32_t kq_qsa_config_block_size(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.block_size : 0U;
}
uint32_t kq_qsa_config_block_selection_limit(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->block_topk : 0U;
}
uint32_t kq_qsa_config_context_limit(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.max_context : 0U;
}
kq_qsa_activation_dtype kq_qsa_config_activation_dtype(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->dimensions.activation_dtype :
        KQ_QSA_ACTIVATION_F32;
}
uint64_t kq_qsa_config_owned_bytes(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? sizeof(*c) : 0U;
}
uint64_t kq_qsa_config_semantic_state_bytes_per_token(const kq_qsa_config *c) {
    return kq_qsa_config_valid(c) ? c->semantic_state_bytes_per_token : 0U;
}

kq_status kq_qsa_state_create(const kq_qsa_config *config,
                              uint64_t token_capacity,
                              kq_qsa_state **out_state,
                              kq_diagnostic *diagnostic) {
    kq_qsa_state *state;
    uint64_t kv_elements;
    uint64_t raw_elements;
    uint64_t element_bytes;
    uint64_t kv_bytes;
    uint64_t raw_bytes;
    uint64_t total;
    kq_diagnostic_clear(diagnostic);
    if (out_state == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "QSA state output is required");
    }
    *out_state = NULL;
    if (!kq_qsa_config_valid(config) || token_capacity == 0U) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid QSA config and bounded positive capacity are required");
    }
    if (token_capacity > config->dimensions.max_context) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_LIMIT_EXCEEDED,
                           "QSA state capacity exceeds the context limit");
    }
    element_bytes = config->dimensions.activation_dtype ==
        KQ_QSA_ACTIVATION_BF16 ? sizeof(uint16_t) : sizeof(float);
    if (!kq_qsa_u64_mul(token_capacity, config->key_values_per_token,
                        &kv_elements) ||
        !kq_qsa_u64_mul(token_capacity, config->raw_index_values_per_token,
                        &raw_elements) ||
        !kq_qsa_u64_mul(kv_elements, element_bytes, &kv_bytes) ||
        !kq_qsa_u64_mul(raw_elements, element_bytes, &raw_bytes) ||
        !kq_qsa_u64_add(sizeof(*state), kv_bytes, &total) ||
        !kq_qsa_u64_add(total, kv_bytes, &total) ||
        !kq_qsa_u64_add(total, raw_bytes, &total) || total > SIZE_MAX) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "QSA state allocation size overflows");
    }
    state = (kq_qsa_state *)calloc(1U, sizeof(*state));
    if (state == NULL) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate QSA state object");
    }
    state->key_cache = calloc(1U, (size_t)kv_bytes);
    state->value_cache = calloc(1U, (size_t)kv_bytes);
    state->raw_index_key_cache = calloc(1U, (size_t)raw_bytes);
    if (state->key_cache == NULL || state->value_cache == NULL ||
        state->raw_index_key_cache == NULL) {
        kq_qsa_state_close(state);
        return kq_qsa_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate bounded QSA cache buffers");
    }
    state->magic = KQ_QSA_STATE_MAGIC;
    state->config = config;
    state->capacity = token_capacity;
    state->owned_bytes = total;
    *out_state = state;
    return KQ_STATUS_OK;
}

void kq_qsa_state_close(kq_qsa_state *state) {
    if (state == NULL) return;
    free(state->key_cache);
    free(state->value_cache);
    free(state->raw_index_key_cache);
    state->magic = 0U;
    free(state);
}

kq_status kq_qsa_state_reset(kq_qsa_state *state,
                             kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    if (!kq_qsa_state_valid(state)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_QSA_STATE,
                           "valid QSA state is required for reset");
    }
    state->length = 0U;
    return KQ_STATUS_OK;
}

uint64_t kq_qsa_state_length(const kq_qsa_state *s) {
    return kq_qsa_state_valid(s) ? s->length : 0U;
}
uint64_t kq_qsa_state_capacity(const kq_qsa_state *s) {
    return kq_qsa_state_valid(s) ? s->capacity : 0U;
}
uint64_t kq_qsa_state_owned_bytes(const kq_qsa_state *s) {
    return kq_qsa_state_valid(s) ? s->owned_bytes : 0U;
}

static float kq_qsa_state_read(const void *source,
                               kq_qsa_activation_dtype dtype,
                               uint64_t index) {
    return dtype == KQ_QSA_ACTIVATION_BF16 ?
        kq_qsa_bf16_to_f32(((const uint16_t *)source)[index]) :
        ((const float *)source)[index];
}

static void kq_qsa_state_write(void *target,
                               kq_qsa_activation_dtype dtype,
                               uint64_t index,
                               float value) {
    if (dtype == KQ_QSA_ACTIVATION_BF16) {
        ((uint16_t *)target)[index] = kq_qsa_f32_to_bf16(value);
    } else {
        ((float *)target)[index] = value;
    }
}

kq_status kq_qsa_state_export_f32(
    const kq_qsa_state *state, float *key_cache, uint64_t key_capacity,
    float *value_cache, uint64_t value_capacity,
    float *raw_cache, uint64_t raw_capacity, uint64_t *length,
    kq_diagnostic *diagnostic) {
    uint64_t kv_count;
    uint64_t raw_count;
    uint64_t index;
    kq_diagnostic_clear(diagnostic);
    if (!kq_qsa_state_valid(state) || key_cache == NULL ||
        value_cache == NULL || raw_cache == NULL || length == NULL ||
        !kq_qsa_u64_mul(state->length,
                        state->config->key_values_per_token, &kv_count) ||
        !kq_qsa_u64_mul(state->length,
                        state->config->raw_index_values_per_token, &raw_count)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid QSA state and export buffers are required");
    }
    if (key_capacity < kv_count || value_capacity < kv_count ||
        raw_capacity < raw_count) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                           "QSA state export capacity is too small");
    }
    for (index = 0U; index < kv_count; ++index) {
        key_cache[index] = kq_qsa_state_read(
            state->key_cache, state->config->dimensions.activation_dtype, index);
        value_cache[index] = kq_qsa_state_read(
            state->value_cache, state->config->dimensions.activation_dtype, index);
    }
    for (index = 0U; index < raw_count; ++index) {
        raw_cache[index] = kq_qsa_state_read(
            state->raw_index_key_cache,
            state->config->dimensions.activation_dtype, index);
    }
    *length = state->length;
    return KQ_STATUS_OK;
}

kq_status kq_qsa_state_import_f32(
    kq_qsa_state *state, const float *key_cache, uint64_t key_count,
    const float *value_cache, uint64_t value_count,
    const float *raw_cache, uint64_t raw_count, uint64_t length,
    kq_diagnostic *diagnostic) {
    uint64_t expected_kv;
    uint64_t expected_raw;
    uint64_t index;
    kq_diagnostic_clear(diagnostic);
    if (!kq_qsa_state_valid(state) || key_cache == NULL ||
        value_cache == NULL || raw_cache == NULL || length > state->capacity ||
        !kq_qsa_u64_mul(length, state->config->key_values_per_token,
                        &expected_kv) ||
        !kq_qsa_u64_mul(length, state->config->raw_index_values_per_token,
                        &expected_raw)) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid bounded QSA import arguments are required");
    }
    if (key_count != expected_kv || value_count != expected_kv ||
        raw_count != expected_raw) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                           "QSA imported cache shape does not match length");
    }
    for (index = 0U; index < expected_kv; ++index) {
        if (!isfinite(key_cache[index]) || !isfinite(value_cache[index])) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "QSA imported K/V state must be finite");
        }
    }
    for (index = 0U; index < expected_raw; ++index) {
        if (!isfinite(raw_cache[index])) {
            return kq_qsa_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "QSA imported index state must be finite");
        }
    }
    for (index = 0U; index < expected_kv; ++index) {
        kq_qsa_state_write(state->key_cache,
                           state->config->dimensions.activation_dtype,
                           index, key_cache[index]);
        kq_qsa_state_write(state->value_cache,
                           state->config->dimensions.activation_dtype,
                           index, value_cache[index]);
    }
    for (index = 0U; index < expected_raw; ++index) {
        kq_qsa_state_write(state->raw_index_key_cache,
                           state->config->dimensions.activation_dtype,
                           index, raw_cache[index]);
    }
    state->length = length;
    return KQ_STATUS_OK;
}

kq_status kq_qsa_calculate_scratch_bytes(
    const kq_qsa_config *config, const kq_qsa_state *state,
    uint64_t sequence_length, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic) {
    uint64_t total = 0U;
    uint64_t q_width;
    uint64_t kv_width = config == NULL ? 0U : config->key_values_per_token;
    uint64_t iq_width;
    uint64_t blocks;
    uint64_t selected;
    uint64_t pending_state;
    uint64_t staged_output;
    uint64_t attention;
    uint64_t state_width;
    uint64_t doubled_query_width;
    uint64_t block_key_values;
    if (!kq_qsa_config_valid(config) || !kq_qsa_state_valid(state) ||
        state->config != config || sequence_length == 0U ||
        scratch_bytes == NULL ||
        !kq_qsa_u64_mul(config->dimensions.query_head_count,
                        config->dimensions.head_dimension, &q_width) ||
        !kq_qsa_u64_mul(config->dimensions.index_query_head_count,
                        config->dimensions.index_head_dimension, &iq_width) ||
        !kq_qsa_u64_add(state->length, sequence_length, &selected) ||
        selected > state->capacity) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid QSA state, sequence, and scratch output are required");
    }
    blocks = state->capacity / config->dimensions.block_size;
    selected = config->selected_token_capacity;
    if (!kq_qsa_u64_add(kv_width, kv_width, &state_width) ||
        !kq_qsa_u64_add(state_width, config->raw_index_values_per_token,
                        &state_width) ||
        !kq_qsa_u64_add(q_width, q_width, &doubled_query_width) ||
        !kq_qsa_u64_mul(blocks, config->raw_index_values_per_token,
                        &block_key_values) ||
        !kq_qsa_u64_mul(sequence_length, state_width, &pending_state) ||
        !kq_qsa_u64_mul(sequence_length, config->dimensions.hidden_size,
                        &staged_output) ||
        !kq_qsa_u64_mul(config->dimensions.query_head_count,
                        selected, &attention) ||
        !kq_qsa_accumulate(&total, pending_state) ||
        !kq_qsa_accumulate(&total, staged_output) ||
        !kq_qsa_accumulate(&total, doubled_query_width) ||
        !kq_qsa_accumulate(&total, q_width) ||
        !kq_qsa_accumulate(&total, kv_width) ||
        !kq_qsa_accumulate(&total, kv_width) ||
        !kq_qsa_accumulate(&total, iq_width) ||
        !kq_qsa_accumulate(&total, config->raw_index_values_per_token) ||
        !kq_qsa_accumulate(&total, block_key_values) ||
        !kq_qsa_accumulate(&total, blocks) ||
        !kq_qsa_accumulate(&total, blocks) ||
        !kq_qsa_accumulate(&total, config->block_topk) ||
        !kq_qsa_accumulate(&total, selected) ||
        !kq_qsa_accumulate(&total, attention) ||
        !kq_qsa_accumulate(&total, attention) ||
        !kq_qsa_accumulate(&total, q_width) ||
        !kq_qsa_accumulate(&total, q_width) ||
        !kq_qsa_accumulate(&total, config->dimensions.hidden_size) ||
        !kq_qsa_u64_mul(total, sizeof(float), scratch_bytes) ||
        *scratch_bytes > SIZE_MAX) {
        return kq_qsa_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "QSA scratch byte count overflows");
    }
    return KQ_STATUS_OK;
}

kq_status kq_qsa_required_scratch_bytes(
    const kq_qsa_config *config, const kq_qsa_state *state,
    uint64_t sequence_length, uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic) {
    kq_diagnostic_clear(diagnostic);
    return kq_qsa_calculate_scratch_bytes(config, state, sequence_length,
                                          scratch_bytes, diagnostic);
}

kq_status kq_qsa_prefill_f32(
    const kq_qsa_config *config, const kq_qsa_weights_f32 *weights,
    const float *hidden_states, uint64_t sequence_length,
    float *output, uint64_t output_capacity, kq_qsa_state *state,
    void *scratch, uint64_t scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    return kq_qsa_execute_f32(
        config, weights, hidden_states, sequence_length, output,
        output_capacity, state, scratch, scratch_bytes, selection_observer,
        checkpoint_observer, observer_user_data, diagnostic);
}

kq_status kq_qsa_decode_f32(
    const kq_qsa_config *config, const kq_qsa_weights_f32 *weights,
    const float *hidden_token, float *output_token, uint64_t output_capacity,
    kq_qsa_state *state, void *scratch, uint64_t scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    return kq_qsa_execute_f32(
        config, weights, hidden_token, 1U, output_token, output_capacity,
        state, scratch, scratch_bytes, selection_observer,
        checkpoint_observer, observer_user_data, diagnostic);
}

const char *kq_qsa_checkpoint_kind_name(kq_qsa_checkpoint_kind kind) {
    static const char *names[] = {
        "INDEX_QUERY", "RAW_INDEX_KEY", "BLOCK_KEYS", "CANDIDATE_SCORES",
        "QUERY", "KEY", "VALUE", "ATTENTION_LOGITS",
        "ATTENTION_PROBABILITIES", "HEAD_CONTEXT", "GATED_CONTEXT",
        "OPERATOR_OUTPUT"
    };
    if ((uint32_t)kind >= sizeof(names) / sizeof(names[0])) return "UNKNOWN";
    return names[(uint32_t)kind];
}
