#include "kq_gdn_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_tensor_view.h"

typedef struct kq_gdn_role_spec {
    kq_semantic_role role;
    const char *suffix;
    kq_binding_relation relation;
    uint32_t rank;
    uint64_t dimensions[3];
    uint32_t physical_type;
} kq_gdn_role_spec;

static const kq_gdn_role_spec kq_gdn_target_specs[KQ_GDN_WEIGHT_ROLE_COUNT] = {
    {KQ_ROLE_GDN_A_LOG, "a_log", KQ_BINDING_TRANSFORMED_LAYOUT,
     1U, {48U, 0U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_CONV, "conv", KQ_BINDING_TRANSFORMED_LAYOUT,
     3U, {10240U, 1U, 4U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_DT_BIAS, "dt_bias", KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {48U, 0U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_ALPHA, "alpha", KQ_BINDING_TRANSFORMED_LAYOUT,
     2U, {48U, 2560U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_BETA, "beta", KQ_BINDING_TRANSFORMED_LAYOUT,
     2U, {48U, 2560U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_QKV, "qkv", KQ_BINDING_TRANSFORMED_LAYOUT,
     2U, {10240U, 2560U, 0U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_GDN_GATE, "gate", KQ_BINDING_TRANSFORMED_LAYOUT,
     2U, {6144U, 2560U, 0U}, KQ_GGUF_TYPE_Q8_0},
    {KQ_ROLE_GDN_NORM, "norm", KQ_BINDING_RENAMED_ONE_TO_ONE,
     1U, {128U, 0U, 0U}, KQ_GGUF_TYPE_F32},
    {KQ_ROLE_GDN_OUT, "out", KQ_BINDING_TRANSFORMED_LAYOUT,
     2U, {2560U, 6144U, 0U}, KQ_GGUF_TYPE_Q8_0}
};

static kq_status kq_gdn_fail(kq_diagnostic *diagnostic,
                             kq_status status,
                             const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_gdn_u64_add(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || UINT64_MAX - left < right) {
        return 0;
    }
    *output = left + right;
    return 1;
}

static int kq_gdn_u64_mul(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || (left != 0U && right > UINT64_MAX / left)) {
        return 0;
    }
    *output = left * right;
    return 1;
}

static int kq_gdn_accumulate(uint64_t *total, uint64_t value) {
    uint64_t next;
    if (!kq_gdn_u64_add(*total, value, &next)) {
        return 0;
    }
    *total = next;
    return 1;
}

static int kq_gdn_memory_overlap(const void *left,
                                 uint64_t left_bytes,
                                 const void *right,
                                 uint64_t right_bytes) {
    uintptr_t left_start;
    uintptr_t right_start;
    uintptr_t left_end;
    uintptr_t right_end;
    if (left == NULL || right == NULL || left_bytes == 0U ||
        right_bytes == 0U || left_bytes > UINTPTR_MAX ||
        right_bytes > UINTPTR_MAX) {
        return 0;
    }
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - (uintptr_t)left_bytes ||
        right_start > UINTPTR_MAX - (uintptr_t)right_bytes) {
        return 1;
    }
    left_end = left_start + (uintptr_t)left_bytes;
    right_end = right_start + (uintptr_t)right_bytes;
    return left_start < right_end && right_start < left_end;
}

static int kq_gdn_config_valid(const kq_gdn_config *config) {
    return config != NULL && config->magic == KQ_GDN_CONFIG_MAGIC;
}

static int kq_gdn_state_valid(const kq_gdn_state *state) {
    return state != NULL && state->magic == KQ_GDN_STATE_MAGIC &&
           kq_gdn_config_valid(state->config) &&
           state->conv_state != NULL && state->recurrent_state != NULL;
}

static uint16_t kq_gdn_f32_to_bf16(float value) {
    uint32_t bits;
    uint32_t rounded;
    memcpy(&bits, &value, sizeof(bits));
    rounded = bits + UINT32_C(0x00007fff) + ((bits >> 16U) & 1U);
    return (uint16_t)(rounded >> 16U);
}

static float kq_gdn_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16U;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static kq_status kq_gdn_validate_dimensions(
    const kq_gdn_dimensions *dimensions,
    uint32_t *key_dimension,
    uint32_t *value_dimension,
    uint32_t *conv_channels,
    uint32_t *head_repeat,
    uint64_t *recurrent_elements,
    uint64_t *conv_state_elements,
    uint64_t *token_scratch_bytes,
    uint64_t *scratch_bytes,
    kq_diagnostic *diagnostic) {
    uint64_t key64;
    uint64_t value64;
    uint64_t conv64;
    uint64_t recurrent64;
    uint64_t conv_state64;
    uint64_t token_floats = 0U;
    uint64_t total_floats;
    uint64_t bytes;

    if (dimensions == NULL || key_dimension == NULL || value_dimension == NULL ||
        conv_channels == NULL || head_repeat == NULL ||
        recurrent_elements == NULL || conv_state_elements == NULL ||
        token_scratch_bytes == NULL || scratch_bytes == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "complete GDN dimensions and outputs are required");
    }
    if (dimensions->hidden_size == 0U ||
        dimensions->key_head_count == 0U ||
        dimensions->value_head_count == 0U ||
        dimensions->key_head_dimension == 0U ||
        dimensions->value_head_dimension == 0U ||
        dimensions->conv_kernel_size != 4U ||
        dimensions->value_head_count % dimensions->key_head_count != 0U ||
        !(dimensions->rms_norm_epsilon > 0.0f) ||
        !isfinite(dimensions->rms_norm_epsilon) ||
        (dimensions->activation_dtype != KQ_GDN_ACTIVATION_F32 &&
         dimensions->activation_dtype != KQ_GDN_ACTIVATION_BF16)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                           "invalid GDN dimensions, head ratio, kernel, epsilon, or dtype");
    }
    if (!kq_gdn_u64_mul(dimensions->key_head_count,
                        dimensions->key_head_dimension, &key64) ||
        !kq_gdn_u64_mul(dimensions->value_head_count,
                        dimensions->value_head_dimension, &value64) ||
        !kq_gdn_u64_add(key64, key64, &conv64) ||
        !kq_gdn_u64_add(conv64, value64, &conv64) ||
        key64 > UINT32_MAX || value64 > UINT32_MAX || conv64 > UINT32_MAX) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN projection dimensions overflow");
    }
    if (!kq_gdn_u64_mul(dimensions->value_head_count,
                        dimensions->key_head_dimension, &recurrent64) ||
        !kq_gdn_u64_mul(recurrent64,
                        dimensions->value_head_dimension, &recurrent64) ||
        !kq_gdn_u64_mul(conv64, dimensions->conv_kernel_size, &conv_state64)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN state dimensions overflow");
    }

    /* Per-token F32 workspace; the mutable state copy is accounted separately. */
    if (!kq_gdn_accumulate(&token_floats, dimensions->hidden_size) ||
        !kq_gdn_accumulate(&token_floats, conv64) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, dimensions->value_head_count) ||
        !kq_gdn_accumulate(&token_floats, dimensions->value_head_count) ||
        !kq_gdn_accumulate(&token_floats, conv64) ||
        !kq_gdn_accumulate(&token_floats,
                           (uint64_t)dimensions->value_head_count *
                           dimensions->key_head_dimension) ||
        !kq_gdn_accumulate(&token_floats,
                           (uint64_t)dimensions->value_head_count *
                           dimensions->key_head_dimension) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, dimensions->value_head_count) ||
        !kq_gdn_accumulate(&token_floats, dimensions->value_head_count) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, value64) ||
        !kq_gdn_accumulate(&token_floats, dimensions->hidden_size)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN token workspace count overflows");
    }
    if (!kq_gdn_u64_mul(token_floats, sizeof(float), &bytes)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN token workspace byte count overflows");
    }
    *token_scratch_bytes = bytes;
    if (!kq_gdn_u64_add(token_floats, recurrent64, &total_floats) ||
        !kq_gdn_u64_add(total_floats, conv_state64, &total_floats) ||
        !kq_gdn_u64_mul(total_floats, sizeof(float), &bytes) ||
        bytes > (uint64_t)SIZE_MAX) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN total workspace byte count overflows size_t");
    }

    *key_dimension = (uint32_t)key64;
    *value_dimension = (uint32_t)value64;
    *conv_channels = (uint32_t)conv64;
    *head_repeat = dimensions->value_head_count / dimensions->key_head_count;
    *recurrent_elements = recurrent64;
    *conv_state_elements = conv_state64;
    *scratch_bytes = bytes;
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_validate_target_semantic(
    const kq_semantic_tensor *semantic,
    const kq_gdn_role_spec *spec,
    uint32_t layer_id,
    kq_diagnostic *diagnostic) {
    uint32_t dimension;
    kq_quant_geometry geometry;

    if (semantic == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                           "required GDN semantic tensor is missing");
    }
    if (semantic->component != KQ_COMPONENT_GDN ||
        semantic->role != spec->role ||
        semantic->layer_type != KQ_MODEL_LAYER_GDN ||
        semantic->layer_id != layer_id ||
        semantic->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
        semantic->canonical_dtype != KQ_CANONICAL_DTYPE_BF16 ||
        semantic->relation != spec->relation ||
        semantic->canonical_rank != spec->rank) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                           "GDN semantic identity, dtype, relation, or rank mismatch");
    }
    for (dimension = 0U; dimension < spec->rank; ++dimension) {
        if (semantic->canonical_dimensions[dimension] !=
            spec->dimensions[dimension]) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                               "GDN canonical tensor shape mismatch");
        }
    }
    if (semantic->binding_count != 1U ||
        semantic->bindings[0].physical == NULL ||
        semantic->bindings[0].part_role != KQ_BINDING_PART_WHOLE ||
        semantic->bindings[0].part_count != 1U ||
        semantic->bindings[0].part_index != 0U ||
        semantic->bindings[0].physical->type_id != spec->physical_type) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                           "GDN physical binding or quantization type mismatch");
    }
    if (kq_quant_geometry_for_type(semantic->bindings[0].physical->type_id,
                                   &geometry, diagnostic) != KQ_STATUS_OK) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                           "GDN physical tensor type has unsupported geometry");
    }
    return KQ_STATUS_OK;
}

kq_status kq_gdn_config_create_from_source(
    const kq_gdn_semantic_source *source,
    int require_target_bindings,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_gdn_config *config;
    kq_status status;
    uint32_t index;

    kq_diagnostic_clear(diagnostic);
    if (source == NULL || out_config == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "GDN semantic source and output are required");
    }
    *out_config = NULL;
    if (source->layer_type != KQ_MODEL_LAYER_GDN) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                           "GDN configuration requested for a non-GDN layer");
    }
    if (require_target_bindings) {
        if (source->model == NULL || source->layer_id >= 48U ||
            source->dimensions.hidden_size != 2560U ||
            source->dimensions.key_head_count != 16U ||
            source->dimensions.value_head_count != 48U ||
            source->dimensions.key_head_dimension != 128U ||
            source->dimensions.value_head_dimension != 128U ||
            (source->dimensions.activation_dtype != KQ_GDN_ACTIVATION_BF16 &&
             source->dimensions.activation_dtype != KQ_GDN_ACTIVATION_F32)) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                               "target Qwen3.8 GDN topology mismatch");
        }
        for (index = 0U; index < KQ_GDN_WEIGHT_ROLE_COUNT; ++index) {
            status = kq_gdn_validate_target_semantic(
                source->tensors[index], &kq_gdn_target_specs[index],
                source->layer_id, diagnostic);
            if (status != KQ_STATUS_OK) {
                return status;
            }
        }
    }

    config = (kq_gdn_config *)calloc(1U, sizeof(*config));
    if (config == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate immutable GDN config");
    }
    status = kq_gdn_validate_dimensions(
        &source->dimensions,
        &config->key_dimension,
        &config->value_dimension,
        &config->conv_channels,
        &config->key_value_head_repeat,
        &config->recurrent_elements,
        &config->conv_state_elements,
        &config->token_scratch_bytes,
        &config->scratch_bytes,
        diagnostic);
    if (status != KQ_STATUS_OK) {
        free(config);
        return status;
    }
    config->magic = KQ_GDN_CONFIG_MAGIC;
    config->model = source->model;
    config->layer_id = source->layer_id;
    config->dimensions = source->dimensions;
    for (index = 0U; index < KQ_GDN_WEIGHT_ROLE_COUNT; ++index) {
        config->tensors[index] = source->tensors[index];
    }
    *out_config = config;
    return KQ_STATUS_OK;
}

kq_status kq_gdn_test_config_create(
    const kq_gdn_dimensions *dimensions,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_gdn_semantic_source source;
    if (dimensions == NULL) {
        kq_diagnostic_clear(diagnostic);
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "test GDN dimensions are required");
    }
    memset(&source, 0, sizeof(source));
    source.layer_type = KQ_MODEL_LAYER_GDN;
    source.dimensions = *dimensions;
    return kq_gdn_config_create_from_source(&source, 0, out_config,
                                            diagnostic);
}

static kq_status kq_gdn_config_create_for_model(
    const kq_model *model,
    uint32_t layer_id,
    kq_gdn_activation_dtype activation_dtype,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic) {
    kq_gdn_semantic_source source;
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    uint32_t index;
    int length;

    kq_diagnostic_clear(diagnostic);
    if (model == NULL || out_config == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "model and GDN config output are required");
    }
    *out_config = NULL;
    if (layer_id >= kq_model_layer_count(model) ||
        kq_model_layer_type_at(model, layer_id) != KQ_MODEL_LAYER_GDN) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
                           "requested layer is not a canonical GDN layer");
    }
    memset(&source, 0, sizeof(source));
    source.model = model;
    source.layer_id = layer_id;
    source.layer_type = KQ_MODEL_LAYER_GDN;
    source.dimensions.hidden_size = kq_model_hidden_size(model);
    source.dimensions.key_head_count = 16U;
    source.dimensions.value_head_count = 48U;
    source.dimensions.key_head_dimension = 128U;
    source.dimensions.value_head_dimension = 128U;
    source.dimensions.conv_kernel_size = 4U;
    source.dimensions.rms_norm_epsilon = 1.0e-6f;
    source.dimensions.activation_dtype = activation_dtype;
    for (index = 0U; index < KQ_GDN_WEIGHT_ROLE_COUNT; ++index) {
        length = snprintf(semantic_id, sizeof(semantic_id),
                          "layer.%02u.gdn.%s", layer_id,
                          kq_gdn_target_specs[index].suffix);
        if (length < 0 || (size_t)length >= sizeof(semantic_id)) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                               "GDN semantic ID formatting overflow");
        }
        source.tensors[index] =
            kq_model_find_semantic_tensor(model, semantic_id);
    }
    return kq_gdn_config_create_from_source(&source, 1, out_config,
                                            diagnostic);
}

kq_status kq_gdn_config_create(const kq_model *model,
                               uint32_t layer_id,
                               kq_gdn_config **out_config,
                               kq_diagnostic *diagnostic) {
    return kq_gdn_config_create_for_model(
        model, layer_id, KQ_GDN_ACTIVATION_BF16,
        out_config, diagnostic);
}

kq_status kq_gdn_config_create_reference_f32(
    const kq_model *model,
    uint32_t layer_id,
    kq_gdn_config **out_config,
    kq_diagnostic *diagnostic) {
    return kq_gdn_config_create_for_model(
        model, layer_id, KQ_GDN_ACTIVATION_F32,
        out_config, diagnostic);
}

void kq_gdn_config_close(kq_gdn_config *config) {
    if (config == NULL) {
        return;
    }
    config->magic = 0U;
    free(config);
}

uint32_t kq_gdn_config_layer_id(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->layer_id : UINT32_MAX;
}

uint32_t kq_gdn_config_hidden_size(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->dimensions.hidden_size : 0U;
}

uint32_t kq_gdn_config_key_head_count(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ?
        config->dimensions.key_head_count : 0U;
}

uint32_t kq_gdn_config_value_head_count(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ?
        config->dimensions.value_head_count : 0U;
}

uint32_t kq_gdn_config_key_head_dimension(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ?
        config->dimensions.key_head_dimension : 0U;
}

uint32_t kq_gdn_config_value_head_dimension(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ?
        config->dimensions.value_head_dimension : 0U;
}

uint32_t kq_gdn_config_conv_channel_count(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->conv_channels : 0U;
}

uint32_t kq_gdn_config_conv_kernel_size(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ?
        config->dimensions.conv_kernel_size : 0U;
}

kq_gdn_activation_dtype kq_gdn_config_activation_dtype(
    const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->dimensions.activation_dtype :
        KQ_GDN_ACTIVATION_F32;
}

uint64_t kq_gdn_config_owned_bytes(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? sizeof(*config) : 0U;
}

uint64_t kq_gdn_config_recurrent_element_count(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->recurrent_elements : 0U;
}

uint64_t kq_gdn_config_conv_state_element_count(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->conv_state_elements : 0U;
}

uint64_t kq_gdn_config_scratch_bytes(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->scratch_bytes : 0U;
}

uint64_t kq_gdn_config_token_scratch_bytes(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? config->token_scratch_bytes : 0U;
}

uint64_t kq_gdn_config_dequant_scratch_bytes(const kq_gdn_config *config) {
    return kq_gdn_config_valid(config) ? 256U * sizeof(float) : 0U;
}

kq_status kq_gdn_state_create(const kq_gdn_config *config,
                              kq_gdn_state **out_state,
                              kq_diagnostic *diagnostic) {
    kq_gdn_state *state;
    uint64_t conv_bytes;
    uint64_t recurrent_bytes;
    uint64_t total;
    uint64_t conv_element_bytes;

    kq_diagnostic_clear(diagnostic);
    if (!kq_gdn_config_valid(config) || out_state == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid GDN config and state output are required");
    }
    *out_state = NULL;
    conv_element_bytes = config->dimensions.activation_dtype ==
        KQ_GDN_ACTIVATION_BF16 ? sizeof(uint16_t) : sizeof(float);
    if (!kq_gdn_u64_mul(config->conv_state_elements,
                        conv_element_bytes, &conv_bytes) ||
        !kq_gdn_u64_mul(config->recurrent_elements,
                        sizeof(float), &recurrent_bytes) ||
        conv_bytes > (uint64_t)SIZE_MAX ||
        recurrent_bytes > (uint64_t)SIZE_MAX ||
        !kq_gdn_u64_add(sizeof(*state), conv_bytes, &total) ||
        !kq_gdn_u64_add(total, recurrent_bytes, &total)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN state allocation size overflows");
    }
    state = (kq_gdn_state *)calloc(1U, sizeof(*state));
    if (state == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate GDN stream state");
    }
    state->conv_state = calloc(1U, (size_t)conv_bytes);
    state->recurrent_state = (float *)calloc(
        (size_t)config->recurrent_elements, sizeof(float));
    if (state->conv_state == NULL || state->recurrent_state == NULL) {
        free(state->conv_state);
        free(state->recurrent_state);
        free(state);
        return kq_gdn_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                           "could not allocate GDN state arrays");
    }
    state->magic = KQ_GDN_STATE_MAGIC;
    state->config = config;
    state->owned_bytes = total;
    *out_state = state;
    return KQ_STATUS_OK;
}

void kq_gdn_state_close(kq_gdn_state *state) {
    if (state == NULL) {
        return;
    }
    state->magic = 0U;
    free(state->conv_state);
    free(state->recurrent_state);
    free(state);
}

kq_status kq_gdn_state_reset(kq_gdn_state *state,
                             kq_diagnostic *diagnostic) {
    uint64_t conv_bytes;
    uint64_t recurrent_bytes;
    uint64_t conv_element_bytes;
    kq_diagnostic_clear(diagnostic);
    if (!kq_gdn_state_valid(state)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_GDN_STATE,
                           "valid GDN state is required for reset");
    }
    conv_element_bytes = state->config->dimensions.activation_dtype ==
        KQ_GDN_ACTIVATION_BF16 ? sizeof(uint16_t) : sizeof(float);
    if (!kq_gdn_u64_mul(state->config->conv_state_elements,
                        conv_element_bytes, &conv_bytes) ||
        !kq_gdn_u64_mul(state->config->recurrent_elements,
                        sizeof(float), &recurrent_bytes)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN reset byte count overflows");
    }
    memset(state->conv_state, 0, (size_t)conv_bytes);
    memset(state->recurrent_state, 0, (size_t)recurrent_bytes);
    state->initialized = 0;
    return KQ_STATUS_OK;
}

uint64_t kq_gdn_state_owned_bytes(const kq_gdn_state *state) {
    return kq_gdn_state_valid(state) ? state->owned_bytes : 0U;
}

kq_status kq_gdn_state_export_f32(const kq_gdn_state *state,
                                  float *conv_state,
                                  uint64_t conv_capacity,
                                  float *recurrent_state,
                                  uint64_t recurrent_capacity,
                                  int *initialized,
                                  kq_diagnostic *diagnostic) {
    uint64_t index;
    uint64_t conv_bytes;
    uint64_t recurrent_bytes;
    kq_diagnostic_clear(diagnostic);
    if (!kq_gdn_state_valid(state) || conv_state == NULL ||
        recurrent_state == NULL || initialized == NULL) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid GDN state and export buffers are required");
    }
    if (conv_capacity < state->config->conv_state_elements ||
        recurrent_capacity < state->config->recurrent_elements) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                           "GDN state export buffer is too small");
    }
    if (!kq_gdn_u64_mul(state->config->conv_state_elements,
                        sizeof(float), &conv_bytes) ||
        !kq_gdn_u64_mul(state->config->recurrent_elements,
                        sizeof(float), &recurrent_bytes)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                           "GDN state export byte count overflows");
    }
    if (kq_gdn_memory_overlap(conv_state, conv_bytes,
                              recurrent_state, recurrent_bytes)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                           "GDN state export buffers overlap");
    }
    if (state->config->dimensions.activation_dtype ==
        KQ_GDN_ACTIVATION_BF16) {
        const uint16_t *source = (const uint16_t *)state->conv_state;
        for (index = 0U; index < state->config->conv_state_elements; ++index) {
            conv_state[index] = kq_gdn_bf16_to_f32(source[index]);
        }
    } else {
        memcpy(conv_state, state->conv_state,
               (size_t)state->config->conv_state_elements * sizeof(float));
    }
    memcpy(recurrent_state, state->recurrent_state,
           (size_t)state->config->recurrent_elements * sizeof(float));
    *initialized = state->initialized;
    return KQ_STATUS_OK;
}

kq_status kq_gdn_state_import_f32(kq_gdn_state *state,
                                  const float *conv_state,
                                  uint64_t conv_count,
                                  const float *recurrent_state,
                                  uint64_t recurrent_count,
                                  int initialized,
                                  kq_diagnostic *diagnostic) {
    uint64_t index;
    kq_diagnostic_clear(diagnostic);
    if (!kq_gdn_state_valid(state) || conv_state == NULL ||
        recurrent_state == NULL || (initialized != 0 && initialized != 1)) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                           "valid GDN state import arguments are required");
    }
    if (conv_count != state->config->conv_state_elements ||
        recurrent_count != state->config->recurrent_elements) {
        return kq_gdn_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                           "GDN state import shape mismatch");
    }
    for (index = 0U; index < conv_count; ++index) {
        if (!isfinite(conv_state[index])) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "GDN convolution state must be finite");
        }
        if (!initialized && conv_state[index] != 0.0f) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_GDN_STATE,
                               "uninitialized GDN convolution state must be zero");
        }
    }
    for (index = 0U; index < recurrent_count; ++index) {
        if (!isfinite(recurrent_state[index])) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                               "GDN recurrent state must be finite");
        }
        if (!initialized && recurrent_state[index] != 0.0f) {
            return kq_gdn_fail(diagnostic, KQ_STATUS_INVALID_GDN_STATE,
                               "uninitialized GDN recurrent state must be zero");
        }
    }
    if (state->config->dimensions.activation_dtype ==
        KQ_GDN_ACTIVATION_BF16) {
        uint16_t *target = (uint16_t *)state->conv_state;
        for (index = 0U; index < conv_count; ++index) {
            target[index] = kq_gdn_f32_to_bf16(conv_state[index]);
        }
    } else {
        memcpy(state->conv_state, conv_state,
               (size_t)conv_count * sizeof(float));
    }
    memcpy(state->recurrent_state, recurrent_state,
           (size_t)recurrent_count * sizeof(float));
    state->initialized = initialized;
    return KQ_STATUS_OK;
}

kq_status kq_gdn_prefill_f32(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    const float *hidden_states,
    uint64_t sequence_length,
    const uint8_t *padding_mask,
    float *output,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic) {
    return kq_gdn_execute_f32(config, weights, hidden_states,
                              sequence_length, padding_mask, output,
                              output_capacity, state, scratch, scratch_bytes,
                              observer, observer_user_data, diagnostic);
}

kq_status kq_gdn_decode_f32(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    const float *hidden_token,
    float *output_token,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic) {
    return kq_gdn_execute_f32(config, weights, hidden_token, 1U, NULL,
                              output_token, output_capacity, state, scratch,
                              scratch_bytes, observer, observer_user_data,
                              diagnostic);
}

const char *kq_gdn_checkpoint_kind_name(kq_gdn_checkpoint_kind kind) {
    static const char *names[] = {
        "MASKED_INPUT", "PROJECTED_QKV", "PROJECTED_GATE",
        "PROJECTED_BETA", "PROJECTED_ALPHA", "CONV_INPUT", "CONV_OUTPUT",
        "QUERY_BEFORE_NORM", "KEY_BEFORE_NORM", "VALUE",
        "NORMALIZED_SCALED_QUERY", "NORMALIZED_KEY", "LOG_DECAY", "BETA",
        "RECURRENT_READ", "RECURRENT_DELTA", "RECURRENT_OUTPUT",
        "RECURRENT_STATE", "CONV_STATE", "GATED_NORM_OUTPUT",
        "OPERATOR_OUTPUT"
    };
    if ((uint32_t)kind >= sizeof(names) / sizeof(names[0])) {
        return "UNKNOWN";
    }
    return names[(uint32_t)kind];
}
