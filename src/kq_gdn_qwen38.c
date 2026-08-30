#include "kq_gdn_internal.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_numeric.h"
#include "kq_weight_provider.h"

typedef struct kq_gdn_workspace {
    float *conv_state;
    float *recurrent_state;
    float *masked_input;
    float *projected_qkv;
    float *projected_gate;
    float *projected_beta;
    float *projected_alpha;
    float *conv_output;
    float *query;
    float *key;
    float *value;
    float *beta;
    float *log_decay;
    float *read;
    float *delta;
    float *core_output;
    float *gated_output;
    float *operator_output;
} kq_gdn_workspace;

static kq_status kq_gdn_execute_fail(kq_diagnostic *diagnostic,
                                     kq_status status,
                                     const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_gdn_exec_u64_mul(uint64_t left, uint64_t right,
                               uint64_t *output) {
    if (output == NULL || (left != 0U && right > UINT64_MAX / left)) {
        return 0;
    }
    *output = left * right;
    return 1;
}

static int kq_gdn_exec_u64_add(uint64_t left, uint64_t right,
                               uint64_t *output) {
    if (output == NULL || UINT64_MAX - left < right) {
        return 0;
    }
    *output = left + right;
    return 1;
}

static int kq_gdn_ranges_overlap(const void *left,
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

static int kq_gdn_weights_overlap(const kq_gdn_weights_f32 *weights,
                                  const void *range,
                                  uint64_t range_bytes) {
    const float *pointers[9] = {
        weights->a_log, weights->conv, weights->dt_bias,
        weights->alpha, weights->beta, weights->qkv, weights->gate,
        weights->norm, weights->output
    };
    const uint64_t counts[9] = {
        weights->a_log_count, weights->conv_count, weights->dt_bias_count,
        weights->alpha_count, weights->beta_count, weights->qkv_count,
        weights->gate_count, weights->norm_count, weights->output_count
    };
    uint32_t index;
    for (index = 0U; index < 9U; ++index) {
        uint64_t bytes;
        if (!kq_gdn_exec_u64_mul(counts[index], sizeof(float), &bytes) ||
            kq_gdn_ranges_overlap(pointers[index], bytes,
                                  range, range_bytes)) {
            return 1;
        }
    }
    return 0;
}

static kq_status kq_gdn_validate_finite(const float *values,
                                        uint64_t count,
                                        const char *message,
                                        kq_diagnostic *diagnostic) {
    uint64_t index;
    if (values == NULL || count == 0U) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                   message);
    }
    for (index = 0U; index < count; ++index) {
        if (!isfinite(values[index])) {
            return kq_gdn_execute_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                       message);
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_validate_weight(const float *values,
                                        uint64_t actual_count,
                                        uint64_t expected_count,
                                        kq_diagnostic *diagnostic) {
    if (actual_count != expected_count) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                                   "GDN F32 weight count mismatch");
    }
    return kq_gdn_validate_finite(values, actual_count,
                                  "GDN F32 weights must be finite",
                                  diagnostic);
}

static kq_status kq_gdn_validate_weights(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    kq_diagnostic *diagnostic) {
    uint64_t count;
    kq_status status;

    if (weights == NULL) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                   "GDN F32 weights are required");
    }
    status = kq_gdn_validate_weight(weights->a_log, weights->a_log_count,
                                    config->dimensions.value_head_count,
                                    diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_gdn_validate_weight(weights->dt_bias, weights->dt_bias_count,
                                    config->dimensions.value_head_count,
                                    diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_gdn_validate_weight(weights->norm, weights->norm_count,
                                    config->dimensions.value_head_dimension,
                                    diagnostic);
    if (status != KQ_STATUS_OK) return status;

    if (!kq_gdn_exec_u64_mul(config->conv_channels,
                             config->dimensions.conv_kernel_size, &count)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN convolution weight count overflows");
    }
    status = kq_gdn_validate_weight(weights->conv, weights->conv_count,
                                    count, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (!kq_gdn_exec_u64_mul(config->dimensions.value_head_count,
                             config->dimensions.hidden_size, &count)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN gate projection count overflows");
    }
    status = kq_gdn_validate_weight(weights->alpha, weights->alpha_count,
                                    count, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    status = kq_gdn_validate_weight(weights->beta, weights->beta_count,
                                    count, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (!kq_gdn_exec_u64_mul(config->conv_channels,
                             config->dimensions.hidden_size, &count)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN QKV projection count overflows");
    }
    status = kq_gdn_validate_weight(weights->qkv, weights->qkv_count,
                                    count, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (!kq_gdn_exec_u64_mul(config->value_dimension,
                             config->dimensions.hidden_size, &count)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN output-gate projection count overflows");
    }
    status = kq_gdn_validate_weight(weights->gate, weights->gate_count,
                                    count, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (!kq_gdn_exec_u64_mul(config->dimensions.hidden_size,
                             config->value_dimension, &count)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN output projection count overflows");
    }
    return kq_gdn_validate_weight(weights->output, weights->output_count,
                                  count, diagnostic);
}

static float *kq_gdn_take(float **cursor, uint64_t count) {
    float *value = *cursor;
    *cursor += count;
    return value;
}

static void kq_gdn_workspace_init(const kq_gdn_config *config,
                                  void *scratch,
                                  kq_gdn_workspace *workspace) {
    float *cursor = (float *)scratch;
    uint64_t repeated_key =
        (uint64_t)config->dimensions.value_head_count *
        config->dimensions.key_head_dimension;

    workspace->conv_state = kq_gdn_take(&cursor, config->conv_state_elements);
    workspace->recurrent_state = kq_gdn_take(&cursor,
                                             config->recurrent_elements);
    workspace->masked_input = kq_gdn_take(&cursor,
                                          config->dimensions.hidden_size);
    workspace->projected_qkv = kq_gdn_take(&cursor, config->conv_channels);
    workspace->projected_gate = kq_gdn_take(&cursor, config->value_dimension);
    workspace->projected_beta = kq_gdn_take(
        &cursor, config->dimensions.value_head_count);
    workspace->projected_alpha = kq_gdn_take(
        &cursor, config->dimensions.value_head_count);
    workspace->conv_output = kq_gdn_take(&cursor, config->conv_channels);
    workspace->query = kq_gdn_take(&cursor, repeated_key);
    workspace->key = kq_gdn_take(&cursor, repeated_key);
    workspace->value = kq_gdn_take(&cursor, config->value_dimension);
    workspace->beta = kq_gdn_take(&cursor,
                                  config->dimensions.value_head_count);
    workspace->log_decay = kq_gdn_take(
        &cursor, config->dimensions.value_head_count);
    workspace->read = kq_gdn_take(&cursor, config->value_dimension);
    workspace->delta = kq_gdn_take(&cursor, config->value_dimension);
    workspace->core_output = kq_gdn_take(&cursor, config->value_dimension);
    workspace->gated_output = kq_gdn_take(&cursor, config->value_dimension);
    workspace->operator_output = kq_gdn_take(
        &cursor, config->dimensions.hidden_size);
}

static void kq_gdn_emit(kq_gdn_checkpoint_observer observer,
                        void *user_data,
                        kq_gdn_checkpoint_kind kind,
                        uint64_t token_index,
                        uint32_t rank,
                        uint64_t first,
                        uint64_t second,
                        uint64_t third,
                        const float *values,
                        uint64_t count) {
    kq_gdn_checkpoint checkpoint;
    if (observer == NULL) {
        return;
    }
    memset(&checkpoint, 0, sizeof(checkpoint));
    checkpoint.kind = kind;
    checkpoint.token_index = token_index;
    checkpoint.rank = rank;
    checkpoint.dimensions[0] = first;
    checkpoint.dimensions[1] = second;
    checkpoint.dimensions[2] = third;
    checkpoint.values = values;
    checkpoint.value_count = count;
    observer(&checkpoint, user_data);
}

static kq_status kq_gdn_project(const kq_gdn_config *config,
                                const kq_gdn_weights_f32 *weights,
                                kq_weight_provider *provider,
                                uint32_t role_index,
                                uint64_t rows,
                                uint64_t columns,
                                const float *input,
                                float *output,
                                void *weight_scratch,
                                uint64_t weight_scratch_bytes,
                                kq_diagnostic *diagnostic) {
    static const size_t offsets[KQ_GDN_WEIGHT_ROLE_COUNT] = {
        offsetof(kq_gdn_weights_f32, a_log),
        offsetof(kq_gdn_weights_f32, conv),
        offsetof(kq_gdn_weights_f32, dt_bias),
        offsetof(kq_gdn_weights_f32, alpha),
        offsetof(kq_gdn_weights_f32, beta),
        offsetof(kq_gdn_weights_f32, qkv),
        offsetof(kq_gdn_weights_f32, gate),
        offsetof(kq_gdn_weights_f32, norm),
        offsetof(kq_gdn_weights_f32, output)
    };
    uint64_t row;
    kq_status status;
    const float *matrix;
    if (provider != NULL) {
        if (config->tensors[role_index] == NULL)
            return kq_gdn_execute_fail(diagnostic,
                KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                "GDN provider matrix semantic is missing");
        return kq_weight_provider_linear_f32(
            provider, config->tensors[role_index], KQ_BINDING_PART_WHOLE,
            KQ_WEIGHT_PROVIDER_NO_EXPERT, rows, columns, input, output, rows,
            weight_scratch, weight_scratch_bytes, diagnostic);
    }
    matrix = *(const float *const *)((const unsigned char *)weights +
                                     offsets[role_index]);
    for (row = 0U; row < rows; ++row) {
        status = kq_f32_dot(matrix + row * columns, input, columns,
                            &output[row], diagnostic);
        if (status != KQ_STATUS_OK) {
            return status;
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_softplus(float input,
                                 float *output,
                                 kq_diagnostic *diagnostic) {
    float value;
    if (!isfinite(input) || output == NULL) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "GDN softplus input must be finite");
    }
    value = input > 20.0f ? input : log1pf(expf(input));
    if (!isfinite(value)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "GDN softplus produced a non-finite value");
    }
    *output = value;
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_l2_normalize(float *values,
                                     uint64_t count,
                                     float extra_scale,
                                     kq_diagnostic *diagnostic) {
    float sum;
    float inverse;
    uint64_t index;
    kq_status status = kq_f32_dot(values, values, count, &sum, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    sum = sum + 1.0e-6f;
    inverse = 1.0f / sqrtf(sum);
    if (!isfinite(inverse) || !isfinite(extra_scale)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "GDN L2 normalization is non-finite");
    }
    for (index = 0U; index < count; ++index) {
        float normalized = values[index] * inverse;
        values[index] = normalized * extra_scale;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_validate_call(
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
    int provider_mode,
    uint64_t *hidden_count,
    kq_diagnostic *diagnostic) {
    uint64_t bytes;
    uint64_t output_bytes;
    uint64_t conv_state_bytes;
    uint64_t recurrent_state_bytes;
    uint64_t index;
    kq_status status;

    if (config == NULL || config->magic != KQ_GDN_CONFIG_MAGIC ||
        (!provider_mode && weights == NULL) || hidden_states == NULL || output == NULL ||
        state == NULL || state->magic != KQ_GDN_STATE_MAGIC ||
        state->config != config || state->conv_state == NULL ||
        state->recurrent_state == NULL || scratch == NULL ||
        hidden_count == NULL ||
        (state->initialized != 0 && state->initialized != 1)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                   "valid compatible GDN call arguments are required");
    }
    if (config->dimensions.activation_dtype != KQ_GDN_ACTIVATION_F32) {
        return kq_gdn_execute_fail(
            diagnostic, KQ_STATUS_INCOMPATIBLE_GDN,
            "F32 scalar execution requires an F32 reference configuration");
    }
    if (sequence_length == 0U ||
        !kq_gdn_exec_u64_mul(sequence_length,
                             config->dimensions.hidden_size, hidden_count) ||
        *hidden_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN sequence element count is invalid or overflows");
    }
    if (output_capacity < *hidden_count) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                   "GDN output capacity is too small");
    }
    if (scratch_bytes < config->scratch_bytes ||
        ((uintptr_t)scratch % (uintptr_t)_Alignof(float)) != 0U) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                   "GDN aligned scratch buffer is too small");
    }
    if (padding_mask != NULL) {
        for (index = 0U; index < sequence_length; ++index) {
            if (padding_mask[index] > 1U) {
                return kq_gdn_execute_fail(diagnostic,
                                           KQ_STATUS_INVALID_ARGUMENT,
                                           "GDN padding mask must contain only zero or one");
            }
        }
    }
    status = kq_gdn_validate_finite(hidden_states, *hidden_count,
                                    "GDN hidden input must be finite",
                                    diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_gdn_validate_finite(
        (const float *)state->conv_state, config->conv_state_elements,
        "GDN convolution state must be finite", diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_gdn_validate_finite(
        state->recurrent_state, config->recurrent_elements,
        "GDN recurrent state must be finite", diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (!provider_mode) {
        status = kq_gdn_validate_weights(config, weights, diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    if (!kq_gdn_exec_u64_mul(*hidden_count, sizeof(float), &bytes) ||
        !kq_gdn_exec_u64_mul(output_capacity, sizeof(float), &output_bytes) ||
        !kq_gdn_exec_u64_mul(config->conv_state_elements, sizeof(float),
                             &conv_state_bytes) ||
        !kq_gdn_exec_u64_mul(config->recurrent_elements, sizeof(float),
                             &recurrent_state_bytes)) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                   "GDN call byte count overflows");
    }
    if (kq_gdn_ranges_overlap(hidden_states, bytes, output, output_bytes) ||
        kq_gdn_ranges_overlap(hidden_states, bytes, scratch, scratch_bytes) ||
        kq_gdn_ranges_overlap(output, output_bytes, scratch, scratch_bytes) ||
        kq_gdn_ranges_overlap(state->conv_state,
                              conv_state_bytes,
                              scratch, scratch_bytes) ||
        kq_gdn_ranges_overlap(state->recurrent_state,
                              recurrent_state_bytes,
                              scratch, scratch_bytes) ||
        kq_gdn_ranges_overlap(hidden_states, bytes, state->conv_state,
                              conv_state_bytes) ||
        kq_gdn_ranges_overlap(hidden_states, bytes, state->recurrent_state,
                              recurrent_state_bytes) ||
        kq_gdn_ranges_overlap(output, output_bytes, state->conv_state,
                              conv_state_bytes) ||
        kq_gdn_ranges_overlap(output, output_bytes, state->recurrent_state,
                              recurrent_state_bytes) ||
        (!provider_mode &&
         (kq_gdn_weights_overlap(weights, output, output_bytes) ||
          kq_gdn_weights_overlap(weights, scratch, scratch_bytes) ||
          kq_gdn_weights_overlap(weights, state->conv_state,
                                 conv_state_bytes) ||
          kq_gdn_weights_overlap(weights, state->recurrent_state,
                                 recurrent_state_bytes)))) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                   "GDN input, output, state, and scratch must not overlap");
    }
    if (fegetround() != FE_TONEAREST) {
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                   "GDN scalar reference requires round-to-nearest");
    }
    return KQ_STATUS_OK;
}

static kq_status kq_gdn_execute_common(
    const kq_gdn_config *config,
    const kq_gdn_weights_f32 *weights,
    kq_weight_provider *provider,
    const float *hidden_states,
    uint64_t sequence_length,
    const uint8_t *padding_mask,
    float *output,
    uint64_t output_capacity,
    kq_gdn_state *state,
    void *scratch,
    uint64_t scratch_bytes,
    void *weight_scratch,
    uint64_t weight_scratch_bytes,
    kq_gdn_checkpoint_observer observer,
    void *observer_user_data,
    kq_diagnostic *diagnostic) {
    kq_gdn_workspace workspace;
    kq_status status;
    uint64_t hidden_count;
    uint64_t token;
    uint64_t index;
    uint64_t head;
    uint64_t feature;
    uint64_t key_feature;
    uint64_t channel;
    uint64_t kernel;
    uint64_t hidden_size;
    uint64_t value_heads;
    uint64_t key_dimension;
    uint64_t value_dimension;
    uint64_t repeated_key_count;
    float query_scale;

    kq_diagnostic_clear(diagnostic);
    status = kq_gdn_validate_call(config, weights, hidden_states,
                                  sequence_length, padding_mask, output,
                                  output_capacity, state, scratch,
                                  scratch_bytes, provider != NULL,
                                  &hidden_count, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    (void)hidden_count;
    kq_gdn_workspace_init(config, scratch, &workspace);
    memcpy(workspace.conv_state, state->conv_state,
           (size_t)config->conv_state_elements * sizeof(float));
    memcpy(workspace.recurrent_state, state->recurrent_state,
           (size_t)config->recurrent_elements * sizeof(float));

    hidden_size = config->dimensions.hidden_size;
    value_heads = config->dimensions.value_head_count;
    key_dimension = config->dimensions.key_head_dimension;
    value_dimension = config->dimensions.value_head_dimension;
    repeated_key_count = value_heads * key_dimension;
    query_scale = 1.0f / sqrtf((float)key_dimension);

    for (token = 0U; token < sequence_length; ++token) {
        const float *input = hidden_states + token * hidden_size;
        float *token_output = output + token * hidden_size;
        float mask_value = padding_mask == NULL ? 1.0f :
            (float)padding_mask[token];

        for (index = 0U; index < hidden_size; ++index) {
            workspace.masked_input[index] = input[index] * mask_value;
        }
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_MASKED_INPUT, token, 1U,
                    hidden_size, 0U, 0U, workspace.masked_input, hidden_size);

        status = kq_gdn_project(config, weights, provider, 5U,
                                config->conv_channels,
                                hidden_size, workspace.masked_input,
                                workspace.projected_qkv, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_project(config, weights, provider, 6U,
                                config->value_dimension,
                                hidden_size, workspace.masked_input,
                                workspace.projected_gate, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_project(config, weights, provider, 4U, value_heads,
                                hidden_size, workspace.masked_input,
                                workspace.projected_beta, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_project(config, weights, provider, 3U, value_heads,
                                hidden_size, workspace.masked_input,
                                workspace.projected_alpha, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;

        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_PROJECTED_QKV, token, 1U,
                    config->conv_channels, 0U, 0U,
                    workspace.projected_qkv, config->conv_channels);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_PROJECTED_GATE, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.projected_gate, config->value_dimension);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_PROJECTED_BETA, token, 1U,
                    value_heads, 0U, 0U,
                    workspace.projected_beta, value_heads);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_PROJECTED_ALPHA, token, 1U,
                    value_heads, 0U, 0U,
                    workspace.projected_alpha, value_heads);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_CONV_INPUT, token, 1U,
                    config->conv_channels, 0U, 0U,
                    workspace.projected_qkv, config->conv_channels);

        for (channel = 0U; channel < config->conv_channels; ++channel) {
            float *history = workspace.conv_state +
                channel * config->dimensions.conv_kernel_size;
            for (kernel = 1U; kernel < config->dimensions.conv_kernel_size;
                 ++kernel) {
                history[kernel - 1U] = history[kernel];
            }
            history[config->dimensions.conv_kernel_size - 1U] =
                workspace.projected_qkv[channel];
            status = kq_f32_dot(
                weights->conv + channel * config->dimensions.conv_kernel_size,
                history, config->dimensions.conv_kernel_size,
                &workspace.conv_output[channel], diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
        status = kq_f32_silu(workspace.conv_output, config->conv_channels,
                             workspace.projected_qkv, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        memcpy(workspace.conv_output, workspace.projected_qkv,
               (size_t)config->conv_channels * sizeof(float));
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_CONV_OUTPUT, token, 1U,
                    config->conv_channels, 0U, 0U,
                    workspace.conv_output, config->conv_channels);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_CONV_STATE, token, 2U,
                    config->conv_channels, config->dimensions.conv_kernel_size,
                    0U, workspace.conv_state, config->conv_state_elements);

        for (head = 0U; head < value_heads; ++head) {
            uint64_t key_head = head / config->key_value_head_repeat;
            uint64_t source_query = key_head * key_dimension;
            uint64_t source_key = config->key_dimension +
                key_head * key_dimension;
            uint64_t target_key = head * key_dimension;
            uint64_t source_value = 2U * config->key_dimension +
                head * value_dimension;
            memcpy(workspace.query + target_key,
                   workspace.conv_output + source_query,
                   (size_t)key_dimension * sizeof(float));
            memcpy(workspace.key + target_key,
                   workspace.conv_output + source_key,
                   (size_t)key_dimension * sizeof(float));
            memcpy(workspace.value + head * value_dimension,
                   workspace.conv_output + source_value,
                   (size_t)value_dimension * sizeof(float));
        }
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_QUERY_BEFORE_NORM, token, 2U,
                    value_heads, key_dimension, 0U,
                    workspace.query, repeated_key_count);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_KEY_BEFORE_NORM, token, 2U,
                    value_heads, key_dimension, 0U,
                    workspace.key, repeated_key_count);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_VALUE, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.value, config->value_dimension);

        status = kq_f32_sigmoid(workspace.projected_beta, value_heads,
                                workspace.beta, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        for (head = 0U; head < value_heads; ++head) {
            float softplus;
            float decay_base = -expf(weights->a_log[head]);
            status = kq_gdn_softplus(workspace.projected_alpha[head] +
                                     weights->dt_bias[head],
                                     &softplus, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            workspace.log_decay[head] = decay_base * softplus;
            if (!isfinite(workspace.log_decay[head]) ||
                workspace.log_decay[head] > 0.0f) {
                return kq_gdn_execute_fail(
                    diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                    "GDN log decay must be finite and non-positive");
            }
            status = kq_gdn_l2_normalize(
                workspace.query + head * key_dimension,
                key_dimension, query_scale, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_gdn_l2_normalize(
                workspace.key + head * key_dimension,
                key_dimension, 1.0f, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_NORMALIZED_SCALED_QUERY, token, 2U,
                    value_heads, key_dimension, 0U,
                    workspace.query, repeated_key_count);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_NORMALIZED_KEY, token, 2U,
                    value_heads, key_dimension, 0U,
                    workspace.key, repeated_key_count);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_LOG_DECAY, token, 1U,
                    value_heads, 0U, 0U,
                    workspace.log_decay, value_heads);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_BETA, token, 1U,
                    value_heads, 0U, 0U,
                    workspace.beta, value_heads);

        for (head = 0U; head < value_heads; ++head) {
            float decay = expf(workspace.log_decay[head]);
            float *head_state = workspace.recurrent_state +
                head * key_dimension * value_dimension;
            float *head_key = workspace.key + head * key_dimension;
            float *head_query = workspace.query + head * key_dimension;
            float *head_value = workspace.value + head * value_dimension;
            float *head_read = workspace.read + head * value_dimension;
            float *head_delta = workspace.delta + head * value_dimension;
            float *head_output = workspace.core_output +
                head * value_dimension;

            if (!isfinite(decay)) {
                return kq_gdn_execute_fail(diagnostic,
                                           KQ_STATUS_NUMERIC_DOMAIN,
                                           "GDN decay is non-finite");
            }
            for (index = 0U; index < key_dimension * value_dimension;
                 ++index) {
                head_state[index] = head_state[index] * decay;
            }
            for (feature = 0U; feature < value_dimension; ++feature) {
                float sum = 0.0f;
                for (key_feature = 0U; key_feature < key_dimension;
                     ++key_feature) {
                    float product = head_state[
                        key_feature * value_dimension + feature] *
                        head_key[key_feature];
                    sum = sum + product;
                }
                head_read[feature] = sum;
                head_delta[feature] =
                    (head_value[feature] - sum) * workspace.beta[head];
            }
            for (key_feature = 0U; key_feature < key_dimension;
                 ++key_feature) {
                for (feature = 0U; feature < value_dimension; ++feature) {
                    uint64_t state_index =
                        key_feature * value_dimension + feature;
                    float update = head_key[key_feature] *
                        head_delta[feature];
                    head_state[state_index] =
                        head_state[state_index] + update;
                }
            }
            for (feature = 0U; feature < value_dimension; ++feature) {
                float sum = 0.0f;
                for (key_feature = 0U; key_feature < key_dimension;
                     ++key_feature) {
                    float product = head_state[
                        key_feature * value_dimension + feature] *
                        head_query[key_feature];
                    sum = sum + product;
                }
                head_output[feature] = sum;
            }
        }
        status = kq_gdn_validate_finite(
            workspace.recurrent_state, config->recurrent_elements,
            "GDN recurrence produced non-finite state", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_validate_finite(
            workspace.core_output, config->value_dimension,
            "GDN recurrence produced non-finite output", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_RECURRENT_READ, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.read, config->value_dimension);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_RECURRENT_DELTA, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.delta, config->value_dimension);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_RECURRENT_OUTPUT, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.core_output, config->value_dimension);
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_RECURRENT_STATE, token, 3U,
                    value_heads, key_dimension, value_dimension,
                    workspace.recurrent_state, config->recurrent_elements);

        status = kq_f32_sigmoid(workspace.projected_gate,
                                config->value_dimension,
                                workspace.gated_output, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        for (head = 0U; head < value_heads; ++head) {
            float sum = 0.0f;
            float inverse;
            float *head_core = workspace.core_output +
                head * value_dimension;
            float *head_gate = workspace.gated_output +
                head * value_dimension;
            for (feature = 0U; feature < value_dimension; ++feature) {
                float square = head_core[feature] * head_core[feature];
                sum = sum + square;
            }
            sum = sum / (float)value_dimension;
            inverse = 1.0f / sqrtf(sum +
                config->dimensions.rms_norm_epsilon);
            if (!isfinite(inverse)) {
                return kq_gdn_execute_fail(
                    diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                    "GDN gated RMS normalization is non-finite");
            }
            for (feature = 0U; feature < value_dimension; ++feature) {
                float normalized = head_core[feature] * inverse;
                float weighted = weights->norm[feature] * normalized;
                head_core[feature] = weighted * head_gate[feature];
            }
        }
        status = kq_gdn_validate_finite(
            workspace.core_output, config->value_dimension,
            "GDN gated normalization produced non-finite output",
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_GATED_NORM_OUTPUT, token, 2U,
                    value_heads, value_dimension, 0U,
                    workspace.core_output, config->value_dimension);

        status = kq_gdn_project(config, weights, provider, 8U, hidden_size,
                                config->value_dimension,
                                workspace.core_output,
                                workspace.operator_output, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_gdn_validate_finite(
            workspace.operator_output, hidden_size,
            "GDN output projection produced non-finite output", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        memcpy(token_output, workspace.operator_output,
               (size_t)hidden_size * sizeof(float));
        kq_gdn_emit(observer, observer_user_data,
                    KQ_GDN_CHECKPOINT_OPERATOR_OUTPUT, token, 1U,
                    hidden_size, 0U, 0U,
                    workspace.operator_output, hidden_size);
    }

    memcpy(state->conv_state, workspace.conv_state,
           (size_t)config->conv_state_elements * sizeof(float));
    memcpy(state->recurrent_state, workspace.recurrent_state,
           (size_t)config->recurrent_elements * sizeof(float));
    state->initialized = 1;
    return KQ_STATUS_OK;
}

kq_status kq_gdn_execute_f32(
    const kq_gdn_config *config, const kq_gdn_weights_f32 *weights,
    const float *hidden_states, uint64_t sequence_length,
    const uint8_t *padding_mask, float *output, uint64_t output_capacity,
    kq_gdn_state *state, void *scratch, uint64_t scratch_bytes,
    kq_gdn_checkpoint_observer observer, void *observer_user_data,
    kq_diagnostic *diagnostic) {
    return kq_gdn_execute_common(config, weights, NULL, hidden_states,
        sequence_length, padding_mask, output, output_capacity, state,
        scratch, scratch_bytes, NULL, 0U, observer, observer_user_data,
        diagnostic);
}

kq_status kq_gdn_execute_quantized(
    const kq_gdn_config *config, kq_weight_provider *provider,
    const float *hidden_states, uint64_t sequence_length,
    const uint8_t *padding_mask, float *output, uint64_t output_capacity,
    kq_gdn_state *state, void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_gdn_checkpoint_observer observer, void *observer_user_data,
    kq_diagnostic *diagnostic) {
    kq_gdn_weights_f32 resolved;
    float *cursor = (float *)weight_scratch;
    float *temporary;
    uint64_t vector_count = 48U + 40960U + 48U + 128U;
    uint64_t vector_bytes = vector_count * sizeof(float);
    uint64_t temporary_count = 40960U;
    uint64_t temporary_bytes = temporary_count * sizeof(float);
    kq_status status;
    if (provider == NULL || weight_scratch == NULL ||
        weight_scratch_bytes < vector_bytes + temporary_bytes + 65536U)
        return kq_gdn_execute_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                   "GDN provider weight scratch is too small");
    memset(&resolved, 0, sizeof(resolved));
    resolved.a_log = cursor; resolved.a_log_count = 48U; cursor += 48U;
    resolved.conv = cursor; resolved.conv_count = 40960U; cursor += 40960U;
    resolved.dt_bias = cursor; resolved.dt_bias_count = 48U; cursor += 48U;
    resolved.norm = cursor; resolved.norm_count = 128U; cursor += 128U;
    temporary = cursor;
#define LOAD_GDN_VECTOR(field, role_index) do { \
    status = kq_weight_provider_vector_f32(provider, config->tensors[role_index], \
        resolved.field##_count, (float *)resolved.field, resolved.field##_count, \
        temporary, temporary_bytes, diagnostic); \
    if (status != KQ_STATUS_OK) return status; \
} while (0)
    LOAD_GDN_VECTOR(a_log, 0U);
    LOAD_GDN_VECTOR(conv, 1U);
    LOAD_GDN_VECTOR(dt_bias, 2U);
    LOAD_GDN_VECTOR(norm, 7U);
#undef LOAD_GDN_VECTOR
    cursor = (float *)((unsigned char *)temporary + temporary_bytes);
    return kq_gdn_execute_common(config, &resolved, provider, hidden_states,
        sequence_length, padding_mask, output, output_capacity, state,
        scratch, scratch_bytes, cursor,
        weight_scratch_bytes - vector_bytes - temporary_bytes,
        observer, observer_user_data, diagnostic);
}
