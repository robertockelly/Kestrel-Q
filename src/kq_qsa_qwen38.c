#include "kq_qsa_internal.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_numeric.h"
#include "kq_weight_provider.h"

typedef struct kq_qsa_workspace {
    float *pending_key;
    float *pending_value;
    float *pending_raw;
    float *staged_output;
    float *projected_query_gate;
    float *query;
    float *key;
    float *value;
    float *index_query;
    float *raw_index_key;
    float *block_keys;
    float *candidate_scores;
    uint32_t *candidate_ids;
    uint32_t *selected_block_ids;
    uint32_t *selected_tokens;
    float *attention_logits;
    float *attention_probabilities;
    float *head_context;
    float *gated_context;
    float *operator_output;
} kq_qsa_workspace;

static kq_status kq_qsa_exec_fail(kq_diagnostic *diagnostic,
                                  kq_status status,
                                  const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_qsa_ranges_overlap(const void *left, uint64_t left_bytes,
                                 const void *right, uint64_t right_bytes) {
    uintptr_t a;
    uintptr_t b;
    if (left == NULL || right == NULL || left_bytes == 0U ||
        right_bytes == 0U || left_bytes > UINTPTR_MAX ||
        right_bytes > UINTPTR_MAX) return 0;
    a = (uintptr_t)left;
    b = (uintptr_t)right;
    if (a > UINTPTR_MAX - (uintptr_t)left_bytes ||
        b > UINTPTR_MAX - (uintptr_t)right_bytes) return 1;
    return a < b + (uintptr_t)right_bytes && b < a + (uintptr_t)left_bytes;
}

static kq_status kq_qsa_validate_finite(const float *values,
                                        uint64_t count,
                                        const char *message,
                                        kq_diagnostic *diagnostic) {
    uint64_t index;
    if (values == NULL || count == 0U) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                message);
    }
    for (index = 0U; index < count; ++index) {
        if (!isfinite(values[index])) {
            return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    message);
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_weight(const float *values, uint64_t actual,
                               uint64_t expected, const char *message,
                               kq_diagnostic *diagnostic) {
    if (actual != expected) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                                message);
    }
    return kq_qsa_validate_finite(values, actual, message, diagnostic);
}

static kq_status kq_qsa_validate_weights(const kq_qsa_config *config,
                                         const kq_qsa_weights_f32 *weights,
                                         kq_diagnostic *diagnostic) {
    const kq_qsa_dimensions *d = &config->dimensions;
    uint64_t q_width;
    uint64_t kv_width = config->key_values_per_token;
    uint64_t index_q_width;
    uint64_t query_rows;
    uint64_t count;
    kq_status status;
    if (weights == NULL ||
        !kq_qsa_u64_mul(d->query_head_count, d->head_dimension, &q_width) ||
        !kq_qsa_u64_mul(d->index_query_head_count,
                        d->index_head_dimension, &index_q_width)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "complete QSA F32 weights are required");
    }
#define KQ_QSA_CHECK_WEIGHT(pointer, actual, expected, message) \
    do { status = kq_qsa_weight((pointer), (actual), (expected), (message), \
                                diagnostic); \
         if (status != KQ_STATUS_OK) return status; } while (0)
    if (!kq_qsa_u64_add(q_width, q_width, &query_rows) ||
        !kq_qsa_u64_mul(query_rows, d->hidden_size, &count)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA query projection count overflows");
    }
    KQ_QSA_CHECK_WEIGHT(weights->query, weights->query_count, count,
                        "QSA query projection shape/data mismatch");
    if (!kq_qsa_u64_mul(kv_width, d->hidden_size, &count)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA K/V projection count overflows");
    }
    KQ_QSA_CHECK_WEIGHT(weights->key, weights->key_count, count,
                        "QSA key projection shape/data mismatch");
    KQ_QSA_CHECK_WEIGHT(weights->value, weights->value_count, count,
                        "QSA value projection shape/data mismatch");
    if (!kq_qsa_u64_mul(d->hidden_size, q_width, &count)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA output projection count overflows");
    }
    KQ_QSA_CHECK_WEIGHT(weights->output, weights->output_count, count,
                        "QSA output projection shape/data mismatch");
    KQ_QSA_CHECK_WEIGHT(weights->query_norm, weights->query_norm_count,
                        d->head_dimension,
                        "QSA query norm shape/data mismatch");
    KQ_QSA_CHECK_WEIGHT(weights->key_norm, weights->key_norm_count,
                        d->head_dimension,
                        "QSA key norm shape/data mismatch");
    if (!kq_qsa_u64_mul(index_q_width, d->hidden_size, &count)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA index query projection count overflows");
    }
    KQ_QSA_CHECK_WEIGHT(weights->index_query, weights->index_query_count,
                        count, "QSA index query projection shape/data mismatch");
    if (!kq_qsa_u64_mul(d->index_head_dimension, d->hidden_size, &count)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA index key projection count overflows");
    }
    KQ_QSA_CHECK_WEIGHT(weights->index_key, weights->index_key_count, count,
                        "QSA index key projection shape/data mismatch");
    KQ_QSA_CHECK_WEIGHT(weights->index_query_norm,
                        weights->index_query_norm_count,
                        d->index_head_dimension,
                        "QSA index query norm shape/data mismatch");
    KQ_QSA_CHECK_WEIGHT(weights->index_key_norm,
                        weights->index_key_norm_count,
                        d->index_head_dimension,
                        "QSA index key norm shape/data mismatch");
#undef KQ_QSA_CHECK_WEIGHT
    return KQ_STATUS_OK;
}

static float *kq_qsa_take(float **cursor, uint64_t count) {
    float *result = *cursor;
    *cursor += count;
    return result;
}

static void kq_qsa_workspace_init(const kq_qsa_config *config,
                                  const kq_qsa_state *state,
                                  uint64_t sequence_length,
                                  void *scratch,
                                  kq_qsa_workspace *workspace) {
    float *cursor = (float *)scratch;
    uint64_t q_width = (uint64_t)config->dimensions.query_head_count *
        config->dimensions.head_dimension;
    uint64_t kv_width = config->key_values_per_token;
    uint64_t iq_width = (uint64_t)config->dimensions.index_query_head_count *
        config->dimensions.index_head_dimension;
    uint64_t blocks = state->capacity / config->dimensions.block_size;
    uint64_t attention = (uint64_t)config->dimensions.query_head_count *
        config->selected_token_capacity;
    workspace->pending_key = kq_qsa_take(&cursor, sequence_length * kv_width);
    workspace->pending_value = kq_qsa_take(&cursor, sequence_length * kv_width);
    workspace->pending_raw = kq_qsa_take(
        &cursor, sequence_length * config->raw_index_values_per_token);
    workspace->staged_output = kq_qsa_take(
        &cursor, sequence_length * config->dimensions.hidden_size);
    workspace->projected_query_gate = kq_qsa_take(&cursor, 2U * q_width);
    workspace->query = kq_qsa_take(&cursor, q_width);
    workspace->key = kq_qsa_take(&cursor, kv_width);
    workspace->value = kq_qsa_take(&cursor, kv_width);
    workspace->index_query = kq_qsa_take(&cursor, iq_width);
    workspace->raw_index_key = kq_qsa_take(
        &cursor, config->raw_index_values_per_token);
    workspace->block_keys = kq_qsa_take(
        &cursor, blocks * config->raw_index_values_per_token);
    workspace->candidate_scores = kq_qsa_take(&cursor, blocks);
    workspace->candidate_ids = (uint32_t *)kq_qsa_take(&cursor, blocks);
    workspace->selected_block_ids = (uint32_t *)kq_qsa_take(
        &cursor, config->block_topk);
    workspace->selected_tokens = (uint32_t *)kq_qsa_take(
        &cursor, config->selected_token_capacity);
    workspace->attention_logits = kq_qsa_take(&cursor, attention);
    workspace->attention_probabilities = kq_qsa_take(&cursor, attention);
    workspace->head_context = kq_qsa_take(&cursor, q_width);
    workspace->gated_context = kq_qsa_take(&cursor, q_width);
    workspace->operator_output = kq_qsa_take(
        &cursor, config->dimensions.hidden_size);
}

static kq_status kq_qsa_project(const kq_qsa_config *config,
                                const float *weights,
                                kq_weight_provider *provider,
                                uint32_t role_index,
                                kq_binding_part_role part_role,
                                uint64_t rows,
                                uint64_t columns, const float *input,
                                float *output, void *weight_scratch,
                                uint64_t weight_scratch_bytes,
                                kq_diagnostic *diagnostic) {
    uint64_t row;
    if (provider != NULL)
        return kq_weight_provider_linear_f32(
            provider, config->tensors[role_index], part_role,
            KQ_WEIGHT_PROVIDER_NO_EXPERT, rows, columns, input, output, rows,
            weight_scratch, weight_scratch_bytes, diagnostic);
    for (row = 0U; row < rows; ++row) {
        kq_status status = kq_f32_dot(weights + row * columns, input,
                                      columns, &output[row], diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_norm(float *values, uint64_t count,
                             const float *delta_weight, float epsilon,
                             kq_diagnostic *diagnostic) {
    float sum = 0.0f;
    float inverse;
    uint64_t index;
    for (index = 0U; index < count; ++index) {
        float square = values[index] * values[index];
        sum = sum + square;
    }
    sum = sum / (float)count;
    inverse = 1.0f / sqrtf(sum + epsilon);
    if (!isfinite(inverse)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "QSA RMS normalization is non-finite");
    }
    for (index = 0U; index < count; ++index) {
        values[index] = (values[index] * inverse) *
            (1.0f + delta_weight[index]);
        if (!isfinite(values[index])) {
            return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "QSA normalized value is non-finite");
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_rope(float *values, uint64_t width,
                             uint32_t rotary_dimension, float theta,
                             uint64_t position,
                             kq_diagnostic *diagnostic) {
    uint32_t half = rotary_dimension / 2U;
    uint32_t index;
    (void)width;
    for (index = 0U; index < half; ++index) {
        float exponent = (2.0f * (float)index) / (float)rotary_dimension;
        float frequency = 1.0f / powf(theta, exponent);
        float angle = (float)position * frequency;
        float cosine = cosf(angle);
        float sine = sinf(angle);
        float first = values[index];
        float second = values[index + half];
        values[index] = first * cosine - second * sine;
        values[index + half] = second * cosine + first * sine;
        if (!isfinite(values[index]) || !isfinite(values[index + half])) {
            return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "QSA RoPE produced a non-finite value");
        }
    }
    return KQ_STATUS_OK;
}

static float kq_qsa_state_f32(const kq_qsa_state *state,
                              const void *cache, uint64_t index) {
    if (state->config->dimensions.activation_dtype == KQ_QSA_ACTIVATION_F32) {
        return ((const float *)cache)[index];
    }
    {
        uint32_t bits = (uint32_t)((const uint16_t *)cache)[index] << 16U;
        float value;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
}

static float kq_qsa_read_combined(const kq_qsa_state *state,
                                  const void *committed,
                                  const float *pending,
                                  uint64_t initial_length,
                                  uint64_t token,
                                  uint64_t width,
                                  uint64_t feature) {
    if (token < initial_length) {
        return kq_qsa_state_f32(state, committed, token * width + feature);
    }
    return pending[(token - initial_length) * width + feature];
}

static void kq_qsa_emit(kq_qsa_checkpoint_observer observer,
                        void *user_data, kq_qsa_checkpoint_kind kind,
                        uint64_t token, uint32_t rank,
                        uint64_t first, uint64_t second, uint64_t third,
                        const float *values, uint64_t count) {
    kq_qsa_checkpoint checkpoint;
    if (observer == NULL) return;
    memset(&checkpoint, 0, sizeof(checkpoint));
    checkpoint.kind = kind;
    checkpoint.token_index = token;
    checkpoint.rank = rank;
    checkpoint.dimensions[0] = first;
    checkpoint.dimensions[1] = second;
    checkpoint.dimensions[2] = third;
    checkpoint.values = values;
    checkpoint.value_count = count;
    observer(&checkpoint, user_data);
}

static int kq_qsa_better(float candidate_score, uint32_t candidate_id,
                         float current_score, uint32_t current_id) {
    return candidate_score > current_score ||
        (candidate_score == current_score && candidate_id < current_id);
}

static uint64_t kq_qsa_select_blocks(const float *scores,
                                     uint64_t candidate_count,
                                     uint64_t limit,
                                     uint32_t *selected) {
    uint64_t selected_count = 0U;
    uint64_t candidate;
    for (candidate = 0U; candidate < candidate_count; ++candidate) {
        uint64_t position = 0U;
        while (position < selected_count &&
               !kq_qsa_better(scores[candidate], (uint32_t)candidate,
                              scores[selected[position]],
                              selected[position])) {
            position += 1U;
        }
        if (position < limit) {
            uint64_t end = selected_count < limit ? selected_count : limit - 1U;
            while (end > position) {
                selected[end] = selected[end - 1U];
                end -= 1U;
            }
            selected[position] = (uint32_t)candidate;
            if (selected_count < limit) selected_count += 1U;
        }
    }
    return selected_count;
}

kq_status kq_qsa_select_blocks_f32(
    const float *candidate_scores, uint64_t candidate_count,
    uint64_t selection_limit, uint32_t *selected_block_ids,
    uint64_t selected_capacity, uint64_t *selected_count,
    kq_diagnostic *diagnostic) {
    uint64_t index;
    uint64_t required = candidate_count < selection_limit ?
        candidate_count : selection_limit;
    kq_diagnostic_clear(diagnostic);
    if (selected_count == NULL || selection_limit == 0U ||
        candidate_count > UINT32_MAX ||
        (candidate_count != 0U && candidate_scores == NULL) ||
        (required != 0U && selected_block_ids == NULL)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "valid bounded QSA selection arguments are required");
    }
    *selected_count = 0U;
    if (selected_capacity < required) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "QSA selected-block output is too small");
    }
    for (index = 0U; index < candidate_count; ++index) {
        if (!isfinite(candidate_scores[index])) {
            return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "QSA candidate scores must be finite");
        }
    }
    *selected_count = kq_qsa_select_blocks(
        candidate_scores, candidate_count, selection_limit,
        selected_block_ids);
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_validate_call(
    const kq_qsa_config *config, const kq_qsa_weights_f32 *weights,
    const float *hidden, uint64_t sequence, float *output,
    uint64_t output_capacity, kq_qsa_state *state, void *scratch,
    uint64_t scratch_bytes, uint64_t *hidden_count, int provider_mode,
    kq_diagnostic *diagnostic) {
    uint64_t required_scratch;
    uint64_t hidden_bytes;
    uint64_t output_bytes;
    uint64_t state_kv_count;
    uint64_t state_kv_bytes;
    uint64_t state_raw_count;
    uint64_t state_raw_bytes;
    uint64_t total_length;
    kq_status status;
    if (!kq_qsa_config_valid(config) || (!provider_mode && weights == NULL) || hidden == NULL ||
        output == NULL || !kq_qsa_state_valid(state) ||
        state->config != config || scratch == NULL || hidden_count == NULL) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "valid compatible QSA call arguments are required");
    }
    if (config->dimensions.activation_dtype != KQ_QSA_ACTIVATION_F32) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_QSA,
                                "scalar execution requires an F32 QSA reference config");
    }
    if (sequence == 0U ||
        !kq_qsa_u64_add(state->length, sequence, &total_length) ||
        total_length > state->capacity ||
        !kq_qsa_u64_mul(sequence, config->dimensions.hidden_size,
                        hidden_count) || *hidden_count > SIZE_MAX / sizeof(float)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_LIMIT_EXCEEDED,
                                "QSA sequence exceeds bounded state capacity or overflows");
    }
    if (output_capacity < *hidden_count) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "QSA output capacity is too small");
    }
    status = kq_qsa_calculate_scratch_bytes(config, state, sequence,
                                            &required_scratch, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (scratch_bytes < required_scratch ||
        ((uintptr_t)scratch % _Alignof(float)) != 0U) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "QSA aligned scratch buffer is too small");
    }
    status = kq_qsa_validate_finite(hidden, *hidden_count,
                                    "QSA input must be finite", diagnostic);
    if (status != KQ_STATUS_OK) return status;
    if (!provider_mode) {
        status = kq_qsa_validate_weights(config, weights, diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    if (state->length != 0U) {
        status = kq_qsa_validate_finite((const float *)state->key_cache,
            state->length * config->key_values_per_token,
            "QSA K cache must be finite", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_validate_finite((const float *)state->value_cache,
            state->length * config->key_values_per_token,
            "QSA V cache must be finite", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_validate_finite((const float *)state->raw_index_key_cache,
            state->length * config->raw_index_values_per_token,
            "QSA index-key cache must be finite", diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    if (!kq_qsa_u64_mul(*hidden_count, sizeof(float), &hidden_bytes) ||
        !kq_qsa_u64_mul(output_capacity, sizeof(float), &output_bytes) ||
        !kq_qsa_u64_mul(state->capacity, config->key_values_per_token,
                        &state_kv_count) ||
        !kq_qsa_u64_mul(state_kv_count, sizeof(float), &state_kv_bytes) ||
        !kq_qsa_u64_mul(state->capacity,
                        config->raw_index_values_per_token,
                        &state_raw_count) ||
        !kq_qsa_u64_mul(state_raw_count, sizeof(float), &state_raw_bytes)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "QSA call byte count overflows");
    }
    if (kq_qsa_ranges_overlap(hidden, hidden_bytes, output, output_bytes) ||
        kq_qsa_ranges_overlap(hidden, hidden_bytes, scratch, scratch_bytes) ||
        kq_qsa_ranges_overlap(output, output_bytes, scratch, scratch_bytes) ||
        kq_qsa_ranges_overlap(state->key_cache, state_kv_bytes,
                              scratch, scratch_bytes) ||
        kq_qsa_ranges_overlap(state->value_cache, state_kv_bytes,
                              scratch, scratch_bytes) ||
        kq_qsa_ranges_overlap(state->raw_index_key_cache, state_raw_bytes,
                              scratch, scratch_bytes)) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "QSA input, output, state, and scratch must not overlap");
    }
    if (fegetround() != FE_TONEAREST) {
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "QSA scalar reference requires round-to-nearest");
    }
    return KQ_STATUS_OK;
}

static kq_status kq_qsa_execute_common(
    const kq_qsa_config *config, const kq_qsa_weights_f32 *weights,
    kq_weight_provider *provider,
    const float *hidden_states, uint64_t sequence_length,
    float *output, uint64_t output_capacity, kq_qsa_state *state,
    void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    const kq_qsa_dimensions *d;
    kq_qsa_workspace w;
    uint64_t hidden_count;
    uint64_t initial_length;
    uint64_t q_width;
    uint64_t kv_width;
    uint64_t iq_width;
    uint64_t token;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    status = kq_qsa_validate_call(
        config, weights, hidden_states, sequence_length, output,
        output_capacity, state, scratch, scratch_bytes, &hidden_count,
        provider != NULL, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    d = &config->dimensions;
    initial_length = state->length;
    q_width = (uint64_t)d->query_head_count * d->head_dimension;
    kv_width = config->key_values_per_token;
    iq_width = (uint64_t)d->index_query_head_count * d->index_head_dimension;
    kq_qsa_workspace_init(config, state, sequence_length, scratch, &w);

    for (token = 0U; token < sequence_length; ++token) {
        const float *input = hidden_states + token * d->hidden_size;
        uint64_t absolute_position = initial_length + token;
        uint64_t head;
        uint64_t feature;
        uint64_t block;
        uint64_t complete_blocks;
        uint64_t selected_block_count;
        uint64_t selected_token_count = 0U;
        uint64_t tail_count;
        float *pending_key = w.pending_key + token * kv_width;
        float *pending_value = w.pending_value + token * kv_width;
        float *pending_raw = w.pending_raw +
            token * config->raw_index_values_per_token;

        status = kq_qsa_project(config, weights->query, provider, 4U,
                                KQ_BINDING_PART_WHOLE, 2U * q_width,
                                d->hidden_size, input,
                                w.projected_query_gate, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_project(config, weights->key, provider, 1U,
                                KQ_BINDING_PART_WHOLE, kv_width, d->hidden_size,
                                input, w.key, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_project(config, weights->value, provider, 5U,
                                KQ_BINDING_PART_WHOLE, kv_width, d->hidden_size,
                                input, w.value, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_project(config, weights->index_query, provider, 6U,
                                KQ_BINDING_PART_INDEX_QUERY, iq_width,
                                d->hidden_size, input, w.index_query,
                                weight_scratch, weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_project(config, weights->index_key, provider, 6U,
                                KQ_BINDING_PART_INDEX_KEY,
                                d->index_head_dimension, d->hidden_size,
                                input, w.raw_index_key, weight_scratch,
                                weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;

        for (head = 0U; head < d->query_head_count; ++head) {
            float *query = w.query + head * d->head_dimension;
            memcpy(query,
                   w.projected_query_gate + head * 2U * d->head_dimension,
                   d->head_dimension * sizeof(float));
            status = kq_qsa_norm(query, d->head_dimension,
                                 weights->query_norm,
                                 d->rms_norm_epsilon, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_qsa_rope(query, d->head_dimension,
                                 d->rotary_dimension, d->rope_theta,
                                 absolute_position, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
        for (head = 0U; head < d->key_value_head_count; ++head) {
            float *key = w.key + head * d->head_dimension;
            status = kq_qsa_norm(key, d->head_dimension,
                                 weights->key_norm, d->rms_norm_epsilon,
                                 diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_qsa_rope(key, d->head_dimension,
                                 d->rotary_dimension, d->rope_theta,
                                 absolute_position, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
        for (head = 0U; head < d->index_query_head_count; ++head) {
            float *query = w.index_query + head * d->index_head_dimension;
            status = kq_qsa_norm(query, d->index_head_dimension,
                                 weights->index_query_norm,
                                 d->rms_norm_epsilon, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_qsa_rope(query, d->index_head_dimension,
                                 d->rotary_dimension, d->rope_theta,
                                 absolute_position, diagnostic);
            if (status != KQ_STATUS_OK) return status;
        }
        memcpy(pending_key, w.key, kv_width * sizeof(float));
        memcpy(pending_value, w.value, kv_width * sizeof(float));
        memcpy(pending_raw, w.raw_index_key,
               d->index_head_dimension * sizeof(float));

        complete_blocks = (absolute_position + 1U) / d->block_size;
        tail_count = (absolute_position + 1U) % d->block_size;
        for (block = 0U; block < complete_blocks; ++block) {
            float *block_key = w.block_keys + block * d->index_head_dimension;
            uint64_t offset;
            float score = 0.0f;
            for (feature = 0U; feature < d->index_head_dimension; ++feature) {
                float sum = 0.0f;
                for (offset = 0U; offset < d->block_size; ++offset) {
                    sum = sum + kq_qsa_read_combined(
                        state, state->raw_index_key_cache, w.pending_raw,
                        initial_length, block * d->block_size + offset,
                        d->index_head_dimension, feature);
                }
                block_key[feature] = sum / (float)d->block_size;
            }
            status = kq_qsa_norm(block_key, d->index_head_dimension,
                                 weights->index_key_norm,
                                 d->rms_norm_epsilon, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            status = kq_qsa_rope(block_key, d->index_head_dimension,
                                 d->rotary_dimension, d->rope_theta,
                                 block * d->block_size, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            for (head = 0U; head < d->index_query_head_count; ++head) {
                float dot;
                status = kq_f32_dot(
                    w.index_query + head * d->index_head_dimension,
                    block_key, d->index_head_dimension, &dot, diagnostic);
                if (status != KQ_STATUS_OK) return status;
                if (dot > 0.0f) score = score + dot;
            }
            score = score / sqrtf((float)d->index_head_dimension);
            if (!isfinite(score)) {
                return kq_qsa_exec_fail(
                    diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                    "QSA candidate score is non-finite");
            }
            w.candidate_scores[block] = score;
            w.candidate_ids[block] = (uint32_t)block;
        }
        selected_block_count = kq_qsa_select_blocks(
            w.candidate_scores, complete_blocks,
            complete_blocks < config->block_topk ?
                complete_blocks : config->block_topk,
            w.selected_block_ids);
        for (block = 0U; block < selected_block_count; ++block) {
            uint64_t offset;
            for (offset = 0U; offset < d->block_size; ++offset) {
                w.selected_tokens[selected_token_count++] =
                    w.selected_block_ids[block] * d->block_size +
                    (uint32_t)offset;
            }
        }
        for (feature = 0U; feature < tail_count; ++feature) {
            w.selected_tokens[selected_token_count++] =
                (uint32_t)(complete_blocks * d->block_size + feature);
        }
        if (selected_token_count == 0U ||
            selected_token_count > config->selected_token_capacity) {
            return kq_qsa_exec_fail(diagnostic, KQ_STATUS_INVALID_QSA_STATE,
                                    "QSA selected-token count is invalid");
        }

        if (selection_observer != NULL) {
            kq_qsa_selection selection;
            memset(&selection, 0, sizeof(selection));
            selection.token_index = token;
            selection.absolute_position = absolute_position;
            selection.candidate_block_ids = w.candidate_ids;
            selection.candidate_scores = w.candidate_scores;
            selection.candidate_count = complete_blocks;
            selection.selected_block_ids = w.selected_block_ids;
            selection.selected_block_count = selected_block_count;
            selection.selected_token_positions = w.selected_tokens;
            selection.selected_token_count = selected_token_count;
            selection.tail_count = tail_count;
            selection_observer(&selection, observer_user_data);
        }
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_INDEX_QUERY, token, 2U,
                    d->index_query_head_count, d->index_head_dimension, 0U,
                    w.index_query, iq_width);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_RAW_INDEX_KEY, token, 1U,
                    d->index_head_dimension, 0U, 0U,
                    w.raw_index_key, d->index_head_dimension);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_BLOCK_KEYS, token, 2U,
                    complete_blocks, d->index_head_dimension, 0U,
                    w.block_keys, complete_blocks * d->index_head_dimension);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_CANDIDATE_SCORES, token, 1U,
                    complete_blocks, 0U, 0U,
                    w.candidate_scores, complete_blocks);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_QUERY, token, 2U,
                    d->query_head_count, d->head_dimension, 0U,
                    w.query, q_width);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_KEY, token, 2U,
                    d->key_value_head_count, d->head_dimension, 0U,
                    w.key, kv_width);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_VALUE, token, 2U,
                    d->key_value_head_count, d->head_dimension, 0U,
                    w.value, kv_width);

        for (head = 0U; head < d->query_head_count; ++head) {
            uint64_t kv_head = head / config->key_value_repeat;
            float maximum = -INFINITY;
            float sum = 0.0f;
            uint64_t selected_index;
            for (selected_index = 0U; selected_index < selected_token_count;
                 ++selected_index) {
                uint64_t selected_token = w.selected_tokens[selected_index];
                float dot = 0.0f;
                for (feature = 0U; feature < d->head_dimension; ++feature) {
                    float key_value = kq_qsa_read_combined(
                        state, state->key_cache, w.pending_key,
                        initial_length, selected_token, kv_width,
                        kv_head * d->head_dimension + feature);
                    dot = dot + w.query[head * d->head_dimension + feature] *
                        key_value;
                }
                dot = dot / sqrtf((float)d->head_dimension);
                w.attention_logits[head * selected_token_count +
                                   selected_index] = dot;
                if (dot > maximum) maximum = dot;
            }
            for (selected_index = 0U; selected_index < selected_token_count;
                 ++selected_index) {
                float probability = expf(
                    w.attention_logits[head * selected_token_count +
                                       selected_index] - maximum);
                w.attention_probabilities[
                    head * selected_token_count + selected_index] =
                    probability;
                sum = sum + probability;
            }
            if (!(sum > 0.0f) || !isfinite(sum)) {
                return kq_qsa_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                        "QSA attention softmax is invalid");
            }
            for (selected_index = 0U; selected_index < selected_token_count;
                 ++selected_index) {
                w.attention_probabilities[
                    head * selected_token_count + selected_index] /=
                    sum;
            }
            for (feature = 0U; feature < d->head_dimension; ++feature) {
                float context = 0.0f;
                for (selected_index = 0U;
                     selected_index < selected_token_count; ++selected_index) {
                    uint64_t selected_token = w.selected_tokens[selected_index];
                    float value = kq_qsa_read_combined(
                        state, state->value_cache, w.pending_value,
                        initial_length, selected_token, kv_width,
                        kv_head * d->head_dimension + feature);
                    context = context + w.attention_probabilities[
                        head * selected_token_count + selected_index] *
                        value;
                }
                w.head_context[head * d->head_dimension + feature] = context;
            }
        }
        for (head = 0U; head < d->query_head_count; ++head) {
            for (feature = 0U; feature < d->head_dimension; ++feature) {
                float gate = w.projected_query_gate[
                    head * 2U * d->head_dimension + d->head_dimension + feature];
                float sigmoid = 1.0f / (1.0f + expf(-gate));
                w.gated_context[head * d->head_dimension + feature] =
                    w.head_context[head * d->head_dimension + feature] * sigmoid;
            }
        }
        status = kq_qsa_project(config, weights->output, provider, 2U,
                                KQ_BINDING_PART_WHOLE, d->hidden_size, q_width,
                                w.gated_context, w.operator_output,
                                weight_scratch, weight_scratch_bytes, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        status = kq_qsa_validate_finite(
            w.operator_output, d->hidden_size,
            "QSA output is non-finite", diagnostic);
        if (status != KQ_STATUS_OK) return status;
        memcpy(w.staged_output + token * d->hidden_size, w.operator_output,
               d->hidden_size * sizeof(float));
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_ATTENTION_LOGITS, token, 2U,
                    d->query_head_count, selected_token_count, 0U,
                    w.attention_logits,
                    (uint64_t)d->query_head_count * selected_token_count);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_ATTENTION_PROBABILITIES, token, 2U,
                    d->query_head_count, selected_token_count, 0U,
                    w.attention_probabilities,
                    (uint64_t)d->query_head_count * selected_token_count);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_HEAD_CONTEXT, token, 2U,
                    d->query_head_count, d->head_dimension, 0U,
                    w.head_context, q_width);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_GATED_CONTEXT, token, 2U,
                    d->query_head_count, d->head_dimension, 0U,
                    w.gated_context, q_width);
        kq_qsa_emit(checkpoint_observer, observer_user_data,
                    KQ_QSA_CHECKPOINT_OPERATOR_OUTPUT, token, 1U,
                    d->hidden_size, 0U, 0U,
                    w.operator_output, d->hidden_size);
    }

    memcpy((float *)state->key_cache + initial_length * kv_width,
           w.pending_key, sequence_length * kv_width * sizeof(float));
    memcpy((float *)state->value_cache + initial_length * kv_width,
           w.pending_value, sequence_length * kv_width * sizeof(float));
    memcpy((float *)state->raw_index_key_cache +
               initial_length * config->raw_index_values_per_token,
           w.pending_raw,
           sequence_length * config->raw_index_values_per_token * sizeof(float));
    memcpy(output, w.staged_output, hidden_count * sizeof(float));
    state->length = initial_length + sequence_length;
    return KQ_STATUS_OK;
}

kq_status kq_qsa_execute_f32(
    const kq_qsa_config *config, const kq_qsa_weights_f32 *weights,
    const float *hidden_states, uint64_t sequence_length,
    float *output, uint64_t output_capacity, kq_qsa_state *state,
    void *scratch, uint64_t scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    return kq_qsa_execute_common(config, weights, NULL, hidden_states,
        sequence_length, output, output_capacity, state, scratch,
        scratch_bytes, NULL, 0U, selection_observer, checkpoint_observer,
        observer_user_data, diagnostic);
}

kq_status kq_qsa_execute_quantized(
    const kq_qsa_config *config, kq_weight_provider *provider,
    const float *hidden_states, uint64_t sequence_length,
    float *output, uint64_t output_capacity, kq_qsa_state *state,
    void *scratch, uint64_t scratch_bytes,
    void *weight_scratch, uint64_t weight_scratch_bytes,
    kq_qsa_selection_observer selection_observer,
    kq_qsa_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    kq_qsa_weights_f32 resolved;
    float *cursor = (float *)weight_scratch;
    float *temporary;
    uint64_t vector_bytes = UINT64_C(768) * sizeof(float);
    uint64_t temporary_bytes = UINT64_C(256) * sizeof(float);
    kq_status status;
    if (provider == NULL || weight_scratch == NULL ||
        weight_scratch_bytes < vector_bytes + temporary_bytes + 65536U)
        return kq_qsa_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "QSA provider weight scratch is too small");
    memset(&resolved, 0, sizeof(resolved));
    resolved.key_norm = cursor; resolved.key_norm_count = 256U; cursor += 256U;
    resolved.query_norm = cursor; resolved.query_norm_count = 256U; cursor += 256U;
    resolved.index_key_norm = cursor; resolved.index_key_norm_count = 128U; cursor += 128U;
    resolved.index_query_norm = cursor; resolved.index_query_norm_count = 128U; cursor += 128U;
    temporary = cursor;
#define LOAD_QSA_VECTOR(field, role_index) do { \
    status = kq_weight_provider_vector_f32(provider, config->tensors[role_index], \
        resolved.field##_count, (float *)resolved.field, resolved.field##_count, \
        temporary, temporary_bytes, diagnostic); \
    if (status != KQ_STATUS_OK) return status; \
} while (0)
    LOAD_QSA_VECTOR(key_norm, 0U);
    LOAD_QSA_VECTOR(query_norm, 3U);
    LOAD_QSA_VECTOR(index_key_norm, 7U);
    LOAD_QSA_VECTOR(index_query_norm, 8U);
#undef LOAD_QSA_VECTOR
    cursor = (float *)((unsigned char *)temporary + temporary_bytes);
    return kq_qsa_execute_common(config, &resolved, provider, hidden_states,
        sequence_length, output, output_capacity, state, scratch, scratch_bytes,
        cursor, weight_scratch_bytes - vector_bytes - temporary_bytes,
        selection_observer, checkpoint_observer, observer_user_data, diagnostic);
}
