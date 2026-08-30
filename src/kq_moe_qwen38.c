#include "kq_moe_internal.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_numeric.h"

typedef struct kq_moe_workspace {
    float *router_logits;
    float *router_probabilities;
    uint32_t *selected_ids;
    float *selected_weights;
    float *expert_gate;
    float *expert_up;
    float *expert_activated;
    float *expert_output;
    float *weighted_output;
    float *routed_sum;
    float *shared_gate;
    float *shared_up;
    float *shared_activated;
    float *shared_output;
    float *gated_shared;
} kq_moe_workspace;

static kq_status kq_moe_exec_fail(kq_diagnostic *diagnostic,
                                  kq_status status,
                                  const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_moe_ranges_overlap(const void *left, uint64_t left_bytes,
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

static kq_status kq_moe_validate_finite(const float *values,
                                        uint64_t count,
                                        const char *message,
                                        kq_diagnostic *diagnostic) {
    uint64_t index;
    if (values == NULL || count == 0U) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                message);
    }
    for (index = 0U; index < count; ++index) {
        if (!isfinite(values[index])) {
            return kq_moe_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    message);
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_moe_weight(const float *values, uint64_t actual,
                               uint64_t expected, const char *message,
                               kq_diagnostic *diagnostic) {
    if (actual != expected) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                                message);
    }
    return kq_moe_validate_finite(values, actual, message, diagnostic);
}

static kq_status kq_moe_validate_weights(const kq_moe_config *config,
                                         const kq_moe_weights_f32 *weights,
                                         kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d = &config->dimensions;
    uint64_t routed_matrix;
    uint64_t routed_stack;
    uint64_t shared_matrix;
    kq_status status;
    if (weights == NULL ||
        !kq_moe_u64_mul(d->routed_intermediate_size, d->hidden_size,
                        &routed_matrix) ||
        !kq_moe_u64_mul(routed_matrix, d->expert_count, &routed_stack)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "MoE routed weight count overflows");
    }
#define KQ_MOE_CHECK_WEIGHT(pointer, actual, expected, message) \
    do { status = kq_moe_weight((pointer), (actual), (expected), (message), \
                                diagnostic); \
         if (status != KQ_STATUS_OK) return status; } while (0)
    KQ_MOE_CHECK_WEIGHT(weights->router, weights->router_count,
                        (uint64_t)d->expert_count * d->hidden_size,
                        "MoE router shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->routed_gate, weights->routed_gate_count,
                        routed_stack, "MoE routed gate shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->routed_up, weights->routed_up_count,
                        routed_stack, "MoE routed up shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->routed_down, weights->routed_down_count,
                        routed_stack, "MoE routed down shape/data mismatch");
    if (!kq_moe_u64_mul(d->shared_intermediate_size, d->hidden_size,
                        &shared_matrix)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "MoE shared weight count overflows");
    }
    KQ_MOE_CHECK_WEIGHT(weights->shared_gate, weights->shared_gate_count,
                        shared_matrix, "MoE shared gate shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->shared_up, weights->shared_up_count,
                        shared_matrix, "MoE shared up shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->shared_down, weights->shared_down_count,
                        shared_matrix, "MoE shared down shape/data mismatch");
    KQ_MOE_CHECK_WEIGHT(weights->shared_gate_weight,
                        weights->shared_gate_weight_count, d->hidden_size,
                        "MoE shared scale gate shape/data mismatch");
#undef KQ_MOE_CHECK_WEIGHT
    return KQ_STATUS_OK;
}

static int kq_moe_weights_overlap_writable(
    const kq_moe_weights_f32 *weights,
    const void *output, uint64_t output_bytes,
    const void *scratch, uint64_t scratch_bytes) {
    const float *pointers[8];
    uint64_t counts[8];
    uint32_t index;
    pointers[0] = weights->router; counts[0] = weights->router_count;
    pointers[1] = weights->routed_gate; counts[1] = weights->routed_gate_count;
    pointers[2] = weights->routed_up; counts[2] = weights->routed_up_count;
    pointers[3] = weights->routed_down; counts[3] = weights->routed_down_count;
    pointers[4] = weights->shared_gate; counts[4] = weights->shared_gate_count;
    pointers[5] = weights->shared_up; counts[5] = weights->shared_up_count;
    pointers[6] = weights->shared_down; counts[6] = weights->shared_down_count;
    pointers[7] = weights->shared_gate_weight;
    counts[7] = weights->shared_gate_weight_count;
    for (index = 0U; index < 8U; ++index) {
        uint64_t bytes;
        if (!kq_moe_u64_mul(counts[index], sizeof(float), &bytes) ||
            kq_moe_ranges_overlap(pointers[index], bytes,
                                  output, output_bytes) ||
            kq_moe_ranges_overlap(pointers[index], bytes,
                                  scratch, scratch_bytes)) return 1;
    }
    return 0;
}

static void *kq_moe_take(unsigned char **cursor, uint64_t bytes) {
    void *result = *cursor;
    *cursor += bytes;
    return result;
}

static void kq_moe_workspace_init(const kq_moe_config *config,
                                  void *scratch,
                                  kq_moe_workspace *workspace) {
    const kq_moe_dimensions *d = &config->dimensions;
    unsigned char *cursor = (unsigned char *)scratch;
    workspace->router_logits = (float *)kq_moe_take(
        &cursor, (uint64_t)d->expert_count * sizeof(float));
    workspace->router_probabilities = (float *)kq_moe_take(
        &cursor, (uint64_t)d->expert_count * sizeof(float));
    workspace->selected_ids = (uint32_t *)kq_moe_take(
        &cursor, (uint64_t)d->top_k * sizeof(uint32_t));
    workspace->selected_weights = (float *)kq_moe_take(
        &cursor, (uint64_t)d->top_k * sizeof(float));
    workspace->expert_gate = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    workspace->expert_up = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    workspace->expert_activated = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    workspace->expert_output = (float *)kq_moe_take(
        &cursor, (uint64_t)d->hidden_size * sizeof(float));
    workspace->weighted_output = (float *)kq_moe_take(
        &cursor, (uint64_t)d->hidden_size * sizeof(float));
    workspace->routed_sum = (float *)kq_moe_take(
        &cursor, (uint64_t)d->hidden_size * sizeof(float));
    workspace->shared_gate = (float *)kq_moe_take(
        &cursor, (uint64_t)d->shared_intermediate_size * sizeof(float));
    workspace->shared_up = (float *)kq_moe_take(
        &cursor, (uint64_t)d->shared_intermediate_size * sizeof(float));
    workspace->shared_activated = (float *)kq_moe_take(
        &cursor, (uint64_t)d->shared_intermediate_size * sizeof(float));
    workspace->shared_output = (float *)kq_moe_take(
        &cursor, (uint64_t)d->hidden_size * sizeof(float));
    workspace->gated_shared = (float *)kq_moe_take(
        &cursor, (uint64_t)d->hidden_size * sizeof(float));
}

static kq_status kq_moe_project(const float *weights, uint64_t rows,
                                uint64_t columns, const float *input,
                                float *output,
                                kq_diagnostic *diagnostic) {
    uint64_t row;
    for (row = 0U; row < rows; ++row) {
        kq_status status = kq_f32_dot(weights + row * columns, input,
                                      columns, &output[row], diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    return KQ_STATUS_OK;
}

static void kq_moe_select_top_k(const float *probabilities,
                                uint32_t expert_count, uint32_t top_k,
                                uint32_t *selected_ids) {
    uint32_t index;
    for (index = 0U; index < top_k; ++index) selected_ids[index] = index;
    for (index = top_k; index < expert_count; ++index) {
        uint32_t position;
        uint32_t worst = 0U;
        for (position = 1U; position < top_k; ++position) {
            uint32_t candidate_id = selected_ids[position];
            uint32_t worst_id = selected_ids[worst];
            if (probabilities[candidate_id] < probabilities[worst_id] ||
                (probabilities[candidate_id] == probabilities[worst_id] &&
                 candidate_id < worst_id)) {
                worst = position;
            }
        }
        if (probabilities[index] > probabilities[selected_ids[worst]]) {
            selected_ids[worst] = index;
        }
    }
    for (index = 1U; index < top_k; ++index) {
        uint32_t value = selected_ids[index];
        uint32_t position = index;
        while (position > 0U) {
            uint32_t previous = selected_ids[position - 1U];
            if (probabilities[previous] > probabilities[value] ||
                (probabilities[previous] == probabilities[value] &&
                 previous < value)) break;
            selected_ids[position] = previous;
            position -= 1U;
        }
        selected_ids[position] = value;
    }
}

kq_status kq_moe_route_f32(
    const kq_moe_config *config, const float *router_weight,
    uint64_t router_weight_count, const float *hidden_token,
    float *router_logits, uint64_t router_logits_capacity,
    float *router_probabilities, uint64_t router_probabilities_capacity,
    uint32_t *selected_expert_ids, uint64_t selected_ids_capacity,
    float *selected_weights, uint64_t selected_weights_capacity,
    kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d;
    uint64_t expected_router;
    uint64_t weight_bytes;
    uint64_t hidden_bytes;
    uint64_t expert_bytes;
    uint64_t top_float_bytes;
    uint64_t top_id_bytes;
    float selected_sum = 0.0f;
    uint32_t position;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (!kq_moe_config_valid(config) || router_weight == NULL ||
        hidden_token == NULL || router_logits == NULL ||
        router_probabilities == NULL || selected_expert_ids == NULL ||
        selected_weights == NULL) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "complete MoE route arguments are required");
    }
    d = &config->dimensions;
    if (d->activation_dtype != KQ_MOE_ACTIVATION_F32) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                                "scalar routing requires an F32 MoE reference config");
    }
    if (!kq_moe_u64_mul(d->expert_count, d->hidden_size, &expected_router) ||
        !kq_moe_u64_mul(expected_router, sizeof(float), &weight_bytes) ||
        !kq_moe_u64_mul(d->hidden_size, sizeof(float), &hidden_bytes) ||
        !kq_moe_u64_mul(d->expert_count, sizeof(float), &expert_bytes) ||
        !kq_moe_u64_mul(d->top_k, sizeof(float), &top_float_bytes) ||
        !kq_moe_u64_mul(d->top_k, sizeof(uint32_t), &top_id_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "MoE route byte geometry overflows");
    }
    if (router_weight_count != expected_router) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_DIMENSION_MISMATCH,
                                "MoE router weight shape mismatch");
    }
    if (router_logits_capacity < d->expert_count ||
        router_probabilities_capacity < d->expert_count ||
        selected_ids_capacity < d->top_k ||
        selected_weights_capacity < d->top_k) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "MoE route output capacity is too small");
    }
    if (kq_moe_ranges_overlap(router_weight, weight_bytes,
                              router_logits, expert_bytes) ||
        kq_moe_ranges_overlap(router_weight, weight_bytes,
                              router_probabilities, expert_bytes) ||
        kq_moe_ranges_overlap(router_weight, weight_bytes,
                              selected_expert_ids, top_id_bytes) ||
        kq_moe_ranges_overlap(router_weight, weight_bytes,
                              selected_weights, top_float_bytes) ||
        kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              router_logits, expert_bytes) ||
        kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              router_probabilities, expert_bytes) ||
        kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              selected_expert_ids, top_id_bytes) ||
        kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              selected_weights, top_float_bytes) ||
        kq_moe_ranges_overlap(router_logits, expert_bytes,
                              router_probabilities, expert_bytes) ||
        kq_moe_ranges_overlap(router_logits, expert_bytes,
                              selected_expert_ids, top_id_bytes) ||
        kq_moe_ranges_overlap(router_logits, expert_bytes,
                              selected_weights, top_float_bytes) ||
        kq_moe_ranges_overlap(router_probabilities, expert_bytes,
                              selected_expert_ids, top_id_bytes) ||
        kq_moe_ranges_overlap(router_probabilities, expert_bytes,
                              selected_weights, top_float_bytes) ||
        kq_moe_ranges_overlap(selected_expert_ids, top_id_bytes,
                              selected_weights, top_float_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "MoE route inputs and outputs must not overlap");
    }
    status = kq_moe_validate_finite(router_weight, expected_router,
                                    "MoE router weights must be finite",
                                    diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_moe_validate_finite(hidden_token, d->hidden_size,
                                        "MoE router input must be finite",
                                        diagnostic);
    }
    if (status != KQ_STATUS_OK) return status;
    for (position = 0U; position < d->expert_count; ++position) {
        status = kq_f32_dot(
            router_weight + (uint64_t)position * d->hidden_size,
            hidden_token, d->hidden_size, &router_logits[position],
            diagnostic);
        if (status != KQ_STATUS_OK) return status;
    }
    status = kq_f32_softmax(router_logits, d->expert_count,
                            router_probabilities, diagnostic);
    if (status != KQ_STATUS_OK) return status;
    kq_moe_select_top_k(router_probabilities, d->expert_count, d->top_k,
                        selected_expert_ids);
    for (position = 0U; position < d->top_k; ++position) {
        selected_weights[position] =
            router_probabilities[selected_expert_ids[position]];
        selected_sum = selected_sum + selected_weights[position];
    }
    if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "MoE selected routing weight sum is invalid");
    }
    for (position = 0U; position < d->top_k; ++position) {
        selected_weights[position] = selected_weights[position] / selected_sum;
        if (!isfinite(selected_weights[position])) {
            return kq_moe_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "MoE selected routing weight is invalid");
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_moe_expert(const kq_moe_config *config,
                               const kq_moe_weights_f32 *weights,
                               uint32_t expert_id, const float *input,
                               float *gate, float *up, float *activated,
                               float *output,
                               kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d = &config->dimensions;
    uint64_t matrix = (uint64_t)d->routed_intermediate_size * d->hidden_size;
    uint64_t down_matrix = (uint64_t)d->hidden_size *
        d->routed_intermediate_size;
    const float *gate_weight = weights->routed_gate +
        (uint64_t)expert_id * matrix;
    const float *up_weight = weights->routed_up +
        (uint64_t)expert_id * matrix;
    const float *down_weight = weights->routed_down +
        (uint64_t)expert_id * down_matrix;
    kq_status status = kq_moe_project(
        gate_weight, d->routed_intermediate_size, d->hidden_size,
        input, gate, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_moe_project(up_weight, d->routed_intermediate_size,
                                d->hidden_size, input, up, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_f32_swiglu(gate, up, d->routed_intermediate_size,
                               activated, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_moe_project(down_weight, d->hidden_size,
                                d->routed_intermediate_size, activated,
                                output, diagnostic);
    }
    return status;
}

static kq_status kq_moe_expert_member(
    const kq_moe_config *config,
    const kq_moe_routed_expert_weights_f32 *weights,
    const float *input, float *gate, float *up, float *activated,
    float *output, kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d = &config->dimensions;
    kq_status status = kq_moe_project(
        weights->gate, d->routed_intermediate_size, d->hidden_size,
        input, gate, diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_moe_project(weights->up, d->routed_intermediate_size,
                                d->hidden_size, input, up, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_f32_swiglu(gate, up, d->routed_intermediate_size,
                               activated, diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_moe_project(weights->down, d->hidden_size,
                                d->routed_intermediate_size, activated,
                                output, diagnostic);
    }
    return status;
}

kq_status kq_moe_execute_routed_expert_f32(
    const kq_moe_config *config,
    const kq_moe_routed_expert_weights_f32 *weights,
    uint32_t expert_id, const float *hidden_token, float *output_token,
    uint64_t output_capacity, void *scratch, uint64_t scratch_bytes,
    kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d;
    unsigned char *cursor;
    float *gate;
    float *up;
    float *activated;
    float *temporary_output;
    uint64_t hidden_bytes;
    uint64_t member_matrix;
    uint64_t down_matrix;
    uint64_t member_bytes;
    uint64_t down_bytes;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (!kq_moe_config_valid(config) || weights == NULL ||
        hidden_token == NULL || output_token == NULL || scratch == NULL) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "complete routed-expert arguments are required");
    }
    d = &config->dimensions;
    if (d->activation_dtype != KQ_MOE_ACTIVATION_F32) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                                "scalar expert execution requires F32 config");
    }
    if (expert_id >= d->expert_count) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "routed expert ID is out of range");
    }
    if (output_capacity < d->hidden_size ||
        scratch_bytes < config->one_expert_workspace_bytes ||
        ((uintptr_t)scratch % _Alignof(float)) != 0U) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "routed expert output or scratch is too small");
    }
    if (!kq_moe_u64_mul(d->routed_intermediate_size, d->hidden_size,
                        &member_matrix) ||
        !kq_moe_u64_mul(d->hidden_size, d->routed_intermediate_size,
                        &down_matrix) ||
        !kq_moe_u64_mul(member_matrix, sizeof(float), &member_bytes) ||
        !kq_moe_u64_mul(down_matrix, sizeof(float), &down_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "routed expert member geometry overflows");
    }
    status = kq_moe_weight(weights->gate, weights->gate_count, member_matrix,
                           "MoE routed member gate shape/data mismatch",
                           diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_moe_weight(weights->up, weights->up_count, member_matrix,
                               "MoE routed member up shape/data mismatch",
                               diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_moe_weight(weights->down, weights->down_count, down_matrix,
                               "MoE routed member down shape/data mismatch",
                               diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_moe_validate_finite(hidden_token, d->hidden_size,
                                        "MoE expert input must be finite",
                                        diagnostic);
    }
    if (status != KQ_STATUS_OK) return status;
    hidden_bytes = (uint64_t)d->hidden_size * sizeof(float);
    if (kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              output_token, hidden_bytes) ||
        kq_moe_ranges_overlap(hidden_token, hidden_bytes,
                              scratch, scratch_bytes) ||
        kq_moe_ranges_overlap(output_token, hidden_bytes,
                              scratch, scratch_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "routed expert buffers must not overlap");
    }
    if (kq_moe_ranges_overlap(weights->gate, member_bytes,
                              output_token, hidden_bytes) ||
        kq_moe_ranges_overlap(weights->gate, member_bytes,
                              scratch, scratch_bytes) ||
        kq_moe_ranges_overlap(weights->up, member_bytes,
                              output_token, hidden_bytes) ||
        kq_moe_ranges_overlap(weights->up, member_bytes,
                              scratch, scratch_bytes) ||
        kq_moe_ranges_overlap(weights->down, down_bytes,
                              output_token, hidden_bytes) ||
        kq_moe_ranges_overlap(weights->down, down_bytes,
                              scratch, scratch_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "MoE weights must not overlap writable expert buffers");
    }
    cursor = (unsigned char *)scratch;
    gate = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    up = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    activated = (float *)kq_moe_take(
        &cursor, (uint64_t)d->routed_intermediate_size * sizeof(float));
    temporary_output = (float *)kq_moe_take(&cursor, hidden_bytes);
    status = kq_moe_expert_member(config, weights, hidden_token,
                                  gate, up, activated, temporary_output,
                                  diagnostic);
    if (status == KQ_STATUS_OK) {
        memcpy(output_token, temporary_output, (size_t)hidden_bytes);
    }
    return status;
}

static void kq_moe_emit(kq_moe_checkpoint_observer observer,
                        void *user_data, kq_moe_checkpoint_kind kind,
                        uint64_t token, uint32_t expert_id,
                        uint32_t top_k_position, uint32_t rank,
                        uint64_t first, uint64_t second,
                        const float *values, uint64_t count) {
    kq_moe_checkpoint checkpoint;
    if (observer == NULL) return;
    memset(&checkpoint, 0, sizeof(checkpoint));
    checkpoint.kind = kind;
    checkpoint.token_index = token;
    checkpoint.expert_id = expert_id;
    checkpoint.top_k_position = top_k_position;
    checkpoint.rank = rank;
    checkpoint.dimensions[0] = first;
    checkpoint.dimensions[1] = second;
    checkpoint.values = values;
    checkpoint.value_count = count;
    observer(&checkpoint, user_data);
}

static int kq_moe_find_selected(const uint32_t *selected,
                                uint32_t top_k, uint32_t expert,
                                uint32_t *position) {
    uint32_t index;
    for (index = 0U; index < top_k; ++index) {
        if (selected[index] == expert) {
            *position = index;
            return 1;
        }
    }
    return 0;
}

kq_status kq_moe_execute_f32(
    const kq_moe_config *config, const kq_moe_weights_f32 *weights,
    const float *hidden_states, uint64_t token_count, float *output,
    uint64_t output_capacity, void *scratch, uint64_t scratch_bytes,
    kq_moe_route_observer route_observer,
    kq_moe_checkpoint_observer checkpoint_observer,
    void *observer_user_data, kq_diagnostic *diagnostic) {
    const kq_moe_dimensions *d;
    kq_moe_workspace w;
    uint64_t value_count;
    uint64_t value_bytes;
    uint64_t token;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (!kq_moe_config_valid(config) || weights == NULL ||
        hidden_states == NULL || output == NULL || scratch == NULL) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "complete MoE execution arguments are required");
    }
    d = &config->dimensions;
    if (d->activation_dtype != KQ_MOE_ACTIVATION_F32) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_MOE,
                                "scalar execution requires an F32 MoE config");
    }
    if (token_count == 0U ||
        !kq_moe_u64_mul(token_count, d->hidden_size, &value_count) ||
        !kq_moe_u64_mul(value_count, sizeof(float), &value_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_LIMIT_EXCEEDED,
                                "MoE token/value count is zero or overflows");
    }
    if (output_capacity < value_count || scratch_bytes < config->scratch_bytes ||
        ((uintptr_t)scratch % _Alignof(float)) != 0U) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "MoE output or aligned scratch is too small");
    }
    status = kq_moe_validate_finite(hidden_states, value_count,
                                    "MoE input must be finite", diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_moe_validate_weights(config, weights, diagnostic);
    }
    if (status != KQ_STATUS_OK) return status;
    if (kq_moe_ranges_overlap(hidden_states, value_bytes,
                              output, value_bytes) ||
        kq_moe_ranges_overlap(hidden_states, value_bytes,
                              scratch, scratch_bytes) ||
        kq_moe_ranges_overlap(output, value_bytes,
                              scratch, scratch_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "MoE input, output, and scratch must not overlap");
    }
    if (kq_moe_weights_overlap_writable(
            weights, output, value_bytes, scratch, scratch_bytes)) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "MoE weights must not overlap writable execution buffers");
    }
    if (fegetround() != FE_TONEAREST) {
        return kq_moe_exec_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "MoE scalar reference requires round-to-nearest");
    }
    kq_moe_workspace_init(config, scratch, &w);
    for (token = 0U; token < token_count; ++token) {
        const float *input = hidden_states + token * d->hidden_size;
        float *token_output = output + token * d->hidden_size;
        uint32_t expert;
        uint64_t feature;
        float shared_scale_logit = 0.0f;
        float shared_scale = 0.0f;

        status = kq_moe_route_f32(
            config, weights->router, weights->router_count, input,
            w.router_logits, d->expert_count, w.router_probabilities,
            d->expert_count, w.selected_ids, d->top_k,
            w.selected_weights, d->top_k, diagnostic);
        if (status != KQ_STATUS_OK) return status;
        if (route_observer != NULL) {
            kq_moe_route route;
            route.token_index = token;
            route.router_logits = w.router_logits;
            route.router_probabilities = w.router_probabilities;
            route.expert_count = d->expert_count;
            route.selected_expert_ids = w.selected_ids;
            route.selected_weights = w.selected_weights;
            route.selected_count = d->top_k;
            route_observer(&route, observer_user_data);
        }
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_ROUTER_LOGITS, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->expert_count, 0U,
                    w.router_logits, d->expert_count);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_ROUTER_PROBABILITIES, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->expert_count, 0U,
                    w.router_probabilities, d->expert_count);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SELECTED_WEIGHTS, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->top_k, 0U,
                    w.selected_weights, d->top_k);
        memset(w.routed_sum, 0, (size_t)d->hidden_size * sizeof(float));
        for (expert = 0U; expert < d->expert_count; ++expert) {
            uint32_t top_k_position = 0U;
            if (!kq_moe_find_selected(w.selected_ids, d->top_k, expert,
                                      &top_k_position)) continue;
            status = kq_moe_expert(config, weights, expert, input,
                                   w.expert_gate, w.expert_up,
                                   w.expert_activated, w.expert_output,
                                   diagnostic);
            if (status != KQ_STATUS_OK) return status;
            kq_moe_emit(checkpoint_observer, observer_user_data,
                        KQ_MOE_CHECKPOINT_ROUTED_GATE, token, expert,
                        top_k_position, 1U, d->routed_intermediate_size, 0U,
                        w.expert_gate, d->routed_intermediate_size);
            kq_moe_emit(checkpoint_observer, observer_user_data,
                        KQ_MOE_CHECKPOINT_ROUTED_UP, token, expert,
                        top_k_position, 1U, d->routed_intermediate_size, 0U,
                        w.expert_up, d->routed_intermediate_size);
            kq_moe_emit(checkpoint_observer, observer_user_data,
                        KQ_MOE_CHECKPOINT_ROUTED_ACTIVATED, token, expert,
                        top_k_position, 1U, d->routed_intermediate_size, 0U,
                        w.expert_activated, d->routed_intermediate_size);
            kq_moe_emit(checkpoint_observer, observer_user_data,
                        KQ_MOE_CHECKPOINT_ROUTED_EXPERT_OUTPUT, token, expert,
                        top_k_position, 1U, d->hidden_size, 0U,
                        w.expert_output, d->hidden_size);
            status = kq_f32_scale(w.expert_output, d->hidden_size,
                                  w.selected_weights[top_k_position],
                                  w.weighted_output, diagnostic);
            if (status != KQ_STATUS_OK) return status;
            kq_moe_emit(checkpoint_observer, observer_user_data,
                        KQ_MOE_CHECKPOINT_ROUTED_WEIGHTED_OUTPUT, token, expert,
                        top_k_position, 1U, d->hidden_size, 0U,
                        w.weighted_output, d->hidden_size);
            for (feature = 0U; feature < d->hidden_size; ++feature) {
                w.routed_sum[feature] =
                    w.routed_sum[feature] + w.weighted_output[feature];
                if (!isfinite(w.routed_sum[feature])) {
                    return kq_moe_exec_fail(
                        diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                        "MoE routed accumulation is non-finite");
                }
            }
        }
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_ROUTED_WEIGHTED_SUM, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->hidden_size, 0U,
                    w.routed_sum, d->hidden_size);

        status = kq_moe_project(weights->shared_gate,
                                d->shared_intermediate_size, d->hidden_size,
                                input, w.shared_gate, diagnostic);
        if (status == KQ_STATUS_OK) {
            status = kq_moe_project(weights->shared_up,
                                    d->shared_intermediate_size,
                                    d->hidden_size, input, w.shared_up,
                                    diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_f32_swiglu(w.shared_gate, w.shared_up,
                                   d->shared_intermediate_size,
                                   w.shared_activated, diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_moe_project(weights->shared_down, d->hidden_size,
                                    d->shared_intermediate_size,
                                    w.shared_activated, w.shared_output,
                                    diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_f32_dot(weights->shared_gate_weight, input,
                                d->hidden_size, &shared_scale_logit,
                                diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_f32_sigmoid(&shared_scale_logit, 1U,
                                    &shared_scale, diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_f32_scale(w.shared_output, d->hidden_size,
                                  shared_scale, w.gated_shared, diagnostic);
        }
        if (status == KQ_STATUS_OK) {
            status = kq_f32_add(w.routed_sum, w.gated_shared,
                                d->hidden_size, token_output, diagnostic);
        }
        if (status != KQ_STATUS_OK) return status;
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_GATE_PROJECTION, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U,
                    d->shared_intermediate_size, 0U,
                    w.shared_gate, d->shared_intermediate_size);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_UP_PROJECTION, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U,
                    d->shared_intermediate_size, 0U,
                    w.shared_up, d->shared_intermediate_size);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_ACTIVATED, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U,
                    d->shared_intermediate_size, 0U,
                    w.shared_activated, d->shared_intermediate_size);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_OUTPUT, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->hidden_size, 0U,
                    w.shared_output, d->hidden_size);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_SCALE_LOGIT, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, 1U, 0U,
                    &shared_scale_logit, 1U);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_SHARED_SCALE, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, 1U, 0U,
                    &shared_scale, 1U);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_GATED_SHARED_OUTPUT, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->hidden_size, 0U,
                    w.gated_shared, d->hidden_size);
        kq_moe_emit(checkpoint_observer, observer_user_data,
                    KQ_MOE_CHECKPOINT_OPERATOR_OUTPUT, token,
                    KQ_MOE_NO_EXPERT, UINT32_MAX, 1U, d->hidden_size, 0U,
                    token_output, d->hidden_size);
    }
    return KQ_STATUS_OK;
}

const char *kq_moe_checkpoint_kind_name(kq_moe_checkpoint_kind kind) {
    static const char *const names[] = {
        "ROUTER_LOGITS", "ROUTER_PROBABILITIES", "SELECTED_WEIGHTS",
        "ROUTED_GATE", "ROUTED_UP", "ROUTED_ACTIVATED",
        "ROUTED_EXPERT_OUTPUT", "ROUTED_WEIGHTED_OUTPUT",
        "ROUTED_WEIGHTED_SUM", "SHARED_GATE_PROJECTION",
        "SHARED_UP_PROJECTION", "SHARED_ACTIVATED", "SHARED_OUTPUT",
        "SHARED_SCALE_LOGIT", "SHARED_SCALE", "GATED_SHARED_OUTPUT",
        "OPERATOR_OUTPUT"
    };
    return (uint32_t)kind < sizeof(names) / sizeof(names[0]) ?
        names[(uint32_t)kind] : "INVALID";
}
