#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_weight_provider_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_numeric.h"
#include "kq_tensor_view.h"

#define KQ_GDN_KEY_HEADS 16U
#define KQ_GDN_VALUE_HEADS 48U
#define KQ_GDN_HEAD_WIDTH 128U
#define KQ_GDN_VALUE_REPEAT 3U

static kq_status fail(kq_diagnostic *d, kq_status s, const char *m) {
    kq_diagnostic_set(d, s, "%s", m);
    return s;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || UINT64_MAX - a < b) return 0;
    *out = a + b;
    return 1;
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL || (a != 0U && b > UINT64_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static int qpc_elapsed_nanoseconds(
    LARGE_INTEGER started,
    LARGE_INTEGER finished,
    LARGE_INTEGER frequency,
    uint64_t *out) {
    uint64_t ticks;
    uint64_t hz;
    uint64_t whole;
    uint64_t remainder;
    uint64_t whole_ns;
    uint64_t remainder_scaled;
    uint64_t fraction_ns;

    if (out == NULL || frequency.QuadPart <= 0 ||
        finished.QuadPart < started.QuadPart)
        return 0;
    ticks = (uint64_t)(finished.QuadPart - started.QuadPart);
    hz = (uint64_t)frequency.QuadPart;
    whole = ticks / hz;
    remainder = ticks % hz;
    if (!mul_u64(whole, UINT64_C(1000000000), &whole_ns) ||
        !mul_u64(remainder, UINT64_C(1000000000), &remainder_scaled))
        return 0;
    fraction_ns = remainder_scaled / hz;
    return add_u64(whole_ns, fraction_ns, out);
}

static int ranges_overlap(const void *left, uint64_t left_bytes,
                          const void *right, uint64_t right_bytes) {
    uintptr_t left_begin;
    uintptr_t right_begin;
    uintptr_t left_end;
    uintptr_t right_end;
    if (left == NULL || right == NULL || left_bytes == 0U || right_bytes == 0U)
        return 0;
    left_begin = (uintptr_t)left;
    right_begin = (uintptr_t)right;
    if (left_bytes > UINTPTR_MAX - left_begin ||
        right_bytes > UINTPTR_MAX - right_begin)
        return 1;
    left_end = left_begin + (uintptr_t)left_bytes;
    right_end = right_begin + (uintptr_t)right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

static int provider_valid(const kq_weight_provider *p) {
    return p != NULL && p->magic == KQ_WEIGHT_PROVIDER_MAGIC &&
           p->gguf != NULL && p->model != NULL;
}

static int semantic_owned(const kq_weight_provider *p,
                          const kq_semantic_tensor *s) {
    const kq_semantic_tensor *found;
    if (!provider_valid(p) || s == NULL) return 0;
    found = kq_model_find_semantic_tensor(p->model, s->semantic_id);
    return found == s;
}

static kq_status account(kq_weight_provider *p,
                         const kq_semantic_tensor *s,
                         uint64_t bytes, uint64_t blocks,
                         uint64_t materialized, kq_diagnostic *d) {
    uint64_t next_bytes;
    uint64_t next_blocks;
    uint32_t i;
    int known = 0;
    if (!add_u64(p->metrics.logical_payload_bytes_touched, bytes, &next_bytes))
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider payload accounting overflows");
    if (next_bytes > p->metrics.payload_budget_bytes)
        return fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                    "weight-provider payload budget would be exceeded");
    if (!add_u64(p->metrics.quantized_blocks_touched, blocks, &next_blocks))
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider block accounting overflows");
    for (i = 0U; i < p->touched_count; ++i)
        if (p->touched[i] == s) known = 1;
    if (!known && p->touched_count >= KQ_WEIGHT_PROVIDER_MAX_SEMANTICS)
        return fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                    "weight-provider semantic accounting capacity exceeded");
    p->metrics.logical_payload_bytes_touched = next_bytes;
    p->metrics.quantized_blocks_touched = next_blocks;
    if (materialized > p->metrics.maximum_f32_weight_bytes_materialized)
        p->metrics.maximum_f32_weight_bytes_materialized = materialized;
    if (!known) p->touched[p->touched_count++] = s;
    p->metrics.unique_semantic_tensors_touched = p->touched_count;
    return KQ_STATUS_OK;
}

static int budget_allows(const kq_weight_provider *p, uint64_t bytes) {
    return provider_valid(p) &&
           p->metrics.logical_payload_bytes_touched <=
               p->metrics.payload_budget_bytes &&
           bytes <= p->metrics.payload_budget_bytes -
                    p->metrics.logical_payload_bytes_touched;
}

static int binding_for_role(const kq_semantic_tensor *s,
                            kq_binding_part_role role, uint32_t *index) {
    uint32_t i;
    if (s == NULL || index == NULL) return 0;
    for (i = 0U; i < s->binding_count; ++i) {
        if (s->bindings[i].part_role == role) {
            *index = i;
            return 1;
        }
    }
    return 0;
}

static uint64_t grouped_to_tiled(uint64_t canonical, uint64_t width) {
    uint64_t group = canonical / (KQ_GDN_VALUE_REPEAT * width);
    uint64_t within = canonical % (KQ_GDN_VALUE_REPEAT * width);
    uint64_t repeat = within / width;
    uint64_t feature = within % width;
    return (repeat * KQ_GDN_KEY_HEADS + group) * width + feature;
}

static int role_reorders_rows(kq_semantic_role role) {
    return role == KQ_ROLE_GDN_ALPHA || role == KQ_ROLE_GDN_BETA ||
           role == KQ_ROLE_GDN_GATE;
}

static uint64_t physical_row_for(const kq_semantic_tensor *s,
                                 uint64_t canonical_row) {
    if (s->role == KQ_ROLE_GDN_QKV) {
        uint64_t qk = UINT64_C(2) * KQ_GDN_KEY_HEADS * KQ_GDN_HEAD_WIDTH;
        return canonical_row < qk ? canonical_row :
            qk + grouped_to_tiled(canonical_row - qk, KQ_GDN_HEAD_WIDTH);
    }
    if (role_reorders_rows(s->role))
        return grouped_to_tiled(canonical_row,
            s->role == KQ_ROLE_GDN_GATE ? KQ_GDN_HEAD_WIDTH : 1U);
    if (s->role == KQ_ROLE_GDN_CONV) {
        uint64_t qk = UINT64_C(2) * KQ_GDN_KEY_HEADS * KQ_GDN_HEAD_WIDTH;
        return canonical_row < qk ? canonical_row :
            qk + grouped_to_tiled(canonical_row - qk, KQ_GDN_HEAD_WIDTH);
    }
    return canonical_row;
}

static kq_status open_request_view(kq_weight_provider *p,
                                   const kq_semantic_tensor *s,
                                   kq_binding_part_role role,
                                   uint32_t expert,
                                   kq_tensor_view **view,
                                   kq_diagnostic *d) {
    uint32_t binding;
    if (!binding_for_role(s, role, &binding))
        return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                    "requested semantic binding part is absent");
    if (expert != KQ_WEIGHT_PROVIDER_NO_EXPERT) {
        if (s->expert_count == 0U || expert >= s->expert_count)
            return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                        "selected expert ID is outside the semantic stack");
        return kq_tensor_view_open_expert_member(
            p->gguf, s, binding, expert, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            view, d);
    }
    if (s->expert_count != 0U)
        return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                    "routed expert stack requires an explicit expert ID");
    return kq_tensor_view_open_binding(
        p->gguf, s, binding, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, view, d);
}

static kq_status validate_matrix_request(const kq_tensor_view_info *info,
                                         uint64_t rows, uint64_t columns,
                                         uint64_t *row_bytes,
                                         kq_diagnostic *d) {
    uint64_t elements;
    if (info == NULL || rows == 0U || columns == 0U ||
        !mul_u64(rows, columns, &elements) ||
        elements != info->requested_element_count)
        return fail(d, KQ_STATUS_DIMENSION_MISMATCH,
                    "semantic linear request does not match physical member geometry");
    if (columns % info->block_elements != 0U ||
        !mul_u64(columns / info->block_elements,
                 info->bytes_per_block, row_bytes))
        return fail(d, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                    "semantic row is not independently block aligned");
    return KQ_STATUS_OK;
}

kq_status kq_weight_provider_open(const kq_gguf *gguf,
    const kq_model *model, uint64_t budget, kq_weight_provider **out,
    kq_diagnostic *d) {
    kq_weight_provider *p;
    if (gguf == NULL || model == NULL || budget == 0U || out == NULL)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                    "valid GGUF, model, budget, and output are required");
    *out = NULL;
    if (kq_model_semantic_tensor_count(model) != 1294U ||
        kq_model_physical_coverage_count(model) != 1224U ||
        kq_model_unknown_physical_count(model) != 0U ||
        kq_model_unbound_required_count(model) != 0U)
        return fail(d, KQ_STATUS_MODEL_TOPOLOGY_MISMATCH,
                    "weight provider requires the verified target semantic registry");
    p = (kq_weight_provider *)calloc(1U, sizeof(*p));
    if (p == NULL) return fail(d, KQ_STATUS_OUT_OF_MEMORY,
                               "could not allocate weight provider");
    p->magic = KQ_WEIGHT_PROVIDER_MAGIC;
    p->gguf = gguf;
    p->model = model;
    p->metrics.payload_budget_bytes = budget;
    p->metrics.provider_owned_bytes = sizeof(*p);
    *out = p;
    return KQ_STATUS_OK;
}

void kq_weight_provider_close(kq_weight_provider *p) {
    if (p != NULL) {
        p->magic = 0U;
        free(p);
    }
}

kq_status kq_weight_provider_linear_f32(kq_weight_provider *p,
    const kq_semantic_tensor *s, kq_binding_part_role role, uint32_t expert,
    uint64_t rows, uint64_t columns, const float *input, float *output,
    uint64_t output_capacity, void *scratch, uint64_t scratch_bytes,
    kq_diagnostic *d) {
    kq_tensor_view *view = NULL;
    const kq_tensor_view_info *info;
    const unsigned char *data;
    const float *activation = input;
    float *permuted = (float *)scratch;
    float *staged;
    uint64_t row_bytes, row, output_bytes, activation_bytes = 0U;
    uint64_t required_scratch;
    uint64_t blocks, touched_blocks, touched_bytes;
    LARGE_INTEGER frequency = {0};
    LARGE_INTEGER started = {0};
    LARGE_INTEGER finished = {0};
    kq_status status;
    if (!semantic_owned(p, s) || input == NULL || output == NULL ||
        scratch == NULL || rows == 0U || columns == 0U)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                    "invalid semantic linear request");
    if (p->metrics.linear_requests == UINT64_MAX ||
        (expert != KQ_WEIGHT_PROVIDER_NO_EXPERT &&
         p->metrics.selected_expert_requests == UINT64_MAX))
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider request counter overflows");
    if (!mul_u64(rows, sizeof(float), &output_bytes) ||
        !mul_u64(columns, sizeof(float), &activation_bytes) ||
        !add_u64(output_bytes, activation_bytes, &required_scratch))
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider linear byte count overflows");
    if (output_capacity < rows || scratch_bytes < required_scratch)
        return fail(d, KQ_STATUS_BUFFER_TOO_SMALL,
                    "weight-provider linear scratch is too small");
    if (ranges_overlap(input, activation_bytes, output, output_bytes) ||
        ranges_overlap(input, activation_bytes, scratch, required_scratch) ||
        ranges_overlap(output, output_bytes, scratch, required_scratch))
        return fail(d, KQ_STATUS_ALIASING_VIOLATION,
                    "weight-provider linear buffers must not overlap");
    staged = (float *)((unsigned char *)scratch + activation_bytes);
    status = open_request_view(p, s, role, expert, &view, d);
    if (status != KQ_STATUS_OK) return status;
    info = kq_tensor_view_get_info(view);
    status = validate_matrix_request(info, rows, columns, &row_bytes, d);
    if (status != KQ_STATUS_OK) goto done;
    blocks = columns / info->block_elements;
    if (!mul_u64(rows, row_bytes, &touched_bytes) ||
        !mul_u64(rows, blocks, &touched_blocks)) {
        status = fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                      "linear payload accounting overflows");
        goto done;
    }
    if (!budget_allows(p, touched_bytes)) {
        status = fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                      "weight-provider payload budget would be exceeded");
        goto done;
    }
    if (s->role == KQ_ROLE_GDN_OUT) {
        uint64_t i;
        for (i = 0U; i < columns; ++i) {
            uint64_t physical_column =
                grouped_to_tiled(i, KQ_GDN_HEAD_WIDTH);
            if (physical_column >= columns) {
                status = fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                              "GDN output permutation exceeds the row width");
                goto done;
            }
            permuted[physical_column] = input[i];
        }
        activation = permuted;
    }
    data = kq_tensor_view_physical_data(view);
    if (data == NULL) {
        status = fail(d, KQ_STATUS_FILE_MAP_FAILED,
                      "weight-provider physical view has no mapped data");
        goto done;
    }
    frequency.QuadPart = 0;
    started.QuadPart = 0;
    finished.QuadPart = 0;
    if (QueryPerformanceFrequency(&frequency) == 0 ||
        QueryPerformanceCounter(&started) == 0)
        frequency.QuadPart = 0;
    for (row = 0U; row < rows; ++row) {
        uint64_t physical_row = physical_row_for(s, row);
        if (physical_row >= rows) {
            status = fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                          "semantic row transform exceeds physical geometry");
            goto done;
        }
        status = kq_quantized_row_dot_f32(
            info->type_id, data + physical_row * row_bytes, row_bytes,
            activation, columns, &staged[row], d);
        if (status != KQ_STATUS_OK) goto done;
    }
    if (frequency.QuadPart > 0 && QueryPerformanceCounter(&finished) == 0)
        frequency.QuadPart = 0;
    status = account(p, s, touched_bytes, touched_blocks, output_bytes, d);
    if (status != KQ_STATUS_OK) goto done;
    memcpy(output, staged, (size_t)output_bytes);
    p->metrics.linear_requests += 1U;
    if (frequency.QuadPart > 0) {
        uint64_t elapsed;
        uint64_t total;
        if (qpc_elapsed_nanoseconds(started, finished, frequency, &elapsed) &&
            add_u64(p->metrics.linear_elapsed_nanoseconds, elapsed, &total))
            p->metrics.linear_elapsed_nanoseconds = total;
    }
    if (expert != KQ_WEIGHT_PROVIDER_NO_EXPERT) {
        p->metrics.selected_expert_requests += 1U;
        if (role == KQ_BINDING_PART_GATE &&
            p->expert_trace_count < KQ_WEIGHT_PROVIDER_EXPERT_TRACE_CAPACITY) {
            p->expert_trace[p->expert_trace_count].layer_id = s->layer_id;
            p->expert_trace[p->expert_trace_count].expert_id = expert;
            p->expert_trace_count += 1U;
        }
    }
done:
    kq_tensor_view_close(view);
    return status;
}

kq_status kq_weight_provider_vector_f32(kq_weight_provider *p,
    const kq_semantic_tensor *s, uint64_t count, float *output,
    uint64_t capacity, void *scratch, uint64_t scratch_bytes,
    kq_diagnostic *d) {
    kq_tensor_view *view = NULL;
    const kq_tensor_view_info *info;
    float *physical = (float *)scratch;
    uint64_t bytes, out_count = 0U, blocks, i;
    kq_status status;
    if (!semantic_owned(p, s) || output == NULL || scratch == NULL ||
        count == 0U)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                    "invalid semantic vector request");
    if (count > KQ_WEIGHT_PROVIDER_VECTOR_LIMIT)
        return fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                    "semantic vector request exceeds its bounded limit");
    if (p->metrics.vector_requests == UINT64_MAX)
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider vector counter overflows");
    if (!mul_u64(count, sizeof(float), &bytes))
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "semantic vector byte count overflows");
    if (capacity < count || scratch_bytes < bytes)
        return fail(d, KQ_STATUS_BUFFER_TOO_SMALL,
                    "semantic vector output or scratch is too small");
    if (ranges_overlap(output, bytes, scratch, bytes))
        return fail(d, KQ_STATUS_ALIASING_VIOLATION,
                    "semantic vector output and scratch must not overlap");
    if (s->canonical_rank != 1U && s->role != KQ_ROLE_GDN_CONV &&
        s->role != KQ_ROLE_PLE_CONV)
        return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                    "complete matrix materialization is forbidden");
    if ((s->role == KQ_ROLE_GDN_CONV || s->role == KQ_ROLE_PLE_CONV) &&
        count % 4U != 0U)
        return fail(d, KQ_STATUS_DIMENSION_MISMATCH,
                    "convolution vector is not divisible by kernel width");
    status = open_request_view(p, s, KQ_BINDING_PART_WHOLE,
                               KQ_WEIGHT_PROVIDER_NO_EXPERT, &view, d);
    if (status != KQ_STATUS_OK) return status;
    info = kq_tensor_view_get_info(view);
    if (info == NULL || info->requested_element_count != count) {
        status = fail(d, KQ_STATUS_DIMENSION_MISMATCH,
                      "semantic vector count does not match physical payload");
        goto done;
    }
    blocks = info->block_count;
    if (!budget_allows(p, info->mapped_logical_length)) {
        status = fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                      "weight-provider payload budget would be exceeded");
        goto done;
    }
    status = kq_dequantize_view_blocks_f32(
        view, 0U, blocks, KQ_NUMERIC_PHYSICAL_ORDER, physical, count,
        &out_count, d);
    if (status != KQ_STATUS_OK || out_count != count) {
        if (status == KQ_STATUS_OK)
            status = fail(d, KQ_STATUS_DIMENSION_MISMATCH,
                          "semantic vector decode count mismatch");
        goto done;
    }
    if (s->role == KQ_ROLE_GDN_A_LOG) {
        for (i = 0U; i < count; ++i) {
            uint64_t physical_index = grouped_to_tiled(i, 1U);
            float stored;
            if (physical_index >= count) {
                status = fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                              "GDN decay transform exceeds physical geometry");
                goto done;
            }
            stored = physical[physical_index];
            if (!(stored < 0.0f) || !isfinite(stored)) {
                status = fail(d, KQ_STATUS_NUMERIC_DOMAIN,
                              "stored GDN decay base must be finite and negative");
                goto done;
            }
        }
    } else if (s->role == KQ_ROLE_GDN_DT_BIAS) {
        for (i = 0U; i < count; ++i)
            if (grouped_to_tiled(i, 1U) >= count) {
                status = fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                              "GDN bias transform exceeds physical geometry");
                goto done;
            }
    } else if (s->role == KQ_ROLE_GDN_CONV) {
        uint64_t channels = count / 4U;
        for (i = 0U; i < channels; ++i) {
            uint64_t physical_row = physical_row_for(s, i);
            if (physical_row >= channels) {
                status = fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                              "GDN convolution transform exceeds physical geometry");
                goto done;
            }
        }
    }
    status = account(p, s, info->mapped_logical_length, blocks, bytes, d);
    if (status == KQ_STATUS_OK) {
        if (s->role == KQ_ROLE_GDN_A_LOG) {
            for (i = 0U; i < count; ++i)
                output[i] = logf(-physical[grouped_to_tiled(i, 1U)]);
        } else if (s->role == KQ_ROLE_GDN_DT_BIAS) {
            for (i = 0U; i < count; ++i)
                output[i] = physical[grouped_to_tiled(i, 1U)];
        } else if (s->role == KQ_ROLE_GDN_CONV) {
            uint64_t channels = count / 4U;
            for (i = 0U; i < channels; ++i) {
                uint64_t physical_row = physical_row_for(s, i);
                memcpy(output + i * 4U, physical + physical_row * 4U,
                       4U * sizeof(float));
            }
        } else if (s->role == KQ_ROLE_PLE_NORM_KEY ||
                   s->role == KQ_ROLE_PLE_NORM_QUERY ||
                   s->role == KQ_ROLE_PLE_NORM_CONV) {
            for (i = 0U; i < count; ++i) output[i] = physical[i] - 1.0f;
        } else {
            memcpy(output, physical, (size_t)bytes);
        }
        p->metrics.vector_requests += 1U;
    }
done:
    kq_tensor_view_close(view);
    return status;
}

static kq_status ple_lookup(void *user, uint32_t member, uint64_t row,
                            float *output, uint64_t capacity,
                            kq_diagnostic *d) {
    kq_weight_provider *p = (kq_weight_provider *)user;
    const kq_semantic_tensor *s;
    kq_tensor_view *view = NULL;
    const kq_tensor_view_info *info;
    float staged[160];
    char id[KQ_SEMANTIC_ID_CAPACITY];
    uint64_t first_block, block_count, out_count = 0U, row_bytes;
    kq_status status;
    if (!provider_valid(p) || member >= 128U || row >= UINT64_C(2500012) ||
        output == NULL || capacity < 160U)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT, "invalid PLE row request");
    if (p->metrics.ple_row_requests == UINT64_MAX)
        return fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                    "weight-provider PLE request counter overflows");
    (void)snprintf(id, sizeof(id), "layer.01.ple.table.%03u", member);
    s = kq_model_find_semantic_tensor(p->model, id);
    if (s == NULL) return fail(d, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                               "PLE table semantic is absent");
    status = kq_tensor_view_open_ple_member(
        p->gguf, s, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, &view, d);
    if (status != KQ_STATUS_OK) return status;
    info = kq_tensor_view_get_info(view);
    if (info == NULL || !mul_u64(row, 5U, &first_block)) {
        status = fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                      "PLE row block offset overflows");
        kq_tensor_view_close(view);
        return status;
    }
    block_count = 5U;
    if (!mul_u64(block_count, info->bytes_per_block, &row_bytes)) {
        status = fail(d, KQ_STATUS_ARITHMETIC_OVERFLOW,
                      "PLE row byte count overflows");
        kq_tensor_view_close(view);
        return status;
    }
    if (!budget_allows(p, row_bytes)) {
        status = fail(d, KQ_STATUS_LIMIT_EXCEEDED,
                      "weight-provider payload budget would be exceeded");
        kq_tensor_view_close(view);
        return status;
    }
    status = kq_dequantize_view_blocks_f32(
        view, first_block, block_count, KQ_NUMERIC_PHYSICAL_ORDER,
        staged, 160U, &out_count, d);
    if (status == KQ_STATUS_OK && out_count != 160U)
        status = fail(d, KQ_STATUS_DIMENSION_MISMATCH,
                      "PLE row decode count mismatch");
    if (status == KQ_STATUS_OK)
        status = account(p, s, row_bytes, block_count,
                         UINT64_C(160) * sizeof(float), d);
    if (status == KQ_STATUS_OK) {
        memcpy(output, staged, sizeof(staged));
        p->metrics.ple_row_requests += 1U;
        if (p->ple_trace_count < KQ_WEIGHT_PROVIDER_PLE_TRACE_CAPACITY) {
            p->ple_trace[p->ple_trace_count].logical_member = member;
            p->ple_trace[p->ple_trace_count].member_row = row;
            p->ple_trace_count += 1U;
        }
    }
    kq_tensor_view_close(view);
    return status;
}

kq_ple_value_lookup_provider kq_weight_provider_ple_lookup_interface(
    kq_weight_provider *p) {
    kq_ple_value_lookup_provider out;
    memset(&out, 0, sizeof(out));
    if (provider_valid(p)) {
        out.user_data = p;
        out.lookup_row = ple_lookup;
        out.logical_member_count = 128U;
        out.member_rows = UINT64_C(2500012);
        out.row_width = 160U;
    }
    return out;
}

const kq_weight_provider_metrics *kq_weight_provider_get_metrics(
    const kq_weight_provider *p) {
    return provider_valid(p) ? &p->metrics : NULL;
}

kq_status kq_weight_provider_copy_expert_requests(
    const kq_weight_provider *p, kq_weight_provider_expert_request *requests,
    uint64_t capacity, uint64_t *required_count) {
    if (!provider_valid(p) || required_count == NULL)
        return KQ_STATUS_INVALID_ARGUMENT;
    *required_count = p->expert_trace_count;
    if (capacity < p->expert_trace_count ||
        (p->expert_trace_count != 0U && requests == NULL))
        return KQ_STATUS_BUFFER_TOO_SMALL;
    if (p->expert_trace_count != 0U)
        memcpy(requests, p->expert_trace,
               (size_t)p->expert_trace_count * sizeof(*requests));
    return KQ_STATUS_OK;
}

kq_status kq_weight_provider_copy_route_requests(
    const kq_weight_provider *p, kq_weight_provider_expert_request *requests,
    uint64_t capacity, uint64_t *required_count) {
    if (!provider_valid(p) || required_count == NULL)
        return KQ_STATUS_INVALID_ARGUMENT;
    *required_count = p->route_trace_count;
    if (capacity < p->route_trace_count ||
        (p->route_trace_count != 0U && requests == NULL))
        return KQ_STATUS_BUFFER_TOO_SMALL;
    if (p->route_trace_count != 0U)
        memcpy(requests, p->route_trace,
               (size_t)p->route_trace_count * sizeof(*requests));
    return KQ_STATUS_OK;
}

kq_status kq_weight_provider_record_route(
    kq_weight_provider *p, uint32_t layer_id, const uint32_t *expert_ids,
    uint32_t expert_count, kq_diagnostic *d) {
    uint32_t index;
    if (!provider_valid(p) || expert_ids == NULL || expert_count == 0U ||
        layer_id >= 48U)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                    "invalid weight-provider route record");
    for (index = 0U; index < expert_count; ++index) {
        if (expert_ids[index] >= 512U)
            return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                        "weight-provider route contains an invalid expert");
        if (p->route_trace_count < KQ_WEIGHT_PROVIDER_EXPERT_TRACE_CAPACITY) {
            p->route_trace[p->route_trace_count].layer_id = layer_id;
            p->route_trace[p->route_trace_count].expert_id = expert_ids[index];
            p->route_trace_count += 1U;
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_weight_provider_copy_ple_requests(
    const kq_weight_provider *p, kq_weight_provider_ple_request *requests,
    uint64_t capacity, uint64_t *required_count) {
    if (!provider_valid(p) || required_count == NULL)
        return KQ_STATUS_INVALID_ARGUMENT;
    *required_count = p->ple_trace_count;
    if (capacity < p->ple_trace_count ||
        (p->ple_trace_count != 0U && requests == NULL))
        return KQ_STATUS_BUFFER_TOO_SMALL;
    if (p->ple_trace_count != 0U)
        memcpy(requests, p->ple_trace,
               (size_t)p->ple_trace_count * sizeof(*requests));
    return KQ_STATUS_OK;
}

static int supported_semantic(const kq_semantic_tensor *s) {
    if (s == NULL || s->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT ||
        s->binding_count == 0U || s->binding_count > 2U) return 0;
    switch (s->component) {
        case KQ_COMPONENT_GATED_RESIDUAL:
        case KQ_COMPONENT_GDN:
        case KQ_COMPONENT_QSA_ATTENTION:
        case KQ_COMPONENT_QSA_INDEXER:
        case KQ_COMPONENT_MOE_ROUTER:
        case KQ_COMPONENT_ROUTED_EXPERT_STACK:
        case KQ_COMPONENT_SHARED_EXPERT:
        case KQ_COMPONENT_PLE_TABLE:
        case KQ_COMPONENT_PLE_DENSE:
            return 1;
        default:
            return 0;
    }
}

kq_status kq_weight_provider_preflight_layer(const kq_weight_provider *p,
    uint32_t layer, kq_diagnostic *d) {
    uint64_t i;
    uint32_t found = 0U;
    if (!provider_valid(p) || layer >= 48U)
        return fail(d, KQ_STATUS_INVALID_ARGUMENT,
                    "invalid weight-provider layer preflight request");
    for (i = 0U; i < kq_model_semantic_tensor_count(p->model); ++i) {
        const kq_semantic_tensor *s = kq_model_semantic_tensor_at(p->model, i);
        uint32_t b;
        if (s == NULL || s->layer_id != layer ||
            s->runtime_scope != KQ_SCOPE_REQUIRED_INITIAL_TEXT) continue;
        if (s->component == KQ_COMPONENT_PLE_ADDRESS_METADATA) continue;
        if (!supported_semantic(s))
            return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                        "layer contains an unsupported semantic role");
        for (b = 0U; b < s->binding_count; ++b) {
            kq_quant_geometry geometry;
            kq_tensor_view *probe = NULL;
            kq_status status;
            if (s->bindings[b].physical == NULL ||
                kq_quant_geometry_for_type(s->bindings[b].physical->type_id,
                                           &geometry, d) != KQ_STATUS_OK)
                return fail(d, KQ_STATUS_UNSUPPORTED_TENSOR_TYPE,
                            "layer binding uses an unsupported physical type");
            if (s->expert_count != 0U) {
                status = kq_tensor_view_open_expert_member(
                    p->gguf, s, b, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
                    &probe, d);
                if (status == KQ_STATUS_OK) kq_tensor_view_close(probe);
                probe = NULL;
                if (status == KQ_STATUS_OK)
                    status = kq_tensor_view_open_expert_member(
                        p->gguf, s, b, s->expert_count - 1U,
                        KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, &probe, d);
                if (status == KQ_STATUS_OK) kq_tensor_view_close(probe);
                if (status != KQ_STATUS_OK)
                    return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                                "expert-stack endpoint view is invalid");
            } else if (s->component == KQ_COMPONENT_PLE_TABLE) {
                status = kq_tensor_view_open_ple_member(
                    p->gguf, s, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, &probe, d);
                if (status == KQ_STATUS_OK) kq_tensor_view_close(probe);
                if (status != KQ_STATUS_OK)
                    return fail(d, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                                "fused PLE member view is invalid");
            }
        }
        found += 1U;
    }
    if (found == 0U)
        return fail(d, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                    "layer preflight found no executable semantics");
    return KQ_STATUS_OK;
}
