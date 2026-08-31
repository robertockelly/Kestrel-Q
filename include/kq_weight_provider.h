#ifndef KQ_WEIGHT_PROVIDER_H
#define KQ_WEIGHT_PROVIDER_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_ple_value.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_WEIGHT_PROVIDER_NO_EXPERT UINT32_MAX

typedef struct kq_weight_provider kq_weight_provider;

typedef struct kq_weight_provider_metrics {
    uint64_t payload_budget_bytes;
    uint64_t logical_payload_bytes_touched;
    uint64_t quantized_blocks_touched;
    uint64_t linear_requests;
    uint64_t row_requests;
    uint64_t vector_requests;
    uint64_t selected_expert_requests;
    uint64_t ple_row_requests;
    uint64_t unique_semantic_tensors_touched;
    uint64_t maximum_f32_weight_bytes_materialized;
    uint64_t provider_owned_bytes;
    uint64_t linear_elapsed_nanoseconds;
} kq_weight_provider_metrics;

typedef struct kq_weight_provider_expert_request {
    uint32_t layer_id;
    uint32_t expert_id;
} kq_weight_provider_expert_request;

typedef struct kq_weight_provider_ple_request {
    uint32_t logical_member;
    uint64_t member_row;
} kq_weight_provider_ple_request;

/* The GGUF/model/file owners must outlive the provider and every call. */
kq_status kq_weight_provider_open(
    const kq_gguf *gguf, const kq_model *model, uint64_t payload_budget_bytes,
    kq_weight_provider **out_provider, kq_diagnostic *diagnostic);
void kq_weight_provider_close(kq_weight_provider *provider);

/* Structural only: views may be mapped, but payload bytes are never read. */
kq_status kq_weight_provider_preflight_layer(
    const kq_weight_provider *provider, uint32_t layer_id,
    kq_diagnostic *diagnostic);

/* Semantic matrix-vector operation. Output is committed only on success. */
kq_status kq_weight_provider_linear_f32(
    kq_weight_provider *provider, const kq_semantic_tensor *semantic,
    kq_binding_part_role part_role, uint32_t expert_id,
    uint64_t rows, uint64_t columns, const float *input,
    float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes, kq_diagnostic *diagnostic);

/* Decode exactly one canonical row. Only entry/output row semantics qualify. */
kq_status kq_weight_provider_row_f32(
    kq_weight_provider *provider, const kq_semantic_tensor *semantic,
    uint64_t row_index, uint64_t columns, float *output,
    uint64_t output_capacity, void *scratch, uint64_t scratch_bytes,
    kq_diagnostic *diagnostic);

/* Bounded small-vector materialization. Complete matrices are rejected. */
kq_status kq_weight_provider_vector_f32(
    kq_weight_provider *provider, const kq_semantic_tensor *semantic,
    uint64_t element_count, float *output, uint64_t output_capacity,
    void *scratch, uint64_t scratch_bytes, kq_diagnostic *diagnostic);

/* Exact fused-PLE row access through the same accounting boundary. */
kq_ple_value_lookup_provider kq_weight_provider_ple_lookup_interface(
    kq_weight_provider *provider);

const kq_weight_provider_metrics *kq_weight_provider_get_metrics(
    const kq_weight_provider *provider);

kq_status kq_weight_provider_copy_expert_requests(
    const kq_weight_provider *provider,
    kq_weight_provider_expert_request *requests, uint64_t capacity,
    uint64_t *required_count);
kq_status kq_weight_provider_copy_route_requests(
    const kq_weight_provider *provider,
    kq_weight_provider_expert_request *requests, uint64_t capacity,
    uint64_t *required_count);
kq_status kq_weight_provider_copy_ple_requests(
    const kq_weight_provider *provider,
    kq_weight_provider_ple_request *requests, uint64_t capacity,
    uint64_t *required_count);

#ifdef __cplusplus
}
#endif

#endif
