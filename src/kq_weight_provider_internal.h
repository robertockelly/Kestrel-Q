#ifndef KQ_WEIGHT_PROVIDER_INTERNAL_H
#define KQ_WEIGHT_PROVIDER_INTERNAL_H

#include "kq_weight_provider.h"

#define KQ_WEIGHT_PROVIDER_MAGIC UINT32_C(0x4b515750)
#define KQ_WEIGHT_PROVIDER_MAX_SEMANTICS 2048U
#define KQ_WEIGHT_PROVIDER_VECTOR_LIMIT 65536U
#define KQ_WEIGHT_PROVIDER_EXPERT_TRACE_CAPACITY 64U
#define KQ_WEIGHT_PROVIDER_PLE_TRACE_CAPACITY 64U

struct kq_weight_provider {
    uint32_t magic;
    const kq_gguf *gguf;
    const kq_model *model;
    kq_weight_provider_metrics metrics;
    const kq_semantic_tensor *touched[KQ_WEIGHT_PROVIDER_MAX_SEMANTICS];
    uint32_t touched_count;
    kq_weight_provider_expert_request
        expert_trace[KQ_WEIGHT_PROVIDER_EXPERT_TRACE_CAPACITY];
    uint32_t expert_trace_count;
    kq_weight_provider_expert_request
        route_trace[KQ_WEIGHT_PROVIDER_EXPERT_TRACE_CAPACITY];
    uint32_t route_trace_count;
    kq_weight_provider_ple_request
        ple_trace[KQ_WEIGHT_PROVIDER_PLE_TRACE_CAPACITY];
    uint32_t ple_trace_count;
    int test_fail_next_request;
};

kq_status kq_weight_provider_record_route(
    kq_weight_provider *provider, uint32_t layer_id,
    const uint32_t *expert_ids, uint32_t expert_count,
    kq_diagnostic *diagnostic);

/* Test-only fault injection used by model-level transaction regressions. */
void kq_weight_provider_test_fail_next_request(kq_weight_provider *provider);

#endif
