#include "kq_moe.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_moe_internal.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", (message)); \
        result = 1; \
        goto cleanup; \
    } \
} while (0)

typedef struct route_capture {
    uint64_t calls;
    uint32_t ids[10];
    float weights[10];
} route_capture;

static void capture_route(const kq_moe_route *route, void *user_data) {
    route_capture *capture = (route_capture *)user_data;
    uint32_t index;
    capture->calls += 1U;
    for (index = 0U; index < route->selected_count && index < 10U; ++index) {
        capture->ids[index] = route->selected_expert_ids[index];
        capture->weights[index] = route->selected_weights[index];
    }
}

static void fill_target_source(kq_moe_semantic_source *source,
                               kq_semantic_tensor semantics[7],
                               kq_gguf_tensor physical[8]) {
    static const kq_semantic_component components[7] = {
        KQ_COMPONENT_MOE_ROUTER, KQ_COMPONENT_ROUTED_EXPERT_STACK,
        KQ_COMPONENT_ROUTED_EXPERT_STACK, KQ_COMPONENT_SHARED_EXPERT,
        KQ_COMPONENT_SHARED_EXPERT, KQ_COMPONENT_SHARED_EXPERT,
        KQ_COMPONENT_SHARED_EXPERT
    };
    static const kq_semantic_role roles[7] = {
        KQ_ROLE_MOE_ROUTER, KQ_ROLE_ROUTED_DOWN, KQ_ROLE_ROUTED_GATE_UP,
        KQ_ROLE_SHARED_DOWN, KQ_ROLE_SHARED_GATE, KQ_ROLE_SHARED_UP,
        KQ_ROLE_SHARED_GATE_WEIGHT
    };
    static const uint32_t ranks[7] = {2U, 3U, 3U, 2U, 2U, 2U, 2U};
    static const uint64_t canonical[7][3] = {
        {512U,2560U,0U}, {512U,2560U,640U}, {512U,1280U,2560U},
        {2560U,640U,0U}, {640U,2560U,0U}, {640U,2560U,0U},
        {1U,2560U,0U}
    };
    static const uint64_t packed[7][3] = {
        {2560U,512U,0U}, {640U,2560U,512U}, {2560U,640U,512U},
        {640U,2560U,0U}, {2560U,640U,0U}, {2560U,640U,0U},
        {2560U,0U,0U}
    };
    static const uint32_t types[7] = {
        KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_Q8_0, KQ_GGUF_TYPE_Q4_K,
        KQ_GGUF_TYPE_Q8_0, KQ_GGUF_TYPE_Q8_0, KQ_GGUF_TYPE_Q8_0,
        KQ_GGUF_TYPE_F32
    };
    uint32_t index;
    memset(source, 0, sizeof(*source));
    memset(semantics, 0, 7U * sizeof(*semantics));
    memset(physical, 0, 8U * sizeof(*physical));
    source->model = (const kq_model *)(uintptr_t)1U;
    source->layer_id = 2U;
    source->layer_type = KQ_MODEL_LAYER_GDN;
    source->dimensions.hidden_size = 2560U;
    source->dimensions.expert_count = 512U;
    source->dimensions.top_k = 10U;
    source->dimensions.routed_intermediate_size = 640U;
    source->dimensions.shared_intermediate_size = 640U;
    source->dimensions.activation_dtype = KQ_MOE_ACTIVATION_BF16;
    for (index = 0U; index < 7U; ++index) {
        uint32_t dim;
        semantics[index].component = components[index];
        semantics[index].role = roles[index];
        semantics[index].relation = index == 2U ?
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL :
            KQ_BINDING_RENAMED_ONE_TO_ONE;
        semantics[index].runtime_scope = KQ_SCOPE_REQUIRED_INITIAL_TEXT;
        semantics[index].layer_type = KQ_MODEL_LAYER_GDN;
        semantics[index].layer_id = 2U;
        semantics[index].canonical_dtype = KQ_CANONICAL_DTYPE_BF16;
        semantics[index].canonical_rank = ranks[index];
        for (dim = 0U; dim < ranks[index]; ++dim) {
            semantics[index].canonical_dimensions[dim] = canonical[index][dim];
        }
        semantics[index].binding_count = index == 2U ? 2U : 1U;
        if (index == 1U || index == 2U) {
            semantics[index].canonical_expert_axis = 0U;
            semantics[index].expert_count = 512U;
        }
        physical[index].type_id = types[index];
        physical[index].rank = index == 6U ? 1U : ranks[index];
        for (dim = 0U; dim < physical[index].rank; ++dim) {
            physical[index].dimensions[dim] = packed[index][dim];
        }
        semantics[index].bindings[0].physical = &physical[index];
        semantics[index].bindings[0].part_role = index == 2U ?
            KQ_BINDING_PART_GATE : KQ_BINDING_PART_WHOLE;
        semantics[index].bindings[0].part_index = 0U;
        semantics[index].bindings[0].part_count = index == 2U ? 2U : 1U;
        if (index == 1U || index == 2U) {
            semantics[index].bindings[0].physical_expert_axis = 2U;
        }
        source->tensors[index] = &semantics[index];
    }
    physical[7] = physical[2];
    semantics[2].bindings[1].physical = &physical[7];
    semantics[2].bindings[1].part_role = KQ_BINDING_PART_UP;
    semantics[2].bindings[1].part_index = 1U;
    semantics[2].bindings[1].part_count = 2U;
    semantics[2].bindings[1].physical_expert_axis = 2U;
}

int main(void) {
    kq_moe_dimensions d;
    kq_moe_config *config = NULL;
    kq_moe_config *large = NULL;
    kq_moe_config *target = NULL;
    kq_moe_semantic_source source;
    kq_semantic_tensor semantics[7];
    kq_gguf_tensor physical[8];
    kq_moe_weights_f32 weights;
    kq_moe_routed_expert_weights_f32 member_weights;
    kq_diagnostic diagnostic;
    kq_status status;
    float router[24] = {0};
    float routed_gate[96] = {0};
    float routed_up[96] = {0};
    float routed_down[96] = {0};
    float shared_gate[24] = {0};
    float shared_up[24] = {0};
    float shared_down[24] = {0};
    float shared_gate_weight[6] = {0};
    float input[12] = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
                       0.0f,1.0f,0.0f,0.0f,0.0f,0.0f};
    float output[12];
    float expert_output[6];
    float logits[64];
    float probabilities[64];
    float selected_weights[10];
    uint32_t selected_ids[10];
    float router64[64] = {0};
    float input1[1] = {1.0f};
    void *scratch = NULL;
    route_capture capture;
    uint64_t scratch_bytes;
    int result = 0;
    uint32_t index;

    memset(&d, 0, sizeof(d));
    d.hidden_size = 6U; d.expert_count = 4U; d.top_k = 2U;
    d.routed_intermediate_size = 4U; d.shared_intermediate_size = 4U;
    d.activation_dtype = KQ_MOE_ACTIVATION_F32;
    status = kq_moe_test_config_create(&d, &config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create reduced MoE config");
    CHECK(kq_moe_config_hidden_size(config) == 6U &&
          kq_moe_config_expert_count(config) == 4U &&
          kq_moe_config_top_k(config) == 2U,
          "reduced config queries");
    weights.router = router; weights.router_count = 24U;
    weights.routed_gate = routed_gate; weights.routed_gate_count = 96U;
    weights.routed_up = routed_up; weights.routed_up_count = 96U;
    weights.routed_down = routed_down; weights.routed_down_count = 96U;
    weights.shared_gate = shared_gate; weights.shared_gate_count = 24U;
    weights.shared_up = shared_up; weights.shared_up_count = 24U;
    weights.shared_down = shared_down; weights.shared_down_count = 24U;
    weights.shared_gate_weight = shared_gate_weight;
    weights.shared_gate_weight_count = 6U;
    member_weights.gate = routed_gate; member_weights.gate_count = 24U;
    member_weights.up = routed_up; member_weights.up_count = 24U;
    member_weights.down = routed_down; member_weights.down_count = 24U;
    routed_gate[0] = 1.0f; routed_up[0] = 1.0f; routed_down[0] = 1.0f;
    scratch_bytes = kq_moe_config_scratch_bytes(config);
    scratch = calloc(1U, (size_t)scratch_bytes);
    CHECK(scratch != NULL, "allocate reduced MoE scratch");
    memset(&capture, 0, sizeof(capture));
    status = kq_moe_execute_f32(config, &weights, input, 2U, output, 12U,
                                scratch, scratch_bytes, capture_route, NULL,
                                &capture, &diagnostic);
    CHECK(status == KQ_STATUS_OK && capture.calls == 2U,
          "execute reduced MoE and observe routes");
    CHECK(capture.ids[0] == 0U && capture.ids[1] == 1U &&
          capture.weights[0] == 0.5f && capture.weights[1] == 0.5f,
          "equal reduced logits have pinned exact routing");
    for (index = 0U; index < 12U; ++index) {
        CHECK(isfinite(output[index]), "reduced MoE output is finite");
    }
    status = kq_moe_execute_routed_expert_f32(
        config, &member_weights, 0U, input, expert_output, 6U, scratch,
        kq_moe_config_one_expert_workspace_bytes(config), &diagnostic);
    CHECK(status == KQ_STATUS_OK, "execute one selected expert");
    status = kq_moe_execute_routed_expert_f32(
        config, &member_weights, 4U, input, expert_output, 6U, scratch,
        kq_moe_config_one_expert_workspace_bytes(config), &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "out-of-range expert fails closed");
    member_weights.gate_count = 23U;
    status = kq_moe_execute_routed_expert_f32(
        config, &member_weights, 0U, input, expert_output, 6U, scratch,
        kq_moe_config_one_expert_workspace_bytes(config), &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "single-member API rejects a malformed member without a stack");
    member_weights.gate_count = 24U;
    status = kq_moe_execute_f32(config, &weights, input, 2U, output, 11U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "short output fails closed");
    weights.router_count = 23U;
    status = kq_moe_execute_f32(config, &weights, input, 1U, output, 6U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong router shape fails closed");
    weights.router_count = 24U;
    router[0] = NAN;
    status = kq_moe_execute_f32(config, &weights, input, 1U, output, 6U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_NUMERIC_DOMAIN,
          "non-finite router data fails closed");
    router[0] = 0.0f;
    input[0] = NAN;
    status = kq_moe_execute_f32(config, &weights, input, 1U, output, 6U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_NUMERIC_DOMAIN,
          "non-finite input fails closed");
    input[0] = 1.0f;
    status = kq_moe_execute_f32(config, &weights, input, 1U, input, 6U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "input/output alias fails closed");
    status = kq_moe_execute_f32(config, &weights, input, 1U, router, 6U,
                                scratch, scratch_bytes, NULL, NULL, NULL,
                                &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "weight/output alias fails closed");
    status = kq_moe_execute_f32(config, &weights, input, UINT64_MAX,
                                output, 12U, scratch, scratch_bytes,
                                NULL, NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_LIMIT_EXCEEDED,
          "token count overflow fails closed");

    memset(&d, 0, sizeof(d));
    d.hidden_size = 1U; d.expert_count = 64U; d.top_k = 10U;
    d.routed_intermediate_size = 1U; d.shared_intermediate_size = 1U;
    d.activation_dtype = KQ_MOE_ACTIVATION_F32;
    status = kq_moe_test_config_create(&d, &large, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create partial-selection routing config");
    router64[63] = 1.0f;
    status = kq_moe_route_f32(large, router64, 64U, input1, logits, 64U,
                              probabilities, 64U, selected_ids, 10U,
                              selected_weights, 10U, &diagnostic);
    CHECK(status == KQ_STATUS_OK && selected_ids[0] == 63U,
          "dominant late expert selected first");
    for (index = 1U; index < 10U; ++index) {
        CHECK(selected_ids[index] == index,
              "pinned partial-selection tie behavior retained");
    }
    status = kq_moe_route_f32(large, router64, 64U, input1, logits, 64U,
                              probabilities, 64U, (uint32_t *)logits, 10U,
                              selected_weights, 10U, &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "overlapping route outputs fail closed");

    fill_target_source(&source, semantics, physical);
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_OK, "valid target binding source");
    kq_moe_config_close(target); target = NULL;
    source.dimensions.top_k = 9U;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_MOE && target == NULL,
          "wrong target top-k fails closed");
    source.dimensions.top_k = 10U;
    source.dimensions.expert_count = 511U;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_MOE && target == NULL,
          "wrong target expert count fails closed");
    source.dimensions.expert_count = 512U;
    source.tensors[0] = NULL;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED && target == NULL,
          "missing router semantic fails closed");
    source.tensors[0] = &semantics[0];
    source.tensors[1] = NULL;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED && target == NULL,
          "missing routed stack fails closed");
    source.tensors[1] = &semantics[1];
    source.tensors[3] = NULL;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED && target == NULL,
          "missing shared expert fails closed");
    source.tensors[3] = &semantics[3];
    source.tensors[6] = NULL;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED && target == NULL,
          "missing shared gate fails closed");
    source.tensors[6] = &semantics[6];
    semantics[1].canonical_expert_axis = 1U;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_MOE && target == NULL,
          "wrong expert axis fails closed");
    semantics[1].canonical_expert_axis = 0U;
    semantics[2].bindings[0].part_role = KQ_BINDING_PART_UP;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH && target == NULL,
          "broken gate/up split fails closed");
    semantics[2].bindings[0].part_role = KQ_BINDING_PART_GATE;
    physical[3].dimensions[0] = 639U;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH && target == NULL,
          "wrong shared tensor shape fails closed");
    physical[3].dimensions[0] = 640U;
    physical[0].type_id = KQ_GGUF_TYPE_Q8_0;
    status = kq_moe_config_create_from_source(&source, 1, &target,
                                               &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH && target == NULL,
          "wrong router physical type fails closed");

    printf("MoE synthetic tests: PASS; routing=exact_discrete, "
           "reduced_full_path=PASS, fail_closed=PASS\n");
cleanup:
    free(scratch);
    kq_moe_config_close(target);
    kq_moe_config_close(large);
    kq_moe_config_close(config);
    return result;
}
