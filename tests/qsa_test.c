#include "kq_qsa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_qsa_internal.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", (message)); \
        result = 1; \
        goto cleanup; \
    } \
} while (0)

typedef struct selection_capture {
    uint64_t calls;
    uint64_t candidate_count;
    uint64_t selected_block_count;
    uint64_t selected_token_count;
    uint64_t tail_count;
    uint32_t blocks[8];
    uint32_t tokens[16];
} selection_capture;

static void capture_selection(const kq_qsa_selection *selection,
                              void *user_data) {
    selection_capture *capture = (selection_capture *)user_data;
    uint64_t index;
    capture->calls += 1U;
    capture->candidate_count = selection->candidate_count;
    capture->selected_block_count = selection->selected_block_count;
    capture->selected_token_count = selection->selected_token_count;
    capture->tail_count = selection->tail_count;
    for (index = 0U; index < selection->selected_block_count && index < 8U;
         ++index) capture->blocks[index] = selection->selected_block_ids[index];
    for (index = 0U; index < selection->selected_token_count && index < 16U;
         ++index) capture->tokens[index] = selection->selected_token_positions[index];
}

static void fill_dimensions(kq_qsa_dimensions *d) {
    memset(d, 0, sizeof(*d));
    d->hidden_size = 8U;
    d->query_head_count = 2U;
    d->key_value_head_count = 1U;
    d->head_dimension = 4U;
    d->index_query_head_count = 2U;
    d->index_head_dimension = 4U;
    d->block_size = 4U;
    d->token_budget = 8U;
    d->max_context = 64U;
    d->rotary_dimension = 2U;
    d->rms_norm_epsilon = 1.0e-6f;
    d->rope_theta = 10000000.0f;
    d->activation_dtype = KQ_QSA_ACTIVATION_F32;
}

static void fill_target_source(kq_qsa_semantic_source *source,
                               kq_semantic_tensor semantics[9],
                               kq_gguf_tensor physical[10]) {
    static const kq_semantic_role roles[9] = {
        KQ_ROLE_QSA_K_NORM, KQ_ROLE_QSA_K, KQ_ROLE_QSA_OUTPUT,
        KQ_ROLE_QSA_Q_NORM, KQ_ROLE_QSA_Q, KQ_ROLE_QSA_V,
        KQ_ROLE_QSA_INDEX_QK, KQ_ROLE_QSA_INDEX_K_NORM,
        KQ_ROLE_QSA_INDEX_Q_NORM
    };
    static const kq_semantic_component components[9] = {
        KQ_COMPONENT_QSA_ATTENTION, KQ_COMPONENT_QSA_ATTENTION,
        KQ_COMPONENT_QSA_ATTENTION, KQ_COMPONENT_QSA_ATTENTION,
        KQ_COMPONENT_QSA_ATTENTION, KQ_COMPONENT_QSA_ATTENTION,
        KQ_COMPONENT_QSA_INDEXER, KQ_COMPONENT_QSA_INDEXER,
        KQ_COMPONENT_QSA_INDEXER
    };
    static const uint32_t ranks[9] = {1U,2U,2U,1U,2U,2U,2U,1U,1U};
    static const uint64_t dims[9][2] = {
        {256U,0U},{512U,2560U},{2560U,6144U},{256U,0U},
        {12288U,2560U},{512U,2560U},{640U,2560U},
        {128U,0U},{128U,0U}
    };
    static const uint32_t types[9] = {
        KQ_GGUF_TYPE_F32,KQ_GGUF_TYPE_Q8_0,KQ_GGUF_TYPE_Q8_0,
        KQ_GGUF_TYPE_F32,KQ_GGUF_TYPE_Q8_0,KQ_GGUF_TYPE_Q8_0,
        KQ_GGUF_TYPE_BF16,KQ_GGUF_TYPE_F32,KQ_GGUF_TYPE_F32
    };
    uint32_t index;
    memset(source, 0, sizeof(*source));
    memset(semantics, 0, 9U * sizeof(*semantics));
    memset(physical, 0, 10U * sizeof(*physical));
    source->model = (const kq_model *)(uintptr_t)1U;
    source->layer_id = 3U;
    source->layer_type = KQ_MODEL_LAYER_QSA;
    source->dimensions.hidden_size = 2560U;
    source->dimensions.query_head_count = 24U;
    source->dimensions.key_value_head_count = 2U;
    source->dimensions.head_dimension = 256U;
    source->dimensions.index_query_head_count = 4U;
    source->dimensions.index_head_dimension = 128U;
    source->dimensions.block_size = 4U;
    source->dimensions.token_budget = 2048U;
    source->dimensions.max_context = 262144U;
    source->dimensions.rotary_dimension = 64U;
    source->dimensions.rms_norm_epsilon = 1.0e-6f;
    source->dimensions.rope_theta = 10000000.0f;
    source->dimensions.activation_dtype = KQ_QSA_ACTIVATION_BF16;
    for (index = 0U; index < 9U; ++index) {
        semantics[index].component = components[index];
        semantics[index].role = roles[index];
        semantics[index].relation = index == 6U ?
            KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL :
            KQ_BINDING_RENAMED_ONE_TO_ONE;
        semantics[index].runtime_scope = KQ_SCOPE_REQUIRED_INITIAL_TEXT;
        semantics[index].layer_type = KQ_MODEL_LAYER_QSA;
        semantics[index].layer_id = 3U;
        semantics[index].canonical_dtype = KQ_CANONICAL_DTYPE_BF16;
        semantics[index].canonical_rank = ranks[index];
        semantics[index].canonical_dimensions[0] = dims[index][0];
        semantics[index].canonical_dimensions[1] = dims[index][1];
        semantics[index].binding_count = index == 6U ? 2U : 1U;
        physical[index].type_id = types[index];
        physical[index].rank = ranks[index];
        physical[index].dimensions[0] = ranks[index] == 2U ?
            dims[index][1] : dims[index][0];
        physical[index].dimensions[1] = ranks[index] == 2U ?
            dims[index][0] : 0U;
        if (index == 6U) physical[index].dimensions[1] = 512U;
        semantics[index].bindings[0].physical = &physical[index];
        semantics[index].bindings[0].part_role = index == 6U ?
            KQ_BINDING_PART_INDEX_QUERY : KQ_BINDING_PART_WHOLE;
        semantics[index].bindings[0].part_index = 0U;
        semantics[index].bindings[0].part_count = index == 6U ? 2U : 1U;
        source->tensors[index] = &semantics[index];
    }
    physical[9].type_id = KQ_GGUF_TYPE_BF16;
    physical[9].rank = 2U;
    physical[9].dimensions[0] = 2560U;
    physical[9].dimensions[1] = 128U;
    semantics[6].bindings[1].physical = &physical[9];
    semantics[6].bindings[1].part_role = KQ_BINDING_PART_INDEX_KEY;
    semantics[6].bindings[1].part_index = 1U;
    semantics[6].bindings[1].part_count = 2U;
}

int main(void) {
    kq_qsa_dimensions dimensions;
    kq_qsa_config *config = NULL;
    kq_qsa_config *other_config = NULL;
    kq_qsa_config *target = NULL;
    kq_qsa_state *state = NULL;
    kq_qsa_state *other_state = NULL;
    kq_qsa_weights_f32 weights;
    kq_qsa_semantic_source source;
    kq_semantic_tensor semantics[9];
    kq_gguf_tensor physical[10];
    kq_diagnostic diagnostic;
    kq_status status;
    float query[128] = {0};
    float key[32] = {0};
    float value[32] = {0};
    float output_weight[64] = {0};
    float norm[4] = {0};
    float index_query[64] = {0};
    float index_key[32] = {0};
    float input[12 * 8] = {0};
    float output[12 * 8];
    float state_key[16 * 4] = {0};
    float state_value[16 * 4] = {0};
    float state_raw[16 * 4] = {0};
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    uint64_t length = 0U;
    selection_capture capture;
    int result = 0;
    uint32_t index;
    float selection_scores[4] = {1.0f, 2.0f, 2.0f, 0.5f};
    uint32_t selected_ids[3] = {99U, 99U, 99U};
    uint64_t selected_count = 0U;

    memset(&weights, 0, sizeof(weights));
    weights.query = query; weights.query_count = 128U;
    weights.key = key; weights.key_count = 32U;
    weights.value = value; weights.value_count = 32U;
    weights.output = output_weight; weights.output_count = 64U;
    weights.query_norm = norm; weights.query_norm_count = 4U;
    weights.key_norm = norm; weights.key_norm_count = 4U;
    weights.index_query = index_query; weights.index_query_count = 64U;
    weights.index_key = index_key; weights.index_key_count = 32U;
    weights.index_query_norm = norm; weights.index_query_norm_count = 4U;
    weights.index_key_norm = norm; weights.index_key_norm_count = 4U;
    fill_dimensions(&dimensions);
    status = kq_qsa_test_config_create(&dimensions, &config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create reduced QSA config");
    status = kq_qsa_state_create(config, 16U, &state, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create bounded QSA state");
    status = kq_qsa_state_create(config, 0U, &other_state, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT && other_state == NULL,
          "zero-capacity QSA state fails closed");
    status = kq_qsa_state_create(config, 65U, &other_state, &diagnostic);
    CHECK(status == KQ_STATUS_LIMIT_EXCEEDED && other_state == NULL,
          "QSA state beyond context limit fails closed");
    status = kq_qsa_select_blocks_f32(
        selection_scores, 4U, 3U, selected_ids, 3U,
        &selected_count, &diagnostic);
    CHECK(status == KQ_STATUS_OK && selected_count == 3U &&
          selected_ids[0] == 1U && selected_ids[1] == 2U &&
          selected_ids[2] == 0U,
          "QSA selection sorts by score then ascending tie ID");
    status = kq_qsa_select_blocks_f32(
        selection_scores, 4U, 3U, selected_ids, 2U,
        &selected_count, &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL && selected_count == 0U,
          "QSA selection capacity fails closed");
    status = kq_qsa_select_blocks_f32(
        selection_scores, 4U, 0U, selected_ids, 3U,
        &selected_count, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "zero QSA selection limit fails closed");
    status = kq_qsa_select_blocks_f32(
        selection_scores, (uint64_t)UINT32_MAX + 1U, 1U,
        selected_ids, 3U, &selected_count, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "QSA candidate count beyond block-ID range fails closed");
    status = kq_qsa_required_scratch_bytes(config, state, 12U,
                                           &scratch_bytes, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "calculate QSA scratch bytes");
    scratch = calloc(1U, (size_t)scratch_bytes);
    CHECK(scratch != NULL, "allocate QSA scratch");
    memset(&capture, 0, sizeof(capture));
    memset(output, 0x5a, sizeof(output));
    status = kq_qsa_prefill_f32(
        config, &weights, input, 12U, output, 96U, state, scratch,
        scratch_bytes, capture_selection, NULL, &capture, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "zero-weight QSA prefill");
    CHECK(kq_qsa_state_length(state) == 12U && capture.calls == 12U,
          "QSA prefill appends exact state length");
    CHECK(capture.candidate_count == 3U &&
          capture.selected_block_count == 2U &&
          capture.blocks[0] == 0U && capture.blocks[1] == 1U &&
          capture.selected_token_count == 8U && capture.tail_count == 0U,
          "QSA tie selection uses ascending block IDs at local limit");
    for (index = 0U; index < 96U; ++index) {
        CHECK(output[index] == 0.0f, "zero weights produce zero QSA output");
    }
    status = kq_qsa_state_export_f32(
        state, state_key, 64U, state_value, 64U, state_raw, 64U,
        &length, &diagnostic);
    CHECK(status == KQ_STATUS_OK && length == 12U,
          "QSA state snapshot exports K/V/index state");
    status = kq_qsa_state_reset(state, &diagnostic);
    CHECK(status == KQ_STATUS_OK && kq_qsa_state_length(state) == 0U,
          "QSA reset clears semantic length");

    status = kq_qsa_required_scratch_bytes(config, state, 8U,
                                           &scratch_bytes, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "calculate prefix scratch");
    free(scratch); scratch = calloc(1U, (size_t)scratch_bytes);
    CHECK(scratch != NULL, "allocate prefix scratch");
    status = kq_qsa_prefill_f32(config, &weights, input, 8U, output, 64U,
                                state, scratch, scratch_bytes, NULL, NULL,
                                NULL, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "QSA prefix execution");
    status = kq_qsa_required_scratch_bytes(config, state, 1U,
                                           &scratch_bytes, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "calculate decode scratch");
    free(scratch); scratch = calloc(1U, (size_t)scratch_bytes);
    CHECK(scratch != NULL, "allocate decode scratch");
    memset(&capture, 0, sizeof(capture));
    status = kq_qsa_decode_f32(config, &weights, input + 64U,
                               output + 64U, 8U, state, scratch,
                               scratch_bytes, capture_selection, NULL,
                               &capture, &diagnostic);
    CHECK(status == KQ_STATUS_OK && kq_qsa_state_length(state) == 9U &&
          capture.selected_token_count == 9U && capture.tail_count == 1U &&
          capture.tokens[8] == 8U,
          "QSA decode inside partial block appends tail exactly");

    input[0] = NAN;
    output[0] = 123.0f;
    length = kq_qsa_state_length(state);
    status = kq_qsa_decode_f32(config, &weights, input, output, 8U,
                               state, scratch, scratch_bytes, NULL, NULL,
                               NULL, &diagnostic);
    CHECK(status == KQ_STATUS_NUMERIC_DOMAIN &&
          kq_qsa_state_length(state) == length && output[0] == 123.0f,
          "QSA failure leaves state and output unchanged");
    input[0] = 0.0f;
    status = kq_qsa_decode_f32(config, &weights, input, output, 7U,
                               state, scratch, scratch_bytes, NULL, NULL,
                               NULL, &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "small QSA output capacity fails closed");
    status = kq_qsa_decode_f32(config, &weights, input, input, 8U,
                               state, scratch, scratch_bytes, NULL, NULL,
                               NULL, &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "QSA input/output alias fails closed");
    status = kq_qsa_decode_f32(config, &weights, input, output, 8U,
                               state, scratch, scratch_bytes - 1U,
                               NULL, NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "small QSA scratch fails closed");
    weights.query_count -= 1U;
    status = kq_qsa_decode_f32(config, &weights, input, output, 8U,
                               state, scratch, scratch_bytes, NULL, NULL,
                               NULL, &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong QSA projection shape fails closed");
    weights.query_count += 1U;
    status = kq_qsa_state_import_f32(state, state_key, 3U, state_value, 4U,
                                     state_raw, 4U, 1U, &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong QSA K cache shape fails closed");
    state_key[0] = NAN;
    status = kq_qsa_state_import_f32(state, state_key, 4U, state_value, 4U,
                                     state_raw, 4U, 1U, &diagnostic);
    CHECK(status == KQ_STATUS_NUMERIC_DOMAIN,
          "non-finite imported QSA state fails closed");
    state_key[0] = 0.0f;

    status = kq_qsa_test_config_create(&dimensions, &other_config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create foreign QSA config");
    status = kq_qsa_state_create(other_config, 16U, &other_state, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create foreign QSA state");
    status = kq_qsa_decode_f32(config, &weights, input, output, 8U,
                               other_state, scratch, scratch_bytes,
                               NULL, NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "foreign QSA state/config pair fails closed");

    fill_target_source(&source, semantics, physical);
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_OK &&
          kq_qsa_config_semantic_state_bytes_per_token(target) == 2304U,
          "valid target QSA bindings and state growth accepted");
    kq_qsa_config_close(target); target = NULL;
    source.layer_type = KQ_MODEL_LAYER_GDN;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_QSA,
          "GDN layer fails QSA construction");
    source.layer_type = KQ_MODEL_LAYER_QSA;
    source.dimensions.query_head_count = 23U;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_QSA,
          "wrong target QSA head count fails closed");
    source.dimensions.query_head_count = 24U;
    source.dimensions.block_size = 5U;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_QSA,
          "wrong target QSA block geometry fails closed");
    source.dimensions.block_size = 4U;
    source.tensors[0] = NULL;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED,
          "missing QSA semantic fails closed");
    source.tensors[0] = &semantics[0];
    semantics[4].canonical_dimensions[0] = 12287U;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_QSA,
          "wrong QSA projection shape fails closed");
    semantics[4].canonical_dimensions[0] = 12288U;
    semantics[6].bindings[1].part_index = 0U;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
          "broken QSA index split fails closed");
    semantics[6].bindings[1].part_index = 1U;
    physical[9].type_id = KQ_GGUF_TYPE_F32;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
          "wrong QSA index physical type fails closed");
    physical[9].type_id = KQ_GGUF_TYPE_BF16;
    physical[9].dimensions[1] = 127U;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
          "wrong QSA split physical shape fails closed");
    physical[9].dimensions[1] = 128U;
    semantics[1].relation = KQ_BINDING_TRANSFORMED_LAYOUT;
    status = kq_qsa_config_create_from_source(&source, 1, &target, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_QSA,
          "transformed QSA binding misuse fails closed");

    puts("QSA synthetic and fail-closed tests: PASS");

cleanup:
    free(scratch);
    kq_qsa_state_close(other_state);
    kq_qsa_config_close(other_config);
    kq_qsa_state_close(state);
    kq_qsa_config_close(config);
    kq_qsa_config_close(target);
    return result;
}
