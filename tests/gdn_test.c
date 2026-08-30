#include "kq_gdn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gdn_internal.h"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
        result = 1; \
        goto cleanup; \
    } \
} while (0)

typedef struct test_buffers {
    float *a_log;
    float *conv;
    float *dt_bias;
    float *alpha;
    float *beta;
    float *qkv;
    float *gate;
    float *norm;
    float *output_weight;
    float *input;
    float *output;
    float *conv_state;
    float *recurrent_state;
    float *before_conv;
    float *before_recurrent;
    void *scratch;
} test_buffers;

static void free_buffers(test_buffers *buffers) {
    free(buffers->a_log);
    free(buffers->conv);
    free(buffers->dt_bias);
    free(buffers->alpha);
    free(buffers->beta);
    free(buffers->qkv);
    free(buffers->gate);
    free(buffers->norm);
    free(buffers->output_weight);
    free(buffers->input);
    free(buffers->output);
    free(buffers->conv_state);
    free(buffers->recurrent_state);
    free(buffers->before_conv);
    free(buffers->before_recurrent);
    free(buffers->scratch);
    memset(buffers, 0, sizeof(*buffers));
}

static void count_observer(const kq_gdn_checkpoint *checkpoint,
                           void *user_data) {
    uint64_t *count = (uint64_t *)user_data;
    if (checkpoint != NULL && checkpoint->values != NULL &&
        checkpoint->value_count != 0U) {
        *count += 1U;
    }
}

static void fill_target_source(kq_gdn_semantic_source *source,
                               kq_semantic_tensor semantics[9],
                               kq_gguf_tensor physical[9]) {
    static const kq_semantic_role roles[9] = {
        KQ_ROLE_GDN_A_LOG, KQ_ROLE_GDN_CONV, KQ_ROLE_GDN_DT_BIAS,
        KQ_ROLE_GDN_ALPHA, KQ_ROLE_GDN_BETA, KQ_ROLE_GDN_QKV,
        KQ_ROLE_GDN_GATE, KQ_ROLE_GDN_NORM, KQ_ROLE_GDN_OUT
    };
    static const kq_binding_relation relations[9] = {
        KQ_BINDING_TRANSFORMED_LAYOUT, KQ_BINDING_TRANSFORMED_LAYOUT,
        KQ_BINDING_RENAMED_ONE_TO_ONE, KQ_BINDING_TRANSFORMED_LAYOUT,
        KQ_BINDING_TRANSFORMED_LAYOUT, KQ_BINDING_TRANSFORMED_LAYOUT,
        KQ_BINDING_TRANSFORMED_LAYOUT, KQ_BINDING_RENAMED_ONE_TO_ONE,
        KQ_BINDING_TRANSFORMED_LAYOUT
    };
    static const uint32_t ranks[9] = {1U, 3U, 1U, 2U, 2U, 2U, 2U, 1U, 2U};
    static const uint64_t dimensions[9][3] = {
        {48U, 0U, 0U}, {10240U, 1U, 4U}, {48U, 0U, 0U},
        {48U, 2560U, 0U}, {48U, 2560U, 0U},
        {10240U, 2560U, 0U}, {6144U, 2560U, 0U},
        {128U, 0U, 0U}, {2560U, 6144U, 0U}
    };
    static const uint32_t types[9] = {
        KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_F32,
        KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_Q8_0,
        KQ_GGUF_TYPE_Q8_0, KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_Q8_0
    };
    uint32_t index;
    uint32_t dimension;
    memset(source, 0, sizeof(*source));
    memset(semantics, 0, sizeof(kq_semantic_tensor) * 9U);
    memset(physical, 0, sizeof(kq_gguf_tensor) * 9U);
    source->model = (const kq_model *)(uintptr_t)1U;
    source->layer_id = 0U;
    source->layer_type = KQ_MODEL_LAYER_GDN;
    source->dimensions.hidden_size = 2560U;
    source->dimensions.key_head_count = 16U;
    source->dimensions.value_head_count = 48U;
    source->dimensions.key_head_dimension = 128U;
    source->dimensions.value_head_dimension = 128U;
    source->dimensions.conv_kernel_size = 4U;
    source->dimensions.rms_norm_epsilon = 1.0e-6f;
    source->dimensions.activation_dtype = KQ_GDN_ACTIVATION_BF16;
    for (index = 0U; index < 9U; ++index) {
        semantics[index].component = KQ_COMPONENT_GDN;
        semantics[index].role = roles[index];
        semantics[index].relation = relations[index];
        semantics[index].runtime_scope = KQ_SCOPE_REQUIRED_INITIAL_TEXT;
        semantics[index].layer_type = KQ_MODEL_LAYER_GDN;
        semantics[index].layer_id = 0U;
        semantics[index].canonical_dtype = KQ_CANONICAL_DTYPE_BF16;
        semantics[index].canonical_rank = ranks[index];
        for (dimension = 0U; dimension < ranks[index]; ++dimension) {
            semantics[index].canonical_dimensions[dimension] =
                dimensions[index][dimension];
        }
        physical[index].type_id = types[index];
        semantics[index].binding_count = 1U;
        semantics[index].bindings[0].physical = &physical[index];
        semantics[index].bindings[0].part_role = KQ_BINDING_PART_WHOLE;
        semantics[index].bindings[0].part_count = 1U;
        source->tensors[index] = &semantics[index];
    }
}

int main(void) {
    kq_gdn_dimensions dimensions = {8U, 2U, 4U, 4U, 4U, 4U,
                                    1.0e-6f, KQ_GDN_ACTIVATION_F32};
    kq_gdn_config *config = NULL;
    kq_gdn_config *other_config = NULL;
    kq_gdn_config *target_config = NULL;
    kq_gdn_state *state = NULL;
    kq_gdn_state *other_state = NULL;
    kq_gdn_weights_f32 weights;
    kq_gdn_semantic_source source;
    kq_semantic_tensor semantics[9];
    kq_gguf_tensor physical[9];
    kq_diagnostic diagnostic;
    kq_status status;
    test_buffers buffers;
    uint64_t conv_count;
    uint64_t recurrent_count;
    uint64_t trace_count = 0U;
    uint64_t index;
    int initialized = 0;
    int result = 0;
    uint8_t mask[5] = {1U, 1U, 0U, 1U, 1U};

    memset(&weights, 0, sizeof(weights));
    memset(&buffers, 0, sizeof(buffers));
    status = kq_gdn_test_config_create(&dimensions, &config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create reduced GDN config");
    CHECK(kq_gdn_config_hidden_size(config) == 8U &&
          kq_gdn_config_conv_channel_count(config) == 32U &&
          kq_gdn_config_recurrent_element_count(config) == 64U &&
          kq_gdn_config_conv_state_element_count(config) == 128U,
          "reduced GDN geometry");
    status = kq_gdn_state_create(config, &state, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create reduced GDN state");
    conv_count = kq_gdn_config_conv_state_element_count(config);
    recurrent_count = kq_gdn_config_recurrent_element_count(config);

    buffers.a_log = (float *)calloc(4U, sizeof(float));
    buffers.conv = (float *)calloc(128U, sizeof(float));
    buffers.dt_bias = (float *)calloc(4U, sizeof(float));
    buffers.alpha = (float *)calloc(32U, sizeof(float));
    buffers.beta = (float *)calloc(32U, sizeof(float));
    buffers.qkv = (float *)calloc(256U, sizeof(float));
    buffers.gate = (float *)calloc(128U, sizeof(float));
    buffers.norm = (float *)calloc(4U, sizeof(float));
    buffers.output_weight = (float *)calloc(128U, sizeof(float));
    buffers.input = (float *)calloc(40U, sizeof(float));
    buffers.output = (float *)calloc(40U, sizeof(float));
    buffers.conv_state = (float *)calloc((size_t)conv_count, sizeof(float));
    buffers.recurrent_state = (float *)calloc((size_t)recurrent_count, sizeof(float));
    buffers.before_conv = (float *)calloc((size_t)conv_count, sizeof(float));
    buffers.before_recurrent = (float *)calloc((size_t)recurrent_count, sizeof(float));
    buffers.scratch = calloc(1U, (size_t)kq_gdn_config_scratch_bytes(config));
    CHECK(buffers.a_log != NULL && buffers.conv != NULL &&
          buffers.dt_bias != NULL && buffers.alpha != NULL &&
          buffers.beta != NULL && buffers.qkv != NULL && buffers.gate != NULL &&
          buffers.norm != NULL && buffers.output_weight != NULL &&
          buffers.input != NULL && buffers.output != NULL &&
          buffers.conv_state != NULL && buffers.recurrent_state != NULL &&
          buffers.before_conv != NULL && buffers.before_recurrent != NULL &&
          buffers.scratch != NULL, "allocate reduced GDN fixtures");
    for (index = 0U; index < 4U; ++index) buffers.norm[index] = 1.0f;
    for (index = 0U; index < 40U; ++index)
        buffers.input[index] = (float)((int)(index % 9U) - 4) / 16.0f;

    weights.a_log = buffers.a_log; weights.a_log_count = 4U;
    weights.conv = buffers.conv; weights.conv_count = 128U;
    weights.dt_bias = buffers.dt_bias; weights.dt_bias_count = 4U;
    weights.alpha = buffers.alpha; weights.alpha_count = 32U;
    weights.beta = buffers.beta; weights.beta_count = 32U;
    weights.qkv = buffers.qkv; weights.qkv_count = 256U;
    weights.gate = buffers.gate; weights.gate_count = 128U;
    weights.norm = buffers.norm; weights.norm_count = 4U;
    weights.output = buffers.output_weight; weights.output_count = 128U;

    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 5U, mask, buffers.output, 40U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        count_observer, &trace_count, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "zero-weight GDN prefill");
    CHECK(trace_count == 21U * 5U, "all prefill checkpoint classes emitted");
    for (index = 0U; index < 40U; ++index)
        CHECK(buffers.output[index] == 0.0f, "zero weights produce zero output");
    status = kq_gdn_state_export_f32(
        state, buffers.conv_state, conv_count,
        buffers.recurrent_state, recurrent_count, &initialized, &diagnostic);
    CHECK(status == KQ_STATUS_OK && initialized == 1,
          "prefill commits explicit state");
    status = kq_gdn_decode_f32(
        config, &weights, buffers.input, buffers.output, 8U, state,
        buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "one-token GDN decode");
    status = kq_gdn_state_reset(state, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "GDN reset");

    status = kq_gdn_state_export_f32(
        state, buffers.before_conv, conv_count,
        buffers.before_recurrent, recurrent_count, &initialized, &diagnostic);
    CHECK(status == KQ_STATUS_OK && initialized == 0,
          "reset exports zero uninitialized state");
    buffers.before_conv[0] = 1.0f;
    status = kq_gdn_state_import_f32(
        state, buffers.before_conv, conv_count,
        buffers.before_recurrent, recurrent_count, 0, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_GDN_STATE,
          "non-zero uninitialized state fails closed");
    buffers.before_conv[0] = 0.0f;
    status = kq_gdn_state_export_f32(
        state, buffers.conv_state, conv_count,
        buffers.conv_state, conv_count, &initialized, &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "overlapping state export buffers fail closed");
    buffers.input[0] = NAN;
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.output, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_NUMERIC_DOMAIN,
          "non-finite input fails closed");
    buffers.input[0] = 0.0f;
    status = kq_gdn_state_export_f32(
        state, buffers.conv_state, conv_count,
        buffers.recurrent_state, recurrent_count, &initialized, &diagnostic);
    CHECK(status == KQ_STATUS_OK && initialized == 0 &&
          memcmp(buffers.conv_state, buffers.before_conv,
                 (size_t)conv_count * sizeof(float)) == 0 &&
          memcmp(buffers.recurrent_state, buffers.before_recurrent,
                 (size_t)recurrent_count * sizeof(float)) == 0,
          "failure leaves state unchanged");

    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 0U, NULL, buffers.output, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_ARITHMETIC_OVERFLOW,
          "zero sequence fails closed");
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.output, 7U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "small output capacity fails closed");
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.input, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "input/output alias fails closed");
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL,
        (float *)state->conv_state, conv_count,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_ALIASING_VIOLATION,
          "output/state alias fails closed");
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.output, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config) - 1U,
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "small scratch fails closed");
    mask[0] = 2U;
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, mask, buffers.output, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "invalid padding mask fails closed");
    mask[0] = 1U;
    weights.qkv_count -= 1U;
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.output, 8U,
        state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong weight shape fails closed");
    weights.qkv_count += 1U;
    status = kq_gdn_state_import_f32(
        state, buffers.conv_state, conv_count - 1U,
        buffers.recurrent_state, recurrent_count, 1, &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong convolution state shape fails closed");
    status = kq_gdn_state_import_f32(
        state, buffers.conv_state, conv_count,
        buffers.recurrent_state, recurrent_count - 1U, 1, &diagnostic);
    CHECK(status == KQ_STATUS_DIMENSION_MISMATCH,
          "wrong recurrent state shape fails closed");

    status = kq_gdn_test_config_create(&dimensions, &other_config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create second GDN config");
    status = kq_gdn_state_create(other_config, &other_state, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "create foreign GDN state");
    status = kq_gdn_prefill_f32(
        config, &weights, buffers.input, 1U, NULL, buffers.output, 8U,
        other_state, buffers.scratch, kq_gdn_config_scratch_bytes(config),
        NULL, NULL, &diagnostic);
    CHECK(status == KQ_STATUS_INVALID_ARGUMENT,
          "foreign state/config pair fails closed");

    fill_target_source(&source, semantics, physical);
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_OK, "valid target semantic bindings accepted");
    CHECK(kq_gdn_config_recurrent_element_count(target_config) == 786432U &&
          kq_gdn_config_conv_state_element_count(target_config) == 40960U,
          "target state geometry");
    kq_gdn_config_close(target_config); target_config = NULL;

    source.layer_type = KQ_MODEL_LAYER_QSA;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_GDN,
          "QSA layer fails GDN construction");
    source.layer_type = KQ_MODEL_LAYER_GDN;
    source.tensors[0] = NULL;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_SEMANTIC_MAPPING_FAILED,
          "missing required semantic fails closed");
    source.tensors[0] = &semantics[0];
    semantics[3].canonical_dimensions[0] = 47U;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_GDN,
          "wrong projection shape fails closed");
    semantics[3].canonical_dimensions[0] = 48U;
    semantics[3].canonical_dtype = KQ_CANONICAL_DTYPE_I64;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_GDN,
          "wrong canonical dtype fails closed");
    semantics[3].canonical_dtype = KQ_CANONICAL_DTYPE_BF16;
    semantics[3].relation = KQ_BINDING_RENAMED_ONE_TO_ONE;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_INCOMPATIBLE_GDN,
          "wrong transformed relation fails closed");
    semantics[3].relation = KQ_BINDING_TRANSFORMED_LAYOUT;
    semantics[3].bindings[0].part_role = KQ_BINDING_PART_GATE;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
          "split binding misuse fails closed");
    semantics[3].bindings[0].part_role = KQ_BINDING_PART_WHOLE;
    physical[3].type_id = KQ_GGUF_TYPE_Q8_0;
    status = kq_gdn_config_create_from_source(
        &source, 1, &target_config, &diagnostic);
    CHECK(status == KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
          "wrong physical type fails closed");

    puts("GDN synthetic and fail-closed tests: PASS");

cleanup:
    kq_gdn_state_close(other_state);
    kq_gdn_config_close(other_config);
    kq_gdn_state_close(state);
    kq_gdn_config_close(config);
    kq_gdn_config_close(target_config);
    free_buffers(&buffers);
    return result;
}
