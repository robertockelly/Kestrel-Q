#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_moe.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_moe_internal.h"

typedef struct probe_data {
    float *router;
    float *routed_gate;
    float *routed_up;
    float *routed_down;
    float *shared_gate;
    float *shared_up;
    float *shared_down;
    float *shared_gate_weight;
    float *input;
    float *output;
    void *scratch;
} probe_data;

typedef struct phase_timing {
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER router_end;
    LARGE_INTEGER routed_end;
    LARGE_INTEGER shared_end;
    int have_router;
    int have_routed;
    int have_shared;
} phase_timing;

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int read_array(const char *expected, uint64_t expected_count,
                      float **output) {
    char label[32];
    unsigned long long count;
    unsigned int bits;
    uint64_t index;
    float *values;
    if (scanf_s("%31s %llu", label, (unsigned)sizeof(label), &count) != 2 ||
        strcmp(label, expected) != 0 || count != expected_count ||
        expected_count > SIZE_MAX / sizeof(float)) return 0;
    values = (float *)calloc((size_t)(expected_count == 0U ? 1U : expected_count),
                             sizeof(float));
    if (values == NULL) return 0;
    for (index = 0U; index < expected_count; ++index) {
        if (scanf_s("%x", &bits) != 1) { free(values); return 0; }
        values[index] = f32_from_bits((uint32_t)bits);
    }
    *output = values;
    return 1;
}

static void print_array(const char *label, const float *values,
                        uint64_t count) {
    uint64_t index;
    printf("%s %" PRIu64, label, count);
    for (index = 0U; index < count; ++index) {
        printf(" %08" PRIx32, f32_bits(values[index]));
    }
    putchar('\n');
}

static void free_data(probe_data *data) {
    if (data == NULL) return;
    free(data->router); free(data->routed_gate); free(data->routed_up);
    free(data->routed_down); free(data->shared_gate); free(data->shared_up);
    free(data->shared_down); free(data->shared_gate_weight);
    free(data->input); free(data->output); free(data->scratch);
    memset(data, 0, sizeof(*data));
}

static void route_observer(const kq_moe_route *route, void *user_data) {
    uint32_t index;
    (void)user_data;
    printf("ROUTE %" PRIu64 " %" PRIu64, route->token_index,
           route->expert_count);
    for (index = 0U; index < route->expert_count; ++index) {
        printf(" %08" PRIx32 ":%08" PRIx32,
               f32_bits(route->router_logits[index]),
               f32_bits(route->router_probabilities[index]));
    }
    printf(" %u", route->selected_count);
    for (index = 0U; index < route->selected_count; ++index) {
        printf(" %u:%08" PRIx32, route->selected_expert_ids[index],
               f32_bits(route->selected_weights[index]));
    }
    putchar('\n');
}

static void checkpoint_observer(const kq_moe_checkpoint *checkpoint,
                                void *user_data) {
    uint32_t dim;
    uint64_t index;
    (void)user_data;
    printf("TRACE %s %" PRIu64 " %u %u %u",
           kq_moe_checkpoint_kind_name(checkpoint->kind),
           checkpoint->token_index, checkpoint->expert_id,
           checkpoint->top_k_position, checkpoint->rank);
    for (dim = 0U; dim < checkpoint->rank; ++dim) {
        printf(" %" PRIu64, checkpoint->dimensions[dim]);
    }
    printf(" %" PRIu64, checkpoint->value_count);
    for (index = 0U; index < checkpoint->value_count; ++index) {
        printf(" %08" PRIx32, f32_bits(checkpoint->values[index]));
    }
    putchar('\n');
}

static void timing_observer(const kq_moe_checkpoint *checkpoint,
                            void *user_data) {
    phase_timing *timing = (phase_timing *)user_data;
    if (checkpoint->token_index != 0U) return;
    if (!timing->have_router &&
        checkpoint->kind == KQ_MOE_CHECKPOINT_ROUTER_LOGITS) {
        QueryPerformanceCounter(&timing->router_end);
        timing->have_router = 1;
    } else if (!timing->have_routed &&
               checkpoint->kind == KQ_MOE_CHECKPOINT_ROUTED_WEIGHTED_SUM) {
        QueryPerformanceCounter(&timing->routed_end);
        timing->have_routed = 1;
    } else if (!timing->have_shared &&
               checkpoint->kind == KQ_MOE_CHECKPOINT_SHARED_GATE_PROJECTION) {
        QueryPerformanceCounter(&timing->shared_end);
        timing->have_shared = 1;
    }
}

static uint64_t phase_elapsed(LARGE_INTEGER start, LARGE_INTEGER end,
                              LARGE_INTEGER frequency) {
    if (frequency.QuadPart <= 0 || end.QuadPart < start.QuadPart) return 0U;
    return ((uint64_t)(end.QuadPart - start.QuadPart) *
            UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
}

static int route_mode(void) {
    char magic[20], config_label[16];
    unsigned int hidden, experts, top_k;
    unsigned long long tokens;
    kq_moe_dimensions dimensions;
    kq_moe_config *config = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    float *router = NULL, *input = NULL, *logits = NULL, *probabilities = NULL;
    float *weights = NULL;
    uint32_t *ids = NULL;
    uint64_t token;
    int result = 2;
    if (scanf_s("%19s", magic, (unsigned)sizeof(magic)) != 1 ||
        strcmp(magic, "KQMOEROUTE1") != 0 ||
        scanf_s("%15s %u %u %u %llu", config_label,
                (unsigned)sizeof(config_label), &hidden, &experts, &top_k,
                &tokens) != 5 || strcmp(config_label, "CONFIG") != 0 ||
        hidden == 0U || hidden > 64U || experts == 0U || experts > 1024U ||
        top_k == 0U || top_k > experts || tokens == 0U || tokens > 64U) {
        fprintf(stderr, "invalid KQMOEROUTE1 probe header\n"); return 2;
    }
    if (!read_array("ROUTER", (uint64_t)experts * hidden, &router) ||
        !read_array("INPUT", tokens * hidden, &input)) goto cleanup;
    logits = (float *)calloc(experts, sizeof(float));
    probabilities = (float *)calloc(experts, sizeof(float));
    weights = (float *)calloc(top_k, sizeof(float));
    ids = (uint32_t *)calloc(top_k, sizeof(uint32_t));
    if (logits == NULL || probabilities == NULL || weights == NULL || ids == NULL)
        goto cleanup;
    memset(&dimensions, 0, sizeof(dimensions));
    dimensions.hidden_size = hidden; dimensions.expert_count = experts;
    dimensions.top_k = top_k; dimensions.routed_intermediate_size = 1U;
    dimensions.shared_intermediate_size = 1U;
    dimensions.activation_dtype = KQ_MOE_ACTIVATION_F32;
    status = kq_moe_test_config_create(&dimensions, &config, &diagnostic);
    if (status != KQ_STATUS_OK) goto cleanup;
    for (token = 0U; token < tokens; ++token) {
        uint32_t index;
        status = kq_moe_route_f32(
            config, router, (uint64_t)experts * hidden,
            input + token * hidden, logits, experts, probabilities, experts,
            ids, top_k, weights, top_k, &diagnostic);
        if (status != KQ_STATUS_OK) goto cleanup;
        printf("ROUTE %" PRIu64 " %u", token, experts);
        for (index = 0U; index < experts; ++index) {
            printf(" %08" PRIx32 ":%08" PRIx32,
                   f32_bits(logits[index]), f32_bits(probabilities[index]));
        }
        printf(" %u", top_k);
        for (index = 0U; index < top_k; ++index) {
            printf(" %u:%08" PRIx32, ids[index], f32_bits(weights[index]));
        }
        putchar('\n');
    }
    result = 0;
cleanup:
    if (result != 0 && config != NULL) fprintf(stderr, "%s\n", diagnostic.message);
    kq_moe_config_close(config); free(router); free(input); free(logits);
    free(probabilities); free(weights); free(ids); return result;
}

int main(int argc, char **argv) {
    char magic[16], config_label[16];
    unsigned int hidden, experts, top_k, routed_i, shared_i;
    unsigned long long tokens;
    uint64_t routed_count, shared_count, input_count, scratch_bytes;
    probe_data data;
    kq_moe_dimensions dimensions;
    kq_moe_weights_f32 weights;
    kq_moe_config *config = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    LARGE_INTEGER frequency = {0}, start = {0}, end = {0};
    uint64_t elapsed = 0U;
    phase_timing phases;

    if (argc == 2 && strcmp(argv[1], "--route") == 0) return route_mode();
    memset(&data, 0, sizeof(data));
    memset(&dimensions, 0, sizeof(dimensions));
    memset(&weights, 0, sizeof(weights));
    if (scanf_s("%15s", magic, (unsigned)sizeof(magic)) != 1 ||
        strcmp(magic, "KQMOE1") != 0 ||
        scanf_s("%15s %u %u %u %u %u %llu", config_label,
                (unsigned)sizeof(config_label), &hidden, &experts, &top_k,
                &routed_i, &shared_i, &tokens) != 7 ||
        strcmp(config_label, "CONFIG") != 0 || hidden == 0U || hidden > 64U ||
        experts == 0U || experts > 64U || top_k == 0U || top_k > experts ||
        routed_i == 0U || routed_i > 64U || shared_i == 0U || shared_i > 64U ||
        tokens == 0U || tokens > 64U) {
        fprintf(stderr, "invalid KQMOE1 probe header\n"); return 2;
    }
    routed_count = (uint64_t)experts * routed_i * hidden;
    shared_count = (uint64_t)shared_i * hidden;
    input_count = tokens * hidden;
    if (!read_array("ROUTER", (uint64_t)experts * hidden, &data.router) ||
        !read_array("ROUTED_GATE", routed_count, &data.routed_gate) ||
        !read_array("ROUTED_UP", routed_count, &data.routed_up) ||
        !read_array("ROUTED_DOWN", routed_count, &data.routed_down) ||
        !read_array("SHARED_GATE", shared_count, &data.shared_gate) ||
        !read_array("SHARED_UP", shared_count, &data.shared_up) ||
        !read_array("SHARED_DOWN", shared_count, &data.shared_down) ||
        !read_array("SHARED_GATE_WEIGHT", hidden, &data.shared_gate_weight) ||
        !read_array("INPUT", input_count, &data.input)) {
        fprintf(stderr, "invalid KQMOE1 probe payload\n"); free_data(&data); return 2;
    }
    dimensions.hidden_size = hidden; dimensions.expert_count = experts;
    dimensions.top_k = top_k; dimensions.routed_intermediate_size = routed_i;
    dimensions.shared_intermediate_size = shared_i;
    dimensions.activation_dtype = KQ_MOE_ACTIVATION_F32;
    status = kq_moe_test_config_create(&dimensions, &config, &diagnostic);
    if (status != KQ_STATUS_OK) goto failure;
    scratch_bytes = kq_moe_config_scratch_bytes(config);
    data.scratch = calloc(1U, (size_t)scratch_bytes);
    data.output = (float *)calloc((size_t)input_count, sizeof(float));
    if (data.scratch == NULL || data.output == NULL) goto failure;
    weights.router = data.router; weights.router_count = (uint64_t)experts * hidden;
    weights.routed_gate = data.routed_gate; weights.routed_gate_count = routed_count;
    weights.routed_up = data.routed_up; weights.routed_up_count = routed_count;
    weights.routed_down = data.routed_down; weights.routed_down_count = routed_count;
    weights.shared_gate = data.shared_gate; weights.shared_gate_count = shared_count;
    weights.shared_up = data.shared_up; weights.shared_up_count = shared_count;
    weights.shared_down = data.shared_down; weights.shared_down_count = shared_count;
    weights.shared_gate_weight = data.shared_gate_weight;
    weights.shared_gate_weight_count = hidden;
    QueryPerformanceFrequency(&frequency); QueryPerformanceCounter(&start);
    status = kq_moe_execute_f32(
        config, &weights, data.input, tokens, data.output, input_count,
        data.scratch, scratch_bytes, route_observer, checkpoint_observer,
        NULL, &diagnostic);
    QueryPerformanceCounter(&end);
    if (status != KQ_STATUS_OK) goto failure;
    if (frequency.QuadPart > 0) {
        elapsed = ((uint64_t)(end.QuadPart - start.QuadPart) *
                   UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
    }
    memset(&phases, 0, sizeof(phases));
    phases.frequency = frequency;
    QueryPerformanceCounter(&phases.start);
    status = kq_moe_execute_f32(
        config, &weights, data.input, tokens, data.output, input_count,
        data.scratch, scratch_bytes, NULL, timing_observer, &phases,
        &diagnostic);
    QueryPerformanceCounter(&end);
    if (status != KQ_STATUS_OK || !phases.have_router ||
        !phases.have_routed || !phases.have_shared) goto failure;
    print_array("OUTPUT", data.output, input_count);
    printf("METRICS config_owned_bytes=%" PRIu64 " scratch_bytes=%" PRIu64
           " tokens=%llu\n", kq_moe_config_owned_bytes(config), scratch_bytes,
           tokens);
    printf("TIMING elapsed_ns=%" PRIu64 "\n", elapsed);
    printf("PHASE_TIMING first_token_router_ns=%" PRIu64
           " first_token_routed_ns=%" PRIu64
           " first_token_shared_plus_final_ns=%" PRIu64
           " sequence_total_ns=%" PRIu64 "\n",
           phase_elapsed(phases.start, phases.router_end, phases.frequency),
           phase_elapsed(phases.router_end, phases.routed_end, phases.frequency),
           phase_elapsed(phases.routed_end, phases.shared_end, phases.frequency),
           phase_elapsed(phases.start, end, phases.frequency));
    kq_moe_config_close(config); free_data(&data); return 0;
failure:
    fprintf(stderr, "MoE probe failed: %s\n", diagnostic.message);
    kq_moe_config_close(config); free_data(&data); return 3;
}
