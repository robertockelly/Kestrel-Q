#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_gdn.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_gdn_internal.h"

typedef struct probe_arrays {
    float *a_log;
    float *conv;
    float *dt_bias;
    float *alpha;
    float *beta;
    float *qkv;
    float *gate;
    float *norm;
    float *output_weight;
    float *initial_conv;
    float *initial_recurrent;
    float *input;
    uint8_t *mask;
    float *output;
    float *final_conv;
    float *final_recurrent;
    void *scratch;
} probe_arrays;

static void probe_free(probe_arrays *arrays) {
    if (arrays == NULL) return;
    free(arrays->a_log);
    free(arrays->conv);
    free(arrays->dt_bias);
    free(arrays->alpha);
    free(arrays->beta);
    free(arrays->qkv);
    free(arrays->gate);
    free(arrays->norm);
    free(arrays->output_weight);
    free(arrays->initial_conv);
    free(arrays->initial_recurrent);
    free(arrays->input);
    free(arrays->mask);
    free(arrays->output);
    free(arrays->final_conv);
    free(arrays->final_recurrent);
    free(arrays->scratch);
    memset(arrays, 0, sizeof(*arrays));
}

static uint32_t probe_f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float probe_f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int probe_read_array(const char *expected,
                            uint64_t expected_count,
                            float **output) {
    char label[32];
    unsigned long long count;
    uint64_t index;
    unsigned int bits;
    float *values;
    if (scanf_s("%31s %llu", label, (unsigned)sizeof(label), &count) != 2 ||
        strcmp(label, expected) != 0 || count != expected_count ||
        expected_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return 0;
    }
    values = (float *)calloc((size_t)expected_count, sizeof(float));
    if (values == NULL) return 0;
    for (index = 0U; index < expected_count; ++index) {
        if (scanf_s("%x", &bits) != 1) {
            free(values);
            return 0;
        }
        values[index] = probe_f32_from_bits((uint32_t)bits);
    }
    *output = values;
    return 1;
}

static int probe_read_mask(uint64_t expected_count, uint8_t **output) {
    char label[32];
    unsigned long long count;
    uint64_t index;
    unsigned int value;
    uint8_t *mask;
    if (scanf_s("%31s %llu", label, (unsigned)sizeof(label), &count) != 2 ||
        strcmp(label, "MASK") != 0 ||
        (count != 0U && count != expected_count)) {
        return 0;
    }
    if (count == 0U) {
        *output = NULL;
        return 1;
    }
    mask = (uint8_t *)calloc((size_t)count, sizeof(uint8_t));
    if (mask == NULL) return 0;
    for (index = 0U; index < expected_count; ++index) {
        if (scanf_s("%u", &value) != 1 || value > 1U) {
            free(mask);
            return 0;
        }
        mask[index] = (uint8_t)value;
    }
    *output = mask;
    return 1;
}

static void probe_print_array(const char *label,
                              const float *values,
                              uint64_t count) {
    uint64_t index;
    printf("%s %" PRIu64, label, count);
    for (index = 0U; index < count; ++index) {
        printf(" %08" PRIx32, probe_f32_bits(values[index]));
    }
    putchar('\n');
}

static void probe_observer(const kq_gdn_checkpoint *checkpoint,
                           void *user_data) {
    uint32_t dimension;
    uint64_t index;
    (void)user_data;
    printf("TRACE %s %" PRIu64 " %u",
           kq_gdn_checkpoint_kind_name(checkpoint->kind),
           checkpoint->token_index, checkpoint->rank);
    for (dimension = 0U; dimension < checkpoint->rank; ++dimension) {
        printf(" %" PRIu64, checkpoint->dimensions[dimension]);
    }
    printf(" %" PRIu64, checkpoint->value_count);
    for (index = 0U; index < checkpoint->value_count; ++index) {
        printf(" %08" PRIx32, probe_f32_bits(checkpoint->values[index]));
    }
    putchar('\n');
}

int main(void) {
    char magic[16];
    char config_label[16];
    unsigned int hidden;
    unsigned int key_heads;
    unsigned int value_heads;
    unsigned int key_dim;
    unsigned int value_dim;
    unsigned int kernel;
    unsigned long long sequence;
    unsigned int initialized;
    uint64_t conv_channels;
    uint64_t value_width;
    uint64_t conv_count;
    uint64_t recurrent_count;
    uint64_t hidden_count;
    uint64_t count;
    probe_arrays arrays;
    kq_gdn_dimensions dimensions;
    kq_gdn_config *config = NULL;
    kq_gdn_state *state = NULL;
    kq_gdn_weights_f32 weights;
    kq_diagnostic diagnostic;
    kq_status status;
    int final_initialized = 0;
    LARGE_INTEGER frequency = {0};
    LARGE_INTEGER start = {0};
    LARGE_INTEGER end = {0};
    uint64_t execution_nanoseconds = 0U;

    memset(&arrays, 0, sizeof(arrays));
    memset(&dimensions, 0, sizeof(dimensions));
    memset(&weights, 0, sizeof(weights));
    if (scanf_s("%15s", magic, (unsigned)sizeof(magic)) != 1 ||
        strcmp(magic, "KQGDN1") != 0 ||
        scanf_s("%15s %u %u %u %u %u %u %llu %u",
                config_label, (unsigned)sizeof(config_label),
                &hidden, &key_heads, &value_heads, &key_dim, &value_dim,
                &kernel, &sequence, &initialized) != 9 ||
        strcmp(config_label, "CONFIG") != 0 ||
        hidden == 0U || hidden > 64U || key_heads == 0U ||
        value_heads == 0U || value_heads > 64U || key_dim == 0U ||
        key_dim > 64U || value_dim == 0U || value_dim > 64U ||
        kernel != 4U || sequence == 0U || sequence > 64U ||
        initialized > 1U || value_heads % key_heads != 0U) {
        fprintf(stderr, "invalid KQGDN1 probe header\n");
        return 2;
    }
    conv_channels = 2U * (uint64_t)key_heads * key_dim +
        (uint64_t)value_heads * value_dim;
    value_width = (uint64_t)value_heads * value_dim;
    conv_count = conv_channels * kernel;
    recurrent_count = (uint64_t)value_heads * key_dim * value_dim;
    hidden_count = sequence * hidden;

    if (!probe_read_array("A_LOG", value_heads, &arrays.a_log) ||
        !probe_read_array("CONV", conv_count, &arrays.conv) ||
        !probe_read_array("DT_BIAS", value_heads, &arrays.dt_bias) ||
        !probe_read_array("ALPHA", (uint64_t)value_heads * hidden,
                          &arrays.alpha) ||
        !probe_read_array("BETA", (uint64_t)value_heads * hidden,
                          &arrays.beta) ||
        !probe_read_array("QKV", conv_channels * hidden, &arrays.qkv) ||
        !probe_read_array("GATE", value_width * hidden, &arrays.gate) ||
        !probe_read_array("NORM", value_dim, &arrays.norm) ||
        !probe_read_array("OUTPUT_WEIGHT", (uint64_t)hidden * value_width,
                          &arrays.output_weight) ||
        !probe_read_array("INITIAL_CONV", conv_count,
                          &arrays.initial_conv) ||
        !probe_read_array("INITIAL_RECURRENT", recurrent_count,
                          &arrays.initial_recurrent) ||
        !probe_read_array("INPUT", hidden_count, &arrays.input) ||
        !probe_read_mask(sequence, &arrays.mask)) {
        fprintf(stderr, "invalid KQGDN1 probe payload\n");
        probe_free(&arrays);
        return 2;
    }

    dimensions.hidden_size = hidden;
    dimensions.key_head_count = key_heads;
    dimensions.value_head_count = value_heads;
    dimensions.key_head_dimension = key_dim;
    dimensions.value_head_dimension = value_dim;
    dimensions.conv_kernel_size = kernel;
    dimensions.rms_norm_epsilon = 1.0e-6f;
    dimensions.activation_dtype = KQ_GDN_ACTIVATION_F32;
    status = kq_gdn_test_config_create(&dimensions, &config, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_state_create(config, &state, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_state_import_f32(
            state, arrays.initial_conv, conv_count,
            arrays.initial_recurrent, recurrent_count, initialized,
            &diagnostic);
    }
    count = config == NULL ? 0U : kq_gdn_config_scratch_bytes(config);
    arrays.scratch = calloc(1U, (size_t)count);
    arrays.output = (float *)calloc((size_t)hidden_count, sizeof(float));
    arrays.final_conv = (float *)calloc((size_t)conv_count, sizeof(float));
    arrays.final_recurrent = (float *)calloc(
        (size_t)recurrent_count, sizeof(float));
    if (status == KQ_STATUS_OK &&
        (arrays.scratch == NULL || arrays.output == NULL ||
         arrays.final_conv == NULL || arrays.final_recurrent == NULL)) {
        status = KQ_STATUS_OUT_OF_MEMORY;
        diagnostic.status = status;
        strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                 "probe allocation failed");
    }

    weights.a_log = arrays.a_log;
    weights.a_log_count = value_heads;
    weights.conv = arrays.conv;
    weights.conv_count = conv_count;
    weights.dt_bias = arrays.dt_bias;
    weights.dt_bias_count = value_heads;
    weights.alpha = arrays.alpha;
    weights.alpha_count = (uint64_t)value_heads * hidden;
    weights.beta = arrays.beta;
    weights.beta_count = (uint64_t)value_heads * hidden;
    weights.qkv = arrays.qkv;
    weights.qkv_count = conv_channels * hidden;
    weights.gate = arrays.gate;
    weights.gate_count = value_width * hidden;
    weights.norm = arrays.norm;
    weights.norm_count = value_dim;
    weights.output = arrays.output_weight;
    weights.output_count = (uint64_t)hidden * value_width;

    if (status == KQ_STATUS_OK && !QueryPerformanceFrequency(&frequency)) {
        status = KQ_STATUS_INVALID_ARGUMENT;
        diagnostic.status = status;
        strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                 "probe high-resolution timer unavailable");
    }
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_prefill_f32(
            config, &weights, arrays.input, sequence, arrays.mask,
            arrays.output, hidden_count, state, arrays.scratch,
            kq_gdn_config_scratch_bytes(config), probe_observer, NULL,
            &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_state_import_f32(
            state, arrays.initial_conv, conv_count,
            arrays.initial_recurrent, recurrent_count, initialized,
            &diagnostic);
    }
    if (status == KQ_STATUS_OK && !QueryPerformanceCounter(&start)) {
        status = KQ_STATUS_INVALID_ARGUMENT;
        diagnostic.status = status;
        strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                 "probe high-resolution timer start failed");
    }
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_prefill_f32(
            config, &weights, arrays.input, sequence, arrays.mask,
            arrays.output, hidden_count, state, arrays.scratch,
            kq_gdn_config_scratch_bytes(config), NULL, NULL,
            &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        if (!QueryPerformanceCounter(&end)) {
            status = KQ_STATUS_INVALID_ARGUMENT;
            diagnostic.status = status;
            strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                     "probe high-resolution timer read failed");
        } else {
            execution_nanoseconds =
                ((uint64_t)(end.QuadPart - start.QuadPart) *
                 UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
        }
    }
    if (status == KQ_STATUS_OK) {
        status = kq_gdn_state_export_f32(
            state, arrays.final_conv, conv_count,
            arrays.final_recurrent, recurrent_count,
            &final_initialized, &diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        printf("RESULT ERROR %d %s\n", (int)status, diagnostic.message);
        kq_gdn_state_close(state);
        kq_gdn_config_close(config);
        probe_free(&arrays);
        return 3;
    }

    puts("RESULT OK");
    probe_print_array("OUTPUT", arrays.output, hidden_count);
    probe_print_array("FINAL_CONV", arrays.final_conv, conv_count);
    probe_print_array("FINAL_RECURRENT", arrays.final_recurrent,
                      recurrent_count);
    printf("INITIALIZED %d\n", final_initialized);
    printf("METRICS %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
           " %" PRIu64 "\n",
           kq_gdn_config_owned_bytes(config),
           kq_gdn_state_owned_bytes(state),
           kq_gdn_config_scratch_bytes(config),
           kq_gdn_config_token_scratch_bytes(config),
           kq_gdn_config_dequant_scratch_bytes(config));
    printf("TIMING %" PRIu64 "\n", execution_nanoseconds);

    kq_gdn_state_close(state);
    kq_gdn_config_close(config);
    probe_free(&arrays);
    return 0;
}
