#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_qsa.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_qsa_internal.h"

typedef struct probe_data {
    float *query;
    float *key;
    float *value;
    float *output_weight;
    float *query_norm;
    float *key_norm;
    float *index_query;
    float *index_key;
    float *index_query_norm;
    float *index_key_norm;
    float *initial_key;
    float *initial_value;
    float *initial_raw;
    float *input;
    float *output;
    float *final_key;
    float *final_value;
    float *final_raw;
    void *scratch;
} probe_data;

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
        if (scanf_s("%x", &bits) != 1) {
            free(values);
            return 0;
        }
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

static void checkpoint_observer(const kq_qsa_checkpoint *checkpoint,
                                void *user_data) {
    uint32_t dim;
    uint64_t index;
    (void)user_data;
    printf("TRACE %s %" PRIu64 " %u", kq_qsa_checkpoint_kind_name(checkpoint->kind),
           checkpoint->token_index, checkpoint->rank);
    for (dim = 0U; dim < checkpoint->rank; ++dim) {
        printf(" %" PRIu64, checkpoint->dimensions[dim]);
    }
    printf(" %" PRIu64, checkpoint->value_count);
    for (index = 0U; index < checkpoint->value_count; ++index) {
        printf(" %08" PRIx32, f32_bits(checkpoint->values[index]));
    }
    putchar('\n');
}

static void selection_observer(const kq_qsa_selection *selection,
                               void *user_data) {
    uint64_t index;
    (void)user_data;
    printf("SELECT %" PRIu64 " %" PRIu64 " %" PRIu64,
           selection->token_index, selection->absolute_position,
           selection->candidate_count);
    for (index = 0U; index < selection->candidate_count; ++index) {
        printf(" %u:%08" PRIx32,
               selection->candidate_block_ids[index],
               f32_bits(selection->candidate_scores[index]));
    }
    printf(" %" PRIu64, selection->selected_block_count);
    for (index = 0U; index < selection->selected_block_count; ++index) {
        printf(" %u", selection->selected_block_ids[index]);
    }
    printf(" %" PRIu64, selection->selected_token_count);
    for (index = 0U; index < selection->selected_token_count; ++index) {
        printf(" %u", selection->selected_token_positions[index]);
    }
    printf(" %" PRIu64 "\n", selection->tail_count);
}

static void free_data(probe_data *data) {
    if (data == NULL) return;
    free(data->query); free(data->key); free(data->value);
    free(data->output_weight); free(data->query_norm); free(data->key_norm);
    free(data->index_query); free(data->index_key);
    free(data->index_query_norm); free(data->index_key_norm);
    free(data->initial_key); free(data->initial_value); free(data->initial_raw);
    free(data->input); free(data->output); free(data->final_key);
    free(data->final_value); free(data->final_raw); free(data->scratch);
    memset(data, 0, sizeof(*data));
}

static int selection_mode(void) {
    char magic[16];
    unsigned long long count;
    unsigned long long limit;
    uint64_t index;
    unsigned int bits;
    float *scores = NULL;
    uint32_t *selected = NULL;
    uint64_t selected_count = 0U;
    kq_diagnostic diagnostic;
    kq_status status;
    if (scanf_s("%15s %llu %llu", magic, (unsigned)sizeof(magic),
                &count, &limit) != 3 || strcmp(magic, "KQQSASEL1") != 0 ||
        count > 1000000U || limit == 0U || limit > count) {
        fprintf(stderr, "invalid QSA selection probe header\n");
        return 2;
    }
    scores = (float *)calloc((size_t)count, sizeof(float));
    selected = (uint32_t *)calloc((size_t)limit, sizeof(uint32_t));
    if (scores == NULL || selected == NULL) {
        free(scores); free(selected); return 2;
    }
    for (index = 0U; index < count; ++index) {
        if (scanf_s("%x", &bits) != 1) {
            free(scores); free(selected); return 2;
        }
        scores[index] = f32_from_bits((uint32_t)bits);
    }
    status = kq_qsa_select_blocks_f32(
        scores, count, limit, selected, limit, &selected_count, &diagnostic);
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "selection failed: %s\n", diagnostic.message);
        free(scores); free(selected); return 3;
    }
    printf("SELECTED %" PRIu64, selected_count);
    for (index = 0U; index < selected_count; ++index) {
        printf(" %u", selected[index]);
    }
    putchar('\n');
    free(scores); free(selected);
    return 0;
}

int main(int argc, char **argv) {
    char magic[16];
    char config_label[16];
    unsigned int hidden, query_heads, kv_heads, head_dim;
    unsigned int index_heads, index_dim, block, budget;
    unsigned long long sequence, capacity, initial_length;
    uint64_t q_width, kv_width, iq_width, hidden_count;
    uint64_t initial_kv, initial_raw, final_kv, final_raw;
    uint64_t scratch_bytes = 0U;
    uint64_t exported_length = 0U;
    probe_data data;
    kq_qsa_dimensions dimensions;
    kq_qsa_weights_f32 weights;
    kq_qsa_config *config = NULL;
    kq_qsa_state *state = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    LARGE_INTEGER frequency = {0}, start = {0}, end = {0};
    uint64_t elapsed = 0U;

    if (argc == 2 && strcmp(argv[1], "--selection") == 0) {
        return selection_mode();
    }
    memset(&data, 0, sizeof(data));
    memset(&dimensions, 0, sizeof(dimensions));
    memset(&weights, 0, sizeof(weights));
    if (scanf_s("%15s", magic, (unsigned)sizeof(magic)) != 1 ||
        strcmp(magic, "KQQSA1") != 0 ||
        scanf_s("%15s %u %u %u %u %u %u %u %u %llu %llu %llu",
            config_label, (unsigned)sizeof(config_label), &hidden,
            &query_heads, &kv_heads, &head_dim, &index_heads, &index_dim,
            &block, &budget, &sequence, &capacity, &initial_length) != 12 ||
        strcmp(config_label, "CONFIG") != 0 || hidden == 0U || hidden > 64U ||
        query_heads == 0U || query_heads > 16U || kv_heads == 0U ||
        head_dim == 0U || head_dim > 64U || index_heads == 0U ||
        index_dim == 0U || index_dim > 64U || block == 0U || budget == 0U ||
        sequence == 0U || sequence > 64U || capacity == 0U || capacity > 64U ||
        initial_length + sequence > capacity) {
        fprintf(stderr, "invalid KQQSA1 probe header\n");
        return 2;
    }
    q_width = (uint64_t)query_heads * head_dim;
    kv_width = (uint64_t)kv_heads * head_dim;
    iq_width = (uint64_t)index_heads * index_dim;
    hidden_count = sequence * hidden;
    initial_kv = initial_length * kv_width;
    initial_raw = initial_length * index_dim;
    final_kv = (initial_length + sequence) * kv_width;
    final_raw = (initial_length + sequence) * index_dim;
    if (!read_array("QUERY", 2U * q_width * hidden, &data.query) ||
        !read_array("KEY", kv_width * hidden, &data.key) ||
        !read_array("VALUE", kv_width * hidden, &data.value) ||
        !read_array("OUTPUT_WEIGHT", hidden * q_width, &data.output_weight) ||
        !read_array("QUERY_NORM", head_dim, &data.query_norm) ||
        !read_array("KEY_NORM", head_dim, &data.key_norm) ||
        !read_array("INDEX_QUERY", iq_width * hidden, &data.index_query) ||
        !read_array("INDEX_KEY", (uint64_t)index_dim * hidden, &data.index_key) ||
        !read_array("INDEX_QUERY_NORM", index_dim, &data.index_query_norm) ||
        !read_array("INDEX_KEY_NORM", index_dim, &data.index_key_norm) ||
        !read_array("INITIAL_KEY", initial_kv, &data.initial_key) ||
        !read_array("INITIAL_VALUE", initial_kv, &data.initial_value) ||
        !read_array("INITIAL_RAW", initial_raw, &data.initial_raw) ||
        !read_array("INPUT", hidden_count, &data.input)) {
        fprintf(stderr, "invalid KQQSA1 probe payload\n");
        free_data(&data);
        return 2;
    }
    dimensions.hidden_size = hidden;
    dimensions.query_head_count = query_heads;
    dimensions.key_value_head_count = kv_heads;
    dimensions.head_dimension = head_dim;
    dimensions.index_query_head_count = index_heads;
    dimensions.index_head_dimension = index_dim;
    dimensions.block_size = block;
    dimensions.token_budget = budget;
    dimensions.max_context = (uint32_t)capacity;
    dimensions.rotary_dimension = head_dim / 2U;
    dimensions.rms_norm_epsilon = 1.0e-6f;
    dimensions.rope_theta = 10000000.0f;
    dimensions.activation_dtype = KQ_QSA_ACTIVATION_F32;
    status = kq_qsa_test_config_create(&dimensions, &config, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_state_create(config, capacity, &state, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_state_import_f32(
            state, data.initial_key, initial_kv, data.initial_value, initial_kv,
            data.initial_raw, initial_raw, initial_length, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_required_scratch_bytes(
            config, state, sequence, &scratch_bytes, &diagnostic);
    }
    data.scratch = calloc(1U, (size_t)scratch_bytes);
    data.output = (float *)calloc((size_t)hidden_count, sizeof(float));
    data.final_key = (float *)calloc((size_t)final_kv, sizeof(float));
    data.final_value = (float *)calloc((size_t)final_kv, sizeof(float));
    data.final_raw = (float *)calloc((size_t)final_raw, sizeof(float));
    if (status == KQ_STATUS_OK &&
        (data.scratch == NULL || data.output == NULL || data.final_key == NULL ||
         data.final_value == NULL || data.final_raw == NULL)) {
        status = KQ_STATUS_OUT_OF_MEMORY;
        strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                 "probe allocation failed");
    }
    weights.query = data.query; weights.query_count = 2U * q_width * hidden;
    weights.key = data.key; weights.key_count = kv_width * hidden;
    weights.value = data.value; weights.value_count = kv_width * hidden;
    weights.output = data.output_weight; weights.output_count = hidden * q_width;
    weights.query_norm = data.query_norm; weights.query_norm_count = head_dim;
    weights.key_norm = data.key_norm; weights.key_norm_count = head_dim;
    weights.index_query = data.index_query;
    weights.index_query_count = iq_width * hidden;
    weights.index_key = data.index_key;
    weights.index_key_count = (uint64_t)index_dim * hidden;
    weights.index_query_norm = data.index_query_norm;
    weights.index_query_norm_count = index_dim;
    weights.index_key_norm = data.index_key_norm;
    weights.index_key_norm_count = index_dim;
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_prefill_f32(
            config, &weights, data.input, sequence, data.output, hidden_count,
            state, data.scratch, scratch_bytes, selection_observer,
            checkpoint_observer, NULL, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_state_export_f32(
            state, data.final_key, final_kv, data.final_value, final_kv,
            data.final_raw, final_raw, &exported_length, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_state_import_f32(
            state, data.initial_key, initial_kv, data.initial_value, initial_kv,
            data.initial_raw, initial_raw, initial_length, &diagnostic);
    }
    if (status == KQ_STATUS_OK &&
        (!QueryPerformanceFrequency(&frequency) ||
         !QueryPerformanceCounter(&start))) {
        status = KQ_STATUS_INVALID_ARGUMENT;
        strcpy_s(diagnostic.message, sizeof(diagnostic.message),
                 "probe high-resolution timer unavailable");
    }
    if (status == KQ_STATUS_OK) {
        status = kq_qsa_prefill_f32(
            config, &weights, data.input, sequence, data.output, hidden_count,
            state, data.scratch, scratch_bytes, NULL, NULL, NULL, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        if (!QueryPerformanceCounter(&end)) {
            status = KQ_STATUS_INVALID_ARGUMENT;
        } else {
            elapsed = ((uint64_t)(end.QuadPart - start.QuadPart) *
                       UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
        }
    }
    if (status != KQ_STATUS_OK) {
        printf("RESULT ERROR %d %s\n", (int)status, diagnostic.message);
        kq_qsa_state_close(state);
        kq_qsa_config_close(config);
        free_data(&data);
        return 3;
    }
    puts("RESULT OK");
    print_array("OUTPUT", data.output, hidden_count);
    print_array("FINAL_KEY", data.final_key, final_kv);
    print_array("FINAL_VALUE", data.final_value, final_kv);
    print_array("FINAL_RAW", data.final_raw, final_raw);
    printf("LENGTH %" PRIu64 "\n", exported_length);
    printf("METRICS %" PRIu64 " %" PRIu64 " %" PRIu64
           " %" PRIu64 "\n",
           kq_qsa_config_owned_bytes(config),
           kq_qsa_state_owned_bytes(state), scratch_bytes,
           kq_qsa_config_semantic_state_bytes_per_token(config));
    printf("TIMING %" PRIu64 "\n", elapsed);
    kq_qsa_state_close(state);
    kq_qsa_config_close(config);
    free_data(&data);
    return 0;
}
