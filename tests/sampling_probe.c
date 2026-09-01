#include <errno.h>
#include <inttypes.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_sampling.h"
#include "kq_sampling_internal.h"

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int parse_u64(const char *text, uint64_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int parse_u32(const char *text, uint32_t *value) {
    uint64_t parsed;
    if (!parse_u64(text, &parsed) || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_float(const char *text, float *value) {
    char *end = NULL;
    float parsed;
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *value = parsed;
    return 1;
}

static int run_rng(int argc, char **argv) {
    kq_sampling_rng_state state;
    kq_diagnostic diagnostic;
    uint64_t seed;
    uint64_t stream;
    uint32_t count;
    uint32_t index;
    if (argc != 5 || !parse_u64(argv[2], &seed) ||
        !parse_u64(argv[3], &stream) || !parse_u32(argv[4], &count)) {
        return 2;
    }
    if (kq_sampling_rng_seed(seed, stream, &state, &diagnostic) != KQ_STATUS_OK) {
        printf("STATUS %d\n", (int)diagnostic.status);
        return 0;
    }
    printf("RNG %016" PRIx64 " %016" PRIx64 "\n",
           state.state, state.increment);
    for (index = 0U; index < count; ++index) {
        uint32_t word = 0U;
        if (kq_sampling_rng_next_u32(&state, &word, &diagnostic) != KQ_STATUS_OK) {
            printf("STATUS %d\n", (int)diagnostic.status);
            return 0;
        }
        printf("WORD %u %08" PRIx32 " %016" PRIx64 "\n",
               index + 1U, word, state.state);
    }
    printf("STATUS 0\n");
    return 0;
}

static int run_info(void) {
    kq_sampling_policy policy;
    kq_sampling_config *config = NULL;
    kq_diagnostic diagnostic;
    kq_sampling_policy_qwen38_default(&policy);
    if (kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) !=
            KQ_STATUS_OK) {
        return 4;
    }
    printf("INFO %" PRIu64 " %" PRIu64 " %zu %zu\n",
           kq_sampling_config_owned_bytes(config),
           kq_sampling_required_scratch_bytes(config),
           sizeof(kq_sampling_rng_state), sizeof(kq_sampling_result));
    kq_sampling_config_close(config);
    return 0;
}

static int run_bench(int argc, char **argv) {
    kq_sampling_policy policy;
    kq_sampling_config *config = NULL;
    kq_sampling_rng_state state;
    kq_sampling_result result;
    kq_diagnostic diagnostic;
    float *logits = NULL;
    void *scratch = NULL;
    uint64_t scratch_bytes;
    uint32_t iterations;
    uint32_t index;
    clock_t start;
    clock_t end;
    double elapsed_ns;
    double config_elapsed_ns;
    const uint32_t config_iterations = 10000U;
    if (argc != 3 || !parse_u32(argv[2], &iterations) || iterations == 0U) {
        return 2;
    }
    kq_sampling_policy_qwen38_default(&policy);
    start = clock();
    for (index = 0U; index < config_iterations; ++index) {
        kq_sampling_config *temporary = NULL;
        if (kq_sampling_config_open_qwen38(&policy, &temporary, &diagnostic) !=
                KQ_STATUS_OK) {
            return 4;
        }
        kq_sampling_config_close(temporary);
    }
    end = clock();
    config_elapsed_ns = ((double)(end - start) * 1000000000.0) /
                        (double)CLOCKS_PER_SEC;
    if (kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) !=
            KQ_STATUS_OK) {
        return 4;
    }
    scratch_bytes = kq_sampling_required_scratch_bytes(config);
    logits = (float *)malloc((size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float));
    scratch = malloc((size_t)scratch_bytes);
    if (logits == NULL || scratch == NULL) {
        free(logits); free(scratch); kq_sampling_config_close(config);
        return 3;
    }
    for (index = 0U; index < KQ_SAMPLING_QWEN38_VOCAB_SIZE; ++index) {
        logits[index] = -80.0f;
    }
    for (index = 0U; index < 25U; ++index) {
        logits[index * 17U] = 6.0f - (float)index * 0.25f;
    }
    if (kq_sampling_rng_seed(0U, KQ_SAMPLING_DEFAULT_STREAM,
                             &state, &diagnostic) != KQ_STATUS_OK) {
        free(logits); free(scratch); kq_sampling_config_close(config);
        return 4;
    }
    start = clock();
    for (index = 0U; index < iterations; ++index) {
        if (kq_sampling_select_f32(config, &state, logits,
                                   KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                   scratch, scratch_bytes, &result,
                                   &diagnostic) != KQ_STATUS_OK) {
            free(logits); free(scratch); kq_sampling_config_close(config);
            return 4;
        }
    }
    end = clock();
    elapsed_ns = ((double)(end - start) * 1000000000.0) / (double)CLOCKS_PER_SEC;
    printf("BENCH %" PRIu32 " %.0f %.0f\n", iterations, elapsed_ns,
           elapsed_ns / (double)iterations);
    printf("CONFIG_BENCH %" PRIu32 " %.0f %.0f\n", config_iterations,
           config_elapsed_ns, config_elapsed_ns / (double)config_iterations);
    free(logits); free(scratch); kq_sampling_config_close(config);
    return 0;
}

static int run_word(int argc, char **argv) {
    float *probabilities;
    uint32_t word;
    uint32_t count;
    uint32_t index;
    uint32_t selected = UINT32_MAX;
    kq_diagnostic diagnostic;
    kq_status status;
    if (argc < 4 || !parse_u32(argv[2], &word)) {
        return 2;
    }
    count = (uint32_t)(argc - 3);
    probabilities = (float *)malloc((size_t)count * sizeof(float));
    if (probabilities == NULL) {
        return 3;
    }
    for (index = 0U; index < count; ++index) {
        uint32_t bits;
        if (!parse_u32(argv[index + 3], &bits)) {
            free(probabilities);
            return 2;
        }
        probabilities[index] = bits_f32(bits);
    }
    status = kq_sampling_categorical_word_f32_for_test(
        probabilities, count, word, &selected, &diagnostic);
    printf("WORD_RESULT %d %" PRIu32 "\n", (int)status, selected);
    free(probabilities);
    return 0;
}

static int run_stat(int argc, char **argv) {
    float *probabilities;
    uint32_t *counts;
    kq_sampling_rng_state state;
    kq_diagnostic diagnostic;
    uint64_t seed;
    uint64_t stream;
    uint32_t trials;
    uint32_t count;
    uint32_t index;
    if (argc < 7 || !parse_u64(argv[2], &seed) ||
        !parse_u64(argv[3], &stream) || !parse_u32(argv[4], &trials)) {
        return 2;
    }
    count = (uint32_t)(argc - 5);
    probabilities = (float *)malloc((size_t)count * sizeof(float));
    counts = (uint32_t *)calloc(count, sizeof(uint32_t));
    if (probabilities == NULL || counts == NULL) {
        free(probabilities); free(counts);
        return 3;
    }
    for (index = 0U; index < count; ++index) {
        uint32_t bits;
        if (!parse_u32(argv[index + 5], &bits)) {
            free(probabilities); free(counts);
            return 2;
        }
        probabilities[index] = bits_f32(bits);
    }
    if (kq_sampling_rng_seed(seed, stream, &state, &diagnostic) != KQ_STATUS_OK) {
        free(probabilities); free(counts);
        return 4;
    }
    for (index = 0U; index < trials; ++index) {
        uint32_t word = 0U;
        uint32_t selected = 0U;
        if (kq_sampling_rng_next_u32(&state, &word, &diagnostic) != KQ_STATUS_OK ||
            kq_sampling_categorical_word_f32_for_test(
                probabilities, count, word, &selected, &diagnostic) !=
                KQ_STATUS_OK) {
            free(probabilities); free(counts);
            return 4;
        }
        counts[selected] += 1U;
    }
    printf("STAT %" PRIu32, count);
    for (index = 0U; index < count; ++index) {
        printf(" %" PRIu32, counts[index]);
    }
    printf("\n");
    free(probabilities); free(counts);
    return 0;
}

static int run_sample(int argc, char **argv) {
    kq_sampling_policy policy;
    kq_sampling_config *config = NULL;
    kq_sampling_rng_state state;
    kq_sampling_rng_state before;
    kq_sampling_result result;
    kq_diagnostic diagnostic;
    float *logits = NULL;
    float *scores = NULL;
    float *probabilities = NULL;
    void *scratch = NULL;
    uint64_t seed;
    uint64_t stream;
    uint32_t top_k;
    float base;
    uint64_t scratch_bytes;
    int argument;
    uint32_t index;
    kq_status status;

    if (argc < 8) {
        return 2;
    }
    kq_sampling_policy_qwen38_default(&policy);
    if (!parse_float(argv[2], &policy.temperature) ||
        !parse_u32(argv[3], &top_k) ||
        !parse_float(argv[4], &policy.top_p) ||
        !parse_u64(argv[5], &seed) || !parse_u64(argv[6], &stream) ||
        !parse_float(argv[7], &base)) {
        return 2;
    }
    policy.top_k = top_k;
    if (kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) !=
            KQ_STATUS_OK) {
        printf("STATUS %d\n", (int)diagnostic.status);
        return 0;
    }
    scratch_bytes = kq_sampling_required_scratch_bytes(config);
    logits = (float *)malloc((size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float));
    scores = (float *)malloc((size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float));
    probabilities = (float *)malloc(
        (size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float));
    scratch = malloc((size_t)scratch_bytes);
    if (logits == NULL || scores == NULL || probabilities == NULL || scratch == NULL) {
        free(logits); free(scores); free(probabilities); free(scratch);
        kq_sampling_config_close(config);
        return 3;
    }
    for (index = 0U; index < KQ_SAMPLING_QWEN38_VOCAB_SIZE; ++index) {
        logits[index] = base;
    }
    for (argument = 8; argument < argc; ++argument) {
        char *separator = strchr(argv[argument], '=');
        uint32_t token_id;
        float value;
        if (separator == NULL) {
            free(logits); free(scores); free(probabilities); free(scratch);
            kq_sampling_config_close(config);
            return 2;
        }
        *separator = '\0';
        if (!parse_u32(argv[argument], &token_id) ||
            !parse_float(separator + 1, &value) ||
            token_id >= KQ_SAMPLING_QWEN38_VOCAB_SIZE) {
            free(logits); free(scores); free(probabilities); free(scratch);
            kq_sampling_config_close(config);
            return 2;
        }
        logits[token_id] = value;
    }
    if (kq_sampling_rng_seed(seed, stream, &state, &diagnostic) != KQ_STATUS_OK) {
        free(logits); free(scores); free(probabilities); free(scratch);
        kq_sampling_config_close(config);
        return 4;
    }
    before = state;
    memset(&result, 0, sizeof(result));
    status = kq_sampling_select_f32(config, &state, logits,
                                    KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                    scratch, scratch_bytes, &result, &diagnostic);
    printf("STATUS %d\n", (int)status);
    printf("STATE %016" PRIx64 " %016" PRIx64 " %" PRIu64 "\n",
           before.state, state.state, state.draws);
    if (status == KQ_STATUS_OK) {
        if (kq_sampling_copy_work_for_test(
                config, scratch, scratch_bytes, scores, probabilities,
                KQ_SAMPLING_QWEN38_VOCAB_SIZE, &diagnostic) != KQ_STATUS_OK) {
            free(logits); free(scores); free(probabilities); free(scratch);
            kq_sampling_config_close(config);
            return 4;
        }
        printf("RESULT %" PRIu32 " %" PRIu32 " %" PRIu32 " %08" PRIx32
               " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %a\n",
               result.selected_token_id, result.top_k_retained_count,
               result.retained_count, result.rng_word,
               f32_bits(result.selected_probability),
               f32_bits(result.maximum_probability),
               f32_bits(result.normalized_sum_f32), result.uniform_value);
        for (index = 0U; index < KQ_SAMPLING_QWEN38_VOCAB_SIZE; ++index) {
            if (probabilities[index] > 0.0f) {
                printf("TOKEN %" PRIu32 " %08" PRIx32 " %08" PRIx32 "\n",
                       index, f32_bits(scores[index]),
                       f32_bits(probabilities[index]));
            }
        }
    }
    free(logits); free(scores); free(probabilities); free(scratch);
    kq_sampling_config_close(config);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        return 2;
    }
    if (strcmp(argv[1], "rng") == 0) {
        return run_rng(argc, argv);
    }
    if (strcmp(argv[1], "info") == 0) {
        return argc == 2 ? run_info() : 2;
    }
    if (strcmp(argv[1], "bench") == 0) {
        return run_bench(argc, argv);
    }
    if (strcmp(argv[1], "word") == 0) {
        return run_word(argc, argv);
    }
    if (strcmp(argv[1], "stat") == 0) {
        return run_stat(argc, argv);
    }
    if (strcmp(argv[1], "sample") == 0) {
        return run_sample(argc, argv);
    }
    return 2;
}
