#include "kq_sampling.h"

#include <fenv.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "kq_internal.h"
#include "kq_sampling_internal.h"

#define KQ_SAMPLING_CONFIG_MAGIC UINT64_C(0x4b5153414d504c31)
#define KQ_PCG32_MULTIPLIER UINT64_C(6364136223846793005)
#define KQ_RNG_INTEGRITY_OFFSET UINT64_C(1469598103934665603)
#define KQ_RNG_INTEGRITY_PRIME UINT64_C(1099511628211)

struct kq_sampling_config {
    uint64_t magic;
    kq_sampling_policy policy;
    uint32_t vocabulary_size;
    uint32_t canonical_token_count;
    uint64_t scratch_bytes;
};

typedef struct kq_sampling_scratch {
    float *scores;
    float *probabilities;
    uint32_t *order;
    uint32_t *temporary;
} kq_sampling_scratch;

static kq_status kq_sampling_fail(kq_diagnostic *diagnostic,
                                  kq_status status,
                                  const char *message) {
    kq_diagnostic_set(diagnostic, status, "%s", message);
    return status;
}

static int kq_sampling_config_valid(const kq_sampling_config *config) {
    return config != NULL && config->magic == KQ_SAMPLING_CONFIG_MAGIC &&
           config->vocabulary_size == KQ_SAMPLING_QWEN38_VOCAB_SIZE &&
           config->canonical_token_count ==
               KQ_SAMPLING_QWEN38_CANONICAL_TOKEN_COUNT;
}

static int kq_u64_add(uint64_t left, uint64_t right, uint64_t *output) {
    if (output == NULL || left > UINT64_MAX - right) {
        return 0;
    }
    *output = left + right;
    return 1;
}

static int kq_u64_multiply(uint64_t left, uint64_t right,
                           uint64_t *output) {
    if (output == NULL || (left != 0U && right > UINT64_MAX / left)) {
        return 0;
    }
    *output = left * right;
    return 1;
}

static int kq_ranges_overlap(const void *left, uint64_t left_bytes,
                             const void *right, uint64_t right_bytes) {
    uintptr_t left_start;
    uintptr_t right_start;
    uintptr_t left_end;
    uintptr_t right_end;

    if (left == NULL || right == NULL || left_bytes == 0U || right_bytes == 0U ||
        left_bytes > (uint64_t)UINTPTR_MAX || right_bytes > (uint64_t)UINTPTR_MAX) {
        return 0;
    }
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - (uintptr_t)left_bytes ||
        right_start > UINTPTR_MAX - (uintptr_t)right_bytes) {
        return 1;
    }
    left_end = left_start + (uintptr_t)left_bytes;
    right_end = right_start + (uintptr_t)right_bytes;
    return left_start < right_end && right_start < left_end;
}

static uint64_t kq_integrity_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= KQ_RNG_INTEGRITY_PRIME;
    return hash;
}

static uint64_t kq_rng_integrity(const kq_sampling_rng_state *state) {
    uint64_t hash = KQ_RNG_INTEGRITY_OFFSET;
    hash = kq_integrity_mix(hash, state->version);
    hash = kq_integrity_mix(hash, state->reserved);
    hash = kq_integrity_mix(hash, state->seed);
    hash = kq_integrity_mix(hash, state->stream);
    hash = kq_integrity_mix(hash, state->state);
    hash = kq_integrity_mix(hash, state->increment);
    hash = kq_integrity_mix(hash, state->draws);
    return hash;
}

static int kq_rng_valid(const kq_sampling_rng_state *state) {
    uint64_t expected_increment;
    if (state == NULL || state->version != KQ_SAMPLING_RNG_STATE_VERSION ||
        state->reserved != 0U || state->stream > KQ_SAMPLING_MAX_STREAM) {
        return 0;
    }
    expected_increment = (state->stream << 1U) | UINT64_C(1);
    return state->increment == expected_increment &&
           (state->increment & UINT64_C(1)) != 0U &&
           state->integrity == kq_rng_integrity(state);
}

static uint32_t kq_pcg32_step(uint64_t *state, uint64_t increment) {
    uint64_t old_state = *state;
    uint32_t xorshifted;
    uint32_t rotation;
    *state = old_state * KQ_PCG32_MULTIPLIER + increment;
    xorshifted = (uint32_t)(((old_state >> 18U) ^ old_state) >> 27U);
    rotation = (uint32_t)(old_state >> 59U);
    return (xorshifted >> rotation) |
           (xorshifted << ((0U - rotation) & 31U));
}

static void kq_rng_seed_value(uint64_t seed, uint64_t stream,
                              kq_sampling_rng_state *output) {
    uint64_t state_value = 0U;
    uint64_t increment = (stream << 1U) | UINT64_C(1);
    (void)kq_pcg32_step(&state_value, increment);
    state_value += seed;
    (void)kq_pcg32_step(&state_value, increment);
    output->version = KQ_SAMPLING_RNG_STATE_VERSION;
    output->reserved = 0U;
    output->seed = seed;
    output->stream = stream;
    output->state = state_value;
    output->increment = increment;
    output->draws = 0U;
    output->integrity = kq_rng_integrity(output);
}

static kq_status kq_rng_next_local(kq_sampling_rng_state *state,
                                    uint32_t *word,
                                    kq_diagnostic *diagnostic) {
    if (!kq_rng_valid(state) || word == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_RNG_STATE,
                                "sampling RNG state is invalid");
    }
    if (state->draws == UINT64_MAX) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "sampling RNG draw count overflowed");
    }
    *word = kq_pcg32_step(&state->state, state->increment);
    state->draws += 1U;
    state->integrity = kq_rng_integrity(state);
    return KQ_STATUS_OK;
}

void kq_sampling_policy_qwen38_default(kq_sampling_policy *policy) {
    if (policy == NULL) {
        return;
    }
    memset(policy, 0, sizeof(*policy));
    policy->version = KQ_SAMPLING_POLICY_VERSION;
    policy->temperature = 1.0f;
    policy->top_p = 0.95f;
    policy->top_k = 20U;
}

kq_status kq_sampling_config_open_qwen38(
    const kq_sampling_policy *policy,
    kq_sampling_config **output,
    kq_diagnostic *diagnostic) {
    kq_sampling_config *config;
    uint64_t arrays_bytes;
    uint64_t scratch_bytes;
    uint32_t index;

    kq_diagnostic_clear(diagnostic);
    if (output == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling config output is null");
    }
    if (policy != NULL &&
        kq_ranges_overlap(policy, sizeof(*policy), output, sizeof(*output))) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "sampling policy and config output overlap");
    }
    *output = NULL;
    if (policy == NULL || policy->version != KQ_SAMPLING_POLICY_VERSION ||
        policy->flags != 0U) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_SAMPLING,
                                "sampling policy version or flags are unsupported");
    }
    for (index = 0U; index < 3U; ++index) {
        if (policy->reserved[index] != 0U) {
            return kq_sampling_fail(diagnostic,
                                    KQ_STATUS_INCOMPATIBLE_SAMPLING,
                                    "sampling policy reserved fields are nonzero");
        }
    }
    if (!isfinite(policy->temperature) || policy->temperature <= 0.0f ||
        !isfinite(policy->top_p) || policy->top_p < 0.0f ||
        policy->top_p > 1.0f ||
        policy->top_k > KQ_SAMPLING_QWEN38_VOCAB_SIZE) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_SAMPLING,
                                "sampling temperature, top-k, or top-p is invalid");
    }
    if (!kq_u64_multiply(KQ_SAMPLING_QWEN38_VOCAB_SIZE, 16U,
                         &arrays_bytes) ||
        !kq_u64_add(arrays_bytes, 7U, &scratch_bytes)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "sampling scratch size overflowed");
    }
    config = (kq_sampling_config *)calloc(1U, sizeof(*config));
    if (config == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_OUT_OF_MEMORY,
                                "sampling config allocation failed");
    }
    config->magic = KQ_SAMPLING_CONFIG_MAGIC;
    config->policy = *policy;
    config->vocabulary_size = KQ_SAMPLING_QWEN38_VOCAB_SIZE;
    config->canonical_token_count = KQ_SAMPLING_QWEN38_CANONICAL_TOKEN_COUNT;
    config->scratch_bytes = scratch_bytes;
    *output = config;
    return KQ_STATUS_OK;
}

void kq_sampling_config_close(kq_sampling_config *config) {
    if (config == NULL) {
        return;
    }
    if (config->magic == KQ_SAMPLING_CONFIG_MAGIC) {
        config->magic = 0U;
    }
    free(config);
}

uint64_t kq_sampling_config_owned_bytes(const kq_sampling_config *config) {
    return kq_sampling_config_valid(config) ? (uint64_t)sizeof(*config) : 0U;
}

uint32_t kq_sampling_config_vocabulary_size(const kq_sampling_config *config) {
    return kq_sampling_config_valid(config) ? config->vocabulary_size : 0U;
}

uint64_t kq_sampling_required_scratch_bytes(const kq_sampling_config *config) {
    return kq_sampling_config_valid(config) ? config->scratch_bytes : 0U;
}

kq_status kq_sampling_rng_seed(uint64_t seed, uint64_t stream,
                                kq_sampling_rng_state *output,
                                kq_diagnostic *diagnostic) {
    kq_sampling_rng_state local;
    kq_diagnostic_clear(diagnostic);
    if (output == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling RNG output is null");
    }
    if (stream > KQ_SAMPLING_MAX_STREAM) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling RNG stream exceeds 63 bits");
    }
    kq_rng_seed_value(seed, stream, &local);
    *output = local;
    return KQ_STATUS_OK;
}

kq_status kq_sampling_rng_reset(kq_sampling_rng_state *state,
                                 kq_diagnostic *diagnostic) {
    kq_sampling_rng_state local;
    kq_diagnostic_clear(diagnostic);
    if (!kq_rng_valid(state)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_RNG_STATE,
                                "sampling RNG reset state is invalid");
    }
    kq_rng_seed_value(state->seed, state->stream, &local);
    *state = local;
    return KQ_STATUS_OK;
}

kq_status kq_sampling_rng_snapshot(const kq_sampling_rng_state *state,
                                    kq_sampling_rng_state *output,
                                    kq_diagnostic *diagnostic) {
    kq_sampling_rng_state local;
    kq_diagnostic_clear(diagnostic);
    if (output == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling RNG snapshot output is null");
    }
    if (!kq_rng_valid(state)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_RNG_STATE,
                                "sampling RNG snapshot source is invalid");
    }
    local = *state;
    *output = local;
    return KQ_STATUS_OK;
}

kq_status kq_sampling_rng_import(const kq_sampling_rng_state *snapshot,
                                  kq_sampling_rng_state *output,
                                  kq_diagnostic *diagnostic) {
    return kq_sampling_rng_snapshot(snapshot, output, diagnostic);
}

kq_status kq_sampling_rng_next_u32(kq_sampling_rng_state *state,
                                    uint32_t *word,
                                    kq_diagnostic *diagnostic) {
    kq_sampling_rng_state local;
    uint32_t local_word = 0U;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (word == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling RNG word output is null");
    }
    if (!kq_rng_valid(state)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_RNG_STATE,
                                "sampling RNG state is invalid");
    }
    local = *state;
    status = kq_rng_next_local(&local, &local_word, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    *state = local;
    *word = local_word;
    return KQ_STATUS_OK;
}

static int kq_id_less_ascending(uint32_t left, uint32_t right,
                                const float *scores) {
    return scores[left] < scores[right] ||
           (scores[left] == scores[right] && left < right);
}

static void kq_sort_ids_ascending(const float *scores, uint32_t *order,
                                  uint32_t *temporary, uint32_t count) {
    uint32_t *current = order;
    uint32_t *next = temporary;
    uint32_t width;
    uint32_t index;

    for (index = 0U; index < count; ++index) {
        order[index] = index;
    }
    for (width = 1U; width < count;) {
        uint32_t left;
        for (left = 0U; left < count;) {
            uint32_t middle = left + width;
            uint32_t right;
            uint32_t first = left;
            uint32_t second;
            uint32_t output = left;
            if (middle > count) {
                middle = count;
            }
            right = middle + width;
            if (right < middle || right > count) {
                right = count;
            }
            second = middle;
            while (first < middle && second < right) {
                if (kq_id_less_ascending(current[second], current[first], scores)) {
                    next[output++] = current[second++];
                } else {
                    next[output++] = current[first++];
                }
            }
            while (first < middle) {
                next[output++] = current[first++];
            }
            while (second < right) {
                next[output++] = current[second++];
            }
            left = right;
        }
        {
            uint32_t *swap = current;
            current = next;
            next = swap;
        }
        if (width > count / 2U) {
            break;
        }
        width *= 2U;
    }
    if (current != order) {
        memcpy(order, current, (size_t)count * sizeof(*order));
    }
}

static kq_status kq_prepare_scratch(const kq_sampling_config *config,
                                    void *scratch, uint64_t scratch_bytes,
                                    kq_sampling_scratch *layout,
                                    kq_diagnostic *diagnostic) {
    uintptr_t start;
    uintptr_t aligned;
    uint64_t leading;
    uint64_t array_bytes;
    unsigned char *cursor;

    if (scratch == NULL || layout == NULL ||
        scratch_bytes < config->scratch_bytes) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "sampling scratch buffer is too small");
    }
    start = (uintptr_t)scratch;
    if (start > UINTPTR_MAX - 7U) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "sampling scratch alignment overflowed");
    }
    aligned = (start + 7U) & ~(uintptr_t)7U;
    leading = (uint64_t)(aligned - start);
    if (!kq_u64_multiply(config->vocabulary_size, sizeof(float),
                         &array_bytes) ||
        leading > scratch_bytes ||
        array_bytes > (scratch_bytes - leading) / 4U) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "sampling scratch layout does not fit");
    }
    cursor = (unsigned char *)aligned;
    layout->scores = (float *)cursor;
    cursor += (size_t)array_bytes;
    layout->probabilities = (float *)cursor;
    cursor += (size_t)array_bytes;
    layout->order = (uint32_t *)cursor;
    cursor += (size_t)array_bytes;
    layout->temporary = (uint32_t *)cursor;
    return KQ_STATUS_OK;
}

static kq_status kq_categorical_word(const float *probabilities,
                                      uint32_t count, uint32_t word,
                                      uint32_t *selected_index,
                                      double *uniform_value,
                                      kq_diagnostic *diagnostic) {
    double total = 0.0;
    double cumulative = 0.0;
    double uniform;
    double threshold;
    uint32_t index;
    uint32_t last = UINT32_MAX;

    if (probabilities == NULL || count == 0U || selected_index == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "categorical selection arguments are invalid");
    }
    for (index = 0U; index < count; ++index) {
        if (!isfinite(probabilities[index]) || probabilities[index] < 0.0f) {
            return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "categorical probability is invalid");
        }
        if (probabilities[index] > 0.0f) {
            total += (double)probabilities[index];
            last = index;
        }
    }
    if (!isfinite(total) || total <= 0.0 || last == UINT32_MAX) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "categorical probability mass is invalid");
    }
    uniform = (double)word / 4294967296.0;
    threshold = uniform * total;
    for (index = 0U; index < count; ++index) {
        cumulative += (double)probabilities[index];
        if (probabilities[index] > 0.0f && threshold < cumulative) {
            *selected_index = index;
            if (uniform_value != NULL) {
                *uniform_value = uniform;
            }
            return KQ_STATUS_OK;
        }
    }
    if (threshold < total) {
        *selected_index = last;
        if (uniform_value != NULL) {
            *uniform_value = uniform;
        }
        return KQ_STATUS_OK;
    }
    return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                            "categorical threshold is outside mass");
}

kq_status kq_sampling_categorical_word_f32_for_test(
    const float *probabilities, uint32_t count, uint32_t word,
    uint32_t *selected_index, kq_diagnostic *diagnostic) {
    uint32_t local = 0U;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    status = kq_categorical_word(probabilities, count, word, &local, NULL,
                                 diagnostic);
    if (status == KQ_STATUS_OK) {
        *selected_index = local;
    }
    return status;
}

kq_status kq_sampling_copy_work_for_test(
    const kq_sampling_config *config, const void *scratch,
    uint64_t scratch_bytes, float *scores, float *probabilities,
    uint64_t output_capacity, kq_diagnostic *diagnostic) {
    kq_sampling_scratch work;
    uint64_t bytes;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (!kq_sampling_config_valid(config) || scratch == NULL || scores == NULL ||
        probabilities == NULL || output_capacity < config->vocabulary_size) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling test-work request is invalid");
    }
    if (!kq_u64_multiply(config->vocabulary_size, sizeof(float), &bytes)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "sampling test-work byte count overflowed");
    }
    if (kq_ranges_overlap(scores, bytes, probabilities, bytes) ||
        kq_ranges_overlap(scores, bytes, scratch, scratch_bytes) ||
        kq_ranges_overlap(probabilities, bytes, scratch, scratch_bytes)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "sampling test-work buffers overlap");
    }
    status = kq_prepare_scratch(config, (void *)scratch, scratch_bytes, &work,
                                diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    memcpy(scores, work.scores, (size_t)bytes);
    memcpy(probabilities, work.probabilities, (size_t)bytes);
    return KQ_STATUS_OK;
}

kq_status kq_sampling_copy_retained_order_for_test(
    const kq_sampling_config *config, const void *scratch,
    uint64_t scratch_bytes, uint32_t *token_ids,
    uint64_t output_capacity, uint64_t *output_count,
    kq_diagnostic *diagnostic) {
    kq_sampling_scratch work;
    uint64_t retained = 0U;
    uint32_t index;
    kq_status status;
    kq_diagnostic_clear(diagnostic);
    if (!kq_sampling_config_valid(config) || scratch == NULL ||
        output_count == NULL) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling retained-order request is invalid");
    }
    status = kq_prepare_scratch(config, (void *)scratch, scratch_bytes, &work,
                                diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < config->vocabulary_size; ++index) {
        uint32_t token_id = work.order[index];
        if (work.probabilities[token_id] > 0.0f) {
            retained += 1U;
        }
    }
    *output_count = retained;
    if (retained != 0U && (token_ids == NULL || output_capacity < retained)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_BUFFER_TOO_SMALL,
                                "sampling retained-order output is too small");
    }
    retained = 0U;
    for (index = 0U; index < config->vocabulary_size; ++index) {
        uint32_t token_id = work.order[index];
        if (work.probabilities[token_id] > 0.0f) {
            token_ids[retained++] = token_id;
        }
    }
    return KQ_STATUS_OK;
}

kq_status kq_sampling_select_f32(const kq_sampling_config *config,
                                  kq_sampling_rng_state *state,
                                  const float *logits,
                                  uint64_t logit_count,
                                  void *scratch,
                                  uint64_t scratch_bytes,
                                  kq_sampling_result *result,
                                  kq_diagnostic *diagnostic) {
    kq_sampling_scratch work;
    kq_sampling_rng_state next_state;
    kq_sampling_result local_result;
    uint64_t logits_bytes;
    uint32_t count;
    uint32_t index;
    uint32_t top_k_start = 0U;
    uint32_t retained_start;
    float maximum;
    float sum;
    float cumulative;
    float cutoff;
    float normalized_sum;
    uint32_t word = 0U;
    uint32_t selected = 0U;
    double uniform = 0.0;
    kq_status status;

    kq_diagnostic_clear(diagnostic);
    if (!kq_sampling_config_valid(config)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INCOMPATIBLE_SAMPLING,
                                "sampling config is invalid");
    }
    if (state == NULL || logits == NULL || result == NULL ||
        logit_count != config->vocabulary_size) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_ARGUMENT,
                                "sampling input or target vocabulary count is invalid");
    }
    if (!kq_rng_valid(state)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_RNG_STATE,
                                "sampling RNG state is invalid");
    }
    if (fegetround() != FE_TONEAREST) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "sampling requires round-to-nearest floating point");
    }
    if (!kq_u64_multiply(logit_count, sizeof(float), &logits_bytes)) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ARITHMETIC_OVERFLOW,
                                "sampling logit byte count overflowed");
    }
    if (kq_ranges_overlap(logits, logits_bytes, scratch, scratch_bytes) ||
        kq_ranges_overlap(state, sizeof(*state), scratch, scratch_bytes) ||
        kq_ranges_overlap(result, sizeof(*result), scratch, scratch_bytes) ||
        kq_ranges_overlap(logits, logits_bytes, state, sizeof(*state)) ||
        kq_ranges_overlap(logits, logits_bytes, result, sizeof(*result)) ||
        kq_ranges_overlap(state, sizeof(*state), result, sizeof(*result)) ||
        kq_ranges_overlap(config, sizeof(*config), logits, logits_bytes) ||
        kq_ranges_overlap(config, sizeof(*config), scratch, scratch_bytes) ||
        kq_ranges_overlap(config, sizeof(*config), state, sizeof(*state)) ||
        kq_ranges_overlap(config, sizeof(*config), result, sizeof(*result))) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_ALIASING_VIOLATION,
                                "sampling buffers overlap");
    }
    status = kq_prepare_scratch(config, scratch, scratch_bytes, &work,
                                diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    count = config->vocabulary_size;
    memset(&local_result, 0, sizeof(local_result));
    memset(work.probabilities, 0, (size_t)count * sizeof(float));
    for (index = 0U; index < count; ++index) {
        float value = logits[index];
        if (!isfinite(value)) {
            return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "sampling logits must all be finite");
        }
        if (config->policy.temperature != 1.0f) {
            value = value / config->policy.temperature;
            if (!isfinite(value)) {
                return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                        "temperature produced a non-finite score");
            }
        }
        work.scores[index] = value;
    }
    kq_sort_ids_ascending(work.scores, work.order, work.temporary, count);
    if (config->policy.top_k != 0U) {
        float threshold = work.scores[
            work.order[count - config->policy.top_k]];
        while (top_k_start < count &&
               work.scores[work.order[top_k_start]] < threshold) {
            top_k_start += 1U;
        }
    }
    if (top_k_start >= count) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "top-k retained no sampling candidate");
    }
    local_result.top_k_retained_count = count - top_k_start;
    maximum = work.scores[work.order[count - 1U]];
    sum = 0.0f;
    for (index = top_k_start; index < count; ++index) {
        uint32_t token_id = work.order[index];
        float weight = expf(work.scores[token_id] - maximum);
        if (!isfinite(weight) || weight < 0.0f) {
            return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                    "top-p softmax produced invalid mass");
        }
        work.probabilities[token_id] = weight;
        sum = sum + weight;
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "top-p normalization mass is invalid");
    }
    retained_start = top_k_start;
    if (config->policy.top_p < 1.0f) {
        cutoff = 1.0f - config->policy.top_p;
        cumulative = 0.0f;
        for (index = top_k_start; index < count; ++index) {
            uint32_t token_id = work.order[index];
            float probability = work.probabilities[token_id] / sum;
            cumulative = cumulative + probability;
            if (index == count - 1U || cumulative > cutoff) {
                retained_start = index;
                break;
            }
            work.probabilities[token_id] = 0.0f;
            retained_start = index + 1U;
        }
    }
    if (retained_start >= count) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "top-p retained no sampling candidate");
    }
    for (index = top_k_start; index < retained_start; ++index) {
        work.probabilities[work.order[index]] = 0.0f;
    }
    local_result.retained_count = count - retained_start;

    sum = 0.0f;
    for (index = 0U; index < count; ++index) {
        if (work.probabilities[index] > 0.0f) {
            float weight = expf(work.scores[index] - maximum);
            work.probabilities[index] = weight;
            sum = sum + weight;
        }
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "final softmax mass is invalid");
    }
    local_result.maximum_probability = 0.0f;
    normalized_sum = 0.0f;
    for (index = 0U; index < count; ++index) {
        if (work.probabilities[index] > 0.0f) {
            float probability = work.probabilities[index] / sum;
            if (!isfinite(probability) || probability < 0.0f) {
                return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                        "final softmax probability is invalid");
            }
            work.probabilities[index] = probability;
            normalized_sum = normalized_sum + probability;
            if (probability > local_result.maximum_probability) {
                local_result.maximum_probability = probability;
            }
        }
    }
    if (!isfinite(normalized_sum) || normalized_sum <= 0.0f) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_NUMERIC_DOMAIN,
                                "normalized probability sum is invalid");
    }

    next_state = *state;
    status = kq_rng_next_local(&next_state, &word, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    status = kq_categorical_word(work.probabilities, count, word, &selected,
                                 &uniform, diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (selected >= config->canonical_token_count) {
        return kq_sampling_fail(diagnostic, KQ_STATUS_INVALID_TOKEN_ID,
                                "sampled model-capacity padding token ID");
    }
    local_result.selected_token_id = selected;
    local_result.rng_word = word;
    local_result.selected_probability = work.probabilities[selected];
    local_result.normalized_sum_f32 = normalized_sum;
    local_result.uniform_value = uniform;
    local_result.rng_draws_before = state->draws;
    local_result.rng_draws_after = next_state.draws;
    *state = next_state;
    *result = local_result;
    return KQ_STATUS_OK;
}
