#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_sampling.h"
#include "kq_sampling_internal.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static void fill_logits(float *logits, float value) {
    uint32_t index;
    for (index = 0U; index < KQ_SAMPLING_QWEN38_VOCAB_SIZE; ++index) {
        logits[index] = value;
    }
}

static int rng_equal(const kq_sampling_rng_state *left,
                     const kq_sampling_rng_state *right) {
    return left->version == right->version &&
           left->reserved == right->reserved &&
           left->seed == right->seed && left->stream == right->stream &&
           left->state == right->state &&
           left->increment == right->increment &&
           left->draws == right->draws &&
           left->integrity == right->integrity;
}

static void test_rng(void) {
    static const uint32_t expected[] = {
        UINT32_C(0xe4c14788), UINT32_C(0x379c6516),
        UINT32_C(0x5c4ab3bb), UINT32_C(0x601d23e0),
        UINT32_C(0x1c382b8c), UINT32_C(0xd1faab16),
        UINT32_C(0x67680a2d), UINT32_C(0x92014a6e),
    };
    kq_sampling_rng_state state;
    kq_sampling_rng_state snapshot;
    kq_sampling_rng_state imported;
    kq_sampling_rng_state before;
    kq_diagnostic diagnostic;
    uint32_t word = 0U;
    uint32_t index;

    check(kq_sampling_rng_seed(0U, 0U, &state, &diagnostic) == KQ_STATUS_OK &&
              state.state == UINT64_C(0x5851f42d4c957f2e),
          "PCG zero seed/stream expansion matches independent vector");
    for (index = 0U; index < 8U; ++index) {
        check(kq_sampling_rng_next_u32(&state, &word, &diagnostic) ==
                  KQ_STATUS_OK && word == expected[index],
              "PCG output word matches independent vector");
    }
    check(state.draws == 8U,
          "PCG successful word generation increments draw count");
    check(kq_sampling_rng_snapshot(&state, &snapshot, &diagnostic) ==
              KQ_STATUS_OK && rng_equal(&state, &snapshot),
          "RNG snapshot preserves semantic state");
    memset(&imported, 0, sizeof(imported));
    check(kq_sampling_rng_import(&snapshot, &imported, &diagnostic) ==
              KQ_STATUS_OK && rng_equal(&snapshot, &imported),
          "RNG import preserves semantic state");
    check(kq_sampling_rng_reset(&state, &diagnostic) == KQ_STATUS_OK &&
              state.draws == 0U &&
              state.state == UINT64_C(0x5851f42d4c957f2e),
          "RNG reset repeats seed expansion");
    before = state;
    state.integrity ^= UINT64_C(1);
    word = UINT32_C(0xaaaaaaaa);
    check(kq_sampling_rng_next_u32(&state, &word, &diagnostic) ==
              KQ_STATUS_INVALID_RNG_STATE && word == UINT32_C(0xaaaaaaaa),
          "corrupted RNG state fails without publishing a word");
    state = before;
    check(kq_sampling_rng_seed(0U, UINT64_MAX, &imported, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT,
          "stream wider than 63 bits fails closed");
}

static void test_categorical_boundaries(void) {
    float probabilities[] = {0.5f, 0.5f};
    uint32_t selected = UINT32_MAX;
    kq_diagnostic diagnostic;
    check(kq_sampling_categorical_word_f32_for_test(
              probabilities, 2U, 0U, &selected, &diagnostic) == KQ_STATUS_OK &&
              selected == 0U,
          "categorical interval start selects the first token");
    check(kq_sampling_categorical_word_f32_for_test(
              probabilities, 2U, UINT32_C(0x80000000), &selected,
              &diagnostic) == KQ_STATUS_OK && selected == 1U,
          "categorical exact boundary belongs to the next token");
    check(kq_sampling_categorical_word_f32_for_test(
              probabilities, 2U, UINT32_MAX, &selected, &diagnostic) ==
              KQ_STATUS_OK && selected == 1U,
          "largest variate below one selects the final interval");
    probabilities[0] = -1.0f;
    selected = UINT32_MAX;
    check(kq_sampling_categorical_word_f32_for_test(
              probabilities, 2U, 0U, &selected, &diagnostic) ==
              KQ_STATUS_NUMERIC_DOMAIN && selected == UINT32_MAX,
          "invalid categorical mass fails without publishing selection");
}

static void test_sampler(void) {
    kq_sampling_policy policy;
    kq_sampling_config *config = NULL;
    kq_sampling_rng_state state;
    kq_sampling_rng_state before_state;
    kq_sampling_result result;
    kq_sampling_result before_result;
    kq_diagnostic diagnostic;
    float *logits = NULL;
    float *copy = NULL;
    void *scratch = NULL;
    uint64_t scratch_bytes;
    int prior_rounding;

    kq_sampling_policy_qwen38_default(&policy);
    check(policy.temperature == 1.0f && policy.top_k == 20U &&
              policy.top_p == 0.95f,
          "official Qwen3.8 sampling policy is exact");
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_OK && config != NULL,
          "official sampling config constructs");
    check(kq_sampling_config_vocabulary_size(config) ==
              KQ_SAMPLING_QWEN38_VOCAB_SIZE &&
              kq_sampling_config_owned_bytes(config) > 0U,
          "sampling config reports target vocabulary and ownership");
    scratch_bytes = kq_sampling_required_scratch_bytes(config);
    logits = (float *)malloc((size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE *
                             sizeof(float));
    copy = (float *)malloc((size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE *
                           sizeof(float));
    scratch = malloc((size_t)scratch_bytes);
    check(logits != NULL && copy != NULL && scratch != NULL,
          "sampling test buffers allocate");
    if (logits == NULL || copy == NULL || scratch == NULL) {
        free(logits); free(copy); free(scratch);
        kq_sampling_config_close(config);
        return;
    }

    fill_logits(logits, -80.0f);
    logits[5] = 3.0f;
    logits[6] = 2.0f;
    logits[7] = 2.0f;
    logits[8] = 2.0f;
    logits[9] = 1.0f;
    memcpy(copy, logits, (size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float));
    kq_sampling_config_close(config);
    config = NULL;
    policy.top_k = 3U;
    policy.top_p = 1.0f;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_OK,
          "top-k tie policy constructs");
    check(kq_sampling_rng_seed(1U, KQ_SAMPLING_DEFAULT_STREAM,
                               &state, &diagnostic) == KQ_STATUS_OK,
          "sampler RNG seeds");
    memset(&result, 0, sizeof(result));
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_OK &&
              result.top_k_retained_count == 4U &&
              result.retained_count == 4U &&
              result.selected_token_id == 6U && state.draws == 1U,
          "top-k threshold ties and fixed selection match independent oracle");
    check(memcmp(logits, copy,
                 (size_t)KQ_SAMPLING_QWEN38_VOCAB_SIZE * sizeof(float)) == 0,
          "sampler does not mutate caller logits");

    fill_logits(logits, -80.0f);
    logits[248077] = 30.0f;
    check(kq_sampling_rng_seed(UINT64_C(0x105), KQ_SAMPLING_DEFAULT_STREAM,
                               &state, &diagnostic) == KQ_STATUS_OK,
          "padded-ID failure RNG seeds");
    before_state = state;
    memset(&result, 0xa5, sizeof(result));
    before_result = result;
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_INVALID_TOKEN_ID &&
              rng_equal(&state, &before_state) &&
              memcmp(&result, &before_result, sizeof(result)) == 0,
          "padded selected ID fails with RNG/result rollback");

    fill_logits(logits, -80.0f);
    logits[248046] = 20.0f;
    check(kq_sampling_rng_seed(6U, KQ_SAMPLING_DEFAULT_STREAM,
                               &state, &diagnostic) == KQ_STATUS_OK &&
              kq_sampling_select_f32(config, &state, logits,
                  KQ_SAMPLING_QWEN38_VOCAB_SIZE, scratch, scratch_bytes,
                  &result, &diagnostic) == KQ_STATUS_OK &&
              result.selected_token_id == 248046U,
          "EOG token remains an eligible canonical selection");

    before_state = state;
    before_result = result;
    logits[0] = NAN;
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_NUMERIC_DOMAIN &&
              rng_equal(&state, &before_state) &&
              memcmp(&result, &before_result, sizeof(result)) == 0,
          "non-finite logits fail transactionally");
    logits[0] = -80.0f;
    logits[1] = INFINITY;
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_NUMERIC_DOMAIN,
          "positive-infinite logit fails closed");
    logits[1] = -INFINITY;
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_NUMERIC_DOMAIN,
          "negative-infinite logit fails closed");
    logits[1] = -80.0f;
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE - 1U,
                                 scratch, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_INVALID_ARGUMENT,
          "wrong target vocabulary count fails closed");
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 scratch, scratch_bytes - 1U, &result,
                                 &diagnostic) == KQ_STATUS_BUFFER_TOO_SMALL,
          "short sampling scratch fails without truncation");
    check(kq_sampling_select_f32(config, &state, logits,
                                 KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                 logits, scratch_bytes, &result,
                                 &diagnostic) == KQ_STATUS_ALIASING_VIOLATION,
          "logit/scratch alias fails closed");
    prior_rounding = fegetround();
    if (fesetround(FE_DOWNWARD) == 0) {
        check(kq_sampling_select_f32(config, &state, logits,
                                     KQ_SAMPLING_QWEN38_VOCAB_SIZE,
                                     scratch, scratch_bytes, &result,
                                     &diagnostic) == KQ_STATUS_NUMERIC_DOMAIN,
              "noncanonical floating environment fails closed");
        (void)fesetround(prior_rounding);
    }

    kq_sampling_config_close(config);
    config = NULL;
    kq_sampling_policy_qwen38_default(&policy);
    policy.temperature = 0.0f;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_INCOMPATIBLE_SAMPLING && config == NULL,
          "zero temperature fails closed");
    kq_sampling_policy_qwen38_default(&policy);
    policy.temperature = INFINITY;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_INCOMPATIBLE_SAMPLING,
          "non-finite temperature fails closed");
    kq_sampling_policy_qwen38_default(&policy);
    policy.top_p = 1.01f;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_INCOMPATIBLE_SAMPLING,
          "invalid top-p fails closed");
    kq_sampling_policy_qwen38_default(&policy);
    policy.top_k = KQ_SAMPLING_QWEN38_VOCAB_SIZE + 1U;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_INCOMPATIBLE_SAMPLING,
          "oversized top-k fails closed instead of clamping");
    kq_sampling_policy_qwen38_default(&policy);
    policy.flags = 1U;
    check(kq_sampling_config_open_qwen38(&policy, &config, &diagnostic) ==
              KQ_STATUS_INCOMPATIBLE_SAMPLING,
          "unsupported sampling option flags fail closed");
    kq_sampling_policy_qwen38_default(&policy);
    check(kq_sampling_config_open_qwen38(
              &policy, (kq_sampling_config **)(void *)&policy,
              &diagnostic) == KQ_STATUS_ALIASING_VIOLATION,
          "policy/config-output alias fails closed");
    free(logits);
    free(copy);
    free(scratch);
}

static void test_statistical(void) {
    static const float distributions[][4] = {
        {0.5f, 0.5f, 0.0f, 0.0f},
        {0.25f, 0.25f, 0.25f, 0.25f},
    };
    static const uint32_t sizes[] = {2U, 4U};
    static const uint64_t seeds[] = {UINT64_C(0x300), UINT64_C(0x301)};
    const double epsilon = 0.007355194859556844;
    kq_sampling_rng_state state;
    kq_diagnostic diagnostic;
    uint32_t distribution;

    for (distribution = 0U; distribution < 2U; ++distribution) {
        uint32_t counts[4] = {0U, 0U, 0U, 0U};
        uint32_t trial;
        check(kq_sampling_rng_seed(seeds[distribution],
                                   KQ_SAMPLING_DEFAULT_STREAM,
                                   &state, &diagnostic) == KQ_STATUS_OK,
              "statistical RNG seeds");
        for (trial = 0U; trial < 100000U; ++trial) {
            uint32_t word = 0U;
            uint32_t selected = UINT32_MAX;
            if (kq_sampling_rng_next_u32(&state, &word, &diagnostic) !=
                    KQ_STATUS_OK ||
                kq_sampling_categorical_word_f32_for_test(
                    distributions[distribution], sizes[distribution], word,
                    &selected, &diagnostic) != KQ_STATUS_OK) {
                check(0, "statistical draw succeeds");
                break;
            }
            counts[selected] += 1U;
        }
        for (trial = 0U; trial < sizes[distribution]; ++trial) {
            double observed = (double)counts[trial] / 100000.0;
            check(fabs(observed - distributions[distribution][trial]) <= epsilon,
                  "predeclared Hoeffding category bound passes");
        }
    }
}

int main(void) {
    test_rng();
    test_categorical_boundaries();
    test_sampler();
    test_statistical();
    if (failures != 0) {
        fprintf(stderr, "%d sampling test(s) failed\n", failures);
        return 1;
    }
    printf("sampling synthetic tests passed\n");
    return 0;
}
