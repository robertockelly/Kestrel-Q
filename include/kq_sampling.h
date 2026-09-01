#ifndef KQ_SAMPLING_H
#define KQ_SAMPLING_H

#include <stdint.h>

#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_SAMPLING_QWEN38_VOCAB_SIZE UINT32_C(248320)
#define KQ_SAMPLING_QWEN38_CANONICAL_TOKEN_COUNT UINT32_C(248077)
#define KQ_SAMPLING_QWEN38_EOS_ID UINT32_C(248046)
#define KQ_SAMPLING_QWEN38_PAD_ID UINT32_C(248044)
#define KQ_SAMPLING_POLICY_VERSION UINT32_C(1)
#define KQ_SAMPLING_RNG_STATE_VERSION UINT32_C(1)
#define KQ_SAMPLING_DEFAULT_STREAM UINT64_C(0x4b515f53414d504c)
#define KQ_SAMPLING_MAX_STREAM UINT64_C(0x7fffffffffffffff)

typedef struct kq_sampling_config kq_sampling_config;

typedef struct kq_sampling_policy {
    uint32_t version;
    uint32_t flags;
    float temperature;
    float top_p;
    uint32_t top_k;
    uint32_t reserved[3];
} kq_sampling_policy;

typedef struct kq_sampling_rng_state {
    uint32_t version;
    uint32_t reserved;
    uint64_t seed;
    uint64_t stream;
    uint64_t state;
    uint64_t increment;
    uint64_t draws;
    uint64_t integrity;
} kq_sampling_rng_state;

typedef struct kq_sampling_result {
    uint32_t selected_token_id;
    uint32_t top_k_retained_count;
    uint32_t retained_count;
    uint32_t rng_word;
    float selected_probability;
    float maximum_probability;
    float normalized_sum_f32;
    uint32_t reserved;
    double uniform_value;
    uint64_t rng_draws_before;
    uint64_t rng_draws_after;
} kq_sampling_result;

void kq_sampling_policy_qwen38_default(kq_sampling_policy *policy);

kq_status kq_sampling_config_open_qwen38(
    const kq_sampling_policy *policy,
    kq_sampling_config **output,
    kq_diagnostic *diagnostic);
void kq_sampling_config_close(kq_sampling_config *config);
uint64_t kq_sampling_config_owned_bytes(const kq_sampling_config *config);
uint32_t kq_sampling_config_vocabulary_size(const kq_sampling_config *config);
uint64_t kq_sampling_required_scratch_bytes(const kq_sampling_config *config);

kq_status kq_sampling_rng_seed(uint64_t seed,
                                uint64_t stream,
                                kq_sampling_rng_state *output,
                                kq_diagnostic *diagnostic);
kq_status kq_sampling_rng_reset(kq_sampling_rng_state *state,
                                 kq_diagnostic *diagnostic);
kq_status kq_sampling_rng_snapshot(const kq_sampling_rng_state *state,
                                    kq_sampling_rng_state *output,
                                    kq_diagnostic *diagnostic);
kq_status kq_sampling_rng_import(const kq_sampling_rng_state *snapshot,
                                  kq_sampling_rng_state *output,
                                  kq_diagnostic *diagnostic);
kq_status kq_sampling_rng_next_u32(kq_sampling_rng_state *state,
                                    uint32_t *word,
                                    kq_diagnostic *diagnostic);

kq_status kq_sampling_select_f32(const kq_sampling_config *config,
                                  kq_sampling_rng_state *state,
                                  const float *logits,
                                  uint64_t logit_count,
                                  void *scratch,
                                  uint64_t scratch_bytes,
                                  kq_sampling_result *result,
                                  kq_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
