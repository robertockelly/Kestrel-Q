#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_model_exec.h"
#include "kq_model_exec_internal.h"
#include "kq_sampling.h"
#include "kq_sampling_internal.h"
#include "kq_tokenizer.h"
#include "kq_weight_provider.h"
#include "kq_weight_provider_internal.h"

#define KQ_TEST_CONTEXT UINT64_C(16)
#define KQ_TEST_STEPS 4U
#define KQ_TEST_PROVIDER_BUDGET \
    (UINT64_C(512) * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024))
#define KQ_TRACE_CAPACITY 65536U
#define KQ_PLE_TRACE_CAPACITY 2048U
#define KQ_TRACE_PAYLOAD_LIMIT \
    (UINT64_C(96) * UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024))

static const uint32_t kq_prompt_ids[] =
    {9419U, 11U, 710U, 467U, 3621U, 27325U, 13U};

typedef struct trace_counts {
    uint64_t routes;
    uint64_t experts;
    uint64_t ple_rows;
} trace_counts;

typedef struct fault_context {
    kq_weight_provider *provider;
    uint32_t layer;
    int fired;
} fault_context;

typedef struct sampled_step {
    kq_model_exec_result model;
    kq_sampling_result sample;
    kq_sampling_rng_state rng_before;
    kq_sampling_rng_state rng_after;
    kq_model_exec_state_summary state;
    unsigned char decoded[256];
    uint64_t route_hash;
    uint64_t ple_hash;
    uint64_t survivor_hash;
    uint64_t survivor_count;
} sampled_step;

typedef struct sampled_trace {
    const char *name;
    uint64_t seed;
    uint64_t stream;
    sampled_step steps[KQ_TEST_STEPS];
    uint32_t step_count;
    uint64_t total_payload;
    uint64_t total_blocks;
    uint64_t total_elapsed;
} sampled_trace;

typedef struct test_environment {
    kq_file *file;
    kq_gguf *gguf;
    kq_model *model;
    kq_tokenizer *tokenizer;
    kq_weight_provider *provider;
    kq_model_exec_config *config;
    kq_sampling_config *sampling_config;
    float *logits;
    float *control_logits;
    void *model_scratch;
    uint64_t model_scratch_bytes;
    void *sampling_scratch;
    uint64_t sampling_scratch_bytes;
    const char *capture_dir;
} test_environment;

static wchar_t *model_path(void) {
    DWORD count = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    wchar_t *path;
    if (count == 0U) return NULL;
    path = (wchar_t *)malloc((size_t)count * sizeof(*path));
    if (path == NULL) return NULL;
    if (GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, count) >= count) {
        free(path);
        return NULL;
    }
    return path;
}

static void inject_failure(const kq_model_exec_progress_event *event,
                           void *user_data) {
    fault_context *fault = (fault_context *)user_data;
    if (event != NULL && fault != NULL && !fault->fired &&
        event->phase == KQ_MODEL_EXEC_PHASE_LAYER_BEGIN &&
        event->layer_id == fault->layer) {
        kq_weight_provider_test_fail_next_request(fault->provider);
        fault->fired = 1;
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

static int state_equal(const kq_model_exec_state_summary *left,
                       const kq_model_exec_state_summary *right) {
    return left->model_position == right->model_position &&
           left->layer_position_min == right->layer_position_min &&
           left->layer_position_max == right->layer_position_max &&
           left->qsa_sequence_length_min == right->qsa_sequence_length_min &&
           left->qsa_sequence_length_max == right->qsa_sequence_length_max &&
           left->qsa_complete_blocks_min == right->qsa_complete_blocks_min &&
           left->qsa_complete_blocks_max == right->qsa_complete_blocks_max &&
           left->qsa_incomplete_tail_min == right->qsa_incomplete_tail_min &&
           left->qsa_incomplete_tail_max == right->qsa_incomplete_tail_max &&
           left->ple_address_position == right->ple_address_position &&
           left->ple_value_position == right->ple_value_position &&
           left->gdn_state_hash == right->gdn_state_hash &&
           left->qsa_state_hash == right->qsa_state_hash &&
           left->ple_address_state_hash == right->ple_address_state_hash &&
           left->ple_value_state_hash == right->ple_value_state_hash &&
           left->structural_hash == right->structural_hash &&
           left->gdn_initialized_layers == right->gdn_initialized_layers &&
           left->qsa_layers == right->qsa_layers;
}

static int get_trace_counts(const kq_weight_provider *provider,
                            trace_counts *counts) {
    kq_status route_status;
    kq_status expert_status;
    kq_status ple_status;
    if (provider == NULL || counts == NULL) return 0;
    route_status = kq_weight_provider_copy_route_requests(
        provider, NULL, 0U, &counts->routes);
    expert_status = kq_weight_provider_copy_expert_requests(
        provider, NULL, 0U, &counts->experts);
    ple_status = kq_weight_provider_copy_ple_requests(
        provider, NULL, 0U, &counts->ple_rows);
    return (route_status == KQ_STATUS_OK ||
            route_status == KQ_STATUS_BUFFER_TOO_SMALL) &&
           (expert_status == KQ_STATUS_OK ||
            expert_status == KQ_STATUS_BUFFER_TOO_SMALL) &&
           (ple_status == KQ_STATUS_OK ||
            ple_status == KQ_STATUS_BUFFER_TOO_SMALL);
}

static uint64_t hash_request(uint64_t hash, uint32_t first,
                             uint64_t second) {
    uint32_t index;
    hash ^= first;
    hash *= UINT64_C(1099511628211);
    for (index = 0U; index < 8U; ++index) {
        hash ^= (second >> (index * 8U)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t hash_u32_order(const uint32_t *values, uint64_t count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    uint64_t index;
    uint32_t byte_index;
    for (index = 0U; index < count; ++index) {
        for (byte_index = 0U; byte_index < 4U; ++byte_index) {
            hash ^= (values[index] >> (byte_index * 8U)) & UINT64_C(0xff);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static int validate_trace_delta(
    const kq_weight_provider *provider, const trace_counts *before,
    uint64_t input_tokens, uint64_t *route_hash, uint64_t *ple_hash) {
    trace_counts after;
    kq_weight_provider_expert_request *routes = NULL;
    kq_weight_provider_expert_request *experts = NULL;
    kq_weight_provider_ple_request *ple = NULL;
    uint64_t route_expected = input_tokens * 48U * 10U;
    uint64_t expert_expected = route_expected;
    uint64_t ple_expected = input_tokens * 16U;
    uint64_t group;
    uint64_t index;
    int passed = 0;
    if (!get_trace_counts(provider, &after) ||
        after.routes < before->routes || after.experts < before->experts ||
        after.ple_rows < before->ple_rows ||
        after.routes - before->routes != route_expected ||
        after.experts - before->experts != expert_expected ||
        after.ple_rows - before->ple_rows != ple_expected) return 0;
    routes = (kq_weight_provider_expert_request *)malloc(
        (size_t)after.routes * sizeof(*routes));
    experts = (kq_weight_provider_expert_request *)malloc(
        (size_t)after.experts * sizeof(*experts));
    ple = (kq_weight_provider_ple_request *)malloc(
        (size_t)after.ple_rows * sizeof(*ple));
    if (routes == NULL || experts == NULL || ple == NULL) goto done;
    if (kq_weight_provider_copy_route_requests(
            provider, routes, after.routes, &index) != KQ_STATUS_OK ||
        index != after.routes ||
        kq_weight_provider_copy_expert_requests(
            provider, experts, after.experts, &index) != KQ_STATUS_OK ||
        index != after.experts ||
        kq_weight_provider_copy_ple_requests(
            provider, ple, after.ple_rows, &index) != KQ_STATUS_OK ||
        index != after.ple_rows) goto done;
    *route_hash = UINT64_C(14695981039346656037);
    for (group = 0U; group < route_expected / 10U; ++group) {
        unsigned char selected[512];
        uint32_t expected_layer =
            (uint32_t)((group / input_tokens) % 48U);
        memset(selected, 0, sizeof(selected));
        for (index = 0U; index < 10U; ++index) {
            const kq_weight_provider_expert_request *request =
                &routes[before->routes + group * 10U + index];
            if (request->layer_id != expected_layer ||
                request->expert_id >= 512U || selected[request->expert_id])
                goto done;
            selected[request->expert_id] = 1U;
            *route_hash = hash_request(*route_hash, request->layer_id,
                                       request->expert_id);
        }
        for (index = 0U; index < 10U; ++index) {
            const kq_weight_provider_expert_request *request =
                &experts[before->experts + group * 10U + index];
            if (request->layer_id != expected_layer ||
                request->expert_id >= 512U || !selected[request->expert_id])
                goto done;
            selected[request->expert_id] = 0U;
        }
        for (index = 0U; index < 512U; ++index)
            if (selected[index]) goto done;
    }
    *ple_hash = UINT64_C(14695981039346656037);
    for (index = before->ple_rows; index < after.ple_rows; ++index)
        *ple_hash = hash_request(*ple_hash, ple[index].logical_member,
                                 ple[index].member_row);
    passed = 1;
done:
    free(ple);
    free(experts);
    free(routes);
    return passed;
}

static int outputs_unchanged(const float *logits,
                             const unsigned char *decoded,
                             const kq_model_exec_result *model,
                             const kq_sampling_result *sample) {
    const unsigned char *bytes;
    uint64_t index;
    for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
        if (logits[index] != -1234.5f) return 0;
    for (index = 0U; index < 256U; ++index)
        if (decoded[index] != 0xa5U) return 0;
    bytes = (const unsigned char *)model;
    for (index = 0U; index < sizeof(*model); ++index)
        if (bytes[index] != 0x5aU) return 0;
    bytes = (const unsigned char *)sample;
    for (index = 0U; index < sizeof(*sample); ++index)
        if (bytes[index] != 0x3cU) return 0;
    return 1;
}

static void prepare_output_sentinels(
    test_environment *env, unsigned char *decoded,
    kq_model_exec_result *model, kq_sampling_result *sample) {
    uint64_t index;
    for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
        env->logits[index] = -1234.5f;
    memset(decoded, 0xa5, 256U);
    memset(model, 0x5a, sizeof(*model));
    memset(sample, 0x3c, sizeof(*sample));
}

static int capture_logits(const test_environment *env, const char *trace_name,
                          uint32_t step) {
    char path[1024];
    FILE *file;
    size_t written;
    int count;
    if (env->capture_dir == NULL) return 1;
    count = snprintf(path, sizeof(path), "%s/%s-step-%u-logits.f32",
                     env->capture_dir, trace_name, (unsigned int)step);
    if (count < 0 || (size_t)count >= sizeof(path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    written = fwrite(env->logits, sizeof(float),
                     KQ_MODEL_EXEC_VOCABULARY_SIZE, file);
    if (fclose(file) != 0) return 0;
    return written == KQ_MODEL_EXEC_VOCABULARY_SIZE;
}

static void print_hex(const unsigned char *data, uint64_t bytes) {
    uint64_t index;
    for (index = 0U; index < bytes; ++index)
        (void)printf("%02x", (unsigned int)data[index]);
}

static int finalize_step(test_environment *env, sampled_trace *trace,
                         uint32_t step_index, uint64_t input_tokens,
                         const trace_counts *before,
                         kq_model_exec_state *state,
                         const kq_sampling_rng_state *rng_before,
                         const kq_sampling_rng_state *rng_after,
                         const kq_model_exec_result *model_result,
                         const kq_sampling_result *sample_result,
                         const unsigned char *decoded,
                         kq_diagnostic *diagnostic) {
    sampled_step *step = &trace->steps[step_index];
    uint32_t *survivors = NULL;
    uint64_t survivor_count = 0U;
    uint64_t expected_position = 7U + step_index;
    uint64_t expected_tail = expected_position % 4U;
    uint64_t expected_complete = expected_position / 4U;
    kq_status status;
    memset(step, 0, sizeof(*step));
    step->model = *model_result;
    step->sample = *sample_result;
    step->rng_before = *rng_before;
    step->rng_after = *rng_after;
    if (model_result->decoded_utf8_bytes > sizeof(step->decoded)) return 0;
    if (model_result->decoded_utf8_bytes != 0U)
        memcpy(step->decoded, decoded,
               (size_t)model_result->decoded_utf8_bytes);
    status = kq_model_exec_state_get_summary(state, &step->state, diagnostic);
    if (status != KQ_STATUS_OK ||
        !validate_trace_delta(env->provider, before, input_tokens,
                              &step->route_hash, &step->ple_hash)) return 0;
    status = kq_sampling_copy_retained_order_for_test(
        env->sampling_config, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, 0U, &survivor_count, diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL || survivor_count == 0U ||
        survivor_count != sample_result->retained_count ||
        survivor_count > SIZE_MAX / sizeof(*survivors)) return 0;
    survivors = (uint32_t *)malloc((size_t)survivor_count * sizeof(*survivors));
    if (survivors == NULL) return 0;
    status = kq_sampling_copy_retained_order_for_test(
        env->sampling_config, env->sampling_scratch,
        env->sampling_scratch_bytes, survivors, survivor_count,
        &survivor_count, diagnostic);
    if (status != KQ_STATUS_OK) {
        free(survivors);
        return 0;
    }
    step->survivor_count = survivor_count;
    step->survivor_hash = hash_u32_order(survivors, survivor_count);
    free(survivors);
    if (sample_result->selected_token_id != model_result->selected_token_id ||
        sample_result->rng_draws_after != sample_result->rng_draws_before + 1U ||
        rng_before->draws + 1U != rng_after->draws ||
        step->state.model_position != expected_position ||
        step->state.layer_position_min != expected_position ||
        step->state.layer_position_max != expected_position ||
        step->state.qsa_sequence_length_min != expected_position ||
        step->state.qsa_sequence_length_max != expected_position ||
        step->state.qsa_complete_blocks_min != expected_complete ||
        step->state.qsa_complete_blocks_max != expected_complete ||
        step->state.qsa_incomplete_tail_min != expected_tail ||
        step->state.qsa_incomplete_tail_max != expected_tail ||
        step->state.ple_address_position != expected_position ||
        step->state.ple_value_position != expected_position ||
        step->state.gdn_initialized_layers != 36U ||
        step->state.qsa_layers != 12U ||
        model_result->metrics.layers_completed != 48U ||
        model_result->metrics.routed_expert_selections != input_tokens * 480U ||
        model_result->metrics.selected_expert_matrix_requests !=
            input_tokens * 1440U ||
        model_result->metrics.routed_expert_member_requests !=
            input_tokens * 1440U ||
        model_result->metrics.ple_row_requests != input_tokens * 16U ||
        model_result->metrics.prompt_prefill_count !=
            (step_index == 0U ? 1U : 0U) ||
        model_result->metrics.incremental_decode_count !=
            (step_index == 0U ? 0U : 1U) ||
        model_result->metrics.maximum_f32_weight_bytes_materialized >
            KQ_MODEL_EXEC_VOCABULARY_SIZE * sizeof(float)) return 0;
    if (step_index == 0U) {
        if (model_result->metrics.qsa_candidate_blocks != 48U ||
            model_result->metrics.qsa_selected_blocks != 48U ||
            model_result->metrics.qsa_selected_tokens != 336U) return 0;
    } else if (model_result->metrics.qsa_candidate_blocks != 24U ||
               model_result->metrics.qsa_selected_blocks != 24U ||
               model_result->metrics.qsa_selected_tokens !=
                   expected_position * 12U) return 0;
    if (!capture_logits(env, trace->name, step_index + 1U)) return 0;
    trace->total_payload += model_result->metrics.logical_payload_bytes_touched;
    trace->total_blocks += model_result->metrics.payload_blocks_touched;
    trace->total_elapsed += model_result->metrics.elapsed_nanoseconds;
    (void)printf(
        "sampled trace=%s step=%u seed=%llu stream=%llu token=%u eog=%d "
        "decoded_hex=", trace->name, (unsigned int)(step_index + 1U),
        (unsigned long long)trace->seed,
        (unsigned long long)trace->stream,
        (unsigned int)model_result->selected_token_id,
        model_result->selected_token_is_eog);
    print_hex(decoded, model_result->decoded_utf8_bytes);
    (void)printf(
        " position=%llu state_hash=%016llx gdn_hash=%016llx "
        "qsa_hash=%016llx ple_address_hash=%016llx ple_value_hash=%016llx "
        "rng_pre_state=%llu rng_pre_increment=%llu rng_pre_draws=%llu "
        "rng_pre_integrity=%llu rng_word=%u rng_post_state=%llu "
        "rng_post_increment=%llu rng_post_draws=%llu rng_post_integrity=%llu "
        "survivors=%llu survivor_hash=%016llx topk_retained=%u "
        "route_hash=%016llx ple_hash=%016llx qsa_candidates=%llu "
        "qsa_blocks=%llu qsa_tokens=%llu payload=%llu blocks=%llu "
        "semantics=%llu embedding_bytes=%llu expert_selections=%llu "
        "expert_matrices=%llu ple_rows=%llu lm_head_bytes=%llu "
        "persistent_state=%llu peak_scratch=%llu logits_bytes=%llu "
        "max_f32_weight=%llu elapsed_ns=%llu\n",
        (unsigned long long)step->state.model_position,
        (unsigned long long)step->state.structural_hash,
        (unsigned long long)step->state.gdn_state_hash,
        (unsigned long long)step->state.qsa_state_hash,
        (unsigned long long)step->state.ple_address_state_hash,
        (unsigned long long)step->state.ple_value_state_hash,
        (unsigned long long)rng_before->state,
        (unsigned long long)rng_before->increment,
        (unsigned long long)rng_before->draws,
        (unsigned long long)rng_before->integrity,
        (unsigned int)sample_result->rng_word,
        (unsigned long long)rng_after->state,
        (unsigned long long)rng_after->increment,
        (unsigned long long)rng_after->draws,
        (unsigned long long)rng_after->integrity,
        (unsigned long long)step->survivor_count,
        (unsigned long long)step->survivor_hash,
        (unsigned int)sample_result->top_k_retained_count,
        (unsigned long long)step->route_hash,
        (unsigned long long)step->ple_hash,
        (unsigned long long)model_result->metrics.qsa_candidate_blocks,
        (unsigned long long)model_result->metrics.qsa_selected_blocks,
        (unsigned long long)model_result->metrics.qsa_selected_tokens,
        (unsigned long long)model_result->metrics.logical_payload_bytes_touched,
        (unsigned long long)model_result->metrics.payload_blocks_touched,
        (unsigned long long)model_result->metrics.unique_semantic_tensors_touched,
        (unsigned long long)model_result->metrics.embedding_logical_bytes_touched,
        (unsigned long long)model_result->metrics.routed_expert_selections,
        (unsigned long long)model_result->metrics.selected_expert_matrix_requests,
        (unsigned long long)model_result->metrics.ple_row_requests,
        (unsigned long long)model_result->metrics.lm_head_logical_bytes_touched,
        (unsigned long long)model_result->metrics.persistent_state_bytes,
        (unsigned long long)model_result->metrics.peak_scratch_bytes,
        (unsigned long long)model_result->metrics.logits_bytes,
        (unsigned long long)model_result->metrics.maximum_f32_weight_bytes_materialized,
        (unsigned long long)model_result->metrics.elapsed_nanoseconds);
    return 1;
}

static int run_failure(
    test_environment *env, kq_model_exec_state *state,
    kq_sampling_rng_state *rng, uint32_t input_token,
    const char *name, uint32_t failure_layer,
    const float *selection_logits, kq_status expected_status,
    kq_diagnostic *diagnostic) {
    kq_model_exec_state_summary before_state;
    kq_model_exec_state_summary after_state;
    kq_sampling_rng_state before_rng;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    unsigned char decoded[256];
    fault_context fault;
    kq_status status;
    fault.provider = env->provider;
    fault.layer = failure_layer;
    fault.fired = 0;
    if (kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
            KQ_STATUS_OK ||
        kq_sampling_rng_snapshot(rng, &before_rng, diagnostic) != KQ_STATUS_OK)
        return 0;
    prepare_output_sentinels(env, decoded, &model_result, &sample_result);
    if (selection_logits != NULL) {
        status = kq_model_exec_sampled_decode_one_with_selection_logits_for_test(
            env->config, state, env->sampling_config, rng, input_token,
            selection_logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            decoded, sizeof(decoded), env->model_scratch,
            env->model_scratch_bytes, env->sampling_scratch,
            env->sampling_scratch_bytes, NULL, NULL, &model_result,
            &sample_result, diagnostic);
    } else {
        status = kq_model_exec_sampled_decode_one_f32(
            env->config, state, env->sampling_config, rng, input_token,
            env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            decoded, sizeof(decoded), env->model_scratch,
            env->model_scratch_bytes, env->sampling_scratch,
            env->sampling_scratch_bytes, inject_failure, &fault,
            &model_result, &sample_result, diagnostic);
    }
    if (status != expected_status ||
        (selection_logits == NULL && !fault.fired) ||
        kq_model_exec_state_get_summary(state, &after_state, diagnostic) !=
            KQ_STATUS_OK ||
        !state_equal(&before_state, &after_state) ||
        !rng_equal(&before_rng, rng) ||
        !outputs_unchanged(env->logits, decoded, &model_result,
                           &sample_result)) return 0;
    (void)printf("sampled-control name=%s status=%s position=%llu "
                 "state_hash=%016llx rng_state=%llu rng_draws=%llu "
                 "model_rollback=PASS rng_rollback=PASS output_rollback=PASS\n",
                 name, kq_status_string(status),
                 (unsigned long long)before_state.model_position,
                 (unsigned long long)before_state.structural_hash,
                 (unsigned long long)before_rng.state,
                 (unsigned long long)before_rng.draws);
    return 1;
}

static int run_corrupt_rng_control(test_environment *env,
                                   kq_model_exec_state *state,
                                   const kq_sampling_rng_state *valid_rng,
                                   uint32_t input_token,
                                   kq_diagnostic *diagnostic) {
    kq_model_exec_state_summary before_state;
    kq_model_exec_state_summary after_state;
    kq_sampling_rng_state corrupt = *valid_rng;
    kq_sampling_rng_state before_corrupt;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    unsigned char decoded[256];
    kq_status status;
    corrupt.integrity ^= UINT64_C(1);
    before_corrupt = corrupt;
    if (kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
        KQ_STATUS_OK) return 0;
    prepare_output_sentinels(env, decoded, &model_result, &sample_result);
    status = kq_model_exec_sampled_decode_one_f32(
        env->config, state, env->sampling_config, &corrupt, input_token,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_INVALID_RNG_STATE ||
        kq_model_exec_state_get_summary(state, &after_state, diagnostic) !=
            KQ_STATUS_OK || !state_equal(&before_state, &after_state) ||
        !rng_equal(&before_corrupt, &corrupt) ||
        !outputs_unchanged(env->logits, decoded, &model_result,
                           &sample_result))
        return 0;
    (void)printf("sampled-control name=corrupt-rng status=%s "
                 "model_rollback=PASS rng_rollback=PASS "
                 "output_rollback=PASS\n", kq_status_string(status));
    return 1;
}

static int run_output_alias_control(test_environment *env,
                                    kq_model_exec_state *state,
                                    kq_sampling_rng_state *rng,
                                    uint32_t input_token,
                                    kq_diagnostic *diagnostic) {
    kq_model_exec_state_summary before_state;
    kq_model_exec_state_summary after_state;
    kq_sampling_rng_state before_rng;
    kq_model_exec_result model_result;
    kq_sampling_result untouched_sample;
    unsigned char decoded[256];
    kq_status status;
    if (kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
            KQ_STATUS_OK ||
        kq_sampling_rng_snapshot(rng, &before_rng, diagnostic) != KQ_STATUS_OK)
        return 0;
    prepare_output_sentinels(
        env, decoded, &model_result, &untouched_sample);
    status = kq_model_exec_sampled_decode_one_f32(
        env->config, state, env->sampling_config, rng, input_token,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        (kq_sampling_result *)env->logits, diagnostic);
    if (status != KQ_STATUS_ALIASING_VIOLATION ||
        kq_model_exec_state_get_summary(state, &after_state, diagnostic) !=
            KQ_STATUS_OK || !state_equal(&before_state, &after_state) ||
        !rng_equal(&before_rng, rng) ||
        !outputs_unchanged(env->logits, decoded, &model_result,
                           &untouched_sample))
        return 0;
    (void)printf("sampled-control name=output-alias status=%s "
                 "model_rollback=PASS rng_rollback=PASS "
                 "output_rollback=PASS\n", kq_status_string(status));
    return 1;
}

static int ensure_model_scratch(test_environment *env,
                                const kq_model_exec_config *config,
                                const kq_model_exec_state *state,
                                uint64_t prompt_tokens,
                                int decode,
                                kq_diagnostic *diagnostic) {
    uint64_t required = 0U;
    void *resized;
    kq_status status = decode ?
        kq_model_exec_required_decode_scratch_bytes(
            config, state, &required, diagnostic) :
        kq_model_exec_required_scratch_bytes(
            config, state, prompt_tokens, &required, diagnostic);
    if (status != KQ_STATUS_OK || required > SIZE_MAX) return 0;
    if (required <= env->model_scratch_bytes) return 1;
    resized = realloc(env->model_scratch, (size_t)required);
    if (resized == NULL) return 0;
    env->model_scratch = resized;
    env->model_scratch_bytes = required;
    return 1;
}

static int run_trace(test_environment *env, kq_model_exec_state *state,
                     sampled_trace *trace, int inject_faults,
                     kq_sampling_rng_state *final_rng,
                     kq_diagnostic *diagnostic) {
    kq_sampling_rng_state rng;
    kq_sampling_rng_state rng_before;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    unsigned char decoded[256];
    trace_counts before;
    uint32_t step;
    kq_status status;
    memset(trace->steps, 0, sizeof(trace->steps));
    trace->step_count = 0U;
    trace->total_payload = 0U;
    trace->total_blocks = 0U;
    trace->total_elapsed = 0U;
    status = kq_sampling_rng_seed(trace->seed, trace->stream, &rng, diagnostic);
    if (status != KQ_STATUS_OK ||
        !ensure_model_scratch(env, env->config, state,
                              sizeof(kq_prompt_ids) / sizeof(kq_prompt_ids[0]),
                              0, diagnostic) ||
        !get_trace_counts(env->provider, &before) ||
        kq_sampling_rng_snapshot(&rng, &rng_before, diagnostic) != KQ_STATUS_OK)
        return 0;
    status = kq_model_exec_sampled_prefill_f32(
        env->config, state, env->sampling_config, &rng,
        kq_prompt_ids, sizeof(kq_prompt_ids) / sizeof(kq_prompt_ids[0]),
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_OK ||
        !finalize_step(env, trace, 0U, 7U, &before, state, &rng_before,
                       &rng, &model_result, &sample_result, decoded,
                       diagnostic)) return 0;
    trace->step_count = 1U;
    if (model_result.selected_token_is_eog) {
        *final_rng = rng;
        return 1;
    }
    if (!ensure_model_scratch(env, env->config, state, 1U, 1, diagnostic))
        return 0;
    for (step = 1U; step < KQ_TEST_STEPS; ++step) {
        uint32_t input_token = trace->steps[step - 1U].model.selected_token_id;
        if (inject_faults && step == 1U) {
            static const uint32_t layers[] = {0U, 24U, 47U};
            static const char *names[] =
                {"early-layer", "middle-layer", "late-layer"};
            uint32_t fault;
            if (!run_output_alias_control(env, state, &rng, input_token,
                                          diagnostic)) return 0;
            if (!run_corrupt_rng_control(env, state, &rng, input_token,
                                         diagnostic)) return 0;
            for (fault = 0U; fault < 3U; ++fault)
                if (!run_failure(env, state, &rng, input_token, names[fault],
                                 layers[fault], NULL, KQ_STATUS_LIMIT_EXCEEDED,
                                 diagnostic)) return 0;
        }
        if (inject_faults && step == 2U) {
            uint64_t index;
            for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
                env->control_logits[index] = 0.0f;
            env->control_logits[0] = NAN;
            if (!run_failure(env, state, &rng, input_token,
                             "post-decode-sampler", UINT32_MAX,
                             env->control_logits, KQ_STATUS_NUMERIC_DOMAIN,
                             diagnostic)) return 0;
            for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
                env->control_logits[index] = -100.0f;
            env->control_logits[KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT] = 100.0f;
            if (!run_failure(env, state, &rng, input_token,
                             "padded-id", UINT32_MAX,
                             env->control_logits, KQ_STATUS_INVALID_TOKEN_ID,
                             diagnostic)) return 0;
        }
        if (!get_trace_counts(env->provider, &before) ||
            kq_sampling_rng_snapshot(&rng, &rng_before, diagnostic) !=
                KQ_STATUS_OK) return 0;
        status = kq_model_exec_sampled_decode_one_f32(
            env->config, state, env->sampling_config, &rng, input_token,
            env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            decoded, sizeof(decoded), env->model_scratch,
            env->model_scratch_bytes, env->sampling_scratch,
            env->sampling_scratch_bytes, NULL, NULL, &model_result,
            &sample_result, diagnostic);
        if (status != KQ_STATUS_OK ||
            !finalize_step(env, trace, step, 1U, &before, state, &rng_before,
                           &rng, &model_result, &sample_result, decoded,
                           diagnostic)) return 0;
        trace->step_count = step + 1U;
        if (model_result.selected_token_is_eog) break;
    }
    if (trace->total_payload > KQ_TRACE_PAYLOAD_LIMIT) return 0;
    *final_rng = rng;
    (void)printf("sampled-trace name=%s steps=%u prefill=1 decode=%u draws=%llu "
                 "replay=0 payload=%llu blocks=%llu rng_state_bytes=%llu "
                 "sampling_config_bytes=%llu sampling_scratch_bytes=%llu "
                 "model_state_owned_bytes=%llu elapsed_ns=%llu\n",
                 trace->name, (unsigned int)trace->step_count,
                 (unsigned int)(trace->step_count - 1U),
                 (unsigned long long)rng.draws,
                 (unsigned long long)trace->total_payload,
                 (unsigned long long)trace->total_blocks,
                 (unsigned long long)sizeof(rng),
                 (unsigned long long)kq_sampling_config_owned_bytes(
                     env->sampling_config),
                 (unsigned long long)env->sampling_scratch_bytes,
                 (unsigned long long)kq_model_exec_state_owned_bytes(state),
                 (unsigned long long)trace->total_elapsed);
    return 1;
}

static int traces_equal(const sampled_trace *left,
                        const sampled_trace *right) {
    uint32_t step;
    if (left->step_count != right->step_count) return 0;
    for (step = 0U; step < left->step_count; ++step) {
        const sampled_step *a = &left->steps[step];
        const sampled_step *b = &right->steps[step];
        if (a->model.selected_token_id != b->model.selected_token_id ||
            a->model.selected_token_is_eog != b->model.selected_token_is_eog ||
            a->model.decoded_utf8_bytes != b->model.decoded_utf8_bytes ||
            memcmp(a->decoded, b->decoded,
                   (size_t)a->model.decoded_utf8_bytes) != 0 ||
            !rng_equal(&a->rng_before, &b->rng_before) ||
            !rng_equal(&a->rng_after, &b->rng_after) ||
            a->sample.rng_word != b->sample.rng_word ||
            a->survivor_count != b->survivor_count ||
            a->survivor_hash != b->survivor_hash ||
            a->route_hash != b->route_hash || a->ple_hash != b->ple_hash ||
            !state_equal(&a->state, &b->state)) return 0;
    }
    return 1;
}

static int run_eog_later_control(test_environment *env,
                                 kq_model_exec_state *state,
                                 kq_sampling_rng_state *rng,
                                 uint32_t input_token,
                                 kq_diagnostic *diagnostic) {
    kq_model_exec_state_summary before_state;
    kq_model_exec_state_summary after_state;
    kq_sampling_rng_state before_rng;
    kq_sampling_rng_state stopped_rng;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    unsigned char decoded[256];
    uint64_t index;
    uint64_t accepted_position_before;
    uint64_t accepted_position_after;
    uint64_t accepted_draws_before;
    uint64_t accepted_draws_after;
    uint32_t accepted_eog_token;
    kq_status status;
    for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
        env->control_logits[index] = -100.0f;
    env->control_logits[KQ_SAMPLING_QWEN38_EOS_ID] = 100.0f;
    if (kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
            KQ_STATUS_OK ||
        kq_sampling_rng_snapshot(rng, &before_rng, diagnostic) != KQ_STATUS_OK)
        return 0;
    status = kq_model_exec_sampled_decode_one_with_selection_logits_for_test(
        env->config, state, env->sampling_config, rng, input_token,
        env->control_logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_OK ||
        model_result.selected_token_id != KQ_SAMPLING_QWEN38_EOS_ID ||
        !model_result.selected_token_is_eog ||
        rng->draws != before_rng.draws + 1U ||
        kq_model_exec_state_get_summary(state, &after_state, diagnostic) !=
            KQ_STATUS_OK ||
        after_state.model_position != before_state.model_position + 1U)
        return 0;
    accepted_eog_token = model_result.selected_token_id;
    accepted_position_before = before_state.model_position;
    accepted_position_after = after_state.model_position;
    accepted_draws_before = before_rng.draws;
    accepted_draws_after = rng->draws;
    if (kq_sampling_rng_snapshot(rng, &stopped_rng, diagnostic) !=
            KQ_STATUS_OK)
        return 0;
    prepare_output_sentinels(env, decoded, &model_result, &sample_result);
    status = kq_model_exec_sampled_decode_one_f32(
        env->config, state, env->sampling_config, rng,
        KQ_SAMPLING_QWEN38_EOS_ID,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_INVALID_MODEL_STATE ||
        kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
            KQ_STATUS_OK || !state_equal(&after_state, &before_state) ||
        !rng_equal(&stopped_rng, rng) ||
        !outputs_unchanged(env->logits, decoded, &model_result,
                           &sample_result))
        return 0;
    (void)printf("sampled-control name=later-eog status=OK token=%u "
                 "position_before=%llu position_after=%llu draws_before=%llu "
                 "draws_after=%llu no_decode_after_eog=PASS "
                 "stopped_state_rejection=PASS\n",
                 (unsigned int)accepted_eog_token,
                 (unsigned long long)accepted_position_before,
                 (unsigned long long)accepted_position_after,
                 (unsigned long long)accepted_draws_before,
                 (unsigned long long)accepted_draws_after);
    return 1;
}

static int run_first_eog_control(test_environment *env,
                                 kq_diagnostic *diagnostic) {
    kq_model_exec_state *state = NULL;
    kq_sampling_rng_state rng;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    kq_model_exec_state_summary summary;
    unsigned char decoded[256];
    uint64_t index;
    kq_status status;
    if (kq_model_exec_state_create(env->config, &state, diagnostic) !=
            KQ_STATUS_OK ||
        kq_sampling_rng_seed(1234U, KQ_SAMPLING_DEFAULT_STREAM,
                             &rng, diagnostic) != KQ_STATUS_OK ||
        !ensure_model_scratch(env, env->config, state, 7U, 0, diagnostic)) {
        kq_model_exec_state_close(state);
        return 0;
    }
    for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
        env->control_logits[index] = -100.0f;
    env->control_logits[KQ_SAMPLING_QWEN38_EOS_ID] = 100.0f;
    status = kq_model_exec_sampled_prefill_with_selection_logits_for_test(
        env->config, state, env->sampling_config, &rng,
        kq_prompt_ids, 7U, env->control_logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, env->logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded, sizeof(decoded),
        env->model_scratch, env->model_scratch_bytes,
        env->sampling_scratch, env->sampling_scratch_bytes, NULL, NULL,
        &model_result, &sample_result, diagnostic);
    if (status != KQ_STATUS_OK ||
        model_result.selected_token_id != KQ_SAMPLING_QWEN38_EOS_ID ||
        !model_result.selected_token_is_eog || rng.draws != 1U ||
        model_result.metrics.prompt_prefill_count != 1U ||
        model_result.metrics.incremental_decode_count != 0U ||
        kq_model_exec_state_get_summary(state, &summary, diagnostic) !=
            KQ_STATUS_OK || summary.model_position != 7U) {
        kq_model_exec_state_close(state);
        return 0;
    }
    (void)printf("sampled-control name=first-eog status=OK token=%u "
                 "position=7 prefill=1 decode=0 draws=1 "
                 "no_decode_after_eog=PASS\n",
                 (unsigned int)model_result.selected_token_id);
    kq_model_exec_state_close(state);
    return 1;
}

static int run_context_exhaustion_control(test_environment *env,
                                          kq_diagnostic *diagnostic) {
    kq_model_exec_config *config = NULL;
    kq_model_exec_state *state = NULL;
    kq_sampling_rng_state rng;
    kq_sampling_rng_state before_rng;
    kq_model_exec_state_summary before_state;
    kq_model_exec_state_summary after_state;
    kq_model_exec_result model_result;
    kq_sampling_result sample_result;
    unsigned char decoded[256];
    uint32_t input_token;
    kq_status status;
    int passed = 0;
    status = kq_model_exec_config_open(
        env->gguf, env->model, env->tokenizer, env->provider, 8U,
        &config, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_state_create(config, &state, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_sampling_rng_seed(999U, KQ_SAMPLING_DEFAULT_STREAM,
                                      &rng, diagnostic);
    if (status != KQ_STATUS_OK ||
        !ensure_model_scratch(env, config, state, 7U, 0, diagnostic))
        goto done;
    status = kq_model_exec_sampled_prefill_f32(
        config, state, env->sampling_config, &rng, kq_prompt_ids, 7U,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_OK || model_result.selected_token_is_eog ||
        !ensure_model_scratch(env, config, state, 1U, 1, diagnostic))
        goto done;
    input_token = model_result.selected_token_id;
    status = kq_model_exec_sampled_decode_one_f32(
        config, state, env->sampling_config, &rng, input_token,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_OK || model_result.selected_token_is_eog ||
        kq_model_exec_state_get_summary(state, &before_state, diagnostic) !=
            KQ_STATUS_OK || before_state.model_position != 8U ||
        kq_sampling_rng_snapshot(&rng, &before_rng, diagnostic) != KQ_STATUS_OK)
        goto done;
    input_token = model_result.selected_token_id;
    prepare_output_sentinels(env, decoded, &model_result, &sample_result);
    status = kq_model_exec_sampled_decode_one_f32(
        config, state, env->sampling_config, &rng, input_token,
        env->logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded, sizeof(decoded), env->model_scratch,
        env->model_scratch_bytes, env->sampling_scratch,
        env->sampling_scratch_bytes, NULL, NULL, &model_result,
        &sample_result, diagnostic);
    if (status != KQ_STATUS_LIMIT_EXCEEDED ||
        kq_model_exec_state_get_summary(state, &after_state, diagnostic) !=
            KQ_STATUS_OK || !state_equal(&before_state, &after_state) ||
        !rng_equal(&before_rng, &rng) ||
        !outputs_unchanged(env->logits, decoded, &model_result,
                           &sample_result))
        goto done;
    (void)printf("sampled-control name=context-exhaustion status=%s "
                 "capacity=8 position=8 model_rollback=PASS "
                 "rng_rollback=PASS output_rollback=PASS\n",
                 kq_status_string(status));
    passed = 1;

done:
    kq_model_exec_state_close(state);
    kq_model_exec_config_close(config);
    return passed;
}

static int open_environment(test_environment *env, wchar_t *path,
                            kq_diagnostic *diagnostic) {
    kq_sampling_policy policy;
    kq_status status;
    memset(env, 0, sizeof(*env));
    status = kq_file_open_readonly(path, &env->file, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_gguf_open(env->file, &env->gguf, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_open_from_gguf(env->gguf, &env->model, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_open_from_gguf(
            env->gguf, &env->tokenizer, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_open(
            env->gguf, env->model, KQ_TEST_PROVIDER_BUDGET,
            &env->provider, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_test_enable_extended_trace(
            env->provider, KQ_TRACE_CAPACITY, KQ_TRACE_CAPACITY,
            KQ_PLE_TRACE_CAPACITY, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_config_open(
            env->gguf, env->model, env->tokenizer, env->provider,
            KQ_TEST_CONTEXT, &env->config, diagnostic);
    kq_sampling_policy_qwen38_default(&policy);
    if (status == KQ_STATUS_OK)
        status = kq_sampling_config_open_qwen38(
            &policy, &env->sampling_config, diagnostic);
    if (status != KQ_STATUS_OK) return 0;
    env->sampling_scratch_bytes =
        kq_sampling_required_scratch_bytes(env->sampling_config);
    if (env->sampling_scratch_bytes == 0U ||
        env->sampling_scratch_bytes > SIZE_MAX) return 0;
    env->sampling_scratch = malloc((size_t)env->sampling_scratch_bytes);
    env->logits = (float *)malloc(
        KQ_MODEL_EXEC_VOCABULARY_SIZE * sizeof(*env->logits));
    env->control_logits = (float *)malloc(
        KQ_MODEL_EXEC_VOCABULARY_SIZE * sizeof(*env->control_logits));
    return env->sampling_scratch != NULL && env->logits != NULL &&
           env->control_logits != NULL;
}

static void close_environment(test_environment *env) {
    free(env->control_logits);
    free(env->logits);
    free(env->sampling_scratch);
    free(env->model_scratch);
    kq_sampling_config_close(env->sampling_config);
    kq_model_exec_config_close(env->config);
    kq_weight_provider_close(env->provider);
    kq_tokenizer_close(env->tokenizer);
    kq_model_close(env->model);
    kq_gguf_close(env->gguf);
    kq_file_close(env->file);
}

int main(int argc, char **argv) {
    wchar_t *path = model_path();
    test_environment env;
    sampled_trace primary;
    sampled_trace holdout;
    sampled_trace replay;
    kq_model_exec_state *state = NULL;
    kq_sampling_rng_state final_rng;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    int passed = 0;
    memset(&env, 0, sizeof(env));
    kq_diagnostic_clear(&diagnostic);
    if (path == NULL) return 77;
    if (argc == 3 && strcmp(argv[1], "--capture-dir") == 0)
        env.capture_dir = argv[2];
    else if (argc != 1) {
        (void)fprintf(stderr, "usage: sampled_generation_integration "
                              "[--capture-dir <ignored-directory>]\n");
        free(path);
        return 2;
    }
    {
        const char *capture_dir = env.capture_dir;
        if (!open_environment(&env, path, &diagnostic)) goto done;
        env.capture_dir = capture_dir;
    }
    free(path);
    path = NULL;

    primary.name = "primary";
    primary.seed = UINT64_C(0);
    primary.stream = KQ_SAMPLING_DEFAULT_STREAM;
    holdout.name = "holdout";
    holdout.seed = UINT64_MAX;
    holdout.stream = KQ_SAMPLING_MAX_STREAM;
    replay.name = "replay";
    replay.seed = primary.seed;
    replay.stream = primary.stream;

    status = kq_model_exec_state_create(env.config, &state, &diagnostic);
    if (status != KQ_STATUS_OK ||
        !run_trace(&env, state, &primary, 1, &final_rng, &diagnostic))
        goto done;
    kq_model_exec_state_close(state);
    state = NULL;

    status = kq_model_exec_state_create(env.config, &state, &diagnostic);
    if (status != KQ_STATUS_OK ||
        !run_trace(&env, state, &holdout, 0, &final_rng, &diagnostic))
        goto done;
    if (holdout.step_count == 0U ||
        !run_eog_later_control(
            &env, state, &final_rng,
            holdout.steps[holdout.step_count - 1U].model.selected_token_id,
            &diagnostic)) goto done;
    kq_model_exec_state_close(state);
    state = NULL;

    status = kq_model_exec_state_create(env.config, &state, &diagnostic);
    if (status != KQ_STATUS_OK ||
        !run_trace(&env, state, &replay, 0, &final_rng, &diagnostic) ||
        !traces_equal(&primary, &replay)) goto done;
    (void)printf("sampled-control name=deterministic-replay "
                 "steps=%u exact=PASS\n", (unsigned int)replay.step_count);
    kq_model_exec_state_close(state);
    state = NULL;

    if (!run_first_eog_control(&env, &diagnostic) ||
        !run_context_exhaustion_control(&env, &diagnostic)) goto done;
    (void)printf("sampled-generation PASS primary_steps=%u holdout_steps=%u "
                 "replay_steps=%u context=16 greedy_unchanged=REQUIRED\n",
                 (unsigned int)primary.step_count,
                 (unsigned int)holdout.step_count,
                 (unsigned int)replay.step_count);
    passed = 1;

done:
    if (path != NULL) free(path);
    kq_model_exec_state_close(state);
    if (!passed)
        (void)fprintf(stderr, "sampled generation integration failed: %s: %s\n",
                      kq_status_string(status), diagnostic.message);
    close_environment(&env);
    return passed ? 0 : 1;
}
