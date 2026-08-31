#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_model_exec.h"
#include "kq_tokenizer.h"
#include "kq_weight_provider.h"
#include "kq_weight_provider_internal.h"

#define KQ_TEST_CONTEXT UINT64_C(16)
#define KQ_TEST_BUDGET (UINT64_C(256) * UINT64_C(1024) * \
                        UINT64_C(1024) * UINT64_C(1024))
#define KQ_STEP_COUNT 4U

typedef struct fault_context {
    kq_weight_provider *provider;
    uint32_t layer;
    int fired;
} fault_context;

typedef struct trace_counts {
    uint64_t routes;
    uint64_t experts;
    uint64_t ple_rows;
} trace_counts;

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
    fault_context *context = (fault_context *)user_data;
    if (event != NULL && context != NULL && !context->fired &&
        event->phase == KQ_MODEL_EXEC_PHASE_LAYER_BEGIN &&
        event->layer_id == context->layer) {
        kq_weight_provider_test_fail_next_request(context->provider);
        context->fired = 1;
    }
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

static uint64_t hash_request(uint64_t hash, uint32_t layer,
                             uint64_t value) {
    uint32_t index;
    hash ^= layer;
    hash *= UINT64_C(1099511628211);
    for (index = 0U; index < 8U; ++index) {
        hash ^= (value >> (index * 8U)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
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
            const kq_weight_provider_expert_request *first =
                &experts[before->experts + group * 10U + index];
            if (first->layer_id != expected_layer ||
                first->expert_id >= 512U || !selected[first->expert_id])
                goto done;
            selected[first->expert_id] = 0U;
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

static int state_summary_equal(const kq_model_exec_state_summary *left,
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

static void capture_top2(const float *logits, uint32_t *runner_id,
                         float *selected_logit, float *runner_logit) {
    uint32_t best = 0U;
    uint32_t second = 1U;
    uint32_t token;
    if (logits[second] > logits[best]) {
        uint32_t swap = best;
        best = second;
        second = swap;
    }
    for (token = 2U; token < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++token) {
        if (logits[token] > logits[best]) {
            second = best;
            best = token;
        } else if (logits[token] > logits[second]) {
            second = token;
        }
    }
    *runner_id = second;
    *selected_logit = logits[best];
    *runner_logit = logits[second];
}

int main(void) {
    static const unsigned char prompt[] = "Hello, Kestrel-Q.";
    static const uint32_t prompt_ids[] =
        {9419U, 11U, 710U, 467U, 3621U, 27325U, 13U};
    static const uint32_t expected_tokens[KQ_STEP_COUNT] =
        {271U, 248068U, 198U, 760U};
    static const unsigned char *expected_pieces[KQ_STEP_COUNT] = {
        (const unsigned char *)"\n\n", (const unsigned char *)"<think>",
        (const unsigned char *)"\n", (const unsigned char *)"The"};
    static const uint64_t expected_piece_bytes[KQ_STEP_COUNT] = {2U,7U,1U,3U};
    static const uint64_t expected_positions[KQ_STEP_COUNT] = {7U,8U,9U,10U};
    static const uint64_t expected_complete_blocks[KQ_STEP_COUNT] = {1U,2U,2U,2U};
    static const uint64_t expected_tails[KQ_STEP_COUNT] = {3U,0U,1U,2U};
    static const uint64_t expected_qsa_candidates[KQ_STEP_COUNT] =
        {48U, 24U, 24U, 24U};
    static const uint64_t expected_qsa_selected_blocks[KQ_STEP_COUNT] =
        {48U, 24U, 24U, 24U};
    static const uint64_t expected_qsa_selected_tokens[KQ_STEP_COUNT] =
        {336U, 96U, 108U, 120U};
    wchar_t *path = model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_weight_provider *provider = NULL;
    kq_model_exec_config *config = NULL;
    kq_model_exec_state *state = NULL;
    kq_model_exec_result results[KQ_STEP_COUNT];
    kq_model_exec_state_summary summaries[KQ_STEP_COUNT];
    uint64_t route_hashes[KQ_STEP_COUNT];
    uint64_t ple_hashes[KQ_STEP_COUNT];
    uint32_t runner_ids[KQ_STEP_COUNT];
    float selected_logits[KQ_STEP_COUNT];
    float runner_logits[KQ_STEP_COUNT];
    unsigned char decoded[KQ_STEP_COUNT][32];
    float *logits = NULL;
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    uint64_t decode_scratch = 0U;
    uint64_t total_payload = 0U;
    uint64_t total_blocks = 0U;
    uint64_t total_elapsed = 0U;
    trace_counts trace_before;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    uint32_t step;
    int passed = 0;

    if (path == NULL) return 77;
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_open(gguf, model, KQ_TEST_BUDGET,
                                         &provider, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_test_enable_extended_trace(
            provider, 8192U, 8192U, 256U, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_config_open(
            gguf, model, tokenizer, provider, KQ_TEST_CONTEXT,
            &config, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_state_create(config, &state, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_required_scratch_bytes(
            config, state, 7U, &scratch_bytes, &diagnostic);
    if (status != KQ_STATUS_OK || scratch_bytes > SIZE_MAX) goto done;
    scratch = malloc((size_t)scratch_bytes);
    logits = (float *)malloc(KQ_MODEL_EXEC_VOCABULARY_SIZE * sizeof(*logits));
    if (scratch == NULL || logits == NULL) goto done;

    status = kq_model_exec_decode_one_f32(
        config, state, 271U, logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
        decoded[0], sizeof(decoded[0]), scratch, scratch_bytes,
        NULL, NULL, &results[0], &diagnostic);
    if (status != KQ_STATUS_INVALID_MODEL_STATE) goto done;
    status = kq_model_exec_decode_one_f32(
        config, state, KQ_MODEL_EXEC_CANONICAL_TOKEN_LIMIT, logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded[0], sizeof(decoded[0]),
        scratch, scratch_bytes, NULL, NULL, &results[0], &diagnostic);
    if (status != KQ_STATUS_INVALID_TOKEN_ID) goto done;

    if (!get_trace_counts(provider, &trace_before)) goto done;
    status = kq_model_exec_prefill_first_token_f32(
        config, state, prompt_ids, 7U, logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded[0], sizeof(decoded[0]),
        scratch, scratch_bytes, NULL, NULL, &results[0], &diagnostic);
    if (status != KQ_STATUS_OK ||
        kq_model_exec_state_get_summary(state, &summaries[0], &diagnostic) !=
            KQ_STATUS_OK ||
        !validate_trace_delta(provider, &trace_before, 7U,
                              &route_hashes[0], &ple_hashes[0])) goto done;
    capture_top2(logits, &runner_ids[0], &selected_logits[0],
                 &runner_logits[0]);
    status = kq_model_exec_prefill_first_token_f32(
        config, state, prompt_ids, 7U, logits,
        KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded[0], sizeof(decoded[0]),
        scratch, scratch_bytes, NULL, NULL, &results[0], &diagnostic);
    if (status != KQ_STATUS_INVALID_MODEL_STATE ||
        kq_model_exec_state_position(state) != 7U) goto done;

    status = kq_model_exec_required_decode_scratch_bytes(
        config, state, &decode_scratch, &diagnostic);
    if (status != KQ_STATUS_OK || decode_scratch > SIZE_MAX) goto done;
    if (decode_scratch > scratch_bytes) {
        void *resized = realloc(scratch, (size_t)decode_scratch);
        if (resized == NULL) goto done;
        scratch = resized;
        scratch_bytes = decode_scratch;
    }

    for (step = 1U; step < KQ_STEP_COUNT; ++step) {
        if (!get_trace_counts(provider, &trace_before)) goto done;
        if (step == 2U) {
            fault_context fault;
            kq_model_exec_state_summary before_failure;
            kq_model_exec_state_summary after_failure;
            uint64_t index;
            fault.provider = provider;
            fault.layer = 24U;
            fault.fired = 0;
            before_failure = summaries[step - 1U];
            for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
                logits[index] = -1234.5f;
            memset(decoded[step], 0xa5, sizeof(decoded[step]));
            memset(&results[step], 0x5a, sizeof(results[step]));
            status = kq_model_exec_decode_one_f32(
                config, state, expected_tokens[step - 1U], logits,
                KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded[step],
                sizeof(decoded[step]), scratch, scratch_bytes,
                inject_failure, &fault, &results[step], &diagnostic);
            if (status != KQ_STATUS_LIMIT_EXCEEDED || !fault.fired ||
                kq_model_exec_state_get_summary(
                    state, &after_failure, &diagnostic) != KQ_STATUS_OK ||
                !state_summary_equal(&before_failure, &after_failure)) goto done;
            for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
                if (logits[index] != -1234.5f) goto done;
            for (index = 0U; index < sizeof(decoded[step]); ++index)
                if (decoded[step][index] != 0xa5U) goto done;
            {
                const unsigned char *bytes =
                    (const unsigned char *)&results[step];
                for (index = 0U; index < sizeof(results[step]); ++index)
                    if (bytes[index] != 0x5aU) goto done;
            }
            /* The failed attempt is observable in provider accounting but is
               outside the successful-path payload gate and trace slice. */
            if (!get_trace_counts(provider, &trace_before)) goto done;
        }
        status = kq_model_exec_decode_one_f32(
            config, state, expected_tokens[step - 1U], logits,
            KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded[step],
            sizeof(decoded[step]), scratch, scratch_bytes,
            NULL, NULL, &results[step], &diagnostic);
        if (status != KQ_STATUS_OK ||
            kq_model_exec_state_get_summary(state, &summaries[step],
                                             &diagnostic) != KQ_STATUS_OK ||
            !validate_trace_delta(provider, &trace_before, 1U,
                                  &route_hashes[step], &ple_hashes[step]))
            goto done;
        capture_top2(logits, &runner_ids[step], &selected_logits[step],
                     &runner_logits[step]);
    }

    for (step = 0U; step < KQ_STEP_COUNT; ++step) {
        if (results[step].selected_token_id != expected_tokens[step] ||
            results[step].selected_token_is_eog ||
            results[step].decoded_utf8_bytes != expected_piece_bytes[step] ||
            memcmp(decoded[step], expected_pieces[step],
                   (size_t)expected_piece_bytes[step]) != 0 ||
            summaries[step].model_position != expected_positions[step] ||
            summaries[step].layer_position_min != expected_positions[step] ||
            summaries[step].layer_position_max != expected_positions[step] ||
            summaries[step].qsa_sequence_length_min != expected_positions[step] ||
            summaries[step].qsa_sequence_length_max != expected_positions[step] ||
            summaries[step].qsa_complete_blocks_min != expected_complete_blocks[step] ||
            summaries[step].qsa_complete_blocks_max != expected_complete_blocks[step] ||
            summaries[step].qsa_incomplete_tail_min != expected_tails[step] ||
            summaries[step].qsa_incomplete_tail_max != expected_tails[step] ||
            summaries[step].ple_address_position != expected_positions[step] ||
            summaries[step].ple_value_position != expected_positions[step] ||
            summaries[step].gdn_initialized_layers != 36U ||
            summaries[step].qsa_layers != 12U ||
            results[step].metrics.layers_completed != 48U ||
            results[step].metrics.qsa_selection_events !=
                (step == 0U ? 84U : 12U) ||
            results[step].metrics.qsa_candidate_blocks !=
                expected_qsa_candidates[step] ||
            results[step].metrics.qsa_selected_blocks !=
                expected_qsa_selected_blocks[step] ||
            results[step].metrics.qsa_selected_tokens !=
                expected_qsa_selected_tokens[step] ||
            results[step].metrics.routed_expert_selections !=
                (step == 0U ? 3360U : 480U) ||
            results[step].metrics.selected_expert_matrix_requests !=
                (step == 0U ? 10080U : 1440U) ||
            results[step].metrics.ple_row_requests !=
                (step == 0U ? 112U : 16U) ||
            results[step].metrics.prompt_prefill_count !=
                (step == 0U ? 1U : 0U) ||
            results[step].metrics.incremental_decode_count !=
                (step == 0U ? 0U : 1U)) goto done;
        total_payload += results[step].metrics.logical_payload_bytes_touched;
        total_blocks += results[step].metrics.payload_blocks_touched;
        total_elapsed += results[step].metrics.elapsed_nanoseconds;
    }
    if (total_payload > UINT64_C(96) * UINT64_C(1024) *
                            UINT64_C(1024) * UINT64_C(1024) ||
        kq_model_exec_state_position(state) != 10U) goto done;

    for (step = 0U; step < KQ_STEP_COUNT; ++step) {
        (void)printf("multi-token step=%u token=%u position=%llu "
                     "state_hash=%016llx gdn_hash=%016llx "
                     "qsa_hash=%016llx ple_address_hash=%016llx "
                     "ple_value_hash=%016llx route_hash=%016llx "
                     "ple_hash=%016llx selected_logit=%.9g "
                     "runner_id=%u runner_logit=%.9g "
                     "qsa_candidates=%llu qsa_blocks=%llu qsa_tokens=%llu "
                     "payload=%llu blocks=%llu\n",
            (unsigned int)(step + 1U), expected_tokens[step],
            (unsigned long long)summaries[step].model_position,
            (unsigned long long)summaries[step].structural_hash,
            (unsigned long long)summaries[step].gdn_state_hash,
            (unsigned long long)summaries[step].qsa_state_hash,
            (unsigned long long)summaries[step].ple_address_state_hash,
            (unsigned long long)summaries[step].ple_value_state_hash,
            (unsigned long long)route_hashes[step],
            (unsigned long long)ple_hashes[step],
            selected_logits[step], runner_ids[step], runner_logits[step],
            (unsigned long long)results[step].metrics.qsa_candidate_blocks,
            (unsigned long long)results[step].metrics.qsa_selected_blocks,
            (unsigned long long)results[step].metrics.qsa_selected_tokens,
            (unsigned long long)results[step].metrics.logical_payload_bytes_touched,
            (unsigned long long)results[step].metrics.payload_blocks_touched);
    }
    (void)printf("multi-token PASS prefill=1 decode=3 rollback=1 "
                 "payload=%llu blocks=%llu state=%llu scratch=%llu "
                 "elapsed_ns=%llu\n",
        (unsigned long long)total_payload,
        (unsigned long long)total_blocks,
        (unsigned long long)results[3].metrics.persistent_state_bytes,
        (unsigned long long)scratch_bytes,
        (unsigned long long)total_elapsed);
    passed = 1;

done:
    if (!passed)
        (void)fprintf(stderr, "multi-token integration failed: %s: %s\n",
                      kq_status_string(status), diagnostic.message);
    free(scratch);
    free(logits);
    kq_model_exec_state_close(state);
    kq_model_exec_config_close(config);
    kq_weight_provider_close(provider);
    kq_tokenizer_close(tokenizer);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return passed ? 0 : 1;
}
