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

#define KQ_TEST_BUDGET (UINT64_C(256) * UINT64_C(1024) * \
                        UINT64_C(1024) * UINT64_C(1024))

typedef struct fault_context {
    kq_weight_provider *provider;
    uint32_t layer;
    int fired;
} fault_context;

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

static int output_is_sentinel(const float *logits, uint64_t count,
                              const unsigned char *decoded,
                              uint64_t decoded_count,
                              const kq_model_exec_result *result) {
    const unsigned char *bytes = (const unsigned char *)result;
    uint64_t index;
    for (index = 0U; index < count; ++index)
        if (logits[index] != -1234.5f) return 0;
    for (index = 0U; index < decoded_count; ++index)
        if (decoded[index] != 0xa5U) return 0;
    for (index = 0U; index < sizeof(*result); ++index)
        if (bytes[index] != 0x5aU) return 0;
    return 1;
}

static int validate_success(const kq_model_exec_result *result,
                            const unsigned char *decoded,
                            const kq_model_exec_state *state) {
    return result->selected_token_id == 271U &&
           result->decoded_utf8_bytes == 2U && decoded[0] == '\n' &&
           decoded[1] == '\n' && kq_model_exec_state_position(state) == 7U &&
           result->metrics.layers_completed == 48U &&
           result->metrics.embedding_logical_bytes_touched == 19040U &&
           result->metrics.routed_expert_selections == 3360U &&
           result->metrics.selected_expert_matrix_requests == 10080U &&
           result->metrics.ple_row_requests == 112U &&
           result->metrics.lm_head_logical_bytes_touched == 675430400U &&
           result->metrics.logical_payload_bytes_touched == 40208768960U &&
           result->metrics.logical_payload_bytes_touched <=
               UINT64_C(64) * UINT64_C(1024) * UINT64_C(1024) *
               UINT64_C(1024);
}

int main(void) {
    static const unsigned char prompt[] = "Hello, Kestrel-Q.";
    static const uint32_t expected_ids[] =
        {9419U, 11U, 710U, 467U, 3621U, 27325U, 13U};
    static const uint32_t failure_layers[] = {0U, 24U, 47U};
    wchar_t *path = model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_weight_provider *provider = NULL;
    kq_model_exec_config *config = NULL;
    kq_model_exec_state *state = NULL;
    uint32_t ids[7];
    uint64_t id_count = 0U;
    kq_tokenizer_encode_options options;
    float *logits = NULL;
    unsigned char decoded[64];
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    kq_model_exec_result result;
    kq_diagnostic diagnostic;
    kq_status status;
    uint32_t failure;
    uint64_t index;
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
        status = kq_model_exec_config_open(gguf, model, tokenizer, provider,
                                           8U, &config, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_state_create(config, &state, &diagnostic);
    options.special_policy = KQ_TOKENIZER_SPECIAL_REJECT;
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_encode(tokenizer, prompt, sizeof(prompt) - 1U,
                                     &options, ids, 7U, &id_count,
                                     &diagnostic);
    if (status != KQ_STATUS_OK || id_count != 7U ||
        memcmp(ids, expected_ids, sizeof(ids)) != 0) goto done;
    status = kq_model_exec_required_scratch_bytes(
        config, state, 7U, &scratch_bytes, &diagnostic);
    if (status != KQ_STATUS_OK || scratch_bytes > SIZE_MAX) goto done;
    logits = (float *)malloc(KQ_MODEL_EXEC_VOCABULARY_SIZE * sizeof(*logits));
    scratch = malloc((size_t)scratch_bytes);
    if (logits == NULL || scratch == NULL) goto done;

    {
        uint32_t invalid_id = 248077U;
        memset(&result, 0x5a, sizeof(result));
        status = kq_model_exec_prefill_first_token_f32(
            config, state, &invalid_id, 1U, logits,
            KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded, sizeof(decoded), scratch,
            scratch_bytes, NULL, NULL, &result, &diagnostic);
        if (status != KQ_STATUS_INVALID_TOKEN_ID ||
            kq_model_exec_state_position(state) != 0U) goto done;
        invalid_id = 248320U;
        status = kq_model_exec_prefill_first_token_f32(
            config, state, &invalid_id, 1U, logits,
            KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded, sizeof(decoded), scratch,
            scratch_bytes, NULL, NULL, &result, &diagnostic);
        if (status != KQ_STATUS_INVALID_TOKEN_ID ||
            kq_model_exec_state_position(state) != 0U) goto done;
        status = kq_model_exec_prefill_first_token_f32(
            config, state, ids, 7U, logits,
            KQ_MODEL_EXEC_VOCABULARY_SIZE - 1U, decoded, sizeof(decoded),
            scratch, scratch_bytes, NULL, NULL, &result, &diagnostic);
        if (status != KQ_STATUS_INVALID_ARGUMENT ||
            kq_model_exec_state_position(state) != 0U) goto done;
        if (kq_model_exec_required_scratch_bytes(
                config, state, 8U, &index, &diagnostic) !=
                KQ_STATUS_INVALID_ARGUMENT) goto done;
    }

    for (failure = 0U; failure < 3U; ++failure) {
        fault_context context;
        for (index = 0U; index < KQ_MODEL_EXEC_VOCABULARY_SIZE; ++index)
            logits[index] = -1234.5f;
        memset(decoded, 0xa5, sizeof(decoded));
        memset(&result, 0x5a, sizeof(result));
        context.provider = provider;
        context.layer = failure_layers[failure];
        context.fired = 0;
        status = kq_model_exec_prefill_first_token_f32(
            config, state, ids, 7U, logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            decoded, sizeof(decoded), scratch, scratch_bytes, inject_failure,
            &context, &result, &diagnostic);
        if (status != KQ_STATUS_LIMIT_EXCEEDED || !context.fired ||
            kq_model_exec_state_position(state) != 0U ||
            !output_is_sentinel(logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
                                decoded, sizeof(decoded), &result)) goto done;

        /* A failed early/middle/late run dirties only private staging state.
           The next call must reset it and reproduce the independent M1 result. */
        status = kq_model_exec_prefill_first_token_f32(
            config, state, ids, 7U, logits, KQ_MODEL_EXEC_VOCABULARY_SIZE,
            decoded, sizeof(decoded), scratch, scratch_bytes, NULL, NULL,
            &result, &diagnostic);
        if (status != KQ_STATUS_OK ||
            !validate_success(&result, decoded, state)) goto done;
        if (failure + 1U < 3U &&
            kq_model_exec_state_reset(state, &diagnostic) != KQ_STATUS_OK)
            goto done;
    }
    (void)printf("first-token real: token=271 recoveries=3 payload=%llu "
                 "blocks=%llu semantics=%llu routes=%llu matrices=%llu "
                 "ple_rows=%llu state=%llu scratch=%llu\n",
        (unsigned long long)result.metrics.logical_payload_bytes_touched,
        (unsigned long long)result.metrics.payload_blocks_touched,
        (unsigned long long)result.metrics.unique_semantic_tensors_touched,
        (unsigned long long)result.metrics.routed_expert_selections,
        (unsigned long long)result.metrics.selected_expert_matrix_requests,
        (unsigned long long)result.metrics.ple_row_requests,
        (unsigned long long)result.metrics.persistent_state_bytes,
        (unsigned long long)result.metrics.peak_scratch_bytes);
    passed = 1;

done:
    if (!passed)
        (void)fprintf(stderr, "first-token integration failed: %s: %s\n",
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
