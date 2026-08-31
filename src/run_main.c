#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_model_exec.h"
#include "kq_status.h"
#include "kq_tokenizer.h"
#include "kq_weight_provider.h"

#define KQ_RUN_CONTEXT_CAPACITY UINT64_C(8)
#define KQ_RUN_PAYLOAD_BUDGET (UINT64_C(64) * UINT64_C(1024) * \
                               UINT64_C(1024) * UINT64_C(1024))

static void usage(void) {
    (void)fprintf(stderr,
        "usage: kq-run <model.gguf> --prompt <text> "
        "--max-new-tokens 1 --greedy\n");
}

static void print_failure(kq_status status, const kq_diagnostic *diagnostic) {
    (void)fprintf(stderr, "kq-run: %s", kq_status_string(status));
    if (diagnostic != NULL && diagnostic->message[0] != '\0')
        (void)fprintf(stderr, ": %s", diagnostic->message);
    (void)fputc('\n', stderr);
}

static unsigned char *wide_to_utf8(const wchar_t *value, uint64_t *bytes) {
    int required;
    int written;
    unsigned char *result;
    if (value == NULL || bytes == NULL) return NULL;
    required = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                                   NULL, 0, NULL, NULL);
    if (required <= 1) return NULL;
    result = (unsigned char *)malloc((size_t)required);
    if (result == NULL) return NULL;
    written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                                  (char *)result, required, NULL, NULL);
    if (written != required) {
        free(result);
        return NULL;
    }
    *bytes = (uint64_t)(required - 1);
    return result;
}

static void progress(const kq_model_exec_progress_event *event,
                     void *user_data) {
    uint64_t value_index;
    double sum = 0.0;
    double squared_sum = 0.0;
    (void)user_data;
    if (event == NULL) return;
    switch (event->phase) {
        case KQ_MODEL_EXEC_PHASE_TOKENIZED:
            (void)fprintf(stderr, "progress: tokenization complete\n");
            break;
        case KQ_MODEL_EXEC_PHASE_EMBEDDING_COMPLETE:
            (void)fprintf(stderr, "progress: embedding complete\n");
            break;
        case KQ_MODEL_EXEC_PHASE_LAYER_BEGIN:
            (void)fprintf(stderr, "progress: layer %u/%u\n",
                          (unsigned int)(event->layer_id + 1U),
                          (unsigned int)event->layer_count);
            break;
        case KQ_MODEL_EXEC_PHASE_FINAL_MIX_COMPLETE:
            (void)fprintf(stderr, "progress: final norm/mix complete\n");
            break;
        case KQ_MODEL_EXEC_PHASE_LOGITS_COMPLETE:
            (void)fprintf(stderr, "progress: LM head complete\n");
            break;
        case KQ_MODEL_EXEC_PHASE_TOKEN_SELECTED:
            (void)fprintf(stderr, "progress: token selected\n");
            break;
        default:
            break;
    }
    if (event->last_token_values != NULL &&
        event->last_token_value_count != 0U &&
        event->phase != KQ_MODEL_EXEC_PHASE_LOGITS_COMPLETE) {
        for (value_index = 0U; value_index < event->last_token_value_count;
             ++value_index) {
            const double value = event->last_token_values[value_index];
            sum += value;
            squared_sum += value * value;
        }
        (void)fprintf(stderr,
            "checkpoint: phase=%u layer=%u elements=%llu sum=%.17g "
            "squared_sum=%.17g first=%.9g,%.9g,%.9g,%.9g\n",
            (unsigned int)event->phase, (unsigned int)event->layer_id,
            (unsigned long long)event->last_token_value_count, sum,
            squared_sum, event->last_token_values[0],
            event->last_token_values[1], event->last_token_values[2],
            event->last_token_values[3]);
    }
}

static void print_top_logits(const float *logits, uint32_t count,
                             uint32_t top_count) {
    uint32_t selected[20];
    uint32_t rank;
    uint32_t token;
    if (top_count > 20U) top_count = 20U;
    for (rank = 0U; rank < top_count; ++rank) {
        uint32_t best = UINT32_MAX;
        for (token = 0U; token < count; ++token) {
            uint32_t prior;
            int used = 0;
            for (prior = 0U; prior < rank; ++prior)
                if (selected[prior] == token) used = 1;
            if (!used && (best == UINT32_MAX || logits[token] > logits[best] ||
                (logits[token] == logits[best] && token < best))) best = token;
        }
        selected[rank] = best;
        (void)printf("top_%u_token_id=%u\ntop_%u_logit=%.9g\n",
                     (unsigned int)(rank + 1U), (unsigned int)best,
                     (unsigned int)(rank + 1U), logits[best]);
    }
}

int wmain(int argc, wchar_t **argv) {
    const wchar_t *model_path;
    const wchar_t *prompt_argument = NULL;
    unsigned char *prompt = NULL;
    uint64_t prompt_bytes = 0U;
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_weight_provider *provider = NULL;
    kq_model_exec_config *config = NULL;
    kq_model_exec_state *state = NULL;
    uint32_t *token_ids = NULL;
    uint64_t token_count = 0U;
    kq_tokenizer_encode_options encode_options;
    float *logits = NULL;
    unsigned char decoded[256];
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    kq_model_exec_result result = {0};
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    int exit_code = 1;

    if (argc != 7) {
        usage();
        return 2;
    }
    model_path = argv[1];
    if (wcscmp(argv[2], L"--prompt") != 0 ||
        wcscmp(argv[4], L"--max-new-tokens") != 0 ||
        wcscmp(argv[5], L"1") != 0 ||
        wcscmp(argv[6], L"--greedy") != 0) {
        usage();
        return 2;
    }
    prompt_argument = argv[3];
    prompt = wide_to_utf8(prompt_argument, &prompt_bytes);
    if (prompt == NULL) {
        (void)fprintf(stderr, "kq-run: prompt is empty or invalid UTF-16\n");
        return 2;
    }
    kq_diagnostic_clear(&diagnostic);

    status = kq_file_open_readonly(model_path, &file, &diagnostic);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_open(gguf, model, KQ_RUN_PAYLOAD_BUDGET,
                                         &provider, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_config_open(
            gguf, model, tokenizer, provider, KQ_RUN_CONTEXT_CAPACITY,
            &config, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_state_create(config, &state, &diagnostic);
    encode_options.special_policy = KQ_TOKENIZER_SPECIAL_REJECT;
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_encode(tokenizer, prompt, prompt_bytes,
                                     &encode_options, NULL, 0U, &token_count,
                                     &diagnostic);
    if (status == KQ_STATUS_BUFFER_TOO_SMALL && token_count != 0U) {
        if (token_count >= KQ_RUN_CONTEXT_CAPACITY ||
            token_count > SIZE_MAX / sizeof(*token_ids)) {
            status = KQ_STATUS_LIMIT_EXCEEDED;
        } else {
            token_ids = (uint32_t *)malloc((size_t)token_count *
                                           sizeof(*token_ids));
            status = token_ids == NULL ? KQ_STATUS_OUT_OF_MEMORY : KQ_STATUS_OK;
        }
    }
    if (status == KQ_STATUS_OK)
        status = kq_tokenizer_encode(tokenizer, prompt, prompt_bytes,
                                     &encode_options, token_ids, token_count,
                                     &token_count, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_required_scratch_bytes(
            config, state, token_count, &scratch_bytes, &diagnostic);
    if (status == KQ_STATUS_OK && scratch_bytes > SIZE_MAX)
        status = KQ_STATUS_LIMIT_EXCEEDED;
    if (status == KQ_STATUS_OK) {
        scratch = malloc((size_t)scratch_bytes);
        logits = (float *)malloc(KQ_MODEL_EXEC_VOCABULARY_SIZE *
                                 sizeof(*logits));
        if (scratch == NULL || logits == NULL) status = KQ_STATUS_OUT_OF_MEMORY;
    }
    if (status == KQ_STATUS_OK)
        status = kq_model_exec_prefill_first_token_f32(
            config, state, token_ids, token_count, logits,
            KQ_MODEL_EXEC_VOCABULARY_SIZE, decoded, sizeof(decoded), scratch,
            scratch_bytes, progress, NULL, &result, &diagnostic);
    if (status != KQ_STATUS_OK) {
        print_failure(status, &diagnostic);
        goto done;
    }
    (void)printf("token_id=%u\ndecoded_utf8=", result.selected_token_id);
    (void)fwrite(decoded, 1U, (size_t)result.decoded_utf8_bytes, stdout);
    (void)printf("\nis_eog=%d\nprompt_tokens=%llu\n"
                 "logical_payload_bytes_touched=%llu\n"
                 "payload_blocks_touched=%llu\n"
                 "unique_semantic_tensors_touched=%llu\n"
                 "embedding_logical_bytes_touched=%llu\n"
                 "routed_expert_selections=%llu\n"
                 "selected_expert_matrix_requests=%llu\n"
                 "routed_expert_member_requests=%llu\n"
                 "ple_row_requests=%llu\n"
                 "lm_head_logical_bytes_touched=%llu\n"
                 "persistent_state_bytes=%llu\npeak_scratch_bytes=%llu\n"
                 "logits_bytes=%llu\n"
                 "maximum_f32_weight_bytes_materialized=%llu\n"
                 "elapsed_nanoseconds=%llu\n",
                 result.selected_token_is_eog,
                 (unsigned long long)result.metrics.prompt_tokens,
                 (unsigned long long)result.metrics.logical_payload_bytes_touched,
                 (unsigned long long)result.metrics.payload_blocks_touched,
                 (unsigned long long)result.metrics.unique_semantic_tensors_touched,
                 (unsigned long long)result.metrics.embedding_logical_bytes_touched,
                 (unsigned long long)result.metrics.routed_expert_selections,
                 (unsigned long long)result.metrics.selected_expert_matrix_requests,
                 (unsigned long long)result.metrics.routed_expert_member_requests,
                 (unsigned long long)result.metrics.ple_row_requests,
                 (unsigned long long)result.metrics.lm_head_logical_bytes_touched,
                 (unsigned long long)result.metrics.persistent_state_bytes,
                 (unsigned long long)result.metrics.peak_scratch_bytes,
                 (unsigned long long)result.metrics.logits_bytes,
                 (unsigned long long)result.metrics.maximum_f32_weight_bytes_materialized,
                 (unsigned long long)result.metrics.elapsed_nanoseconds);
    print_top_logits(logits, KQ_MODEL_EXEC_VOCABULARY_SIZE, 20U);
    (void)printf("oracle_selected_token_271_logit=%.9g\n", logits[271]);
    exit_code = 0;

done:
    free(logits);
    free(scratch);
    free(token_ids);
    kq_model_exec_state_close(state);
    kq_model_exec_config_close(config);
    kq_weight_provider_close(provider);
    kq_tokenizer_close(tokenizer);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    free(prompt);
    return exit_code;
}
