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
#include "kq_ple.h"
#include "kq_status.h"
#include "kq_tokenizer.h"

#define KQ_TEST_SKIP 77

static wchar_t *load_model_path(void) {
    DWORD required = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    wchar_t *path;
    DWORD copied;

    if (required == 0U ||
        (uint64_t)required > (uint64_t)(SIZE_MAX / sizeof(*path))) {
        return NULL;
    }
    path = (wchar_t *)malloc((size_t)required * sizeof(*path));
    if (path == NULL) {
        return NULL;
    }
    copied = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, required);
    if (copied == 0U || copied >= required) {
        free(path);
        return NULL;
    }
    return path;
}

int wmain(void) {
    static const unsigned char text[] = "Hello, Kestrel-Q.";
    static const uint32_t expected_ids[] = {
        9419U, 11U, 710U, 467U, 3621U, 27325U, 13U
    };
    static const uint32_t prefill_ids[] = {101U, 102U, 103U};
    wchar_t *path = load_model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_ple_config *config = NULL;
    kq_ple_stream_state state;
    kq_ple_address_intent prefill[48];
    kq_ple_address_intent decoded[16];
    kq_ple_address_intent text_intents[112];
    kq_ple_run_metrics prefill_metrics = {0};
    kq_ple_run_metrics decode_metrics = {0};
    kq_ple_run_metrics text_metrics = {0};
    kq_tokenizer_encode_options encode_options = {
        KQ_TOKENIZER_SPECIAL_RECOGNIZE
    };
    kq_diagnostic diagnostic;
    kq_status status;
    uint32_t token_ids[7];
    uint64_t required = 0U;
    uint64_t payload_before = 0U;
    int result = 1;

    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; PLE integration skipped\n");
        return KQ_TEST_SKIP;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) {
        status = kq_gguf_open(file, &gguf, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        payload_before = kq_gguf_payload_bytes_accessed(gguf);
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_ple_config_open_from_model(model, &config, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "PLE integration construction failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }

    status = kq_tokenizer_encode(tokenizer,
                                 text,
                                 sizeof(text) - 1U,
                                 &encode_options,
                                 token_ids,
                                 7U,
                                 &required,
                                 &diagnostic);
    if (status != KQ_STATUS_OK || required != 7U ||
        memcmp(token_ids, expected_ids, sizeof(expected_ids)) != 0) {
        fprintf(stderr, "tokenizer-to-PLE integration IDs differ\n");
        goto cleanup;
    }
    status = kq_ple_state_reset(config, &state, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_ple_generate_prefill(config,
                                         &state,
                                         token_ids,
                                         7U,
                                         text_intents,
                                         112U,
                                         &required,
                                         &text_metrics,
                                         &diagnostic);
    }
    if (status != KQ_STATUS_OK || required != 112U ||
        state.position != 7U || text_intents[0].token_id != 9419U) {
        fprintf(stderr, "tokenizer-to-PLE address generation failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }

    status = kq_ple_state_reset(config, &state, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_ple_generate_prefill(config,
                                         &state,
                                         prefill_ids,
                                         3U,
                                         prefill,
                                         48U,
                                         &required,
                                         &prefill_metrics,
                                         &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_ple_generate_decode_step(config,
                                             &state,
                                             104U,
                                             decoded,
                                             16U,
                                             &required,
                                             &decode_metrics,
                                             &diagnostic);
    }
    if (status != KQ_STATUS_OK || required != 16U ||
        decoded[0].global_address != UINT64_C(18639696) ||
        decoded[15].global_address != UINT64_C(318589180)) {
        fprintf(stderr, "PLE real-config canonical address check failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }
    if (payload_before != 0U || kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr, "PLE integration crossed the tensor payload boundary\n");
        goto cleanup;
    }

    printf("PLE real integration: PASS; config_ns=%llu, config_owned_bytes=%llu, "
           "state_bytes=%llu, addresses_per_token=16, prefill_ns=%llu, "
           "decode_ns=%llu, PLE_payload_views_opened=0, "
           "model_tensor_payload_bytes_touched=0\n",
           (unsigned long long)kq_ple_config_get_metrics(config)->construction_nanoseconds,
           (unsigned long long)kq_ple_config_get_metrics(config)->owned_heap_bytes,
           (unsigned long long)kq_ple_config_get_metrics(config)->stream_state_bytes,
           (unsigned long long)prefill_metrics.elapsed_nanoseconds,
           (unsigned long long)decode_metrics.elapsed_nanoseconds);
    result = 0;

cleanup:
    kq_ple_config_close(config);
    kq_tokenizer_close(tokenizer);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
