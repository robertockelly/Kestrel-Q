#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_qsa.h"
#include "kq_status.h"

#define KQ_TEST_SKIP 77

static wchar_t *load_model_path(void) {
    DWORD required = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    wchar_t *path;
    DWORD copied;
    if (required == 0U ||
        (uint64_t)required > (uint64_t)(SIZE_MAX / sizeof(*path))) return NULL;
    path = (wchar_t *)malloc((size_t)required * sizeof(*path));
    if (path == NULL) return NULL;
    copied = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, required);
    if (copied == 0U || copied >= required) {
        free(path);
        return NULL;
    }
    return path;
}

static uint64_t elapsed_ns(LARGE_INTEGER start, LARGE_INTEGER end,
                           LARGE_INTEGER frequency) {
    return ((uint64_t)(end.QuadPart - start.QuadPart) *
            UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
}

int wmain(void) {
    wchar_t *path = load_model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_qsa_config *config = NULL;
    kq_qsa_state *state = NULL;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    uint32_t layer;
    uint32_t qsa_count = 0U;
    uint32_t gdn_rejected = 0U;
    uint32_t f32_configs = 0U;
    uint64_t payload_before = 0U;
    uint64_t config_bytes = 0U;
    uint64_t state_bytes = 0U;
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    int result = 1;

    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; QSA integration skipped\n");
        return KQ_TEST_SKIP;
    }
    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "high-resolution timer unavailable\n");
        free(path);
        return 1;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK) {
        payload_before = kq_gguf_payload_bytes_accessed(gguf);
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "QSA integration construction failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }
    if (!QueryPerformanceCounter(&start)) goto cleanup;
    for (layer = 0U; layer < kq_model_layer_count(model); ++layer) {
        status = kq_qsa_config_create(model, layer, &config, &diagnostic);
        if (kq_model_layer_type_at(model, layer) == KQ_MODEL_LAYER_QSA) {
            if (status != KQ_STATUS_OK || config == NULL ||
                kq_qsa_config_layer_id(config) != layer ||
                kq_qsa_config_hidden_size(config) != 2560U ||
                kq_qsa_config_query_head_count(config) != 24U ||
                kq_qsa_config_key_value_head_count(config) != 2U ||
                kq_qsa_config_head_dimension(config) != 256U ||
                kq_qsa_config_index_query_head_count(config) != 4U ||
                kq_qsa_config_index_head_dimension(config) != 128U ||
                kq_qsa_config_block_size(config) != 4U ||
                kq_qsa_config_block_selection_limit(config) != 512U ||
                kq_qsa_config_context_limit(config) != 262144U ||
                kq_qsa_config_activation_dtype(config) !=
                    KQ_QSA_ACTIVATION_BF16 ||
                kq_qsa_config_semantic_state_bytes_per_token(config) != 2304U) {
                fprintf(stderr, "QSA layer %u structural validation failed: %s\n",
                        layer, diagnostic.message);
                goto cleanup;
            }
            if (qsa_count == 0U) {
                status = kq_qsa_state_create(config, 1U, &state, &diagnostic);
                if (status != KQ_STATUS_OK) {
                    fprintf(stderr, "target QSA state construction failed: %s\n",
                            diagnostic.message);
                    goto cleanup;
                }
                state_bytes = kq_qsa_state_owned_bytes(state);
                kq_qsa_state_close(state);
                state = NULL;
            }
            config_bytes += kq_qsa_config_owned_bytes(config);
            qsa_count += 1U;
        } else {
            if (status != KQ_STATUS_INCOMPATIBLE_QSA || config != NULL) {
                fprintf(stderr, "GDN layer %u did not reject QSA config\n", layer);
                goto cleanup;
            }
            gdn_rejected += 1U;
        }
        kq_qsa_config_close(config);
        config = NULL;
    }
    if (!QueryPerformanceCounter(&end)) goto cleanup;
    for (layer = 0U; layer < kq_model_layer_count(model); ++layer) {
        if (kq_model_layer_type_at(model, layer) != KQ_MODEL_LAYER_QSA) continue;
        status = kq_qsa_config_create_reference_f32(
            model, layer, &config, &diagnostic);
        if (status != KQ_STATUS_OK || config == NULL ||
            kq_qsa_config_activation_dtype(config) != KQ_QSA_ACTIVATION_F32) {
            fprintf(stderr, "QSA F32 config failed at layer %u: %s\n",
                    layer, diagnostic.message);
            goto cleanup;
        }
        f32_configs += 1U;
        kq_qsa_config_close(config);
        config = NULL;
    }
    if (qsa_count != 12U || gdn_rejected != 36U || f32_configs != 12U ||
        payload_before != 0U || kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr, "QSA real integration counts/payload mismatch\n");
        goto cleanup;
    }
    printf("QSA real integration: PASS; layers=12, f32_reference_configs=12, "
           "gdn_rejected=36, config_total_ns=%llu, "
           "config_owned_bytes_total=%llu, capacity1_state_owned_bytes=%llu, "
           "semantic_state_bytes_per_token_layer=2304, "
           "semantic_state_bytes_per_token_all_qsa=27648, "
           "real_model_payload_logical_bytes_touched=0, "
           "real_model_payload_blocks_touched=0\n",
           (unsigned long long)elapsed_ns(start, end, frequency),
           (unsigned long long)config_bytes,
           (unsigned long long)state_bytes);
    result = 0;

cleanup:
    kq_qsa_state_close(state);
    kq_qsa_config_close(config);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
