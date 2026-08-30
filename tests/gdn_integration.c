#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gdn.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_status.h"

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
    if (path == NULL) return NULL;
    copied = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, required);
    if (copied == 0U || copied >= required) {
        free(path);
        return NULL;
    }
    return path;
}

static uint64_t elapsed_nanoseconds(LARGE_INTEGER start,
                                    LARGE_INTEGER end,
                                    LARGE_INTEGER frequency) {
    uint64_t ticks = (uint64_t)(end.QuadPart - start.QuadPart);
    return (ticks * UINT64_C(1000000000)) / (uint64_t)frequency.QuadPart;
}

int wmain(void) {
    wchar_t *path = load_model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_gdn_config *config = NULL;
    kq_gdn_state *state = NULL;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    uint32_t layer;
    uint32_t gdn_count = 0U;
    uint32_t qsa_rejected = 0U;
    uint64_t payload_before = 0U;
    uint64_t config_bytes = 0U;
    uint64_t state_bytes = 0U;
    uint64_t scratch_bytes = 0U;
    uint32_t reference_f32_configs = 0U;
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    int result = 1;

    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; GDN integration skipped\n");
        return KQ_TEST_SKIP;
    }
    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "high-resolution timer unavailable\n");
        free(path);
        return 1;
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
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "GDN integration construction failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }

    if (!QueryPerformanceCounter(&start)) {
        fprintf(stderr, "high-resolution timer start failed\n");
        goto cleanup;
    }

    for (layer = 0U; layer < kq_model_layer_count(model); ++layer) {
        status = kq_gdn_config_create(model, layer, &config, &diagnostic);
        if (kq_model_layer_type_at(model, layer) == KQ_MODEL_LAYER_GDN) {
            if (status != KQ_STATUS_OK || config == NULL ||
                kq_gdn_config_hidden_size(config) != 2560U ||
                kq_gdn_config_key_head_count(config) != 16U ||
                kq_gdn_config_value_head_count(config) != 48U ||
                kq_gdn_config_key_head_dimension(config) != 128U ||
                kq_gdn_config_value_head_dimension(config) != 128U ||
                kq_gdn_config_conv_channel_count(config) != 10240U ||
                kq_gdn_config_conv_kernel_size(config) != 4U ||
                kq_gdn_config_activation_dtype(config) !=
                    KQ_GDN_ACTIVATION_BF16 ||
                kq_gdn_config_recurrent_element_count(config) != 786432U ||
                kq_gdn_config_conv_state_element_count(config) != 40960U) {
                fprintf(stderr, "GDN layer %u structural validation failed: %s\n",
                        layer, diagnostic.message);
                goto cleanup;
            }
            if (gdn_count == 0U) {
                status = kq_gdn_state_create(config, &state, &diagnostic);
                if (status != KQ_STATUS_OK) {
                    fprintf(stderr, "target GDN state construction failed: %s\n",
                            diagnostic.message);
                    goto cleanup;
                }
                state_bytes = kq_gdn_state_owned_bytes(state);
                scratch_bytes = kq_gdn_config_scratch_bytes(config);
                kq_gdn_state_close(state);
                state = NULL;
            }
            config_bytes += kq_gdn_config_owned_bytes(config);
            gdn_count += 1U;
        } else {
            if (status != KQ_STATUS_INCOMPATIBLE_GDN || config != NULL) {
                fprintf(stderr, "QSA layer %u did not reject GDN config\n", layer);
                goto cleanup;
            }
            qsa_rejected += 1U;
        }
        kq_gdn_config_close(config);
        config = NULL;
    }
    if (!QueryPerformanceCounter(&end)) {
        fprintf(stderr, "high-resolution timer read failed\n");
        goto cleanup;
    }
    for (layer = 0U; layer < kq_model_layer_count(model); ++layer) {
        if (kq_model_layer_type_at(model, layer) != KQ_MODEL_LAYER_GDN) {
            continue;
        }
        status = kq_gdn_config_create_reference_f32(
            model, layer, &config, &diagnostic);
        if (status != KQ_STATUS_OK || config == NULL ||
            kq_gdn_config_activation_dtype(config) != KQ_GDN_ACTIVATION_F32) {
            fprintf(stderr, "F32 reference config failed at layer %u: %s\n",
                    layer, diagnostic.message);
            goto cleanup;
        }
        reference_f32_configs += 1U;
        kq_gdn_config_close(config);
        config = NULL;
    }
    if (gdn_count != 36U || qsa_rejected != 12U ||
        reference_f32_configs != 36U ||
        payload_before != 0U || kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr,
                "GDN real integration counts/payload boundary mismatch\n");
        goto cleanup;
    }

    printf("GDN real integration: PASS; layers=36, f32_reference_configs=36, "
           "qsa_rejected=12, "
           "config_total_ns=%llu, config_owned_bytes_total=%llu, "
           "batch1_state_owned_bytes=%llu, scratch_bytes=%llu, "
           "real_model_payload_logical_bytes_touched=0, "
           "real_model_payload_blocks_touched=0\n",
           (unsigned long long)elapsed_nanoseconds(start, end, frequency),
           (unsigned long long)config_bytes,
           (unsigned long long)state_bytes,
           (unsigned long long)scratch_bytes);
    result = 0;

cleanup:
    kq_gdn_state_close(state);
    kq_gdn_config_close(config);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
