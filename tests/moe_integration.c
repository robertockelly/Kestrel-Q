#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_moe.h"
#include "kq_status.h"
#include "kq_tensor_view.h"

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
    if (copied == 0U || copied >= required) { free(path); return NULL; }
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
    kq_moe_config *config = NULL;
    kq_tensor_view *view = NULL;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    LARGE_INTEGER frequency, start, end;
    uint32_t layer;
    uint32_t configs = 0U, f32_configs = 0U, views = 0U;
    uint32_t common = 0U, layer2 = 0U, special = 0U;
    uint64_t payload_before = 0U, config_bytes = 0U;
    uint64_t selected_packed_total = 0U;
    int result = 1;
    static const uint32_t expert_id = 7U;

    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; MoE integration skipped\n");
        return KQ_TEST_SKIP;
    }
    if (!QueryPerformanceFrequency(&frequency)) { free(path); return 1; }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK) {
        payload_before = kq_gguf_payload_bytes_accessed(gguf);
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "MoE integration construction failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }
    if (!QueryPerformanceCounter(&start)) goto cleanup;
    for (layer = 0U; layer < kq_model_layer_count(model); ++layer) {
        char id[KQ_SEMANTIC_ID_CAPACITY];
        const kq_semantic_tensor *down;
        const kq_semantic_tensor *gate_up;
        uint64_t packed = 0U;
        const kq_tensor_view_info *info;
        int length;
        status = kq_moe_config_create(model, layer, &config, &diagnostic);
        if (status != KQ_STATUS_OK || config == NULL ||
            kq_moe_config_layer_id(config) != layer ||
            kq_moe_config_hidden_size(config) != 2560U ||
            kq_moe_config_expert_count(config) != 512U ||
            kq_moe_config_top_k(config) != 10U ||
            kq_moe_config_routed_intermediate_size(config) != 640U ||
            kq_moe_config_shared_intermediate_size(config) != 640U ||
            kq_moe_config_activation_dtype(config) != KQ_MOE_ACTIVATION_BF16) {
            fprintf(stderr, "MoE layer %u config failed: %s\n",
                    layer, diagnostic.message);
            goto cleanup;
        }
        config_bytes += kq_moe_config_owned_bytes(config);
        configs += 1U;
        kq_moe_config_close(config); config = NULL;
        status = kq_moe_config_create_reference_f32(
            model, layer, &config, &diagnostic);
        if (status != KQ_STATUS_OK || config == NULL ||
            kq_moe_config_activation_dtype(config) != KQ_MOE_ACTIVATION_F32) {
            fprintf(stderr, "MoE F32 config %u failed: %s\n",
                    layer, diagnostic.message);
            goto cleanup;
        }
        f32_configs += 1U;
        kq_moe_config_close(config); config = NULL;

        length = snprintf(id, sizeof(id), "layer.%02u.moe.routed.down", layer);
        if (length < 0 || (size_t)length >= sizeof(id)) goto cleanup;
        down = kq_model_find_semantic_tensor(model, id);
        length = snprintf(id, sizeof(id), "layer.%02u.moe.routed.gate_up", layer);
        if (length < 0 || (size_t)length >= sizeof(id)) goto cleanup;
        gate_up = kq_model_find_semantic_tensor(model, id);
        if (down == NULL || gate_up == NULL) goto cleanup;
        status = kq_tensor_view_open_expert_member(
            gguf, down, 0U, expert_id, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            &view, &diagnostic);
        if (status != KQ_STATUS_OK) goto cleanup;
        info = kq_tensor_view_get_info(view);
        packed += info->mapped_logical_length;
        kq_tensor_view_close(view); view = NULL; views += 1U;
        status = kq_tensor_view_open_expert_member(
            gguf, gate_up, 0U, expert_id, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            &view, &diagnostic);
        if (status != KQ_STATUS_OK) goto cleanup;
        info = kq_tensor_view_get_info(view);
        packed += info->mapped_logical_length;
        kq_tensor_view_close(view); view = NULL; views += 1U;
        status = kq_tensor_view_open_expert_member(
            gguf, gate_up, 1U, expert_id, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            &view, &diagnostic);
        if (status != KQ_STATUS_OK) goto cleanup;
        info = kq_tensor_view_get_info(view);
        packed += info->mapped_logical_length;
        kq_tensor_view_close(view); view = NULL; views += 1U;
        selected_packed_total += packed;
        if (layer == 2U && packed == UINT64_C(3993600)) layer2 += 1U;
        else if ((layer == 4U || layer == 30U || layer == 46U || layer == 47U) &&
                 packed == UINT64_C(3584000)) special += 1U;
        else if (packed == UINT64_C(3072000)) common += 1U;
        else {
            fprintf(stderr, "MoE layer %u selected expert packed bytes=%llu\n",
                    layer, (unsigned long long)packed);
            goto cleanup;
        }
    }
    if (!QueryPerformanceCounter(&end)) goto cleanup;
    if (configs != 48U || f32_configs != 48U || views != 144U ||
        common != 43U || layer2 != 1U || special != 4U ||
        payload_before != 0U || kq_gguf_payload_bytes_accessed(gguf) != 0U) {
        fprintf(stderr, "MoE integration counts/payload mismatch\n");
        goto cleanup;
    }
    printf("MoE real integration: PASS; layers=48, f32_configs=48, "
           "expert_views=144, common_layers=43, layer2_mix=1, "
           "special_layers=4, expert_id=7, selected_packed_total=%llu, "
           "config_total_ns=%llu, config_owned_bytes_total=%llu, "
           "real_model_payload_logical_bytes_touched=0, "
           "real_model_payload_blocks_touched=0\n",
           (unsigned long long)selected_packed_total,
           (unsigned long long)elapsed_ns(start, end, frequency),
           (unsigned long long)config_bytes);
    result = 0;
cleanup:
    kq_tensor_view_close(view);
    kq_moe_config_close(config);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
