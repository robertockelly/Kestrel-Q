#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_layer.h"
#include "kq_model.h"
#include "kq_sha256.h"
#include "kq_weight_provider.h"

#define KQ_TASK211_BUDGET (UINT64_C(768) * UINT64_C(1024) * UINT64_C(1024))
#define KQ_TARGET_WIDTH UINT64_C(10240)

static wchar_t *path_from_env(void) {
    DWORD count = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    wchar_t *path;
    DWORD got;
    if (count == 0U) return NULL;
    path = (wchar_t *)malloc((size_t)count * sizeof(*path));
    if (path == NULL) return NULL;
    got = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, count);
    if (got == 0U || got >= count) { free(path); return NULL; }
    return path;
}

static void print_digest(const float *values, uint64_t count) {
    kq_sha256 sha;
    unsigned char digest[32];
    uint32_t i;
    kq_sha256_init(&sha);
    kq_sha256_update(&sha, (const unsigned char *)values,
                     (size_t)(count * sizeof(*values)));
    kq_sha256_final(&sha, digest);
    for (i = 0U; i < 32U; ++i) (void)printf("%02x", digest[i]);
}

static void print_values(const char *family, const char *phase,
                         const float *values, uint64_t count) {
    uint64_t index;
    (void)printf("VALUES family=%s phase=%s count=%llu", family, phase,
                 (unsigned long long)count);
    for (index = 0U; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        (void)printf(" %08x", bits);
    }
    (void)printf("\n");
}

static int run_layer(kq_weight_provider *provider, const kq_model *model,
                     uint32_t layer_id, const char *family,
                     int input_profile, int dump_values,
                     kq_diagnostic *diagnostic) {
    kq_layer_config *config = NULL;
    kq_layer_state *state = NULL;
    float *input = NULL;
    float *output = NULL;
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    uint32_t token = 100U + layer_id;
    uint64_t index;
    kq_layer_metrics metrics;
    kq_status status;
    int result = 0;
    status = kq_layer_config_create_reference_f32(
        model, layer_id, &config, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    status = kq_layer_state_create(config, 8U, &state, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    status = kq_layer_required_quantized_scratch_bytes(
        config, state, 1U, &scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    input = (float *)malloc((size_t)KQ_TARGET_WIDTH * sizeof(*input));
    output = (float *)malloc((size_t)KQ_TARGET_WIDTH * sizeof(*output));
    scratch = malloc((size_t)scratch_bytes);
    if (input == NULL || output == NULL || scratch == NULL) {
        status = KQ_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (index = 0U; index < KQ_TARGET_WIDTH; ++index)
        input[index] = input_profile == 2
            ? (float)((int)(index % 41U) - 20) * 0.000244140625f
            : input_profile == 4
                ? (float)((int)((index * 17U) % 67U) - 33) * 0.000244140625f
            : input_profile == 3
                ? (float)((int)(index % 43U) - 21) * 0.000244140625f
            : input_profile == 1
                ? (float)((int)(index % 53U) - 26) * 0.000244140625f
                : (float)((int)(index % 31U) - 15) * 0.0009765625f;
    status = kq_layer_prefill_quantized_f32(
        config, provider, input,
        kq_layer_config_family(config) == KQ_LAYER_FAMILY_PLE_GDN
            ? &token : NULL,
        1U, NULL, output, KQ_TARGET_WIDTH, state, scratch, scratch_bytes,
        NULL, NULL, &metrics, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    if ((layer_id == 3U &&
         (metrics.qsa_selection_events != 1U ||
          metrics.qsa_candidate_blocks != 0U ||
          metrics.qsa_selected_blocks != 0U ||
          metrics.qsa_selected_tokens != 1U)) ||
        (layer_id != 3U && metrics.qsa_selection_events != 0U)) {
        (void)fprintf(stderr,
            "unexpected QSA prefill selection metrics: events=%llu "
            "candidates=%llu blocks=%llu tokens=%llu\n",
            (unsigned long long)metrics.qsa_selection_events,
            (unsigned long long)metrics.qsa_candidate_blocks,
            (unsigned long long)metrics.qsa_selected_blocks,
            (unsigned long long)metrics.qsa_selected_tokens);
        status = KQ_STATUS_INCOMPATIBLE_LAYER;
        goto done;
    }
    (void)printf("TARGET family=%s layer=%u phase=prefill output_sha256=",
                 family, layer_id);
    print_digest(output, KQ_TARGET_WIDTH);
    (void)printf(" position=%llu elapsed_ns=%llu\n",
                 (unsigned long long)kq_layer_state_position(state),
                 (unsigned long long)metrics.elapsed_nanoseconds);
    if (dump_values) print_values(family, "prefill", output, KQ_TARGET_WIDTH);
    for (index = 0U; index < KQ_TARGET_WIDTH; ++index)
        input[index] = input_profile == 2
            ? (float)((int)(index % 47U) - 23) * 0.0001220703125f
            : input_profile == 4
                ? (float)((int)((index * 29U) % 71U) - 35) * 0.0001220703125f
            : input_profile == 3
                ? (float)((int)(index % 61U) - 30) * 0.0001220703125f
            : input_profile == 1
                ? (float)((int)(index % 59U) - 29) * 0.0001220703125f
                : (float)((int)(index % 37U) - 18) * 0.00048828125f;
    token += 1U;
    status = kq_layer_decode_quantized_f32(
        config, provider, input, token, output, KQ_TARGET_WIDTH, state,
        scratch, scratch_bytes, NULL, NULL, &metrics, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    if ((layer_id == 3U &&
         (metrics.qsa_selection_events != 1U ||
          metrics.qsa_candidate_blocks != 0U ||
          metrics.qsa_selected_blocks != 0U ||
          metrics.qsa_selected_tokens != 2U)) ||
        (layer_id != 3U && metrics.qsa_selection_events != 0U)) {
        (void)fprintf(stderr,
            "unexpected QSA decode selection metrics: events=%llu "
            "candidates=%llu blocks=%llu tokens=%llu\n",
            (unsigned long long)metrics.qsa_selection_events,
            (unsigned long long)metrics.qsa_candidate_blocks,
            (unsigned long long)metrics.qsa_selected_blocks,
            (unsigned long long)metrics.qsa_selected_tokens);
        status = KQ_STATUS_INCOMPATIBLE_LAYER;
        goto done;
    }
    for (index = 0U; index < KQ_TARGET_WIDTH; ++index)
        if (!isfinite(output[index])) {
            status = KQ_STATUS_NUMERIC_DOMAIN;
            goto done;
        }
    (void)printf("TARGET family=%s layer=%u phase=decode output_sha256=",
                 family, layer_id);
    print_digest(output, KQ_TARGET_WIDTH);
    (void)printf(" position=%llu scratch_bytes=%llu state_bytes=%llu "
                 "elapsed_ns=%llu\n",
                 (unsigned long long)kq_layer_state_position(state),
                 (unsigned long long)scratch_bytes,
                 (unsigned long long)kq_layer_state_owned_bytes(state),
                 (unsigned long long)metrics.elapsed_nanoseconds);
    if (dump_values) print_values(family, "decode", output, KQ_TARGET_WIDTH);
    result = 1;
done:
    if (!result && status != KQ_STATUS_OK)
    {
        const kq_weight_provider_metrics *m =
            kq_weight_provider_get_metrics(provider);
        (void)fprintf(stderr,
                      "layer %u failed: %s: %s; payload=%llu blocks=%llu\n",
                      layer_id, kq_status_string(status), diagnostic->message,
                      (unsigned long long)(m != NULL
                          ? m->logical_payload_bytes_touched : 0U),
                      (unsigned long long)(m != NULL
                          ? m->quantized_blocks_touched : 0U));
    }
    free(scratch);
    free(output);
    free(input);
    kq_layer_state_close(state);
    kq_layer_config_close(config);
    return result;
}

static int run_failure_case(const kq_gguf *gguf, const kq_model *model,
                            uint64_t budget, uint64_t *payload_touched,
                            kq_diagnostic *diagnostic) {
    kq_weight_provider *provider = NULL;
    kq_layer_config *config = NULL;
    kq_layer_state *state = NULL;
    float *input = NULL;
    float *output = NULL;
    void *scratch = NULL;
    uint64_t scratch_bytes = 0U;
    uint64_t index;
    const kq_weight_provider_metrics *metrics;
    kq_status status;
    int passed = 0;
    status = kq_weight_provider_open(gguf, model, budget, &provider,
                                     diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_layer_config_create_reference_f32(
            model, 0U, &config, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_layer_state_create(config, 8U, &state, diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_layer_required_quantized_scratch_bytes(
            config, state, 1U, &scratch_bytes, diagnostic);
    if (status != KQ_STATUS_OK) goto done;
    input = (float *)calloc((size_t)KQ_TARGET_WIDTH, sizeof(*input));
    output = (float *)malloc((size_t)KQ_TARGET_WIDTH * sizeof(*output));
    scratch = malloc((size_t)scratch_bytes);
    if (input == NULL || output == NULL || scratch == NULL) goto done;
    for (index = 0U; index < KQ_TARGET_WIDTH; ++index) output[index] = 123.5f;
    status = kq_layer_prefill_quantized_f32(
        config, provider, input, NULL, 1U, NULL, output, KQ_TARGET_WIDTH,
        state, scratch, scratch_bytes, NULL, NULL, NULL, diagnostic);
    metrics = kq_weight_provider_get_metrics(provider);
    if (metrics == NULL || status != KQ_STATUS_LIMIT_EXCEEDED ||
        kq_layer_state_position(state) != 0U) goto done;
    for (index = 0U; index < KQ_TARGET_WIDTH; ++index)
        if (output[index] != 123.5f) goto done;
    *payload_touched = metrics->logical_payload_bytes_touched;
    passed = 1;
done:
    free(scratch);
    free(output);
    free(input);
    kq_layer_state_close(state);
    kq_layer_config_close(config);
    kq_weight_provider_close(provider);
    return passed;
}

static int validate_provider_fail_closed(kq_weight_provider *provider,
                                         const kq_model *model,
                                         kq_diagnostic *diagnostic) {
    const kq_semantic_tensor *semantic = kq_model_find_semantic_tensor(
        model, "layer.00.moe.routed.gate_up");
    float *input = (float *)calloc(2560U, sizeof(float));
    float *output = (float *)calloc(640U, sizeof(float));
    void *scratch = calloc(3200U, sizeof(float));
    kq_status status;
    int passed = 0;
    if (semantic == NULL || input == NULL || output == NULL || scratch == NULL)
        goto done;
    status = kq_weight_provider_linear_f32(
        provider, semantic, KQ_BINDING_PART_GATE, 512U, 640U, 2560U,
        input, output, 640U, scratch, 3200U * sizeof(float), diagnostic);
    if (status != KQ_STATUS_INVALID_ARGUMENT) goto done;
    status = kq_weight_provider_linear_f32(
        provider, semantic, KQ_BINDING_PART_GATE, 0U, 639U, 2560U,
        input, output, 640U, scratch, 3200U * sizeof(float), diagnostic);
    if (status != KQ_STATUS_DIMENSION_MISMATCH) goto done;
    status = kq_weight_provider_linear_f32(
        provider, semantic, KQ_BINDING_PART_WHOLE, 0U, 640U, 2560U,
        input, output, 640U, scratch, 3200U * sizeof(float), diagnostic);
    if (status != KQ_STATUS_TENSOR_LAYOUT_MISMATCH) goto done;
    status = kq_weight_provider_linear_f32(
        provider, semantic, KQ_BINDING_PART_GATE, 0U, 640U, 2560U,
        input, output, 639U, scratch, 3200U * sizeof(float), diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL) goto done;
    status = kq_weight_provider_linear_f32(
        provider, semantic, KQ_BINDING_PART_GATE, 0U, 640U, 2560U,
        input, input, 2560U, scratch, 3200U * sizeof(float), diagnostic);
    if (status != KQ_STATUS_ALIASING_VIOLATION) goto done;
    passed = 1;
done:
    free(scratch);
    free(output);
    free(input);
    return passed;
}

int wmain(int argc, wchar_t **argv) {
    static const uint32_t layers[3] = {0U, 3U, 1U};
    static const char *families[3] = {"GDN", "QSA", "PLE_GDN"};
    wchar_t *path = path_from_env();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_weight_provider *provider = NULL;
    const kq_weight_provider_metrics *metrics;
    kq_weight_provider_expert_request expert_trace[64];
    kq_weight_provider_expert_request route_trace[64];
    kq_weight_provider_ple_request ple_trace[64];
    uint64_t expert_count = 0U;
    uint64_t route_count = 0U;
    uint64_t ple_count = 0U;
    uint64_t failure_payload_a = 0U;
    uint64_t failure_payload_b = 0U;
    kq_diagnostic diagnostic;
    kq_status status = KQ_STATUS_OK;
    uint32_t i;
    int input_profile = 0;
    int dump_values_enabled = 0;
    for (i = 1U; i < (uint32_t)argc; ++i) {
        if (wcscmp(argv[i], L"--holdout") == 0) input_profile = 2;
        else if (wcscmp(argv[i], L"--calibration-permuted") == 0)
            input_profile = 4;
        else if (wcscmp(argv[i], L"--calibration-tertiary") == 0)
            input_profile = 3;
        else if (wcscmp(argv[i], L"--calibration-secondary") == 0)
            input_profile = 1;
        else if (wcscmp(argv[i], L"--dump-values") == 0)
            dump_values_enabled = 1;
        else return 2;
    }
    if (path == NULL) {
        (void)puts("KQ_GGUF_PATH unavailable; target layer integration skipped");
        return 77;
    }
    kq_diagnostic_clear(&diagnostic);
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_open(gguf, model, KQ_TASK211_BUDGET,
                                         &provider, &diagnostic);
    if (status != KQ_STATUS_OK) goto fail;
    if (!run_failure_case(gguf, model, 1U, &failure_payload_a,
                          &diagnostic) ||
        !run_failure_case(gguf, model, UINT64_C(8) * 1024U * 1024U,
                          &failure_payload_b, &diagnostic))
        goto fail;
    for (i = 0U; i < 48U; ++i) {
        status = kq_weight_provider_preflight_layer(provider, i, &diagnostic);
        if (status != KQ_STATUS_OK) goto fail;
    }
    if (kq_gguf_payload_bytes_accessed(gguf) != 0U) goto fail;
    if (!validate_provider_fail_closed(provider, model, &diagnostic) ||
        kq_gguf_payload_bytes_accessed(gguf) != 0U)
        goto fail;
    for (i = 0U; i < 3U; ++i)
        if (!run_layer(provider, model, layers[i], families[i], input_profile,
                       dump_values_enabled, &diagnostic))
            goto fail;
    metrics = kq_weight_provider_get_metrics(provider);
    status = kq_weight_provider_copy_expert_requests(
        provider, expert_trace, 64U, &expert_count);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_copy_route_requests(
            provider, route_trace, 64U, &route_count);
    if (status == KQ_STATUS_OK)
        status = kq_weight_provider_copy_ple_requests(
            provider, ple_trace, 64U, &ple_count);
    if (metrics == NULL ||
        status != KQ_STATUS_OK || expert_count != 60U || route_count != 60U ||
        ple_count != 32U ||
        metrics->logical_payload_bytes_touched + failure_payload_a +
            failure_payload_b > KQ_TASK211_BUDGET ||
        metrics->selected_expert_requests != UINT64_C(180) ||
        metrics->ple_row_requests != UINT64_C(32) ||
        metrics->maximum_f32_weight_bytes_materialized > UINT64_C(163840))
        goto fail;
    (void)printf("EXPERT_TRACE count=%llu", (unsigned long long)expert_count);
    for (i = 0U; i < expert_count; ++i)
        (void)printf(" %u:%u", expert_trace[i].layer_id,
                     expert_trace[i].expert_id);
    (void)printf("\nPLE_TRACE count=%llu", (unsigned long long)ple_count);
    for (i = 0U; i < ple_count; ++i)
        (void)printf(" %u:%llu", ple_trace[i].logical_member,
                     (unsigned long long)ple_trace[i].member_row);
    (void)printf("\n");
    (void)printf("ROUTE_TRACE count=%llu", (unsigned long long)route_count);
    for (i = 0U; i < route_count; ++i)
        (void)printf(" %u:%u", route_trace[i].layer_id,
                     route_trace[i].expert_id);
    (void)printf("\n");
    (void)printf(
        "target quantized layer integration: PASS; preflight=48/48 "
        "logical_payload_bytes_touched=%llu quantized_blocks_touched=%llu "
        "unique_semantics=%llu selected_expert_requests=%llu "
        "ple_row_requests=%llu maximum_f32_weight_bytes_materialized=%llu "
        "provider_owned_bytes=%llu linear_elapsed_ns=%llu budget=%llu\n",
        (unsigned long long)metrics->logical_payload_bytes_touched,
        (unsigned long long)metrics->quantized_blocks_touched,
        (unsigned long long)metrics->unique_semantic_tensors_touched,
        (unsigned long long)metrics->selected_expert_requests,
        (unsigned long long)metrics->ple_row_requests,
        (unsigned long long)metrics->maximum_f32_weight_bytes_materialized,
        (unsigned long long)metrics->provider_owned_bytes,
        (unsigned long long)metrics->linear_elapsed_nanoseconds,
        (unsigned long long)metrics->payload_budget_bytes);
    (void)printf(
        "provider rollback injection: PASS; first_request_payload=%llu "
        "post_gr_payload=%llu aggregate_payload=%llu\n",
        (unsigned long long)failure_payload_a,
        (unsigned long long)failure_payload_b,
        (unsigned long long)(metrics->logical_payload_bytes_touched +
                             failure_payload_a + failure_payload_b));
    kq_weight_provider_close(provider);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return 0;
fail:
    (void)fprintf(stderr, "target layer integration failed: %s: %s\n",
                  kq_status_string(status), diagnostic.message);
    kq_weight_provider_close(provider);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return 1;
}
