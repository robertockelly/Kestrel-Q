#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_numeric.h"
#include "kq_sha256.h"
#include "kq_status.h"
#include "kq_tensor_view.h"

#define KQ_TEST_SKIP 77
#define KQ_TASK25_REAL_PAYLOAD_SAMPLE_BUDGET UINT64_C(1048576)

typedef enum sample_view_kind {
    SAMPLE_BINDING = 0,
    SAMPLE_EXPERT,
    SAMPLE_PLE
} sample_view_kind;

typedef struct sample_spec {
    const char *id;
    const char *semantic_id;
    uint32_t expected_type;
    sample_view_kind view_kind;
    uint32_t binding_index;
    uint32_t member_id;
    uint64_t block_index;
} sample_spec;

static const sample_spec samples[] = {
    {"real-f32-router", "layer.00.moe.router", KQ_GGUF_TYPE_F32,
     SAMPLE_BINDING, 0U, 0U, 0U},
    {"real-bf16-qsa-query", "layer.11.qsa.indexer.qk", KQ_GGUF_TYPE_BF16,
     SAMPLE_BINDING, 0U, 0U, 0U},
    {"real-q5-1-expert", "layer.00.moe.routed.down", KQ_GGUF_TYPE_Q5_1,
     SAMPLE_EXPERT, 0U, 7U, 0U},
    {"real-q4-k-expert-gate", "layer.00.moe.routed.gate_up", KQ_GGUF_TYPE_Q4_K,
     SAMPLE_EXPERT, 0U, 7U, 0U},
    {"real-q8-0-shared", "layer.00.moe.shared.down", KQ_GGUF_TYPE_Q8_0,
     SAMPLE_BINDING, 0U, 0U, 0U},
    {"real-layer2-q8-down", "layer.02.moe.routed.down", KQ_GGUF_TYPE_Q8_0,
     SAMPLE_EXPERT, 0U, 7U, 0U},
    {"real-layer2-q5-k-gate", "layer.02.moe.routed.gate_up", KQ_GGUF_TYPE_Q5_K,
     SAMPLE_EXPERT, 0U, 7U, 0U},
    {"real-layer2-q5-k-up", "layer.02.moe.routed.gate_up", KQ_GGUF_TYPE_Q5_K,
     SAMPLE_EXPERT, 1U, 7U, 0U},
    {"real-iq4-nl-ple", "layer.01.ple.table.064", KQ_GGUF_TYPE_IQ4_NL,
     SAMPLE_PLE, 0U, 64U, 0U}
};

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

static void print_digest(const unsigned char digest[32]) {
    uint32_t index;
    for (index = 0U; index < 32U; ++index)
        printf("%02x", (unsigned int)digest[index]);
}

static void print_hex(const unsigned char *bytes, uint64_t count) {
    uint64_t index;
    for (index = 0U; index < count; ++index)
        printf("%02x", (unsigned int)bytes[index]);
}

static const char *type_name(uint32_t type_id) {
    const kq_gguf_type_info *info = kq_gguf_type_info_for(type_id);
    return info == NULL ? "UNKNOWN" : info->name;
}

static kq_status open_sample_view(const kq_gguf *gguf,
                                  const kq_model *model,
                                  const sample_spec *sample,
                                  kq_tensor_view **view,
                                  kq_diagnostic *diagnostic) {
    const kq_semantic_tensor *semantic =
        kq_model_find_semantic_tensor(model, sample->semantic_id);
    if (semantic == NULL) return KQ_STATUS_SEMANTIC_MAPPING_FAILED;
    if (sample->view_kind == SAMPLE_BINDING)
        return kq_tensor_view_open_binding(gguf, semantic,
                                           sample->binding_index,
                                           KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
                                           view, diagnostic);
    if (sample->view_kind == SAMPLE_EXPERT)
        return kq_tensor_view_open_expert_member(
            gguf, semantic, sample->binding_index, sample->member_id,
            KQ_TENSOR_VIEW_PHYSICAL_LAYOUT, view, diagnostic);
    return kq_tensor_view_open_ple_member(gguf, semantic,
                                          KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
                                          view, diagnostic);
}

int wmain(int argc, wchar_t **argv) {
    wchar_t *path = load_model_path();
    int raw_mode = argc == 2 && wcscmp(argv[1], L"--oracle-raw") == 0;
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    uint64_t total_bytes = 0U;
    uint64_t total_blocks = 0U;
    uint64_t sample_index;
    int result = 1;

    if (argc > 2 || (argc == 2 && !raw_mode)) {
        fprintf(stderr, "usage: numeric_integration [--oracle-raw]\n");
        return 2;
    }
    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; numeric integration skipped\n");
        return KQ_TEST_SKIP;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status == KQ_STATUS_OK) status = kq_gguf_open(file, &gguf, &diagnostic);
    if (status == KQ_STATUS_OK)
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "numeric integration construction failed: %s\n",
                diagnostic.message);
        goto cleanup;
    }

    for (sample_index = 0U;
         sample_index < (uint64_t)(sizeof(samples) / sizeof(samples[0]));
         ++sample_index) {
        const sample_spec *sample = &samples[sample_index];
        kq_tensor_view *view = NULL;
        const kq_tensor_view_info *info;
        const unsigned char *packed;
        float decoded[KQ_NUMERIC_MAX_BLOCK_ELEMENTS];
        uint64_t decoded_count = 0U;
        unsigned char digest[32];
        kq_sha256 sha;
        float minimum = FLT_MAX;
        float maximum = -FLT_MAX;
        double sum = 0.0;
        uint64_t index;

        status = open_sample_view(gguf, model, sample, &view, &diagnostic);
        if (status != KQ_STATUS_OK) {
            fprintf(stderr, "open %s failed: %s\n", sample->id,
                    diagnostic.message);
            kq_tensor_view_close(view);
            goto cleanup;
        }
        info = kq_tensor_view_get_info(view);
        packed = kq_tensor_view_physical_data(view);
        if (info == NULL || packed == NULL ||
            info->type_id != sample->expected_type ||
            sample->block_index >= info->block_count ||
            total_bytes > KQ_TASK25_REAL_PAYLOAD_SAMPLE_BUDGET -
                              info->bytes_per_block) {
            fprintf(stderr, "sample %s violated its type/span/budget guard\n",
                    sample->id);
            kq_tensor_view_close(view);
            goto cleanup;
        }
        status = kq_dequantize_view_blocks_f32(
            view, sample->block_index, 1U, KQ_NUMERIC_PHYSICAL_ORDER,
            decoded, KQ_NUMERIC_MAX_BLOCK_ELEMENTS, &decoded_count,
            &diagnostic);
        if (status != KQ_STATUS_OK || decoded_count != info->block_elements) {
            fprintf(stderr, "decode %s failed: %s\n", sample->id,
                    diagnostic.message);
            kq_tensor_view_close(view);
            goto cleanup;
        }
        for (index = 0U; index < decoded_count; ++index) {
            if (decoded[index] < minimum) minimum = decoded[index];
            if (decoded[index] > maximum) maximum = decoded[index];
            sum += (double)decoded[index];
        }
        kq_sha256_init(&sha);
        kq_sha256_update(&sha, (const unsigned char *)decoded,
                         (size_t)decoded_count * sizeof(float));
        kq_sha256_final(&sha, digest);
        printf("sample\t%s\t%s\t%s\t%llu\t%llu\t",
               sample->id, sample->semantic_id, type_name(info->type_id),
               (unsigned long long)sample->block_index,
               (unsigned long long)info->bytes_per_block);
        print_digest(digest);
        printf("\t%a\t%a\t%a", (double)minimum, (double)maximum, sum);
        if (raw_mode) {
            printf("\t");
            print_hex(packed + (size_t)(sample->block_index *
                                       info->bytes_per_block),
                      info->bytes_per_block);
        }
        putchar('\n');
        total_bytes += info->bytes_per_block;
        total_blocks += 1U;
        kq_tensor_view_close(view);
    }
    {
        const kq_semantic_tensor *transformed =
            kq_model_find_semantic_tensor(model, "layer.00.gdn.qkv");
        kq_tensor_view *view = NULL;
        float decoded[KQ_NUMERIC_MAX_BLOCK_ELEMENTS];
        uint64_t decoded_count = 0U;
        if (transformed == NULL) {
            fprintf(stderr, "transformed-layout regression semantic missing\n");
            goto cleanup;
        }
        status = kq_tensor_view_open_binding(
            gguf, transformed, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            &view, &diagnostic);
        if (status == KQ_STATUS_OK) {
            status = kq_dequantize_view_blocks_f32(
                view, 0U, 1U, KQ_NUMERIC_REQUIRE_CANONICAL_ORDER,
                decoded, KQ_NUMERIC_MAX_BLOCK_ELEMENTS,
                &decoded_count, &diagnostic);
        }
        if (status != KQ_STATUS_TENSOR_LAYOUT_MISMATCH) {
            fprintf(stderr, "transformed canonical-order misuse did not fail closed\n");
            kq_tensor_view_close(view);
            goto cleanup;
        }
        status = kq_dequantize_view_blocks_f32(
            view, kq_tensor_view_get_info(view)->block_count, 1U,
            KQ_NUMERIC_PHYSICAL_ORDER, decoded,
            KQ_NUMERIC_MAX_BLOCK_ELEMENTS, &decoded_count, &diagnostic);
        if (status != KQ_STATUS_SPAN_OUT_OF_RANGE) {
            fprintf(stderr, "out-of-view block request did not fail closed\n");
            kq_tensor_view_close(view);
            goto cleanup;
        }
        kq_tensor_view_close(view);
    }
    if (total_bytes > KQ_TASK25_REAL_PAYLOAD_SAMPLE_BUDGET) {
        fprintf(stderr, "real payload sample budget exceeded\n");
        goto cleanup;
    }
    printf("summary\treal_model_payload_logical_bytes_touched\t%llu\t"
           "real_model_payload_blocks_touched\t%llu\tbudget\t%llu\n",
           (unsigned long long)total_bytes,
           (unsigned long long)total_blocks,
           (unsigned long long)KQ_TASK25_REAL_PAYLOAD_SAMPLE_BUDGET);
    result = 0;

cleanup:
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
