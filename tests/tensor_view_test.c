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
#include "kq_gguf_internal.h"
#include "kq_model.h"
#include "kq_status.h"
#include "kq_tensor_view.h"

#define VIEW_FIXTURE_CAPACITY 16384U
#define VIEW_TENSOR_COUNT 12U
#define VIEW_ALIGNMENT 32U

typedef struct view_tensor_spec {
    const char *name;
    uint32_t type_id;
    uint32_t rank;
    uint64_t dimensions[3];
    uint64_t relative_offset;
    uint64_t packed_bytes;
    unsigned char pattern;
} view_tensor_spec;

typedef struct view_fixture {
    unsigned char bytes[VIEW_FIXTURE_CAPACITY];
    size_t size;
    size_t data_section_offset;
    view_tensor_spec tensors[VIEW_TENSOR_COUNT];
} view_fixture;

static int failures = 0;
static uint64_t synthetic_payload_bytes_touched = 0U;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static int append_bytes(view_fixture *fixture,
                        const void *bytes,
                        size_t length) {
    if (fixture->size > VIEW_FIXTURE_CAPACITY ||
        length > VIEW_FIXTURE_CAPACITY - fixture->size) {
        return 0;
    }
    if (length != 0U) {
        memcpy(fixture->bytes + fixture->size, bytes, length);
    }
    fixture->size += length;
    return 1;
}

static int append_u8(view_fixture *fixture, uint8_t value) {
    return append_bytes(fixture, &value, sizeof(value));
}

static int append_u32(view_fixture *fixture, uint32_t value) {
    unsigned char bytes[4];
    unsigned int index;
    for (index = 0U; index < 4U; ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8U)) &
                                       UINT32_C(0xff));
    }
    return append_bytes(fixture, bytes, sizeof(bytes));
}

static int append_u64(view_fixture *fixture, uint64_t value) {
    unsigned char bytes[8];
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8U)) &
                                       UINT64_C(0xff));
    }
    return append_bytes(fixture, bytes, sizeof(bytes));
}

static int append_string(view_fixture *fixture, const char *text) {
    size_t length = strlen(text);
    return append_u64(fixture, (uint64_t)length) &&
           append_bytes(fixture, text, length);
}

static int pad_to(view_fixture *fixture, size_t alignment) {
    while ((fixture->size % alignment) != 0U) {
        if (!append_u8(fixture, 0U)) {
            return 0;
        }
    }
    return 1;
}

static uint64_t align_u64(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static int set_spec(view_tensor_spec *spec,
                    const char *name,
                    uint32_t type_id,
                    uint32_t rank,
                    uint64_t d0,
                    uint64_t d1,
                    uint64_t d2,
                    unsigned char pattern) {
    const kq_gguf_type_info *type_info = kq_gguf_type_info_for(type_id);
    uint64_t elements;

    if (type_info == NULL || rank == 0U || rank > 3U) {
        return 0;
    }
    spec->name = name;
    spec->type_id = type_id;
    spec->rank = rank;
    spec->dimensions[0] = d0;
    spec->dimensions[1] = rank >= 2U ? d1 : 1U;
    spec->dimensions[2] = rank >= 3U ? d2 : 1U;
    elements = spec->dimensions[0] * spec->dimensions[1] *
               spec->dimensions[2];
    if ((elements % type_info->block_elements) != 0U) {
        return 0;
    }
    spec->packed_bytes = (elements / type_info->block_elements) *
                         type_info->bytes_per_block;
    spec->pattern = pattern;
    return 1;
}

static int build_fixture(view_fixture *fixture) {
    uint64_t relative_cursor = 0U;
    uint64_t payload_end;
    uint32_t tensor_index;
    uint32_t dimension;
    size_t byte_index;

    memset(fixture, 0, sizeof(*fixture));
    if (!set_spec(&fixture->tensors[0], "whole_f32", KQ_GGUF_TYPE_F32,
                  1U, 4U, 1U, 1U, 0x10U) ||
        !set_spec(&fixture->tensors[1], "whole_bf16", KQ_GGUF_TYPE_BF16,
                  1U, 4U, 1U, 1U, 0x20U) ||
        !set_spec(&fixture->tensors[2], "whole_q5_1", KQ_GGUF_TYPE_Q5_1,
                  1U, 32U, 1U, 1U, 0x30U) ||
        !set_spec(&fixture->tensors[3], "whole_q8_0", KQ_GGUF_TYPE_Q8_0,
                  1U, 32U, 1U, 1U, 0x40U) ||
        !set_spec(&fixture->tensors[4], "whole_q4_k", KQ_GGUF_TYPE_Q4_K,
                  1U, 256U, 1U, 1U, 0x50U) ||
        !set_spec(&fixture->tensors[5], "whole_q5_k", KQ_GGUF_TYPE_Q5_K,
                  1U, 256U, 1U, 1U, 0x60U) ||
        !set_spec(&fixture->tensors[6], "whole_iq4_nl", KQ_GGUF_TYPE_IQ4_NL,
                  1U, 32U, 1U, 1U, 0x70U) ||
        !set_spec(&fixture->tensors[7], "split_gate", KQ_GGUF_TYPE_BF16,
                  1U, 4U, 1U, 1U, 0x80U) ||
        !set_spec(&fixture->tensors[8], "split_up", KQ_GGUF_TYPE_BF16,
                  1U, 4U, 1U, 1U, 0x90U) ||
        !set_spec(&fixture->tensors[9], "expert_q8", KQ_GGUF_TYPE_Q8_0,
                  3U, 32U, 2U, 3U, 0xa0U) ||
        !set_spec(&fixture->tensors[10], "ple_iq4", KQ_GGUF_TYPE_IQ4_NL,
                  2U, 32U, 6U, 1U, 0xb0U) ||
        !set_spec(&fixture->tensors[11], "transformed_f32",
                  KQ_GGUF_TYPE_F32, 1U, 4U, 1U, 1U, 0xc0U)) {
        return 0;
    }
    for (tensor_index = 0U; tensor_index < VIEW_TENSOR_COUNT;
         ++tensor_index) {
        fixture->tensors[tensor_index].relative_offset = relative_cursor;
        relative_cursor = align_u64(
            relative_cursor + fixture->tensors[tensor_index].packed_bytes,
            VIEW_ALIGNMENT);
    }

    if (!append_bytes(fixture, "GGUF", 4U) ||
        !append_u32(fixture, 3U) ||
        !append_u64(fixture, VIEW_TENSOR_COUNT) ||
        !append_u64(fixture, 1U) ||
        !append_string(fixture, "general.architecture") ||
        !append_u32(fixture, KQ_GGUF_VALUE_STRING) ||
        !append_string(fixture, "viewtest")) {
        return 0;
    }
    for (tensor_index = 0U; tensor_index < VIEW_TENSOR_COUNT;
         ++tensor_index) {
        const view_tensor_spec *spec = &fixture->tensors[tensor_index];
        if (!append_string(fixture, spec->name) ||
            !append_u32(fixture, spec->rank)) {
            return 0;
        }
        for (dimension = 0U; dimension < spec->rank; ++dimension) {
            if (!append_u64(fixture, spec->dimensions[dimension])) {
                return 0;
            }
        }
        if (!append_u32(fixture, spec->type_id) ||
            !append_u64(fixture, spec->relative_offset)) {
            return 0;
        }
    }
    if (!pad_to(fixture, VIEW_ALIGNMENT)) {
        return 0;
    }
    fixture->data_section_offset = fixture->size;
    payload_end = fixture->tensors[VIEW_TENSOR_COUNT - 1U].relative_offset +
                  fixture->tensors[VIEW_TENSOR_COUNT - 1U].packed_bytes;
    if (payload_end > VIEW_FIXTURE_CAPACITY - fixture->size) {
        return 0;
    }
    memset(fixture->bytes + fixture->size, 0xee, (size_t)payload_end);
    fixture->size += (size_t)payload_end;
    for (tensor_index = 0U; tensor_index < VIEW_TENSOR_COUNT;
         ++tensor_index) {
        const view_tensor_spec *spec = &fixture->tensors[tensor_index];
        unsigned char *payload = fixture->bytes +
            fixture->data_section_offset + (size_t)spec->relative_offset;
        for (byte_index = 0U; byte_index < (size_t)spec->packed_bytes;
             ++byte_index) {
            payload[byte_index] =
                (unsigned char)(spec->pattern + (unsigned char)byte_index);
        }
    }
    return 1;
}

static int write_fixture(const view_fixture *fixture,
                         wchar_t path[MAX_PATH]) {
    wchar_t directory[MAX_PATH];
    DWORD directory_length;
    FILE *stream = NULL;
    size_t written;

    directory_length = GetTempPathW(MAX_PATH, directory);
    if (directory_length == 0U || directory_length >= MAX_PATH ||
        GetTempFileNameW(directory, L"kqv", 0U, path) == 0U) {
        return 0;
    }
    if (_wfopen_s(&stream, path, L"wb") != 0 || stream == NULL) {
        DeleteFileW(path);
        return 0;
    }
    written = fwrite(fixture->bytes, 1U, fixture->size, stream);
    if (written != fixture->size || fclose(stream) != 0) {
        if (written != fixture->size) {
            (void)fclose(stream);
        }
        DeleteFileW(path);
        return 0;
    }
    return 1;
}

static void init_single(kq_semantic_tensor *semantic,
                        const kq_gguf_tensor *physical,
                        kq_binding_relation relation) {
    memset(semantic, 0, sizeof(*semantic));
    (void)snprintf(semantic->semantic_id,
                   sizeof(semantic->semantic_id),
                   "test.semantic");
    semantic->component = KQ_COMPONENT_TOKEN_EMBEDDING;
    semantic->relation = relation;
    semantic->canonical_rank = physical->rank;
    semantic->binding_count = 1U;
    semantic->bindings[0].physical = physical;
    semantic->bindings[0].part_role = KQ_BINDING_PART_WHOLE;
    semantic->bindings[0].part_count = 1U;
    if (physical->rank == 1U) {
        semantic->canonical_dimensions[0] = physical->dimensions[0];
    }
}

static int payload_matches(const kq_tensor_view *view,
                           const unsigned char *expected,
                           uint64_t length,
                           const char *message) {
    const kq_tensor_view_info *info = kq_tensor_view_get_info(view);
    const unsigned char *data = kq_tensor_view_physical_data(view);
    int matches = info != NULL && data != NULL &&
                  info->mapped_logical_length == length &&
                  memcmp(data, expected, (size_t)length) == 0;
    synthetic_payload_bytes_touched += length;
    check(matches, message);
    return matches;
}

static void expect_status(kq_status actual,
                          kq_status expected,
                          const char *name,
                          const kq_diagnostic *diagnostic) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s expected %s, got %s (%s)\n",
                name, kq_status_string(expected), kq_status_string(actual),
                diagnostic == NULL ? "" : diagnostic->message);
        failures += 1;
    }
}

static void test_quant_geometry(void) {
    static const uint32_t types[] = {
        KQ_GGUF_TYPE_F32, KQ_GGUF_TYPE_BF16, KQ_GGUF_TYPE_Q5_1,
        KQ_GGUF_TYPE_Q8_0, KQ_GGUF_TYPE_Q4_K, KQ_GGUF_TYPE_Q5_K,
        KQ_GGUF_TYPE_IQ4_NL
    };
    static const uint64_t blocks[] = {1U, 1U, 32U, 32U, 256U, 256U, 32U};
    static const uint64_t bytes[] = {4U, 2U, 24U, 34U, 144U, 176U, 18U};
    kq_quant_geometry geometry;
    kq_quant_block_span span;
    kq_diagnostic diagnostic;
    uint64_t packed;
    uint64_t block_index;
    uint64_t element_in_block;
    kq_status status;
    size_t index;

    for (index = 0U; index < sizeof(types) / sizeof(types[0]); ++index) {
        status = kq_quant_geometry_for_type(types[index], &geometry,
                                            &diagnostic);
        check(status == KQ_STATUS_OK &&
                  geometry.block_elements == blocks[index] &&
                  geometry.bytes_per_block == bytes[index],
              "all seven target types must expose exact block geometry");
        status = kq_quant_packed_size(types[index], blocks[index] * 3U,
                                      &packed, &diagnostic);
        check(status == KQ_STATUS_OK && packed == bytes[index] * 3U,
              "packed size must derive from checked block geometry");
    }
    status = kq_quant_element_to_block(KQ_GGUF_TYPE_Q8_0, 64U, 33U,
                                       &block_index, &element_in_block,
                                       &diagnostic);
    check(status == KQ_STATUS_OK && block_index == 1U &&
              element_in_block == 1U,
          "element lookup must return containing quant block");
    status = kq_quant_range_to_blocks(KQ_GGUF_TYPE_Q8_0, 64U, 31U, 2U,
                                      &span, &diagnostic);
    check(status == KQ_STATUS_OK && span.first_block == 0U &&
              span.block_count == 2U && span.physical_byte_length == 68U &&
              span.leading_elements == 31U && span.trailing_elements == 31U,
          "logical range must expand to its minimal block-aligned span");
    expect_status(kq_quant_packed_size(KQ_GGUF_TYPE_F32, UINT64_MAX,
                                      &packed, &diagnostic),
                  KQ_STATUS_ARITHMETIC_OVERFLOW,
                  "packed byte arithmetic overflow", &diagnostic);
    expect_status(kq_quant_packed_size(KQ_GGUF_TYPE_Q8_0, 33U,
                                      &packed, &diagnostic),
                  KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "partial quant block", &diagnostic);
    expect_status(kq_quant_range_to_blocks(KQ_GGUF_TYPE_Q8_0, 64U,
                                           UINT64_MAX, 2U, &span,
                                           &diagnostic),
                  KQ_STATUS_ARITHMETIC_OVERFLOW,
                  "range arithmetic overflow", &diagnostic);
    expect_status(kq_quant_element_to_block(KQ_GGUF_TYPE_Q8_0, 64U, 64U,
                                            &block_index,
                                            &element_in_block, &diagnostic),
                  KQ_STATUS_SPAN_OUT_OF_RANGE,
                  "element out of range", &diagnostic);
}

static void test_fixture_views(const view_fixture *fixture,
                               const wchar_t *path) {
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_tensor_view *view = NULL;
    kq_file_view *raw_view = NULL;
    kq_semantic_tensor semantic;
    kq_semantic_tensor split;
    kq_semantic_tensor expert;
    kq_semantic_tensor ple;
    kq_semantic_tensor metadata;
    kq_semantic_tensor invalid;
    kq_gguf_tensor copied_tensor;
    const kq_gguf_tensor *physical;
    const kq_tensor_view_info *info;
    kq_diagnostic diagnostic;
    kq_status status;
    SYSTEM_INFO system_info;
    uint32_t index;

    status = kq_file_open_readonly(path, &file, &diagnostic);
    check(status == KQ_STATUS_OK, "payload fixture file must open");
    if (status == KQ_STATUS_OK) {
        status = kq_gguf_open(file, &gguf, &diagnostic);
    }
    check(status == KQ_STATUS_OK, "payload fixture GGUF must parse");
    if (status != KQ_STATUS_OK) {
        kq_file_close(file);
        return;
    }
    check(kq_gguf_payload_bytes_accessed(gguf) == 0U,
          "fixture parsing must not touch payload");
    GetSystemInfo(&system_info);
    physical = kq_gguf_find_tensor(gguf, "whole_f32");
    check(system_info.dwAllocationGranularity != 0U && physical != NULL &&
              (physical->data_offset %
               (uint64_t)system_info.dwAllocationGranularity) != 0U,
          "payload fixture must exercise a non-allocation-granularity file offset");
    check(fixture->bytes[fixture->data_section_offset +
                         (size_t)fixture->tensors[3].relative_offset - 1U] ==
              0xeeU &&
              fixture->bytes[fixture->data_section_offset +
                         (size_t)fixture->tensors[3].relative_offset +
                         (size_t)fixture->tensors[3].packed_bytes] == 0xeeU,
          "known guard bytes must surround the selected quantized payload");

    for (index = 0U; index < 7U; ++index) {
        const view_tensor_spec *spec = &fixture->tensors[index];
        physical = kq_gguf_find_tensor(gguf, spec->name);
        check(physical != NULL, "target-type fixture tensor must exist");
        if (physical == NULL) {
            continue;
        }
        init_single(&semantic, physical, KQ_BINDING_RENAMED_ONE_TO_ONE);
        status = kq_tensor_view_open_binding(
            gguf, &semantic, 0U,
            KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
            &view, &diagnostic);
        check(status == KQ_STATUS_OK,
              "whole physical tensor view must open for every target type");
        if (status == KQ_STATUS_OK) {
            const unsigned char *expected = fixture->bytes +
                fixture->data_section_offset +
                (size_t)spec->relative_offset;
            info = kq_tensor_view_get_info(view);
            check(info->kind == KQ_TENSOR_VIEW_WHOLE_PHYSICAL &&
                      info->tensor_byte_offset == 0U &&
                      info->logical_unpacked_bytes ==
                          physical->element_count * 2U &&
                      info->mapped_logical_offset == physical->data_offset &&
                      info->mapped_logical_length == physical->packed_bytes,
                  "whole view must expose exactly the physical payload span");
            payload_matches(view, expected, spec->packed_bytes,
                            "whole view bytes must match guarded fixture payload");
        }
        kq_tensor_view_close(view);
        view = NULL;
    }

    physical = kq_gguf_find_tensor(gguf, "whole_q8_0");
    init_single(&semantic, physical, KQ_BINDING_RENAMED_ONE_TO_ONE);
    copied_tensor = *physical;
    semantic.bindings[0].physical = &copied_tensor;
    status = kq_tensor_view_open_binding(
        gguf, &semantic, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_TENSOR_OWNERSHIP_MISMATCH,
                  "foreign descriptor", &diagnostic);
    copied_tensor.block_elements = 31U;
    status = kq_tensor_view_open_binding(
        gguf, &semantic, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "corrupt descriptor geometry", &diagnostic);
    status = kq_gguf_open_tensor_span(gguf, physical,
                                      physical->packed_bytes - 1U, 2U,
                                      &raw_view, &diagnostic);
    expect_status(status, KQ_STATUS_SPAN_OUT_OF_RANGE,
                  "span past packed tensor", &diagnostic);
    status = kq_gguf_open_tensor_span(gguf, physical, UINT64_MAX, 2U,
                                      &raw_view, &diagnostic);
    expect_status(status, KQ_STATUS_ARITHMETIC_OVERFLOW,
                  "span byte overflow", &diagnostic);

    memset(&split, 0, sizeof(split));
    split.component = KQ_COMPONENT_ROUTED_EXPERT_STACK;
    split.relation = KQ_BINDING_ONE_CANONICAL_TO_MULTIPLE_PHYSICAL;
    split.canonical_rank = 1U;
    split.canonical_dimensions[0] = 8U;
    split.binding_count = 2U;
    split.bindings[0].physical = kq_gguf_find_tensor(gguf, "split_gate");
    split.bindings[0].part_role = KQ_BINDING_PART_GATE;
    split.bindings[0].part_index = 0U;
    split.bindings[0].part_count = 2U;
    split.bindings[1].physical = kq_gguf_find_tensor(gguf, "split_up");
    split.bindings[1].part_role = KQ_BINDING_PART_UP;
    split.bindings[1].part_index = 1U;
    split.bindings[1].part_count = 2U;
    for (index = 0U; index < 2U; ++index) {
        status = kq_tensor_view_open_binding(
            gguf, &split, index, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
            &view, &diagnostic);
        check(status == KQ_STATUS_OK,
              "ordered split part must open as its own segment");
        if (status == KQ_STATUS_OK) {
            info = kq_tensor_view_get_info(view);
            check(info->kind == KQ_TENSOR_VIEW_SPLIT_PART &&
                      info->layout == KQ_TENSOR_LAYOUT_SPLIT_SEGMENT &&
                      info->binding_index == index,
                  "split view must retain part identity without concatenation");
        }
        kq_tensor_view_close(view);
        view = NULL;
    }
    status = kq_tensor_view_open_binding(
        gguf, &split, 0U,
        KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                  "split canonical-contiguous request", &diagnostic);
    invalid = split;
    invalid.bindings[1].part_index = 0U;
    status = kq_tensor_view_open_binding(
        gguf, &invalid, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                  "duplicate split part", &diagnostic);
    invalid = split;
    invalid.binding_count = 1U;
    status = kq_tensor_view_open_binding(
        gguf, &invalid, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_SEMANTIC_MAPPING_FAILED,
                  "missing split part", &diagnostic);
    invalid = split;
    invalid.canonical_dimensions[0] = 9U;
    status = kq_tensor_view_open_binding(
        gguf, &invalid, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "broken split geometry", &diagnostic);

    physical = kq_gguf_find_tensor(gguf, "expert_q8");
    memset(&expert, 0, sizeof(expert));
    expert.component = KQ_COMPONENT_ROUTED_EXPERT_STACK;
    expert.relation = KQ_BINDING_RENAMED_ONE_TO_ONE;
    expert.canonical_rank = 3U;
    expert.canonical_dimensions[0] = 3U;
    expert.canonical_dimensions[1] = 2U;
    expert.canonical_dimensions[2] = 32U;
    expert.canonical_expert_axis = 0U;
    expert.expert_count = 3U;
    expert.binding_count = 1U;
    expert.bindings[0].physical = physical;
    expert.bindings[0].part_role = KQ_BINDING_PART_WHOLE;
    expert.bindings[0].part_count = 1U;
    expert.bindings[0].physical_expert_axis = 2U;
    status = kq_tensor_view_open_expert_member(
        gguf, &expert, 0U, 1U,
        KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
        &view, &diagnostic);
    check(status == KQ_STATUS_OK,
          "contiguous expert member must open independently");
    if (status == KQ_STATUS_OK) {
        const unsigned char *expected = fixture->bytes +
            fixture->data_section_offset +
            (size_t)fixture->tensors[9].relative_offset + 68U;
        info = kq_tensor_view_get_info(view);
        check(info->kind == KQ_TENSOR_VIEW_EXPERT_MEMBER &&
                  info->expert_id == 1U && info->tensor_byte_offset == 68U &&
                  info->mapped_logical_length == 68U,
              "expert member must map only one checked contiguous stride");
        payload_matches(view, expected, 68U,
                        "expert member bytes must match the selected member");
    }
    kq_tensor_view_close(view);
    view = NULL;
    status = kq_tensor_view_open_expert_member(
        gguf, &expert, 0U, 3U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_SPAN_OUT_OF_RANGE,
                  "expert id out of range", &diagnostic);
    invalid = expert;
    invalid.bindings[0].physical_expert_axis = 0U;
    status = kq_tensor_view_open_expert_member(
        gguf, &invalid, 0U, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "invalid expert axis", &diagnostic);
    invalid = expert;
    invalid.expert_count = 2U;
    invalid.canonical_dimensions[0] = 2U;
    invalid.canonical_dimensions[1] = 3U;
    invalid.bindings[0].physical_expert_axis = 1U;
    status = kq_tensor_view_open_expert_member(
        gguf, &invalid, 0U, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_NONCONTIGUOUS_TENSOR_VIEW,
                  "non-contiguous expert axis", &diagnostic);

    physical = kq_gguf_find_tensor(gguf, "ple_iq4");
    memset(&ple, 0, sizeof(ple));
    ple.component = KQ_COMPONENT_PLE_TABLE;
    ple.relation = KQ_BINDING_MULTIPLE_CANONICAL_TO_ONE_PHYSICAL;
    ple.canonical_rank = 2U;
    ple.canonical_dimensions[0] = 2U;
    ple.canonical_dimensions[1] = 32U;
    ple.binding_count = 1U;
    ple.bindings[0].physical = physical;
    ple.bindings[0].part_role = KQ_BINDING_PART_FUSED_MEMBER;
    ple.bindings[0].fused_member_count = 3U;
    for (index = 0U; index < 3U; ++index) {
        const unsigned char *expected = fixture->bytes +
            fixture->data_section_offset +
            (size_t)fixture->tensors[10].relative_offset +
            (size_t)index * 36U;
        ple.bindings[0].fused_member_index = index;
        status = kq_tensor_view_open_ple_member(
            gguf, &ple, KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
            &view, &diagnostic);
        check(status == KQ_STATUS_OK,
              "first, middle and last PLE members must open independently");
        if (status == KQ_STATUS_OK) {
            info = kq_tensor_view_get_info(view);
            check(info->kind == KQ_TENSOR_VIEW_PLE_FUSED_MEMBER &&
                      info->fused_member_index == index &&
                      info->mapped_logical_length == 36U &&
                      info->mapped_logical_length < physical->packed_bytes,
                  "PLE view must map one member, never the fused tensor");
            payload_matches(view, expected, 36U,
                            "PLE member bytes must match fused member boundaries");
        }
        kq_tensor_view_close(view);
        view = NULL;
    }
    status = kq_tensor_view_open_binding(
        gguf, &ple, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                  "generic fused tensor request", &diagnostic);
    invalid = ple;
    invalid.bindings[0].fused_member_index = 3U;
    status = kq_tensor_view_open_ple_member(
        gguf, &invalid, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "PLE member out of range", &diagnostic);
    invalid = ple;
    invalid.bindings[0].fused_member_count = 4U;
    status = kq_tensor_view_open_ple_member(
        gguf, &invalid, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_INVALID_QUANTIZED_GEOMETRY,
                  "invalid PLE fusion", &diagnostic);

    memset(&metadata, 0, sizeof(metadata));
    metadata.component = KQ_COMPONENT_PLE_ADDRESS_METADATA;
    metadata.relation = KQ_BINDING_METADATA_DERIVED;
    metadata.binding_count = 1U;
    metadata.bindings[0].metadata = kq_gguf_metadata_at(gguf, 0U);
    metadata.bindings[0].part_role = KQ_BINDING_PART_METADATA;
    status = kq_tensor_view_open_binding(
        gguf, &metadata, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_NO_TENSOR_PAYLOAD,
                  "metadata-derived payload request", &diagnostic);

    physical = kq_gguf_find_tensor(gguf, "transformed_f32");
    init_single(&semantic, physical, KQ_BINDING_TRANSFORMED_LAYOUT);
    status = kq_tensor_view_open_binding(
        gguf, &semantic, 0U, KQ_TENSOR_VIEW_PHYSICAL_LAYOUT,
        &view, &diagnostic);
    check(status == KQ_STATUS_OK &&
              kq_tensor_view_get_info(view)->layout ==
                  KQ_TENSOR_LAYOUT_TRANSFORMED_PHYSICAL &&
              kq_tensor_view_get_info(view)->canonical_contiguous == 0U,
          "transformed physical view must carry explicit layout metadata");
    kq_tensor_view_close(view);
    view = NULL;
    status = kq_tensor_view_open_binding(
        gguf, &semantic, 0U,
        KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS,
        &view, &diagnostic);
    expect_status(status, KQ_STATUS_TENSOR_LAYOUT_MISMATCH,
                  "transformed canonical-contiguous request", &diagnostic);

    kq_gguf_close(gguf);
    kq_file_close(file);
}

int main(void) {
    view_fixture fixture;
    wchar_t path[MAX_PATH];

    test_quant_geometry();
    if (!build_fixture(&fixture) || !write_fixture(&fixture, path)) {
        fprintf(stderr, "FAIL: could not create payload-bearing fixture\n");
        return 1;
    }
    test_fixture_views(&fixture, path);
    test_fixture_views(&fixture, path);
    check(DeleteFileW(path) != 0,
          "all payload views must release mappings for fixture deletion");
    if (failures != 0) {
        fprintf(stderr, "%d tensor-view tests failed\n", failures);
        return 1;
    }
    printf("tensor view synthetic tests passed; payload_bytes_touched=%llu\n",
           (unsigned long long)synthetic_payload_bytes_touched);
    return 0;
}
