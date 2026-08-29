#ifndef KQ_TENSOR_VIEW_H
#define KQ_TENSOR_VIEW_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kq_tensor_view_access {
    KQ_TENSOR_VIEW_PHYSICAL_LAYOUT = 0,
    KQ_TENSOR_VIEW_REQUIRE_CANONICAL_CONTIGUOUS
} kq_tensor_view_access;

typedef enum kq_tensor_view_kind {
    KQ_TENSOR_VIEW_WHOLE_PHYSICAL = 0,
    KQ_TENSOR_VIEW_SPLIT_PART,
    KQ_TENSOR_VIEW_EXPERT_MEMBER,
    KQ_TENSOR_VIEW_PLE_FUSED_MEMBER
} kq_tensor_view_kind;

typedef enum kq_tensor_view_layout {
    KQ_TENSOR_LAYOUT_CANONICAL_CONTIGUOUS = 0,
    KQ_TENSOR_LAYOUT_TRANSFORMED_PHYSICAL,
    KQ_TENSOR_LAYOUT_SPLIT_SEGMENT,
    KQ_TENSOR_LAYOUT_FUSED_MEMBER
} kq_tensor_view_layout;

typedef struct kq_quant_geometry {
    uint32_t type_id;
    uint64_t block_elements;
    uint64_t bytes_per_block;
} kq_quant_geometry;

typedef struct kq_quant_block_span {
    uint64_t requested_element_offset;
    uint64_t requested_element_count;
    uint64_t first_block;
    uint64_t block_count;
    uint64_t physical_byte_offset;
    uint64_t physical_byte_length;
    uint64_t leading_elements;
    uint64_t trailing_elements;
} kq_quant_block_span;

typedef struct kq_tensor_view_info {
    char semantic_id[KQ_SEMANTIC_ID_CAPACITY];
    kq_tensor_view_kind kind;
    kq_tensor_view_layout layout;
    kq_binding_relation relation;
    kq_binding_part_role part_role;
    uint32_t binding_index;
    uint32_t type_id;
    uint64_t block_elements;
    uint64_t bytes_per_block;
    uint32_t canonical_rank;
    uint64_t canonical_dimensions[KQ_SEMANTIC_MAX_DIMS];
    uint32_t logical_rank;
    uint64_t logical_dimensions[KQ_SEMANTIC_MAX_DIMS];
    uint32_t physical_rank;
    uint64_t physical_dimensions[KQ_GGUF_MAX_DIMS];
    uint64_t physical_element_count;
    uint64_t requested_element_offset;
    uint64_t requested_element_count;
    uint64_t logical_unpacked_bytes;
    uint64_t first_block;
    uint64_t block_count;
    uint64_t tensor_byte_offset;
    uint64_t file_byte_offset;
    uint64_t mapped_logical_offset;
    uint64_t mapped_logical_length;
    uint64_t leading_elements;
    uint64_t trailing_elements;
    uint32_t canonical_contiguous;
    uint32_t expert_id;
    uint32_t fused_member_index;
} kq_tensor_view_info;

typedef struct kq_tensor_view kq_tensor_view;

kq_status kq_quant_geometry_for_type(uint32_t type_id,
                                     kq_quant_geometry *out_geometry,
                                     kq_diagnostic *diagnostic);
kq_status kq_quant_packed_size(uint32_t type_id,
                               uint64_t element_count,
                               uint64_t *out_packed_bytes,
                               kq_diagnostic *diagnostic);
kq_status kq_quant_element_to_block(uint32_t type_id,
                                    uint64_t total_elements,
                                    uint64_t element_index,
                                    uint64_t *out_block_index,
                                    uint64_t *out_element_in_block,
                                    kq_diagnostic *diagnostic);
kq_status kq_quant_range_to_blocks(uint32_t type_id,
                                   uint64_t total_elements,
                                   uint64_t element_offset,
                                   uint64_t element_count,
                                   kq_quant_block_span *out_span,
                                   kq_diagnostic *diagnostic);

kq_status kq_tensor_view_open_binding(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    uint32_t binding_index,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic);
kq_status kq_tensor_view_open_expert_member(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    uint32_t binding_index,
    uint32_t expert_id,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic);
kq_status kq_tensor_view_open_ple_member(
    const kq_gguf *gguf,
    const kq_semantic_tensor *semantic,
    kq_tensor_view_access access,
    kq_tensor_view **out_view,
    kq_diagnostic *diagnostic);

const kq_tensor_view_info *kq_tensor_view_get_info(
    const kq_tensor_view *view);
const unsigned char *kq_tensor_view_physical_data(
    const kq_tensor_view *view);
void kq_tensor_view_close(kq_tensor_view *view);

const char *kq_tensor_view_kind_name(kq_tensor_view_kind kind);
const char *kq_tensor_view_layout_name(kq_tensor_view_layout layout);

#ifdef __cplusplus
}
#endif

#endif
