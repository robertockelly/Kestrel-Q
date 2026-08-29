#ifndef KQ_GGUF_H
#define KQ_GGUF_H

#include <stdint.h>

#include "kq_file.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_GGUF_MAX_DIMS 8U

enum {
    KQ_GGUF_TYPE_F32 = 0,
    KQ_GGUF_TYPE_Q5_1 = 7,
    KQ_GGUF_TYPE_Q8_0 = 8,
    KQ_GGUF_TYPE_Q4_K = 12,
    KQ_GGUF_TYPE_Q5_K = 13,
    KQ_GGUF_TYPE_IQ4_NL = 20,
    KQ_GGUF_TYPE_BF16 = 30
};

enum {
    KQ_GGUF_VALUE_UINT16 = 2,
    KQ_GGUF_VALUE_UINT32 = 4,
    KQ_GGUF_VALUE_INT32 = 5,
    KQ_GGUF_VALUE_FLOAT32 = 6,
    KQ_GGUF_VALUE_BOOL = 7,
    KQ_GGUF_VALUE_STRING = 8,
    KQ_GGUF_VALUE_ARRAY = 9,
    KQ_GGUF_VALUE_UINT64 = 10
};

typedef struct kq_string_view {
    const unsigned char *data;
    uint64_t length;
} kq_string_view;

typedef struct kq_gguf_type_info {
    uint32_t type_id;
    const char *name;
    uint64_t block_elements;
    uint64_t bytes_per_block;
} kq_gguf_type_info;

typedef struct kq_gguf_metadata {
    kq_string_view key;
    uint32_t value_type;
    uint32_t array_element_type;
    uint64_t array_length;
    uint64_t scalar_value;
    kq_string_view string_value;
    kq_string_view array_data;
} kq_gguf_metadata;

typedef struct kq_gguf_tensor {
    kq_string_view name;
    uint32_t rank;
    uint64_t dimensions[KQ_GGUF_MAX_DIMS];
    uint32_t type_id;
    uint64_t block_elements;
    uint64_t bytes_per_block;
    uint64_t element_count;
    uint64_t relative_offset;
    uint64_t data_offset;
    uint64_t packed_bytes;
} kq_gguf_tensor;

typedef struct kq_gguf kq_gguf;

kq_status kq_gguf_open(kq_file *file,
                       kq_gguf **out_gguf,
                       kq_diagnostic *diagnostic);
void kq_gguf_close(kq_gguf *gguf);

uint32_t kq_gguf_version(const kq_gguf *gguf);
uint64_t kq_gguf_metadata_count(const kq_gguf *gguf);
uint64_t kq_gguf_tensor_count(const kq_gguf *gguf);
uint64_t kq_gguf_alignment(const kq_gguf *gguf);
uint64_t kq_gguf_directory_end_offset(const kq_gguf *gguf);
uint64_t kq_gguf_data_section_offset(const kq_gguf *gguf);
uint64_t kq_gguf_packed_tensor_bytes(const kq_gguf *gguf);
uint64_t kq_gguf_format_overhead_bytes(const kq_gguf *gguf);
uint64_t kq_gguf_directory_bytes_parsed(const kq_gguf *gguf);
uint64_t kq_gguf_payload_bytes_accessed(const kq_gguf *gguf);
kq_string_view kq_gguf_architecture(const kq_gguf *gguf);

const kq_gguf_metadata *kq_gguf_metadata_at(const kq_gguf *gguf,
                                            uint64_t index);
const kq_gguf_metadata *kq_gguf_find_metadata(const kq_gguf *gguf,
                                              const char *key);
const kq_gguf_tensor *kq_gguf_tensor_at(const kq_gguf *gguf,
                                        uint64_t index);
const kq_gguf_tensor *kq_gguf_find_tensor(const kq_gguf *gguf,
                                          const char *name);

const kq_gguf_type_info *kq_gguf_type_info_for(uint32_t type_id);
int kq_string_view_equal(const kq_string_view *left,
                         const kq_string_view *right);
int kq_string_view_equal_cstr(const kq_string_view *view,
                              const char *text);
int kq_gguf_metadata_u16(const kq_gguf_metadata *metadata, uint16_t *value);
int kq_gguf_metadata_u32(const kq_gguf_metadata *metadata, uint32_t *value);
int kq_gguf_metadata_i32(const kq_gguf_metadata *metadata, int32_t *value);
int kq_gguf_metadata_array_i32_at(const kq_gguf_metadata *metadata,
                                  uint64_t index,
                                  int32_t *value);
int kq_gguf_metadata_array_u64_at(const kq_gguf_metadata *metadata,
                                  uint64_t index,
                                  uint64_t *value);

#ifdef __cplusplus
}
#endif

#endif
