#ifndef KQ_MODEL_INTERNAL_H
#define KQ_MODEL_INTERNAL_H

#include "kq_model.h"

typedef struct kq_model_source {
    void *context;
    kq_string_view architecture;
    uint64_t metadata_count;
    const kq_gguf_metadata *(*find_metadata)(void *context, const char *key);
    uint64_t tensor_count;
    const kq_gguf_tensor *(*tensor_at)(void *context, uint64_t index);
    uint64_t payload_bytes_accessed;
} kq_model_source;

kq_status kq_model_open_from_source(const kq_model_source *source,
                                    kq_model **out_model,
                                    kq_diagnostic *diagnostic);

#endif
