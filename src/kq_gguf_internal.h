#ifndef KQ_GGUF_INTERNAL_H
#define KQ_GGUF_INTERNAL_H

#include "kq_gguf.h"

kq_status kq_gguf_open_tensor_span(const kq_gguf *gguf,
                                    const kq_gguf_tensor *tensor,
                                    uint64_t tensor_byte_offset,
                                    uint64_t byte_length,
                                    kq_file_view **out_view,
                                    kq_diagnostic *diagnostic);

#endif
