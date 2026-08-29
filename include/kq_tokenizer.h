#ifndef KQ_TOKENIZER_H
#define KQ_TOKENIZER_H

#include <stdint.h>

#include "kq_gguf.h"
#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KQ_TOKENIZER_BASE_VOCABULARY 248044U
#define KQ_TOKENIZER_ADDED_TOKENS 33U
#define KQ_TOKENIZER_LENGTH 248077U
#define KQ_TOKENIZER_MODEL_VOCABULARY 248320U
#define KQ_TOKENIZER_PADDED_IDS 243U

typedef struct kq_tokenizer kq_tokenizer;

typedef enum kq_tokenizer_special_policy {
    KQ_TOKENIZER_SPECIAL_REJECT = 0,
    KQ_TOKENIZER_SPECIAL_RECOGNIZE = 1
} kq_tokenizer_special_policy;

typedef enum kq_tokenizer_decode_policy {
    KQ_TOKENIZER_DECODE_KEEP_SPECIAL = 0,
    KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL = 1
} kq_tokenizer_decode_policy;

typedef struct kq_tokenizer_encode_options {
    kq_tokenizer_special_policy special_policy;
} kq_tokenizer_encode_options;

typedef struct kq_tokenizer_decode_options {
    kq_tokenizer_decode_policy special_policy;
} kq_tokenizer_decode_options;

typedef struct kq_tokenizer_metrics {
    uint64_t construction_nanoseconds;
    uint64_t owned_heap_bytes;
    uint64_t vocabulary_lookup_bytes;
    uint64_t merge_lookup_bytes;
} kq_tokenizer_metrics;

kq_status kq_tokenizer_open_from_gguf(const kq_gguf *gguf,
                                      kq_tokenizer **out_tokenizer,
                                      kq_diagnostic *diagnostic);
void kq_tokenizer_close(kq_tokenizer *tokenizer);

kq_status kq_tokenizer_encode(const kq_tokenizer *tokenizer,
                              const unsigned char *utf8,
                              uint64_t utf8_length,
                              const kq_tokenizer_encode_options *options,
                              uint32_t *token_ids,
                              uint64_t token_capacity,
                              uint64_t *required_tokens,
                              kq_diagnostic *diagnostic);

kq_status kq_tokenizer_decode(const kq_tokenizer *tokenizer,
                              const uint32_t *token_ids,
                              uint64_t token_count,
                              const kq_tokenizer_decode_options *options,
                              unsigned char *utf8,
                              uint64_t utf8_capacity,
                              uint64_t *required_utf8_bytes,
                              kq_diagnostic *diagnostic);

const kq_tokenizer_metrics *kq_tokenizer_get_metrics(
    const kq_tokenizer *tokenizer);
const char *kq_tokenizer_nfc_unicode_version(void);
const char *kq_tokenizer_property_unicode_version(void);

#ifdef __cplusplus
}
#endif

#endif
