#include "kq_tokenizer.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "utf8proc.h"

#include "kq_internal.h"
#include "kq_sha256.h"
#include "kq_tokenizer_internal.h"

#define KQ_TOKENIZER_MERGE_COUNT 247587U
#define KQ_TOKENIZER_HASH_CAPACITY 524288U
#define KQ_TOKENIZER_MAX_INPUT_BYTES (16ULL * 1024ULL * 1024ULL)

typedef struct kq_token_slot {
    uint64_t hash;
    uint32_t token_id;
    int occupied;
} kq_token_slot;

typedef struct kq_merge_slot {
    uint64_t key;
    uint32_t rank;
    uint32_t result_id;
    int occupied;
} kq_merge_slot;

typedef struct kq_string_cursor {
    const unsigned char *data;
    uint64_t length;
    uint64_t offset;
    uint64_t index;
    uint64_t count;
} kq_string_cursor;

typedef struct kq_codepoint {
    int32_t value;
    uint64_t byte_offset;
    uint64_t byte_length;
} kq_codepoint;

typedef struct kq_u32_buffer {
    uint32_t *data;
    uint64_t count;
    uint64_t capacity;
} kq_u32_buffer;

typedef struct kq_byte_buffer {
    unsigned char *data;
    uint64_t count;
    uint64_t capacity;
} kq_byte_buffer;

#include "kq_unicode9_assigned.inc"

struct kq_tokenizer {
    const kq_gguf *gguf;
    kq_string_view *tokens;
    kq_token_slot *token_slots;
    kq_merge_slot *merge_slots;
    uint32_t byte_token_ids[256];
    kq_tokenizer_metrics metrics;
};

typedef struct kq_special_definition {
    uint32_t token_id;
    const char *text;
    int skipped_by_canonical_decode;
} kq_special_definition;

static const kq_special_definition kq_special_tokens[KQ_TOKENIZER_ADDED_TOKENS] = {
    {248044U, "<|endoftext|>", 1},
    {248045U, "<|im_start|>", 1},
    {248046U, "<|im_end|>", 1},
    {248047U, "<|object_ref_start|>", 1},
    {248048U, "<|object_ref_end|>", 1},
    {248049U, "<|box_start|>", 1},
    {248050U, "<|box_end|>", 1},
    {248051U, "<|quad_start|>", 1},
    {248052U, "<|quad_end|>", 1},
    {248053U, "<|vision_start|>", 1},
    {248054U, "<|vision_end|>", 1},
    {248055U, "<|vision_pad|>", 1},
    {248056U, "<|image_pad|>", 1},
    {248057U, "<|video_pad|>", 1},
    {248058U, "<tool_call>", 0},
    {248059U, "</tool_call>", 0},
    {248060U, "<|fim_prefix|>", 0},
    {248061U, "<|fim_middle|>", 0},
    {248062U, "<|fim_suffix|>", 0},
    {248063U, "<|fim_pad|>", 0},
    {248064U, "<|repo_name|>", 0},
    {248065U, "<|file_sep|>", 0},
    {248066U, "<tool_response>", 0},
    {248067U, "</tool_response>", 0},
    {248068U, "<think>", 0},
    {248069U, "</think>", 0},
    {248070U, "<|audio_start|>", 1},
    {248071U, "<|audio_end|>", 1},
    {248072U, "<tts_pad>", 1},
    {248073U, "<tts_text_bos>", 1},
    {248074U, "<tts_text_eod>", 1},
    {248075U, "<tts_text_bos_single>", 1},
    {248076U, "<|audio_pad|>", 1}
};

static const unsigned char kq_expected_token_hash[32] = {
    0x90U, 0x53U, 0x6bU, 0xd9U, 0x26U, 0xc0U, 0xd6U, 0xb3U,
    0x23U, 0x0bU, 0x68U, 0xcdU, 0x9fU, 0xd9U, 0x00U, 0x6aU,
    0x59U, 0xeaU, 0x90U, 0x95U, 0x0cU, 0x42U, 0x82U, 0x7aU,
    0xa5U, 0xf1U, 0x79U, 0x5bU, 0x6eU, 0x84U, 0xb3U, 0x40U
};
static const unsigned char kq_expected_merge_hash[32] = {
    0xdcU, 0x39U, 0xbdU, 0xbcU, 0xf9U, 0xd1U, 0x82U, 0xb1U,
    0x4eU, 0x39U, 0xb1U, 0xaaU, 0x9aU, 0x54U, 0xfbU, 0xc8U,
    0x75U, 0x73U, 0x42U, 0xcfU, 0xddU, 0x2cU, 0x5fU, 0x55U,
    0x36U, 0x7fU, 0x46U, 0x15U, 0xccU, 0x9aU, 0x64U, 0xb4U
};
static const unsigned char kq_expected_type_hash[32] = {
    0x2cU, 0xc0U, 0x7dU, 0x08U, 0x07U, 0x48U, 0x9cU, 0xe9U,
    0xc9U, 0xe3U, 0xa0U, 0x4dU, 0x53U, 0xfcU, 0x05U, 0x5aU,
    0x8dU, 0xe4U, 0x81U, 0x76U, 0xe6U, 0x1dU, 0xbfU, 0x11U,
    0x30U, 0x05U, 0xb5U, 0x59U, 0x7fU, 0x80U, 0x8cU, 0x84U
};

static int kq_u64_add_checked(uint64_t left, uint64_t right, uint64_t *result) {
    if (result == NULL || UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int kq_allocation_size(uint64_t count,
                              size_t item_size,
                              size_t *bytes) {
    if (bytes == NULL || item_size == 0U ||
        count > (uint64_t)(SIZE_MAX / item_size)) {
        return 0;
    }
    *bytes = (size_t)count * item_size;
    return 1;
}

static uint64_t kq_hash_bytes(const unsigned char *data, uint64_t length) {
    uint64_t hash = 1469598103934665603ULL;
    uint64_t index;
    for (index = 0U; index < length; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t kq_hash_concat(const kq_string_view *left,
                               const kq_string_view *right) {
    uint64_t hash = 1469598103934665603ULL;
    uint64_t index;
    for (index = 0U; index < left->length; ++index) {
        hash ^= left->data[index];
        hash *= 1099511628211ULL;
    }
    for (index = 0U; index < right->length; ++index) {
        hash ^= right->data[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void kq_sha256_u64(kq_sha256 *hash, uint64_t value) {
    unsigned char encoded[8];
    unsigned int index;
    for (index = 0U; index < 8U; ++index) {
        encoded[index] = (unsigned char)(value >> (index * 8U));
    }
    kq_sha256_update(hash, encoded, sizeof(encoded));
}

static void kq_sha256_i32(kq_sha256 *hash, int32_t value) {
    unsigned char encoded[4];
    uint32_t bits;
    unsigned int index;
    memcpy(&bits, &value, sizeof(bits));
    for (index = 0U; index < 4U; ++index) {
        encoded[index] = (unsigned char)(bits >> (index * 8U));
    }
    kq_sha256_update(hash, encoded, sizeof(encoded));
}

static int kq_string_cursor_begin(const kq_gguf_metadata *metadata,
                                  kq_string_cursor *cursor) {
    if (metadata == NULL || cursor == NULL ||
        metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        metadata->array_element_type != KQ_GGUF_VALUE_STRING ||
        metadata->array_data.data == NULL) {
        return 0;
    }
    cursor->data = metadata->array_data.data;
    cursor->length = metadata->array_data.length;
    cursor->offset = 0U;
    cursor->index = 0U;
    cursor->count = metadata->array_length;
    return 1;
}

static int kq_string_cursor_next(kq_string_cursor *cursor,
                                 kq_string_view *value) {
    uint64_t length = 0U;
    unsigned int byte_index;
    if (cursor == NULL || value == NULL || cursor->index >= cursor->count ||
        cursor->offset > cursor->length ||
        cursor->length - cursor->offset < 8U) {
        return 0;
    }
    for (byte_index = 0U; byte_index < 8U; ++byte_index) {
        length |= (uint64_t)cursor->data[(size_t)(cursor->offset + byte_index)]
                  << (byte_index * 8U);
    }
    cursor->offset += 8U;
    if (length > cursor->length - cursor->offset) {
        return 0;
    }
    value->data = cursor->data + (size_t)cursor->offset;
    value->length = length;
    cursor->offset += length;
    cursor->index += 1U;
    return 1;
}

static int kq_metadata_string_equals(const kq_gguf_metadata *metadata,
                                     const char *expected) {
    return metadata != NULL && metadata->value_type == KQ_GGUF_VALUE_STRING &&
           kq_string_view_equal_cstr(&metadata->string_value, expected);
}

static int kq_metadata_u32_equals(const kq_gguf_metadata *metadata,
                                  uint32_t expected) {
    uint32_t value = 0U;
    return kq_gguf_metadata_u32(metadata, &value) && value == expected;
}

static int kq_metadata_bool_equals(const kq_gguf_metadata *metadata,
                                   int expected) {
    return metadata != NULL && metadata->value_type == KQ_GGUF_VALUE_BOOL &&
           metadata->scalar_value == (uint64_t)(expected != 0);
}

static int kq_token_view_equals(const kq_string_view *view, const char *text) {
    size_t length = strlen(text);
    return view != NULL && view->length == (uint64_t)length &&
           (length == 0U || memcmp(view->data, text, length) == 0);
}

static int kq_token_slot_insert(kq_tokenizer *tokenizer,
                                uint32_t token_id,
                                kq_diagnostic *diagnostic) {
    const kq_string_view *token = &tokenizer->tokens[token_id];
    uint64_t hash = kq_hash_bytes(token->data, token->length);
    uint64_t slot = hash & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    uint64_t probe;
    for (probe = 0U; probe < KQ_TOKENIZER_HASH_CAPACITY; ++probe) {
        kq_token_slot *entry = &tokenizer->token_slots[slot];
        if (!entry->occupied) {
            entry->occupied = 1;
            entry->hash = hash;
            entry->token_id = token_id;
            return 1;
        }
        if (entry->hash == hash) {
            const kq_string_view *existing =
                &tokenizer->tokens[entry->token_id];
            if (existing->length == token->length &&
                (token->length == 0U ||
                 memcmp(existing->data, token->data, (size_t)token->length) == 0)) {
                kq_diagnostic_set(diagnostic,
                                  KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                                  "duplicate tokenizer vocabulary string at IDs %u and %u",
                                  entry->token_id,
                                  token_id);
                return 0;
            }
        }
        slot = (slot + 1U) & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    }
    kq_diagnostic_set(diagnostic,
                      KQ_STATUS_LIMIT_EXCEEDED,
                      "tokenizer vocabulary hash table is full");
    return 0;
}

static int kq_token_lookup(const kq_tokenizer *tokenizer,
                           const unsigned char *data,
                           uint64_t length,
                           uint32_t *token_id) {
    uint64_t hash = kq_hash_bytes(data, length);
    uint64_t slot = hash & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    uint64_t probe;
    for (probe = 0U; probe < KQ_TOKENIZER_HASH_CAPACITY; ++probe) {
        const kq_token_slot *entry = &tokenizer->token_slots[slot];
        const kq_string_view *candidate;
        if (!entry->occupied) {
            return 0;
        }
        candidate = &tokenizer->tokens[entry->token_id];
        if (entry->hash == hash && candidate->length == length &&
            (length == 0U || memcmp(candidate->data, data, (size_t)length) == 0)) {
            *token_id = entry->token_id;
            return 1;
        }
        slot = (slot + 1U) & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    }
    return 0;
}

static int kq_token_lookup_concat(const kq_tokenizer *tokenizer,
                                  uint32_t left_id,
                                  uint32_t right_id,
                                  uint32_t *token_id) {
    const kq_string_view *left = &tokenizer->tokens[left_id];
    const kq_string_view *right = &tokenizer->tokens[right_id];
    uint64_t length;
    uint64_t hash;
    uint64_t slot;
    uint64_t probe;
    if (!kq_u64_add_checked(left->length, right->length, &length)) {
        return 0;
    }
    hash = kq_hash_concat(left, right);
    slot = hash & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    for (probe = 0U; probe < KQ_TOKENIZER_HASH_CAPACITY; ++probe) {
        const kq_token_slot *entry = &tokenizer->token_slots[slot];
        const kq_string_view *candidate;
        if (!entry->occupied) {
            return 0;
        }
        candidate = &tokenizer->tokens[entry->token_id];
        if (entry->hash == hash && candidate->length == length &&
            (left->length == 0U ||
             memcmp(candidate->data, left->data, (size_t)left->length) == 0) &&
            (right->length == 0U ||
             memcmp(candidate->data + (size_t)left->length,
                    right->data,
                    (size_t)right->length) == 0)) {
            *token_id = entry->token_id;
            return 1;
        }
        slot = (slot + 1U) & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    }
    return 0;
}

static uint64_t kq_merge_key(uint32_t left, uint32_t right) {
    return ((uint64_t)left << 32U) | (uint64_t)right;
}

static int kq_merge_insert(kq_tokenizer *tokenizer,
                           uint32_t left,
                           uint32_t right,
                           uint32_t rank,
                           uint32_t result,
                           kq_diagnostic *diagnostic) {
    uint64_t key = kq_merge_key(left, right);
    uint64_t slot = (key * 11400714819323198485ULL) &
                    (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    uint64_t probe;
    for (probe = 0U; probe < KQ_TOKENIZER_HASH_CAPACITY; ++probe) {
        kq_merge_slot *entry = &tokenizer->merge_slots[slot];
        if (!entry->occupied) {
            entry->occupied = 1;
            entry->key = key;
            entry->rank = rank;
            entry->result_id = result;
            return 1;
        }
        if (entry->key == key) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "duplicate tokenizer merge pair at rank %u",
                              rank);
            return 0;
        }
        slot = (slot + 1U) & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    }
    kq_diagnostic_set(diagnostic,
                      KQ_STATUS_LIMIT_EXCEEDED,
                      "tokenizer merge hash table is full");
    return 0;
}

static const kq_merge_slot *kq_merge_find(const kq_tokenizer *tokenizer,
                                          uint32_t left,
                                          uint32_t right) {
    uint64_t key = kq_merge_key(left, right);
    uint64_t slot = (key * 11400714819323198485ULL) &
                    (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    uint64_t probe;
    for (probe = 0U; probe < KQ_TOKENIZER_HASH_CAPACITY; ++probe) {
        const kq_merge_slot *entry = &tokenizer->merge_slots[slot];
        if (!entry->occupied) {
            return NULL;
        }
        if (entry->key == key) {
            return entry;
        }
        slot = (slot + 1U) & (KQ_TOKENIZER_HASH_CAPACITY - 1U);
    }
    return NULL;
}

static int kq_digest_equal(const unsigned char left[32],
                           const unsigned char right[32]) {
    unsigned char difference = 0U;
    unsigned int index;
    for (index = 0U; index < 32U; ++index) {
        difference |= (unsigned char)(left[index] ^ right[index]);
    }
    return difference == 0U;
}

static int kq_apply_string_mutation(
    const kq_tokenizer_test_mutation *mutation,
    kq_tokenizer_test_mutation_kind kind,
    uint64_t index,
    kq_string_view *value) {
    if (mutation != NULL && mutation->kind == kind && mutation->index == index) {
        *value = mutation->replacement_string;
        return 1;
    }
    return 0;
}

static int kq_byte_is_direct(unsigned int byte_value) {
    return (byte_value >= 33U && byte_value <= 126U) ||
           (byte_value >= 161U && byte_value <= 172U) ||
           (byte_value >= 174U && byte_value <= 255U);
}

static int32_t kq_byte_to_unicode(unsigned int byte_value) {
    unsigned int candidate;
    unsigned int extra = 0U;
    if (kq_byte_is_direct(byte_value)) {
        return (int32_t)byte_value;
    }
    for (candidate = 0U; candidate < byte_value; ++candidate) {
        if (!kq_byte_is_direct(candidate)) {
            extra += 1U;
        }
    }
    return (int32_t)(256U + extra);
}

static int kq_unicode_to_byte(int32_t codepoint, unsigned char *byte_value) {
    unsigned int candidate;
    if (codepoint >= 0 && codepoint <= 255 &&
        kq_byte_is_direct((unsigned int)codepoint)) {
        *byte_value = (unsigned char)codepoint;
        return 1;
    }
    for (candidate = 0U; candidate < 256U; ++candidate) {
        if (kq_byte_to_unicode(candidate) == codepoint) {
            *byte_value = (unsigned char)candidate;
            return 1;
        }
    }
    return 0;
}

static kq_status kq_validate_and_build(kq_tokenizer *tokenizer,
                                       const kq_gguf *gguf,
                                       const kq_tokenizer_test_mutation *mutation,
                                       kq_diagnostic *diagnostic) {
    const kq_gguf_metadata *tokens_metadata;
    const kq_gguf_metadata *types_metadata;
    const kq_gguf_metadata *merges_metadata;
    kq_string_cursor cursor;
    kq_string_view value;
    kq_sha256 hash;
    unsigned char digest[32];
    uint64_t token_count;
    uint64_t merge_count;
    uint64_t index;
    uint64_t space_index;
    int32_t token_type;
    uint32_t left_id;
    uint32_t right_id;
    uint32_t result_id;
    unsigned char encoded[4];
    utf8proc_ssize_t encoded_length;

    if (!kq_string_view_equal_cstr(&((kq_string_view){
            kq_gguf_architecture(gguf).data,
            kq_gguf_architecture(gguf).length}), "qwen4exp") ||
        !kq_metadata_string_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.model"), "gpt2") ||
        !kq_metadata_string_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.pre"), "qwen35") ||
        !kq_metadata_u32_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.eos_token_id"), 248046U) ||
        !kq_metadata_u32_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.padding_token_id"), 248044U) ||
        !kq_metadata_u32_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.bos_token_id"), 248044U) ||
        !kq_metadata_bool_equals(
            kq_gguf_find_metadata(gguf, "tokenizer.ggml.add_bos_token"), 0)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer identity/default metadata does not match the registered substrate");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }

    tokens_metadata = kq_gguf_find_metadata(gguf, "tokenizer.ggml.tokens");
    types_metadata = kq_gguf_find_metadata(gguf, "tokenizer.ggml.token_type");
    merges_metadata = kq_gguf_find_metadata(gguf, "tokenizer.ggml.merges");
    token_count = tokens_metadata != NULL ? tokens_metadata->array_length : 0U;
    merge_count = merges_metadata != NULL ? merges_metadata->array_length : 0U;
    if (mutation != NULL && mutation->kind == KQ_TOKENIZER_TEST_MUTATION_TOKEN_COUNT) {
        token_count -= token_count != 0U ? 1U : 0U;
    }
    if (mutation != NULL && mutation->kind == KQ_TOKENIZER_TEST_MUTATION_MERGE_COUNT) {
        merge_count -= merge_count != 0U ? 1U : 0U;
    }
    if (tokens_metadata == NULL || types_metadata == NULL ||
        merges_metadata == NULL ||
        tokens_metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        tokens_metadata->array_element_type != KQ_GGUF_VALUE_STRING ||
        token_count != KQ_TOKENIZER_MODEL_VOCABULARY ||
        types_metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        types_metadata->array_element_type != KQ_GGUF_VALUE_INT32 ||
        types_metadata->array_length != KQ_TOKENIZER_MODEL_VOCABULARY ||
        merges_metadata->value_type != KQ_GGUF_VALUE_ARRAY ||
        merges_metadata->array_element_type != KQ_GGUF_VALUE_STRING ||
        merge_count != KQ_TOKENIZER_MERGE_COUNT) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer array type/count mismatch");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }

    if (!kq_string_cursor_begin(tokens_metadata, &cursor)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer token array is not addressable");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }
    kq_sha256_init(&hash);
    kq_sha256_update(&hash,
                     (const unsigned char *)"KQ-TOKEN-STRINGS-v1\0",
                     sizeof("KQ-TOKEN-STRINGS-v1\0") - 1U);
    kq_sha256_u64(&hash, KQ_TOKENIZER_MODEL_VOCABULARY);
    for (index = 0U; index < KQ_TOKENIZER_MODEL_VOCABULARY; ++index) {
        if (!kq_string_cursor_next(&cursor, &value)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "GGUF tokenizer token array is malformed at ID %llu",
                              (unsigned long long)index);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
        (void)kq_apply_string_mutation(mutation,
                                       KQ_TOKENIZER_TEST_MUTATION_TOKEN_STRING,
                                       index,
                                       &value);
        tokenizer->tokens[index] = value;
        kq_sha256_u64(&hash, value.length);
        kq_sha256_update(&hash, value.data, (size_t)value.length);
    }
    kq_sha256_final(&hash, digest);
    if (!kq_digest_equal(digest, kq_expected_token_hash)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer vocabulary strings/IDs differ from the canonical substrate");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }

    kq_sha256_init(&hash);
    kq_sha256_update(&hash,
                     (const unsigned char *)"KQ-TOKEN-TYPES-v1\0",
                     sizeof("KQ-TOKEN-TYPES-v1\0") - 1U);
    kq_sha256_u64(&hash, KQ_TOKENIZER_MODEL_VOCABULARY);
    for (index = 0U; index < KQ_TOKENIZER_MODEL_VOCABULARY; ++index) {
        if (!kq_gguf_metadata_array_i32_at(types_metadata, index, &token_type)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "GGUF tokenizer type array is malformed at ID %llu",
                              (unsigned long long)index);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
        if (mutation != NULL &&
            mutation->kind == KQ_TOKENIZER_TEST_MUTATION_TOKEN_TYPE &&
            mutation->index == index) {
            token_type = mutation->replacement_i32;
        }
        kq_sha256_i32(&hash, token_type);
    }
    kq_sha256_final(&hash, digest);
    if (!kq_digest_equal(digest, kq_expected_type_hash)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer type/padded-ID substrate differs from the registered artifact");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }

    for (index = 0U; index < KQ_TOKENIZER_LENGTH; ++index) {
        if (!kq_token_slot_insert(tokenizer, (uint32_t)index, diagnostic)) {
            return diagnostic != NULL ? diagnostic->status :
                   KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
    }
    for (index = 0U; index < KQ_TOKENIZER_ADDED_TOKENS; ++index) {
        const kq_special_definition *definition = &kq_special_tokens[index];
        if (!kq_token_view_equals(&tokenizer->tokens[definition->token_id],
                                  definition->text)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "canonical special token ID %u is missing or conflicting",
                              definition->token_id);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
    }
    for (index = KQ_TOKENIZER_LENGTH;
         index < KQ_TOKENIZER_MODEL_VOCABULARY;
         ++index) {
        char expected[32];
        int written = snprintf(expected, sizeof(expected), "[PAD%llu]",
                               (unsigned long long)index);
        if (written <= 0 || (size_t)written >= sizeof(expected) ||
            !kq_token_view_equals(&tokenizer->tokens[index], expected)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "unused padded tokenizer ID %llu is incompatible",
                              (unsigned long long)index);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
    }

    for (index = 0U; index < 256U; ++index) {
        encoded_length = utf8proc_encode_char(kq_byte_to_unicode((unsigned int)index), encoded);
        if (encoded_length <= 0 ||
            !kq_token_lookup(tokenizer,
                             encoded,
                             (uint64_t)encoded_length,
                             &tokenizer->byte_token_ids[index])) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "byte-level alphabet token for byte %llu is missing",
                              (unsigned long long)index);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
    }

    if (!kq_string_cursor_begin(merges_metadata, &cursor)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer merge array is not addressable");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }
    kq_sha256_init(&hash);
    kq_sha256_update(&hash,
                     (const unsigned char *)"KQ-MERGES-v1\0",
                     sizeof("KQ-MERGES-v1\0") - 1U);
    kq_sha256_u64(&hash, KQ_TOKENIZER_MERGE_COUNT);
    for (index = 0U; index < KQ_TOKENIZER_MERGE_COUNT; ++index) {
        if (!kq_string_cursor_next(&cursor, &value)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "GGUF tokenizer merge array is malformed at rank %llu",
                              (unsigned long long)index);
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
        (void)kq_apply_string_mutation(mutation,
                                       KQ_TOKENIZER_TEST_MUTATION_MERGE_STRING,
                                       index,
                                       &value);
        kq_sha256_u64(&hash, value.length);
        kq_sha256_update(&hash, value.data, (size_t)value.length);
        space_index = 0U;
        while (space_index < value.length && value.data[space_index] != ' ') {
            space_index += 1U;
        }
        if (space_index == 0U || space_index + 1U >= value.length ||
            memchr(value.data + (size_t)(space_index + 1U),
                   ' ',
                   (size_t)(value.length - space_index - 1U)) != NULL ||
            !kq_token_lookup(tokenizer,
                             value.data,
                             space_index,
                             &left_id) ||
            !kq_token_lookup(tokenizer,
                             value.data + (size_t)(space_index + 1U),
                             value.length - space_index - 1U,
                             &right_id) ||
            !kq_token_lookup_concat(tokenizer, left_id, right_id, &result_id) ||
            !kq_merge_insert(tokenizer,
                             left_id,
                             right_id,
                             (uint32_t)index,
                             result_id,
                             diagnostic)) {
            if (diagnostic == NULL || diagnostic->status == KQ_STATUS_OK) {
                kq_diagnostic_set(diagnostic,
                                  KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                                  "invalid tokenizer merge relation at rank %llu",
                                  (unsigned long long)index);
            }
            return diagnostic != NULL ? diagnostic->status :
                   KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
    }
    kq_sha256_final(&hash, digest);
    if (!kq_digest_equal(digest, kq_expected_merge_hash)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                          "GGUF tokenizer ordered merges differ from the canonical substrate");
        return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
    }
    return KQ_STATUS_OK;
}

static kq_status kq_tokenizer_open_impl(
    const kq_gguf *gguf,
    const kq_tokenizer_test_mutation *mutation,
    kq_tokenizer **out_tokenizer,
    kq_diagnostic *diagnostic) {
    kq_tokenizer *tokenizer = NULL;
    size_t token_bytes;
    size_t token_slot_bytes;
    size_t merge_slot_bytes;
    LARGE_INTEGER frequency;
    LARGE_INTEGER started;
    LARGE_INTEGER finished;
    kq_status status;

    if (out_tokenizer != NULL) {
        *out_tokenizer = NULL;
    }
    kq_diagnostic_clear(diagnostic);
    if (gguf == NULL || out_tokenizer == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "GGUF and tokenizer output are required");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    if (!kq_allocation_size(KQ_TOKENIZER_MODEL_VOCABULARY,
                            sizeof(kq_string_view),
                            &token_bytes) ||
        !kq_allocation_size(KQ_TOKENIZER_HASH_CAPACITY,
                            sizeof(kq_token_slot),
                            &token_slot_bytes) ||
        !kq_allocation_size(KQ_TOKENIZER_HASH_CAPACITY,
                            sizeof(kq_merge_slot),
                            &merge_slot_bytes)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "tokenizer allocation size overflows");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    (void)QueryPerformanceFrequency(&frequency);
    (void)QueryPerformanceCounter(&started);
    tokenizer = (kq_tokenizer *)calloc(1U, sizeof(*tokenizer));
    if (tokenizer != NULL) {
        tokenizer->tokens = (kq_string_view *)calloc(1U, token_bytes);
        tokenizer->token_slots = (kq_token_slot *)calloc(1U, token_slot_bytes);
        tokenizer->merge_slots = (kq_merge_slot *)calloc(1U, merge_slot_bytes);
    }
    if (tokenizer == NULL || tokenizer->tokens == NULL ||
        tokenizer->token_slots == NULL || tokenizer->merge_slots == NULL) {
        kq_tokenizer_close(tokenizer);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot allocate tokenizer lookup structures");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    tokenizer->gguf = gguf;
    status = kq_validate_and_build(tokenizer, gguf, mutation, diagnostic);
    if (status != KQ_STATUS_OK) {
        kq_tokenizer_close(tokenizer);
        return status;
    }
    (void)QueryPerformanceCounter(&finished);
    tokenizer->metrics.owned_heap_bytes =
        (uint64_t)sizeof(*tokenizer) + token_bytes + token_slot_bytes +
        merge_slot_bytes;
    tokenizer->metrics.vocabulary_lookup_bytes = token_bytes + token_slot_bytes;
    tokenizer->metrics.merge_lookup_bytes = merge_slot_bytes;
    if (frequency.QuadPart > 0 && finished.QuadPart >= started.QuadPart) {
        tokenizer->metrics.construction_nanoseconds =
            (uint64_t)(((finished.QuadPart - started.QuadPart) * 1000000000ULL) /
                       (uint64_t)frequency.QuadPart);
    }
    *out_tokenizer = tokenizer;
    return KQ_STATUS_OK;
}

kq_status kq_tokenizer_open_from_gguf(const kq_gguf *gguf,
                                      kq_tokenizer **out_tokenizer,
                                      kq_diagnostic *diagnostic) {
    return kq_tokenizer_open_impl(gguf, NULL, out_tokenizer, diagnostic);
}

kq_status kq_tokenizer_open_from_gguf_for_test(
    const kq_gguf *gguf,
    const kq_tokenizer_test_mutation *mutation,
    kq_tokenizer **out_tokenizer,
    kq_diagnostic *diagnostic) {
    if (mutation == NULL || mutation->kind == KQ_TOKENIZER_TEST_MUTATION_NONE) {
        return kq_tokenizer_open_impl(gguf, NULL, out_tokenizer, diagnostic);
    }
    return kq_tokenizer_open_impl(gguf, mutation, out_tokenizer, diagnostic);
}

void kq_tokenizer_close(kq_tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }
    free(tokenizer->merge_slots);
    free(tokenizer->token_slots);
    free(tokenizer->tokens);
    memset(tokenizer, 0, sizeof(*tokenizer));
    free(tokenizer);
}

static int kq_u32_buffer_reserve(kq_u32_buffer *buffer, uint64_t required) {
    uint64_t capacity;
    size_t bytes;
    uint32_t *grown;
    if (required <= buffer->capacity) {
        return 1;
    }
    capacity = buffer->capacity == 0U ? 64U : buffer->capacity;
    while (capacity < required) {
        if (capacity > UINT64_MAX / 2U) {
            return 0;
        }
        capacity *= 2U;
    }
    if (!kq_allocation_size(capacity, sizeof(*buffer->data), &bytes)) {
        return 0;
    }
    grown = (uint32_t *)realloc(buffer->data, bytes);
    if (grown == NULL) {
        return 0;
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return 1;
}

static int kq_u32_buffer_append(kq_u32_buffer *buffer, uint32_t value) {
    if (!kq_u32_buffer_reserve(buffer, buffer->count + 1U)) {
        return 0;
    }
    buffer->data[buffer->count++] = value;
    return 1;
}

static int kq_byte_buffer_reserve(kq_byte_buffer *buffer, uint64_t required) {
    uint64_t capacity;
    size_t bytes;
    unsigned char *grown;
    if (required <= buffer->capacity) {
        return 1;
    }
    capacity = buffer->capacity == 0U ? 256U : buffer->capacity;
    while (capacity < required) {
        if (capacity > UINT64_MAX / 2U) {
            return 0;
        }
        capacity *= 2U;
    }
    if (!kq_allocation_size(capacity, sizeof(*buffer->data), &bytes)) {
        return 0;
    }
    grown = (unsigned char *)realloc(buffer->data, bytes);
    if (grown == NULL) {
        return 0;
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return 1;
}

static int kq_byte_buffer_append(kq_byte_buffer *buffer,
                                 const unsigned char *data,
                                 uint64_t length) {
    uint64_t required;
    if (!kq_u64_add_checked(buffer->count, length, &required) ||
        !kq_byte_buffer_reserve(buffer, required)) {
        return 0;
    }
    if (length != 0U) {
        memcpy(buffer->data + (size_t)buffer->count, data, (size_t)length);
    }
    buffer->count = required;
    return 1;
}

static int kq_unicode9_is_assigned(int32_t codepoint) {
    uint64_t low = 0U;
    uint64_t high = KQ_UNICODE9_ASSIGNED_RANGE_COUNT;
    uint64_t middle;
    uint32_t value;
    if (codepoint < 0) {
        return 0;
    }
    value = (uint32_t)codepoint;
    while (low < high) {
        middle = low + (high - low) / 2U;
        if (value < kq_unicode9_assigned_ranges[middle][0]) {
            high = middle;
        } else if (value > kq_unicode9_assigned_ranges[middle][1]) {
            low = middle + 1U;
        } else {
            return 1;
        }
    }
    return 0;
}

static kq_status kq_nfc9_append_run(kq_byte_buffer *output,
                                    const unsigned char *input,
                                    uint64_t start,
                                    uint64_t end,
                                    kq_diagnostic *diagnostic) {
    utf8proc_uint8_t *mapped = NULL;
    utf8proc_ssize_t mapped_length;
    if (start == end) {
        return KQ_STATUS_OK;
    }
    mapped_length = utf8proc_map(
        input + (size_t)start,
        (utf8proc_ssize_t)(end - start),
        &mapped,
        (utf8proc_option_t)(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (mapped_length < 0) {
        kq_diagnostic_set(diagnostic,
                          mapped_length == UTF8PROC_ERROR_INVALIDUTF8 ?
                              KQ_STATUS_INVALID_UTF8 : KQ_STATUS_OUT_OF_MEMORY,
                          "Unicode-9 NFC run failed: %s",
                          utf8proc_errmsg(mapped_length));
        return mapped_length == UTF8PROC_ERROR_INVALIDUTF8 ?
               KQ_STATUS_INVALID_UTF8 : KQ_STATUS_OUT_OF_MEMORY;
    }
    if (!kq_byte_buffer_append(output, mapped, (uint64_t)mapped_length)) {
        free(mapped);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot grow Unicode-9 NFC output");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    free(mapped);
    return KQ_STATUS_OK;
}

static kq_status kq_nfc9_map(const unsigned char *input,
                             uint64_t length,
                             utf8proc_uint8_t **mapped,
                             utf8proc_ssize_t *mapped_length,
                             kq_diagnostic *diagnostic) {
    kq_byte_buffer output = {0};
    uint64_t offset = 0U;
    uint64_t assigned_start = 0U;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t consumed;
    kq_status status;
    while (offset < length) {
        consumed = utf8proc_iterate(input + (size_t)offset,
                                    (utf8proc_ssize_t)(length - offset),
                                    &codepoint);
        if (consumed <= 0) {
            free(output.data);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INVALID_UTF8,
                              "tokenizer input contains invalid UTF-8");
            return KQ_STATUS_INVALID_UTF8;
        }
        if (!kq_unicode9_is_assigned(codepoint)) {
            status = kq_nfc9_append_run(&output,
                                        input,
                                        assigned_start,
                                        offset,
                                        diagnostic);
            if (status != KQ_STATUS_OK ||
                !kq_byte_buffer_append(&output,
                                       input + (size_t)offset,
                                       (uint64_t)consumed)) {
                free(output.data);
                if (status == KQ_STATUS_OK) {
                    kq_diagnostic_set(diagnostic,
                                      KQ_STATUS_OUT_OF_MEMORY,
                                      "cannot grow Unicode-9 NFC output");
                    status = KQ_STATUS_OUT_OF_MEMORY;
                }
                return status;
            }
            assigned_start = offset + (uint64_t)consumed;
        }
        offset += (uint64_t)consumed;
    }
    status = kq_nfc9_append_run(&output,
                                input,
                                assigned_start,
                                length,
                                diagnostic);
    if (status != KQ_STATUS_OK) {
        free(output.data);
        return status;
    }
    if (output.count > (uint64_t)PTRDIFF_MAX) {
        free(output.data);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "Unicode-9 NFC output exceeds ptrdiff_t");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    *mapped = output.data;
    *mapped_length = (utf8proc_ssize_t)output.count;
    return KQ_STATUS_OK;
}

static int kq_is_letter(int32_t codepoint) {
    utf8proc_category_t category = utf8proc_category(codepoint);
    return category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO;
}

static int kq_is_number(int32_t codepoint) {
    utf8proc_category_t category = utf8proc_category(codepoint);
    return category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO;
}

static int kq_is_space(int32_t codepoint) {
    return (codepoint >= 0x09 && codepoint <= 0x0d) || codepoint == 0x20 ||
           codepoint == 0x85 || codepoint == 0xa0 || codepoint == 0x1680 ||
           (codepoint >= 0x2000 && codepoint <= 0x200a) ||
           codepoint == 0x2028 || codepoint == 0x2029 ||
           codepoint == 0x202f || codepoint == 0x205f || codepoint == 0x3000;
}

static int kq_is_crlf(int32_t codepoint) {
    return codepoint == 0x0a || codepoint == 0x0d;
}

static int kq_ascii_lower(int32_t codepoint) {
    return codepoint >= 'A' && codepoint <= 'Z' ? codepoint + ('a' - 'A') :
           codepoint;
}

static uint64_t kq_match_contraction(const kq_codepoint *points,
                                     uint64_t count,
                                     uint64_t offset) {
    static const char *suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    size_t suffix_index;
    if (offset >= count || points[offset].value != '\'') {
        return 0U;
    }
    for (suffix_index = 0U;
         suffix_index < sizeof(suffixes) / sizeof(suffixes[0]);
         ++suffix_index) {
        const char *suffix = suffixes[suffix_index];
        size_t length = strlen(suffix);
        size_t index;
        if (offset + 1U + length > count) {
            continue;
        }
        for (index = 0U; index < length; ++index) {
            if (kq_ascii_lower(points[offset + 1U + index].value) != suffix[index]) {
                break;
            }
        }
        if (index == length) {
            return (uint64_t)length + 1U;
        }
    }
    return 0U;
}

static uint64_t kq_next_pretoken(const kq_codepoint *points,
                                 uint64_t count,
                                 uint64_t offset) {
    uint64_t length;
    uint64_t index;
    uint64_t whitespace_end;
    uint64_t last_crlf;
    int32_t current = points[offset].value;

    length = kq_match_contraction(points, count, offset);
    if (length != 0U) {
        return length;
    }
    if (kq_is_letter(current)) {
        index = offset + 1U;
        while (index < count && kq_is_letter(points[index].value)) {
            index += 1U;
        }
        return index - offset;
    }
    if (!kq_is_crlf(current) && !kq_is_letter(current) &&
        !kq_is_number(current) && offset + 1U < count &&
        kq_is_letter(points[offset + 1U].value)) {
        index = offset + 2U;
        while (index < count && kq_is_letter(points[index].value)) {
            index += 1U;
        }
        return index - offset;
    }
    if (kq_is_number(current)) {
        return 1U;
    }
    index = offset;
    if (current == 0x20 && index + 1U < count &&
        !kq_is_space(points[index + 1U].value) &&
        !kq_is_letter(points[index + 1U].value) &&
        !kq_is_number(points[index + 1U].value)) {
        index += 1U;
    }
    if (index < count && !kq_is_space(points[index].value) &&
        !kq_is_letter(points[index].value) &&
        !kq_is_number(points[index].value)) {
        do {
            index += 1U;
        } while (index < count && !kq_is_space(points[index].value) &&
                 !kq_is_letter(points[index].value) &&
                 !kq_is_number(points[index].value));
        while (index < count && kq_is_crlf(points[index].value)) {
            index += 1U;
        }
        return index - offset;
    }
    if (kq_is_space(current)) {
        whitespace_end = offset;
        last_crlf = UINT64_MAX;
        while (whitespace_end < count && kq_is_space(points[whitespace_end].value)) {
            if (kq_is_crlf(points[whitespace_end].value)) {
                last_crlf = whitespace_end;
            }
            whitespace_end += 1U;
        }
        if (last_crlf != UINT64_MAX) {
            return last_crlf - offset + 1U;
        }
        if (whitespace_end == count) {
            return whitespace_end - offset;
        }
        if (whitespace_end - offset >= 2U) {
            return whitespace_end - offset - 1U;
        }
        return 1U;
    }
    return 1U;
}

static kq_status kq_bpe_piece(const kq_tokenizer *tokenizer,
                              const unsigned char *bytes,
                              uint64_t length,
                              kq_u32_buffer *output,
                              uint64_t *temporary_bytes,
                              kq_diagnostic *diagnostic) {
    uint32_t *symbols = NULL;
    uint32_t *merged = NULL;
    uint64_t symbol_count = length;
    uint64_t index;
    uint64_t out_count;
    size_t allocation_bytes;
    const kq_merge_slot *candidate;
    const kq_merge_slot *best;
    uint32_t best_left;
    uint32_t best_right;

    if (length == 0U) {
        return KQ_STATUS_OK;
    }
    if (!kq_allocation_size(length, sizeof(*symbols), &allocation_bytes)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "BPE symbol allocation overflows");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    symbols = (uint32_t *)malloc(allocation_bytes);
    merged = (uint32_t *)malloc(allocation_bytes);
    if (symbols == NULL || merged == NULL) {
        free(symbols);
        free(merged);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot allocate BPE temporary symbols");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    if (*temporary_bytes < (uint64_t)allocation_bytes * 2U) {
        *temporary_bytes = (uint64_t)allocation_bytes * 2U;
    }
    for (index = 0U; index < length; ++index) {
        symbols[index] = tokenizer->byte_token_ids[bytes[index]];
    }
    while (symbol_count > 1U) {
        best = NULL;
        best_left = 0U;
        best_right = 0U;
        for (index = 0U; index + 1U < symbol_count; ++index) {
            candidate = kq_merge_find(tokenizer, symbols[index], symbols[index + 1U]);
            if (candidate != NULL && (best == NULL || candidate->rank < best->rank)) {
                best = candidate;
                best_left = symbols[index];
                best_right = symbols[index + 1U];
            }
        }
        if (best == NULL) {
            break;
        }
        out_count = 0U;
        index = 0U;
        while (index < symbol_count) {
            if (index + 1U < symbol_count && symbols[index] == best_left &&
                symbols[index + 1U] == best_right) {
                merged[out_count++] = best->result_id;
                index += 2U;
            } else {
                merged[out_count++] = symbols[index++];
            }
        }
        memcpy(symbols, merged, (size_t)out_count * sizeof(*symbols));
        symbol_count = out_count;
    }
    if (!kq_u32_buffer_reserve(output, output->count + symbol_count)) {
        free(symbols);
        free(merged);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot grow tokenizer output");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    memcpy(output->data + (size_t)output->count,
           symbols,
           (size_t)symbol_count * sizeof(*symbols));
    output->count += symbol_count;
    free(symbols);
    free(merged);
    return KQ_STATUS_OK;
}

static kq_status kq_encode_ordinary(const kq_tokenizer *tokenizer,
                                    const unsigned char *utf8,
                                    uint64_t length,
                                    kq_u32_buffer *output,
                                    uint64_t *temporary_bytes,
                                    kq_diagnostic *diagnostic) {
    utf8proc_uint8_t *normalized = NULL;
    utf8proc_ssize_t normalized_length;
    kq_codepoint *points = NULL;
    uint64_t point_count = 0U;
    uint64_t byte_offset = 0U;
    uint64_t point_offset;
    uint64_t piece_points;
    uint64_t piece_start;
    uint64_t piece_end;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t consumed;
    size_t points_bytes;
    kq_status status;

    if (length == 0U) {
        return KQ_STATUS_OK;
    }
    if (length > (uint64_t)PTRDIFF_MAX) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "tokenizer input exceeds utf8proc length range");
        return KQ_STATUS_LIMIT_EXCEEDED;
    }
    status = kq_nfc9_map(utf8,
                         length,
                         &normalized,
                         &normalized_length,
                         diagnostic);
    if (status != KQ_STATUS_OK) {
        return status;
    }
    if (!kq_allocation_size((uint64_t)normalized_length,
                            sizeof(*points),
                            &points_bytes)) {
        free(normalized);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "pre-tokenization codepoint allocation overflows");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    points = (kq_codepoint *)malloc(points_bytes == 0U ? 1U : points_bytes);
    if (points == NULL) {
        free(normalized);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot allocate pre-tokenization codepoints");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    while (byte_offset < (uint64_t)normalized_length) {
        consumed = utf8proc_iterate(normalized + (size_t)byte_offset,
                                    normalized_length - (utf8proc_ssize_t)byte_offset,
                                    &codepoint);
        if (consumed <= 0) {
            free(points);
            free(normalized);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INVALID_UTF8,
                              "NFC result contains invalid UTF-8");
            return KQ_STATUS_INVALID_UTF8;
        }
        points[point_count].value = codepoint;
        points[point_count].byte_offset = byte_offset;
        points[point_count].byte_length = (uint64_t)consumed;
        point_count += 1U;
        byte_offset += (uint64_t)consumed;
    }
    if (*temporary_bytes < (uint64_t)normalized_length + (uint64_t)points_bytes) {
        *temporary_bytes = (uint64_t)normalized_length + (uint64_t)points_bytes;
    }
    point_offset = 0U;
    while (point_offset < point_count) {
        piece_points = kq_next_pretoken(points, point_count, point_offset);
        if (piece_points == 0U || point_offset + piece_points > point_count) {
            free(points);
            free(normalized);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INCOMPATIBLE_TOKENIZER,
                              "canonical pre-tokenizer made no progress");
            return KQ_STATUS_INCOMPATIBLE_TOKENIZER;
        }
        piece_start = points[point_offset].byte_offset;
        piece_end = points[point_offset + piece_points - 1U].byte_offset +
                    points[point_offset + piece_points - 1U].byte_length;
        status = kq_bpe_piece(tokenizer,
                              normalized + (size_t)piece_start,
                              piece_end - piece_start,
                              output,
                              temporary_bytes,
                              diagnostic);
        if (status != KQ_STATUS_OK) {
            free(points);
            free(normalized);
            return status;
        }
        point_offset += piece_points;
    }
    free(points);
    free(normalized);
    return KQ_STATUS_OK;
}

static const kq_special_definition *kq_special_at(const unsigned char *utf8,
                                                  uint64_t remaining) {
    const kq_special_definition *best = NULL;
    size_t index;
    for (index = 0U; index < KQ_TOKENIZER_ADDED_TOKENS; ++index) {
        size_t length = strlen(kq_special_tokens[index].text);
        if ((uint64_t)length <= remaining &&
            memcmp(utf8, kq_special_tokens[index].text, length) == 0 &&
            (best == NULL || length > strlen(best->text))) {
            best = &kq_special_tokens[index];
        }
    }
    return best;
}

kq_status kq_tokenizer_encode(const kq_tokenizer *tokenizer,
                              const unsigned char *utf8,
                              uint64_t utf8_length,
                              const kq_tokenizer_encode_options *options,
                              uint32_t *token_ids,
                              uint64_t token_capacity,
                              uint64_t *required_tokens,
                              kq_diagnostic *diagnostic) {
    kq_u32_buffer output = {0};
    uint64_t offset = 0U;
    uint64_t ordinary_start = 0U;
    uint64_t temporary_bytes = 0U;
    const kq_special_definition *special;
    size_t special_length;
    kq_status status = KQ_STATUS_OK;

    kq_diagnostic_clear(diagnostic);
    if (required_tokens != NULL) {
        *required_tokens = 0U;
    }
    if (tokenizer == NULL || options == NULL || required_tokens == NULL ||
        (utf8 == NULL && utf8_length != 0U) ||
        (token_ids == NULL && token_capacity != 0U)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "tokenizer encode arguments are invalid");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    if (utf8_length > KQ_TOKENIZER_MAX_INPUT_BYTES) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "tokenizer input exceeds the 16 MiB defensive limit");
        return KQ_STATUS_LIMIT_EXCEEDED;
    }
    if (options->special_policy != KQ_TOKENIZER_SPECIAL_REJECT &&
        options->special_policy != KQ_TOKENIZER_SPECIAL_RECOGNIZE) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
                          "unknown special-token encode policy");
        return KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION;
    }
    while (offset < utf8_length) {
        special = kq_special_at(utf8 + (size_t)offset, utf8_length - offset);
        if (special == NULL) {
            offset += 1U;
            continue;
        }
        status = kq_encode_ordinary(tokenizer,
                                    utf8 + (size_t)ordinary_start,
                                    offset - ordinary_start,
                                    &output,
                                    &temporary_bytes,
                                    diagnostic);
        if (status != KQ_STATUS_OK) {
            goto cleanup;
        }
        if (options->special_policy == KQ_TOKENIZER_SPECIAL_REJECT) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
                              "special-token literal is forbidden by encode policy");
            status = KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION;
            goto cleanup;
        }
        if (!kq_u32_buffer_append(&output, special->token_id)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_OUT_OF_MEMORY,
                              "cannot grow tokenizer output");
            status = KQ_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
        special_length = strlen(special->text);
        offset += (uint64_t)special_length;
        ordinary_start = offset;
    }
    status = kq_encode_ordinary(tokenizer,
                                utf8 != NULL ? utf8 + (size_t)ordinary_start :
                                               (const unsigned char *)"",
                                utf8_length - ordinary_start,
                                &output,
                                &temporary_bytes,
                                diagnostic);
    if (status != KQ_STATUS_OK) {
        goto cleanup;
    }
    *required_tokens = output.count;
    if (token_capacity < output.count) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_BUFFER_TOO_SMALL,
                          "token output requires %llu entries; capacity is %llu",
                          (unsigned long long)output.count,
                          (unsigned long long)token_capacity);
        status = KQ_STATUS_BUFFER_TOO_SMALL;
        goto cleanup;
    }
    if (output.count != 0U) {
        memcpy(token_ids,
               output.data,
               (size_t)output.count * sizeof(*token_ids));
    }
cleanup:
    free(output.data);
    return status;
}

static const kq_special_definition *kq_special_by_id(uint32_t token_id) {
    if (token_id < KQ_TOKENIZER_BASE_VOCABULARY ||
        token_id >= KQ_TOKENIZER_LENGTH) {
        return NULL;
    }
    return &kq_special_tokens[token_id - KQ_TOKENIZER_BASE_VOCABULARY];
}

static int kq_append_replacement(kq_byte_buffer *output) {
    static const unsigned char replacement[] = {0xefU, 0xbfU, 0xbdU};
    return kq_byte_buffer_append(output, replacement, sizeof(replacement));
}

static kq_status kq_raw_bytes_to_utf8(const kq_byte_buffer *raw,
                                      kq_byte_buffer *output,
                                      kq_diagnostic *diagnostic) {
    uint64_t offset = 0U;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t consumed;
    while (offset < raw->count) {
        consumed = utf8proc_iterate(raw->data + (size_t)offset,
                                    (utf8proc_ssize_t)(raw->count - offset),
                                    &codepoint);
        if (consumed > 0) {
            if (!kq_byte_buffer_append(output,
                                       raw->data + (size_t)offset,
                                       (uint64_t)consumed)) {
                kq_diagnostic_set(diagnostic,
                                  KQ_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate decoded UTF-8 output");
                return KQ_STATUS_OUT_OF_MEMORY;
            }
            offset += (uint64_t)consumed;
        } else {
            if (!kq_append_replacement(output)) {
                kq_diagnostic_set(diagnostic,
                                  KQ_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate decoded UTF-8 replacement");
                return KQ_STATUS_OUT_OF_MEMORY;
            }
            offset += 1U;
            while (offset < raw->count &&
                   (raw->data[offset] & 0xc0U) == 0x80U) {
                offset += 1U;
            }
        }
    }
    return KQ_STATUS_OK;
}

static kq_status kq_flush_raw_decode(kq_byte_buffer *raw,
                                     kq_byte_buffer *decoded,
                                     kq_diagnostic *diagnostic) {
    kq_status status;
    if (raw->count == 0U) {
        return KQ_STATUS_OK;
    }
    status = kq_raw_bytes_to_utf8(raw, decoded, diagnostic);
    if (status == KQ_STATUS_OK) {
        raw->count = 0U;
    }
    return status;
}

kq_status kq_tokenizer_decode(const kq_tokenizer *tokenizer,
                              const uint32_t *token_ids,
                              uint64_t token_count,
                              const kq_tokenizer_decode_options *options,
                              unsigned char *utf8,
                              uint64_t utf8_capacity,
                              uint64_t *required_utf8_bytes,
                              kq_diagnostic *diagnostic) {
    kq_byte_buffer raw = {0};
    kq_byte_buffer decoded = {0};
    uint64_t token_index;
    uint64_t byte_offset;
    const kq_string_view *token;
    const kq_special_definition *special;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t consumed;
    unsigned char byte_value;
    kq_status status = KQ_STATUS_OK;

    kq_diagnostic_clear(diagnostic);
    if (required_utf8_bytes != NULL) {
        *required_utf8_bytes = 0U;
    }
    if (tokenizer == NULL || options == NULL || required_utf8_bytes == NULL ||
        (token_ids == NULL && token_count != 0U) ||
        (utf8 == NULL && utf8_capacity != 0U)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "tokenizer decode arguments are invalid");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    if (options->special_policy != KQ_TOKENIZER_DECODE_KEEP_SPECIAL &&
        options->special_policy != KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
                          "unknown special-token decode policy");
        return KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION;
    }
    for (token_index = 0U; token_index < token_count; ++token_index) {
        uint32_t token_id = token_ids[token_index];
        if (token_id >= KQ_TOKENIZER_LENGTH) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INVALID_TOKEN_ID,
                              "token ID %u is padded/unused or out of range",
                              token_id);
            status = KQ_STATUS_INVALID_TOKEN_ID;
            goto cleanup;
        }
        special = kq_special_by_id(token_id);
        if (special != NULL) {
            status = kq_flush_raw_decode(&raw, &decoded, diagnostic);
            if (status != KQ_STATUS_OK) {
                goto cleanup;
            }
            if (options->special_policy ==
                    KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL &&
                special->skipped_by_canonical_decode) {
                continue;
            }
            if (!kq_byte_buffer_append(&decoded,
                                       (const unsigned char *)special->text,
                                       (uint64_t)strlen(special->text))) {
                status = KQ_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            continue;
        }
        token = &tokenizer->tokens[token_id];
        byte_offset = 0U;
        while (byte_offset < token->length) {
            int byte_ok;
            consumed = utf8proc_iterate(token->data + (size_t)byte_offset,
                                        (utf8proc_ssize_t)(token->length - byte_offset),
                                        &codepoint);
            byte_ok = consumed > 0 && kq_unicode_to_byte(codepoint, &byte_value);
            if (!byte_ok || !kq_byte_buffer_append(&raw, &byte_value, 1U)) {
                kq_diagnostic_set(diagnostic,
                                  !byte_ok ?
                                      KQ_STATUS_INCOMPATIBLE_TOKENIZER :
                                      KQ_STATUS_OUT_OF_MEMORY,
                                  "token ID %u is not valid byte-level vocabulary data",
                                  token_id);
                status = !byte_ok ?
                         KQ_STATUS_INCOMPATIBLE_TOKENIZER : KQ_STATUS_OUT_OF_MEMORY;
                goto cleanup;
            }
            byte_offset += (uint64_t)consumed;
        }
    }
    status = kq_flush_raw_decode(&raw, &decoded, diagnostic);
    if (status != KQ_STATUS_OK) {
        goto cleanup;
    }
    *required_utf8_bytes = decoded.count;
    if (utf8_capacity < decoded.count) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_BUFFER_TOO_SMALL,
                          "decoded UTF-8 requires %llu bytes; capacity is %llu",
                          (unsigned long long)decoded.count,
                          (unsigned long long)utf8_capacity);
        status = KQ_STATUS_BUFFER_TOO_SMALL;
        goto cleanup;
    }
    if (decoded.count != 0U) {
        memcpy(utf8, decoded.data, (size_t)decoded.count);
    }
cleanup:
    free(decoded.data);
    free(raw.data);
    return status;
}

const kq_tokenizer_metrics *kq_tokenizer_get_metrics(
    const kq_tokenizer *tokenizer) {
    return tokenizer != NULL ? &tokenizer->metrics : NULL;
}

const char *kq_tokenizer_nfc_unicode_version(void) {
    return "9.0.0";
}

const char *kq_tokenizer_property_unicode_version(void) {
    return utf8proc_unicode_version();
}
