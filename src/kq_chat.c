#include "kq_chat.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "utf8proc.h"

#include "kq_internal.h"

typedef struct kq_chat_buffer {
    unsigned char *data;
    uint64_t count;
    uint64_t capacity;
} kq_chat_buffer;

static int kq_chat_add_checked(uint64_t left,
                               uint64_t right,
                               uint64_t *result) {
    if (result == NULL || UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int kq_chat_reserve(kq_chat_buffer *buffer, uint64_t required) {
    uint64_t capacity;
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
    if (capacity > (uint64_t)SIZE_MAX) {
        return 0;
    }
    grown = (unsigned char *)realloc(buffer->data, (size_t)capacity);
    if (grown == NULL) {
        return 0;
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return 1;
}

static int kq_chat_append(kq_chat_buffer *buffer,
                          const unsigned char *data,
                          uint64_t length) {
    uint64_t required;
    if (!kq_chat_add_checked(buffer->count, length, &required) ||
        !kq_chat_reserve(buffer, required)) {
        return 0;
    }
    if (length != 0U) {
        memcpy(buffer->data + (size_t)buffer->count, data, (size_t)length);
    }
    buffer->count = required;
    return 1;
}

static int kq_chat_append_cstr(kq_chat_buffer *buffer, const char *text) {
    return kq_chat_append(buffer,
                          (const unsigned char *)text,
                          (uint64_t)strlen(text));
}

static int kq_chat_is_space(int32_t codepoint) {
    return (codepoint >= 0x09 && codepoint <= 0x0d) || codepoint == 0x20 ||
           codepoint == 0x85 || codepoint == 0xa0 || codepoint == 0x1680 ||
           (codepoint >= 0x2000 && codepoint <= 0x200a) ||
           codepoint == 0x2028 || codepoint == 0x2029 ||
           codepoint == 0x202f || codepoint == 0x205f || codepoint == 0x3000;
}

static kq_status kq_chat_trim(const unsigned char *content,
                              uint64_t length,
                              uint64_t *trimmed_offset,
                              uint64_t *trimmed_length,
                              kq_diagnostic *diagnostic) {
    uint64_t offset = 0U;
    uint64_t first_non_space = UINT64_MAX;
    uint64_t last_non_space_end = 0U;
    utf8proc_int32_t codepoint;
    utf8proc_ssize_t consumed;
    if (content == NULL && length != 0U) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_MALFORMED_CHAT,
                          "chat text content pointer is null");
        return KQ_STATUS_MALFORMED_CHAT;
    }
    if (length > (uint64_t)PTRDIFF_MAX) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_LIMIT_EXCEEDED,
                          "chat content exceeds UTF-8 processing range");
        return KQ_STATUS_LIMIT_EXCEEDED;
    }
    while (offset < length) {
        consumed = utf8proc_iterate(content + (size_t)offset,
                                    (utf8proc_ssize_t)(length - offset),
                                    &codepoint);
        if (consumed <= 0) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_INVALID_UTF8,
                              "chat content contains invalid UTF-8");
            return KQ_STATUS_INVALID_UTF8;
        }
        if (!kq_chat_is_space(codepoint)) {
            if (first_non_space == UINT64_MAX) {
                first_non_space = offset;
            }
            last_non_space_end = offset + (uint64_t)consumed;
        }
        offset += (uint64_t)consumed;
    }
    if (first_non_space == UINT64_MAX) {
        *trimmed_offset = 0U;
        *trimmed_length = 0U;
    } else {
        *trimmed_offset = first_non_space;
        *trimmed_length = last_non_space_end - first_non_space;
    }
    return KQ_STATUS_OK;
}

static const char *kq_chat_role_name(kq_chat_role role) {
    switch (role) {
        case KQ_CHAT_ROLE_SYSTEM:
            return "system";
        case KQ_CHAT_ROLE_USER:
            return "user";
        case KQ_CHAT_ROLE_ASSISTANT:
            return "assistant";
        default:
            return NULL;
    }
}

kq_status kq_chat_render(const kq_chat_message *messages,
                         uint64_t message_count,
                         const kq_chat_options *options,
                         unsigned char *utf8,
                         uint64_t utf8_capacity,
                         uint64_t *required_utf8_bytes,
                         kq_diagnostic *diagnostic) {
    kq_chat_buffer rendered = {0};
    uint64_t index;
    uint64_t trimmed_offset;
    uint64_t trimmed_length;
    uint64_t first_conversation = 0U;
    kq_chat_role expected_role = KQ_CHAT_ROLE_USER;
    const char *role_name;
    kq_status status = KQ_STATUS_OK;

    kq_diagnostic_clear(diagnostic);
    if (required_utf8_bytes != NULL) {
        *required_utf8_bytes = 0U;
    }
    if (messages == NULL || message_count == 0U || options == NULL ||
        required_utf8_bytes == NULL || (utf8 == NULL && utf8_capacity != 0U)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_MALFORMED_CHAT,
                          "chat messages, options, and output size are required");
        return KQ_STATUS_MALFORMED_CHAT;
    }
    if ((options->add_generation_prompt != 0 &&
         options->add_generation_prompt != 1) ||
        options->reserved_flags != 0U) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_UNSUPPORTED_CHAT,
                          "unsupported chat-template option");
        return KQ_STATUS_UNSUPPORTED_CHAT;
    }
    if (messages[0].role == KQ_CHAT_ROLE_SYSTEM) {
        first_conversation = 1U;
    }
    if (first_conversation == message_count) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_MALFORMED_CHAT,
                          "chat requires at least one user message");
        return KQ_STATUS_MALFORMED_CHAT;
    }
    for (index = 0U; index < message_count; ++index) {
        const kq_chat_message *message = &messages[index];
        if (message->content_type != KQ_CHAT_CONTENT_TEXT) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_UNSUPPORTED_CHAT,
                              "only text chat content is supported");
            status = KQ_STATUS_UNSUPPORTED_CHAT;
            goto cleanup;
        }
        role_name = kq_chat_role_name(message->role);
        if (role_name == NULL) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_UNSUPPORTED_CHAT,
                              "chat role is outside the canonical initial subset");
            status = KQ_STATUS_UNSUPPORTED_CHAT;
            goto cleanup;
        }
        if ((message->role == KQ_CHAT_ROLE_SYSTEM && index != 0U) ||
            (index >= first_conversation && message->role != expected_role)) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_MALFORMED_CHAT,
                              "chat roles must be optional-system then alternating user/assistant");
            status = KQ_STATUS_MALFORMED_CHAT;
            goto cleanup;
        }
        if (index >= first_conversation) {
            expected_role = expected_role == KQ_CHAT_ROLE_USER ?
                            KQ_CHAT_ROLE_ASSISTANT : KQ_CHAT_ROLE_USER;
        }
        status = kq_chat_trim(message->content,
                              message->content_length,
                              &trimmed_offset,
                              &trimmed_length,
                              diagnostic);
        if (status != KQ_STATUS_OK) {
            goto cleanup;
        }
        if (!kq_chat_append_cstr(&rendered, "<|im_start|>") ||
            !kq_chat_append_cstr(&rendered, role_name) ||
            !kq_chat_append_cstr(&rendered, "\n") ||
            !kq_chat_append(&rendered,
                            message->content != NULL ?
                                message->content + (size_t)trimmed_offset :
                                (const unsigned char *)"",
                            trimmed_length) ||
            !kq_chat_append_cstr(&rendered, "<|im_end|>\n")) {
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_OUT_OF_MEMORY,
                              "cannot allocate rendered chat prompt");
            status = KQ_STATUS_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    if (expected_role != KQ_CHAT_ROLE_ASSISTANT) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_MALFORMED_CHAT,
                          "initial chat subset requires the final message to be user");
        status = KQ_STATUS_MALFORMED_CHAT;
        goto cleanup;
    }
    if (options->add_generation_prompt &&
        !kq_chat_append_cstr(
            &rendered,
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot allocate chat generation prompt");
        status = KQ_STATUS_OUT_OF_MEMORY;
        goto cleanup;
    }
    *required_utf8_bytes = rendered.count;
    if (utf8_capacity < rendered.count) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_BUFFER_TOO_SMALL,
                          "rendered chat requires %llu bytes; capacity is %llu",
                          (unsigned long long)rendered.count,
                          (unsigned long long)utf8_capacity);
        status = KQ_STATUS_BUFFER_TOO_SMALL;
        goto cleanup;
    }
    if (rendered.count != 0U) {
        memcpy(utf8, rendered.data, (size_t)rendered.count);
    }
cleanup:
    free(rendered.data);
    return status;
}

kq_status kq_chat_tokenize(const kq_tokenizer *tokenizer,
                           const kq_chat_message *messages,
                           uint64_t message_count,
                           const kq_chat_options *options,
                           uint32_t *token_ids,
                           uint64_t token_capacity,
                           uint64_t *required_tokens,
                           kq_diagnostic *diagnostic) {
    unsigned char *rendered = NULL;
    uint64_t rendered_bytes = 0U;
    kq_tokenizer_encode_options encode_options;
    kq_status status;
    if (required_tokens != NULL) {
        *required_tokens = 0U;
    }
    status = kq_chat_render(messages,
                            message_count,
                            options,
                            NULL,
                            0U,
                            &rendered_bytes,
                            diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL) {
        return status;
    }
    if (rendered_bytes > (uint64_t)SIZE_MAX) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "rendered chat allocation overflows size_t");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    rendered = (unsigned char *)malloc((size_t)(rendered_bytes == 0U ? 1U : rendered_bytes));
    if (rendered == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "cannot allocate rendered chat prompt");
        return KQ_STATUS_OUT_OF_MEMORY;
    }
    status = kq_chat_render(messages,
                            message_count,
                            options,
                            rendered,
                            rendered_bytes,
                            &rendered_bytes,
                            diagnostic);
    if (status == KQ_STATUS_OK) {
        encode_options.special_policy = KQ_TOKENIZER_SPECIAL_RECOGNIZE;
        status = kq_tokenizer_encode(tokenizer,
                                     rendered,
                                     rendered_bytes,
                                     &encode_options,
                                     token_ids,
                                     token_capacity,
                                     required_tokens,
                                     diagnostic);
    }
    free(rendered);
    return status;
}
