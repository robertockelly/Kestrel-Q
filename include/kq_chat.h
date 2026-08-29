#ifndef KQ_CHAT_H
#define KQ_CHAT_H

#include <stdint.h>

#include "kq_status.h"
#include "kq_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum kq_chat_role {
    KQ_CHAT_ROLE_SYSTEM = 0,
    KQ_CHAT_ROLE_USER = 1,
    KQ_CHAT_ROLE_ASSISTANT = 2,
    KQ_CHAT_ROLE_DEVELOPER = 3,
    KQ_CHAT_ROLE_TOOL = 4
} kq_chat_role;

typedef enum kq_chat_content_type {
    KQ_CHAT_CONTENT_TEXT = 0,
    KQ_CHAT_CONTENT_MULTIMODAL = 1,
    KQ_CHAT_CONTENT_TOOL_RESULT = 2
} kq_chat_content_type;

typedef struct kq_chat_message {
    kq_chat_role role;
    kq_chat_content_type content_type;
    const unsigned char *content;
    uint64_t content_length;
} kq_chat_message;

typedef struct kq_chat_options {
    int add_generation_prompt;
    uint32_t reserved_flags;
} kq_chat_options;

kq_status kq_chat_render(const kq_chat_message *messages,
                         uint64_t message_count,
                         const kq_chat_options *options,
                         unsigned char *utf8,
                         uint64_t utf8_capacity,
                         uint64_t *required_utf8_bytes,
                         kq_diagnostic *diagnostic);

kq_status kq_chat_tokenize(const kq_tokenizer *tokenizer,
                           const kq_chat_message *messages,
                           uint64_t message_count,
                           const kq_chat_options *options,
                           uint32_t *token_ids,
                           uint64_t token_capacity,
                           uint64_t *required_tokens,
                           kq_diagnostic *diagnostic);

#ifdef __cplusplus
}
#endif

#endif
