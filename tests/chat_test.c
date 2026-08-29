#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "kq_chat.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

int main(void) {
    static const unsigned char user_text[] = "  hello  \n";
    static const unsigned char expected[] =
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    static const unsigned char invalid_utf8[] = {0xc0U, 0xafU};
    kq_chat_message messages[2];
    kq_chat_options options = {1, 0U};
    kq_diagnostic diagnostic;
    unsigned char output[sizeof(expected) - 1U];
    uint64_t required = 0U;
    kq_status status;

    messages[0].role = KQ_CHAT_ROLE_USER;
    messages[0].content_type = KQ_CHAT_CONTENT_TEXT;
    messages[0].content = user_text;
    messages[0].content_length = sizeof(user_text) - 1U;

    status = kq_chat_render(messages,
                            1U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL &&
              required == sizeof(expected) - 1U,
          "chat required-size query must be exact");
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            output,
                            sizeof(output) - 1U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "chat render must not silently truncate output");
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            output,
                            sizeof(output),
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_OK &&
              memcmp(output, expected, sizeof(output)) == 0,
          "chat render must match the canonical text-only subset");

    messages[0].content = invalid_utf8;
    messages[0].content_length = sizeof(invalid_utf8);
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_INVALID_UTF8,
          "invalid UTF-8 chat content must fail closed");

    messages[0].content = user_text;
    messages[0].content_length = sizeof(user_text) - 1U;
    messages[0].content_type = KQ_CHAT_CONTENT_TOOL_RESULT;
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT,
          "tool-result content must fail closed");
    messages[0].content_type = KQ_CHAT_CONTENT_TEXT;
    messages[0].role = KQ_CHAT_ROLE_TOOL;
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT,
          "tool roles must fail closed");

    messages[0].role = KQ_CHAT_ROLE_USER;
    options.add_generation_prompt = 2;
    status = kq_chat_render(messages,
                            1U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT,
          "unknown generation-prompt modes must fail closed");

    messages[1] = messages[0];
    messages[1].role = KQ_CHAT_ROLE_USER;
    options.add_generation_prompt = 0;
    status = kq_chat_render(messages,
                            2U,
                            &options,
                            NULL,
                            0U,
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_MALFORMED_CHAT,
          "non-alternating roles must fail closed");

    if (failures != 0) {
        fprintf(stderr, "%d chat test(s) failed\n", failures);
        return 1;
    }
    printf("chat synthetic tests passed\n");
    return 0;
}
