#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "kq_chat.h"
#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_status.h"
#include "kq_tokenizer.h"

static int hex_nibble(wchar_t value) {
    if (value >= L'0' && value <= L'9') {
        return (int)(value - L'0');
    }
    if (value >= L'a' && value <= L'f') {
        return (int)(value - L'a') + 10;
    }
    if (value >= L'A' && value <= L'F') {
        return (int)(value - L'A') + 10;
    }
    return -1;
}

static unsigned char *parse_hex(const wchar_t *text, uint64_t *length) {
    size_t chars = wcslen(text);
    unsigned char *bytes;
    size_t index;
    if ((chars & 1U) != 0U || chars / 2U > SIZE_MAX) {
        return NULL;
    }
    bytes = (unsigned char *)malloc(chars == 0U ? 1U : chars / 2U);
    if (bytes == NULL) {
        return NULL;
    }
    for (index = 0U; index < chars; index += 2U) {
        int high = hex_nibble(text[index]);
        int low = hex_nibble(text[index + 1U]);
        if (high < 0 || low < 0) {
            free(bytes);
            return NULL;
        }
        bytes[index / 2U] = (unsigned char)((high << 4) | low);
    }
    *length = (uint64_t)(chars / 2U);
    return bytes;
}

static uint32_t *parse_ids(const wchar_t *text, uint64_t *count) {
    size_t length = wcslen(text);
    wchar_t *copy;
    wchar_t *context = NULL;
    wchar_t *item;
    uint32_t *values = NULL;
    uint64_t used = 0U;
    uint64_t capacity = 0U;
    if (length > SIZE_MAX / sizeof(*copy) - 1U) {
        return NULL;
    }
    copy = (wchar_t *)malloc((length + 1U) * sizeof(*copy));
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, text, (length + 1U) * sizeof(*copy));
    item = wcstok_s(copy, L",", &context);
    while (item != NULL) {
        wchar_t *end = NULL;
        unsigned long value = wcstoul(item, &end, 10);
        uint32_t *grown;
        if (end == item || *end != L'\0' || value > UINT32_MAX) {
            free(values);
            free(copy);
            return NULL;
        }
        if (used == capacity) {
            capacity = capacity == 0U ? 16U : capacity * 2U;
            if (capacity > (uint64_t)(SIZE_MAX / sizeof(*values))) {
                free(values);
                free(copy);
                return NULL;
            }
            grown = (uint32_t *)realloc(values, (size_t)capacity * sizeof(*values));
            if (grown == NULL) {
                free(values);
                free(copy);
                return NULL;
            }
            values = grown;
        }
        values[used++] = (uint32_t)value;
        item = wcstok_s(NULL, L",", &context);
    }
    free(copy);
    *count = used;
    return values;
}

static void print_ids(const uint32_t *ids, uint64_t count) {
    uint64_t index;
    for (index = 0U; index < count; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%u", ids[index]);
    }
    putchar('\n');
}

static void print_hex(const unsigned char *bytes, uint64_t length) {
    static const char digits[] = "0123456789abcdef";
    uint64_t index;
    for (index = 0U; index < length; ++index) {
        putchar(digits[bytes[index] >> 4U]);
        putchar(digits[bytes[index] & 0x0fU]);
    }
    putchar('\n');
}

static void print_failure(kq_status status, const kq_diagnostic *diagnostic) {
    printf("ERROR:%s", kq_status_string(status));
    if (diagnostic != NULL && diagnostic->message[0] != '\0') {
        printf(":%s", diagnostic->message);
    }
    putchar('\n');
}

static kq_chat_role parse_role(const wchar_t *text, int *valid) {
    *valid = 1;
    if (wcscmp(text, L"system") == 0) {
        return KQ_CHAT_ROLE_SYSTEM;
    }
    if (wcscmp(text, L"user") == 0) {
        return KQ_CHAT_ROLE_USER;
    }
    if (wcscmp(text, L"assistant") == 0) {
        return KQ_CHAT_ROLE_ASSISTANT;
    }
    if (wcscmp(text, L"developer") == 0) {
        return KQ_CHAT_ROLE_DEVELOPER;
    }
    if (wcscmp(text, L"tool") == 0) {
        return KQ_CHAT_ROLE_TOOL;
    }
    *valid = 0;
    return KQ_CHAT_ROLE_USER;
}

int wmain(int argc, wchar_t **argv) {
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    int result = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: tokenizer-probe <gguf> <operation> ...\n");
        return 1;
    }
    status = kq_file_open_readonly(argv[1], &file, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_gguf_open(file, &gguf, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic);
    }
    if (status != KQ_STATUS_OK) {
        print_failure(status, &diagnostic);
        goto cleanup;
    }

    if (wcscmp(argv[2], L"encode") == 0 && argc == 4) {
        unsigned char *input;
        uint64_t input_length = 0U;
        uint64_t count = 0U;
        uint32_t *ids;
        kq_tokenizer_encode_options options = {KQ_TOKENIZER_SPECIAL_RECOGNIZE};
        input = parse_hex(argv[3], &input_length);
        if (input == NULL) {
            goto cleanup;
        }
        status = kq_tokenizer_encode(tokenizer, input, input_length, &options,
                                     NULL, 0U, &count, &diagnostic);
        ids = (uint32_t *)malloc(count == 0U ? 1U : (size_t)count * sizeof(*ids));
        if ((status != KQ_STATUS_BUFFER_TOO_SMALL &&
             !(status == KQ_STATUS_OK && count == 0U)) || ids == NULL) {
            free(input);
            free(ids);
            print_failure(status, &diagnostic);
            result = 2;
            goto cleanup;
        }
        status = kq_tokenizer_encode(tokenizer, input, input_length, &options,
                                     ids, count, &count, &diagnostic);
        free(input);
        if (status != KQ_STATUS_OK) {
            free(ids);
            print_failure(status, &diagnostic);
            result = 2;
            goto cleanup;
        }
        print_ids(ids, count);
        free(ids);
        result = 0;
    } else if (wcscmp(argv[2], L"decode") == 0 && argc == 5) {
        uint64_t count = 0U;
        uint64_t bytes = 0U;
        uint32_t *ids = parse_ids(argv[4], &count);
        unsigned char *output;
        kq_tokenizer_decode_options options;
        if (ids == NULL && argv[4][0] != L'\0') {
            goto cleanup;
        }
        options.special_policy = wcscmp(argv[3], L"skip") == 0 ?
            KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL :
            KQ_TOKENIZER_DECODE_KEEP_SPECIAL;
        status = kq_tokenizer_decode(tokenizer, ids, count, &options,
                                     NULL, 0U, &bytes, &diagnostic);
        output = (unsigned char *)malloc(bytes == 0U ? 1U : (size_t)bytes);
        if ((status != KQ_STATUS_BUFFER_TOO_SMALL &&
             !(status == KQ_STATUS_OK && bytes == 0U)) || output == NULL) {
            free(ids);
            free(output);
            print_failure(status, &diagnostic);
            result = 2;
            goto cleanup;
        }
        status = kq_tokenizer_decode(tokenizer, ids, count, &options,
                                     output, bytes, &bytes, &diagnostic);
        free(ids);
        if (status != KQ_STATUS_OK) {
            free(output);
            print_failure(status, &diagnostic);
            result = 2;
            goto cleanup;
        }
        print_hex(output, bytes);
        free(output);
        result = 0;
    } else if ((wcscmp(argv[2], L"chat-render") == 0 ||
                wcscmp(argv[2], L"chat-token") == 0) &&
               argc >= 6 && ((argc - 4) % 2) == 0) {
        uint64_t message_count = (uint64_t)(argc - 4) / 2U;
        kq_chat_message *messages =
            (kq_chat_message *)calloc((size_t)message_count, sizeof(*messages));
        unsigned char **contents =
            (unsigned char **)calloc((size_t)message_count, sizeof(*contents));
        kq_chat_options options = {wcscmp(argv[3], L"1") == 0, 0U};
        uint64_t index;
        int valid = 1;
        if (messages == NULL || contents == NULL) {
            free(messages);
            free(contents);
            goto cleanup;
        }
        for (index = 0U; index < message_count; ++index) {
            messages[index].role = parse_role(argv[4 + (int)index * 2], &valid);
            messages[index].content_type = KQ_CHAT_CONTENT_TEXT;
            contents[index] = parse_hex(argv[5 + (int)index * 2],
                                        &messages[index].content_length);
            messages[index].content = contents[index];
            if (!valid || contents[index] == NULL) {
                break;
            }
        }
        if (valid) {
            uint64_t required = 0U;
            if (wcscmp(argv[2], L"chat-render") == 0) {
                unsigned char *output;
                status = kq_chat_render(messages, message_count, &options,
                                        NULL, 0U, &required, &diagnostic);
                output = (unsigned char *)malloc(required == 0U ? 1U : (size_t)required);
                if ((status == KQ_STATUS_BUFFER_TOO_SMALL ||
                     (status == KQ_STATUS_OK && required == 0U)) && output != NULL) {
                    status = kq_chat_render(messages, message_count, &options,
                                            output, required, &required, &diagnostic);
                }
                if (status == KQ_STATUS_OK) {
                    print_hex(output, required);
                    result = 0;
                } else {
                    print_failure(status, &diagnostic);
                    result = 2;
                }
                free(output);
            } else {
                uint32_t *ids;
                status = kq_chat_tokenize(tokenizer, messages, message_count,
                                          &options, NULL, 0U, &required, &diagnostic);
                ids = (uint32_t *)malloc(required == 0U ? 1U :
                                         (size_t)required * sizeof(*ids));
                if ((status == KQ_STATUS_BUFFER_TOO_SMALL ||
                     (status == KQ_STATUS_OK && required == 0U)) && ids != NULL) {
                    status = kq_chat_tokenize(tokenizer, messages, message_count,
                                              &options, ids, required, &required,
                                              &diagnostic);
                }
                if (status == KQ_STATUS_OK) {
                    print_ids(ids, required);
                    result = 0;
                } else {
                    print_failure(status, &diagnostic);
                    result = 2;
                }
                free(ids);
            }
        }
        for (index = 0U; index < message_count; ++index) {
            free(contents[index]);
        }
        free(contents);
        free(messages);
    }

cleanup:
    kq_tokenizer_close(tokenizer);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
