#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_model.h"
#include "kq_ple.h"
#include "kq_status.h"
#include "kq_tokenizer.h"

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
            uint64_t next = capacity == 0U ? 16U : capacity * 2U;
            if (next < capacity || next > SIZE_MAX / sizeof(*values)) {
                free(values);
                free(copy);
                return NULL;
            }
            grown = (uint32_t *)realloc(values,
                                        (size_t)next * sizeof(*values));
            if (grown == NULL) {
                free(values);
                free(copy);
                return NULL;
            }
            values = grown;
            capacity = next;
        }
        values[used++] = (uint32_t)value;
        item = wcstok_s(NULL, L",", &context);
    }
    free(copy);
    *count = used;
    return values;
}

static unsigned char *parse_hex(const wchar_t *text, uint64_t *length) {
    size_t chars = wcslen(text);
    unsigned char *bytes;
    size_t index;

    if ((chars & 1U) != 0U) {
        return NULL;
    }
    bytes = (unsigned char *)malloc(chars == 0U ? 1U : chars / 2U);
    if (bytes == NULL) {
        return NULL;
    }
    for (index = 0U; index < chars; index += 2U) {
        wchar_t high = text[index];
        wchar_t low = text[index + 1U];
        int high_value = high >= L'0' && high <= L'9' ? (int)(high - L'0') :
                         high >= L'a' && high <= L'f' ? (int)(high - L'a') + 10 :
                         high >= L'A' && high <= L'F' ? (int)(high - L'A') + 10 : -1;
        int low_value = low >= L'0' && low <= L'9' ? (int)(low - L'0') :
                        low >= L'a' && low <= L'f' ? (int)(low - L'a') + 10 :
                        low >= L'A' && low <= L'F' ? (int)(low - L'A') + 10 : -1;
        if (high_value < 0 || low_value < 0) {
            free(bytes);
            return NULL;
        }
        bytes[index / 2U] = (unsigned char)((high_value << 4) | low_value);
    }
    *length = (uint64_t)(chars / 2U);
    return bytes;
}

static void print_intents(const kq_ple_address_intent *intents,
                          uint64_t count) {
    uint64_t index;
    putchar('[');
    for (index = 0U; index < count; ++index) {
        const kq_ple_address_intent *intent = &intents[(size_t)index];
        if (index != 0U) {
            putchar(',');
        }
        printf("{\"position\":%llu,\"token_id\":%u,\"ngram_order\":%u,"
               "\"local_head\":%u,\"global_head\":%u,"
               "\"global_address\":%llu,\"head_offset\":%llu,"
               "\"head_vocab_size\":%llu,\"partition_id\":%u,"
               "\"partition_row\":%llu}",
               (unsigned long long)intent->position,
               (unsigned int)intent->token_id,
               (unsigned int)intent->ngram_order,
               (unsigned int)intent->local_head,
               (unsigned int)intent->global_head,
               (unsigned long long)intent->global_address,
               (unsigned long long)intent->head_offset,
               (unsigned long long)intent->head_vocab_size,
               (unsigned int)intent->logical_member,
               (unsigned long long)intent->member_row);
    }
    putchar(']');
}

static int generate(kq_ple_config *config,
                    const uint32_t *tokens,
                    uint64_t count,
                    kq_ple_stream_state *state,
                    kq_ple_address_intent **out_intents,
                    uint64_t *out_count,
                    kq_diagnostic *diagnostic) {
    kq_ple_address_intent *intents = NULL;
    uint64_t required = 0U;
    kq_status status;

    status = kq_ple_generate_prefill(config, state, tokens, count,
                                     NULL, 0U, &required, NULL, diagnostic);
    if (status != KQ_STATUS_BUFFER_TOO_SMALL &&
        !(status == KQ_STATUS_OK && required == 0U)) {
        return 0;
    }
    if (required != 0U) {
        if (required > SIZE_MAX / sizeof(*intents)) {
            return 0;
        }
        intents = (kq_ple_address_intent *)malloc(
            (size_t)required * sizeof(*intents));
        if (intents == NULL) {
            return 0;
        }
    }
    status = kq_ple_generate_prefill(config, state, tokens, count,
                                     intents, required, &required,
                                     NULL, diagnostic);
    if (status != KQ_STATUS_OK) {
        free(intents);
        return 0;
    }
    *out_intents = intents;
    *out_count = required;
    return 1;
}

int wmain(int argc, wchar_t **argv) {
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_model *model = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_ple_config *config = NULL;
    kq_ple_stream_state state;
    kq_diagnostic diagnostic;
    kq_status status;
    uint32_t *tokens = NULL;
    uint32_t *steps = NULL;
    uint64_t token_count = 0U;
    uint64_t step_count = 0U;
    kq_ple_address_intent *intents = NULL;
    uint64_t intent_count = 0U;
    int result = 1;

    if (argc < 4) {
        fprintf(stderr, "usage: ple-probe <gguf> <sequence|decode|text> ...\n");
        return 1;
    }
    status = kq_file_open_readonly(argv[1], &file, &diagnostic);
    if (status == KQ_STATUS_OK) {
        status = kq_gguf_open(file, &gguf, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_model_open_from_gguf(gguf, &model, &diagnostic);
    }
    if (status == KQ_STATUS_OK) {
        status = kq_ple_config_open_from_model(model, &config, &diagnostic);
    }
    if (status != KQ_STATUS_OK ||
        kq_ple_state_reset(config, &state, &diagnostic) != KQ_STATUS_OK) {
        fprintf(stderr, "PLE probe construction failed: %s\n", diagnostic.message);
        goto cleanup;
    }

    if (wcscmp(argv[2], L"text") == 0) {
        unsigned char *text;
        uint64_t text_length = 0U;
        uint64_t required = 0U;
        kq_tokenizer_encode_options options = {
            KQ_TOKENIZER_SPECIAL_RECOGNIZE
        };
        text = parse_hex(argv[3], &text_length);
        if (text == NULL ||
            kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic) !=
                KQ_STATUS_OK) {
            free(text);
            goto cleanup;
        }
        status = kq_tokenizer_encode(tokenizer, text, text_length, &options,
                                     NULL, 0U, &required, &diagnostic);
        free(text);
        if (status != KQ_STATUS_BUFFER_TOO_SMALL ||
            required > SIZE_MAX / sizeof(*tokens)) {
            goto cleanup;
        }
        tokens = (uint32_t *)malloc((size_t)required * sizeof(*tokens));
        if (tokens == NULL) {
            goto cleanup;
        }
        text = parse_hex(argv[3], &text_length);
        if (text == NULL) {
            goto cleanup;
        }
        status = kq_tokenizer_encode(tokenizer, text, text_length, &options,
                                     tokens, required, &token_count, &diagnostic);
        free(text);
        if (status != KQ_STATUS_OK || token_count != required) {
            goto cleanup;
        }
    } else {
        tokens = parse_ids(argv[3], &token_count);
        if (tokens == NULL && argv[3][0] != L'\0') {
            goto cleanup;
        }
    }

    if (!generate(config, tokens, token_count, &state,
                  &intents, &intent_count, &diagnostic)) {
        goto cleanup;
    }
    printf("{\"tokens\":[");
    for (uint64_t index = 0U; index < token_count; ++index) {
        printf("%s%u", index == 0U ? "" : ",", tokens[(size_t)index]);
    }
    printf("],\"intents\":");
    print_intents(intents, intent_count);

    if (wcscmp(argv[2], L"decode") == 0) {
        uint64_t step_index;
        if (argc != 5) {
            goto cleanup;
        }
        steps = parse_ids(argv[4], &step_count);
        if (steps == NULL && argv[4][0] != L'\0') {
            goto cleanup;
        }
        printf(",\"steps\":[");
        for (step_index = 0U; step_index < step_count; ++step_index) {
            kq_ple_address_intent step_intents[16];
            uint64_t required = 0U;
            if (kq_ple_generate_decode_step(config, &state,
                                            steps[(size_t)step_index],
                                            step_intents, 16U, &required,
                                            NULL, &diagnostic) != KQ_STATUS_OK) {
                goto cleanup;
            }
            if (step_index != 0U) {
                putchar(',');
            }
            printf("{\"token_id\":%u,\"intents\":",
                   steps[(size_t)step_index]);
            print_intents(step_intents, required);
            printf(",\"state\":{\"position\":%llu,\"history\":[%u,%u]}}",
                   (unsigned long long)state.position,
                   state.history[0], state.history[1]);
        }
        putchar(']');
    } else if (wcscmp(argv[2], L"sequence") != 0 &&
               wcscmp(argv[2], L"text") != 0) {
        goto cleanup;
    }
    printf(",\"state\":{\"position\":%llu,\"history\":[%u,%u]},"
           "\"payload_bytes_accessed\":%llu}\n",
           (unsigned long long)state.position,
           state.history[0], state.history[1],
           (unsigned long long)kq_gguf_payload_bytes_accessed(gguf));
    result = 0;

cleanup:
    free(intents);
    free(steps);
    free(tokens);
    kq_ple_config_close(config);
    kq_tokenizer_close(tokenizer);
    kq_model_close(model);
    kq_gguf_close(gguf);
    kq_file_close(file);
    return result;
}
