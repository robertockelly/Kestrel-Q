#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_chat.h"
#include "kq_file.h"
#include "kq_gguf.h"
#include "kq_status.h"
#include "kq_tokenizer.h"
#include "kq_tokenizer_internal.h"

#define KQ_TEST_SKIP 77

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static wchar_t *load_model_path(void) {
    DWORD required = GetEnvironmentVariableW(L"KQ_GGUF_PATH", NULL, 0U);
    wchar_t *path;
    DWORD copied;
    if (required == 0U) {
        return NULL;
    }
    if ((uint64_t)required > (uint64_t)(SIZE_MAX / sizeof(*path))) {
        return NULL;
    }
    path = (wchar_t *)malloc((size_t)required * sizeof(*path));
    if (path == NULL) {
        return NULL;
    }
    copied = GetEnvironmentVariableW(L"KQ_GGUF_PATH", path, required);
    if (copied == 0U || copied >= required) {
        free(path);
        return NULL;
    }
    return path;
}

static int encode_equals(kq_tokenizer *tokenizer,
                         const unsigned char *text,
                         uint64_t text_length,
                         const uint32_t *expected,
                         uint64_t expected_count) {
    kq_tokenizer_encode_options options = {KQ_TOKENIZER_SPECIAL_RECOGNIZE};
    kq_diagnostic diagnostic;
    uint64_t required = 0U;
    uint32_t *actual = NULL;
    kq_status status = kq_tokenizer_encode(tokenizer,
                                           text,
                                           text_length,
                                           &options,
                                           NULL,
                                           0U,
                                           &required,
                                           &diagnostic);
    if (expected_count != 0U) {
        if (status != KQ_STATUS_BUFFER_TOO_SMALL || required != expected_count) {
            return 0;
        }
        actual = (uint32_t *)malloc((size_t)required * sizeof(*actual));
        if (actual == NULL) {
            return 0;
        }
    } else if (status != KQ_STATUS_OK || required != 0U) {
        return 0;
    }
    status = kq_tokenizer_encode(tokenizer,
                                 text,
                                 text_length,
                                 &options,
                                 actual,
                                 required,
                                 &required,
                                 &diagnostic);
    if (status != KQ_STATUS_OK || required != expected_count ||
        (required != 0U &&
         memcmp(actual, expected, (size_t)required * sizeof(*actual)) != 0)) {
        free(actual);
        return 0;
    }
    free(actual);
    return 1;
}

static int decode_equals(kq_tokenizer *tokenizer,
                         const uint32_t *tokens,
                         uint64_t token_count,
                         kq_tokenizer_decode_policy policy,
                         const char *expected) {
    kq_tokenizer_decode_options options = {policy};
    kq_diagnostic diagnostic;
    uint64_t required = 0U;
    unsigned char *actual;
    size_t expected_length = strlen(expected);
    kq_status status = kq_tokenizer_decode(tokenizer,
                                           tokens,
                                           token_count,
                                           &options,
                                           NULL,
                                           0U,
                                           &required,
                                           &diagnostic);
    if ((expected_length == 0U && status != KQ_STATUS_OK) ||
        (expected_length != 0U && status != KQ_STATUS_BUFFER_TOO_SMALL) ||
        required != (uint64_t)expected_length) {
        return 0;
    }
    actual = (unsigned char *)malloc(required == 0U ? 1U : (size_t)required);
    if (actual == NULL) {
        return 0;
    }
    status = kq_tokenizer_decode(tokenizer,
                                 tokens,
                                 token_count,
                                 &options,
                                 actual,
                                 required,
                                 &required,
                                 &diagnostic);
    if (status != KQ_STATUS_OK || required != (uint64_t)expected_length ||
        (required != 0U && memcmp(actual, expected, expected_length) != 0)) {
        free(actual);
        return 0;
    }
    free(actual);
    return 1;
}

static void test_chat(kq_tokenizer *tokenizer) {
    static const unsigned char system_text[] = "Answer with one concise sentence.";
    static const unsigned char user_text[] = "Explain why a byte has eight bits.";
    static const unsigned char expected_rendered[] =
        "<|im_start|>system\nAnswer with one concise sentence.<|im_end|>\n"
        "<|im_start|>user\nExplain why a byte has eight bits.<|im_end|>\n";
    static const uint32_t expected_tokens[] = {
        248045U, 8678U, 198U, 15666U, 440U, 799U, 61446U, 11316U, 13U,
        248046U, 198U, 248045U, 846U, 198U, 814U, 20139U, 3069U, 264U,
        4763U, 682U, 7810U, 9192U, 13U, 248046U, 198U
    };
    kq_chat_message messages[2];
    kq_chat_options options = {0, 0U};
    kq_diagnostic diagnostic;
    uint64_t required = 0U;
    unsigned char rendered[sizeof(expected_rendered) - 1U];
    uint32_t tokens[sizeof(expected_tokens) / sizeof(expected_tokens[0])];
    kq_status status;

    messages[0].role = KQ_CHAT_ROLE_SYSTEM;
    messages[0].content_type = KQ_CHAT_CONTENT_TEXT;
    messages[0].content = system_text;
    messages[0].content_length = sizeof(system_text) - 1U;
    messages[1].role = KQ_CHAT_ROLE_USER;
    messages[1].content_type = KQ_CHAT_CONTENT_TEXT;
    messages[1].content = user_text;
    messages[1].content_length = sizeof(user_text) - 1U;

    status = kq_chat_render(messages, 2U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL &&
              required == sizeof(expected_rendered) - 1U,
          "chat required-size query");
    status = kq_chat_render(messages,
                            2U,
                            &options,
                            rendered,
                            sizeof(rendered),
                            &required,
                            &diagnostic);
    check(status == KQ_STATUS_OK &&
              memcmp(rendered, expected_rendered, sizeof(rendered)) == 0,
          "canonical chat rendering");
    status = kq_chat_tokenize(tokenizer,
                              messages,
                              2U,
                              &options,
                              tokens,
                              sizeof(tokens) / sizeof(tokens[0]),
                              &required,
                              &diagnostic);
    check(status == KQ_STATUS_OK &&
              required == sizeof(expected_tokens) / sizeof(expected_tokens[0]) &&
              memcmp(tokens, expected_tokens, sizeof(tokens)) == 0,
          "canonical chat token IDs");

    messages[0].role = KQ_CHAT_ROLE_DEVELOPER;
    status = kq_chat_render(messages, 2U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT, "developer role fails closed");
    messages[0].role = KQ_CHAT_ROLE_SYSTEM;
    messages[1].content_type = KQ_CHAT_CONTENT_MULTIMODAL;
    status = kq_chat_render(messages, 2U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT, "multimodal chat fails closed");
    messages[1].content_type = KQ_CHAT_CONTENT_TEXT;
    options.reserved_flags = 1U;
    status = kq_chat_render(messages, 2U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_UNSUPPORTED_CHAT, "unknown chat option fails closed");
    options.reserved_flags = 0U;
    status = kq_chat_render(NULL, 0U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_MALFORMED_CHAT, "empty chat fails closed");
    messages[0].role = KQ_CHAT_ROLE_USER;
    messages[1].role = KQ_CHAT_ROLE_SYSTEM;
    status = kq_chat_render(messages, 2U, &options, NULL, 0U, &required, &diagnostic);
    check(status == KQ_STATUS_MALFORMED_CHAT, "late system role fails closed");
}

static void expect_mutation_rejected(
    const kq_gguf *gguf,
    const kq_tokenizer_test_mutation *mutation,
    const char *message) {
    kq_tokenizer *tokenizer = NULL;
    kq_diagnostic diagnostic;
    kq_status status = kq_tokenizer_open_from_gguf_for_test(
        gguf, mutation, &tokenizer, &diagnostic);
    check(status == KQ_STATUS_INCOMPATIBLE_TOKENIZER && tokenizer == NULL,
          message);
    kq_tokenizer_close(tokenizer);
}

int wmain(void) {
    static const unsigned char decomposed_e[] = {'e', 0xccU, 0x81U};
    static const uint32_t e_acute_token[] = {933U};
    static const unsigned char special_text[] =
        "<|fim_prefix|><|fim_middle|><|fim_suffix|><|fim_pad|><|repo_name|><|file_sep|>";
    static const uint32_t special_ids[] = {
        248060U, 248061U, 248062U, 248063U, 248064U, 248065U
    };
    static const unsigned char invalid_utf8[] = {0xc0U, 0xafU};
    static const unsigned char changed[] = "changed";
    wchar_t *path = load_model_path();
    kq_file *file = NULL;
    kq_gguf *gguf = NULL;
    kq_tokenizer *tokenizer = NULL;
    kq_diagnostic diagnostic;
    kq_status status;
    kq_tokenizer_encode_options encode_options = {KQ_TOKENIZER_SPECIAL_REJECT};
    kq_tokenizer_decode_options decode_options = {
        KQ_TOKENIZER_DECODE_KEEP_SPECIAL
    };
    kq_tokenizer_test_mutation mutation;
    const kq_gguf_metadata *tokens_metadata;
    kq_string_view duplicate_token = {0};
    const kq_tokenizer_metrics *metrics;
    uint64_t required = 0U;
    uint64_t payload_before;
    uint32_t padded_id = 248077U;
    unsigned char one_byte[1];

    if (path == NULL) {
        printf("KQ_GGUF_PATH unavailable; tokenizer integration skipped\n");
        return KQ_TEST_SKIP;
    }
    status = kq_file_open_readonly(path, &file, &diagnostic);
    free(path);
    if (status != KQ_STATUS_OK) {
        fprintf(stderr, "cannot open tokenizer integration artifact: %s\n",
                diagnostic.message);
        return 1;
    }
    status = kq_gguf_open(file, &gguf, &diagnostic);
    check(status == KQ_STATUS_OK, "open GGUF for tokenizer integration");
    if (status != KQ_STATUS_OK) {
        kq_file_close(file);
        return 1;
    }
    payload_before = kq_gguf_payload_bytes_accessed(gguf);
    status = kq_tokenizer_open_from_gguf(gguf, &tokenizer, &diagnostic);
    check(status == KQ_STATUS_OK && tokenizer != NULL,
          "construct canonical tokenizer from exact GGUF substrate");
    if (status == KQ_STATUS_OK) {
        metrics = kq_tokenizer_get_metrics(tokenizer);
        check(metrics != NULL && metrics->construction_nanoseconds > 0U &&
                  metrics->owned_heap_bytes > 0U &&
                  metrics->vocabulary_lookup_bytes > 0U &&
                  metrics->merge_lookup_bytes > 0U,
              "tokenizer construction metrics");
        check(strcmp(kq_tokenizer_nfc_unicode_version(), "9.0.0") == 0,
              "pinned canonical NFC Unicode version");
        check(strcmp(kq_tokenizer_property_unicode_version(), "16.0.0") == 0,
              "pinned canonical regex-property Unicode version");
        check(encode_equals(tokenizer,
                            decomposed_e,
                            sizeof(decomposed_e),
                            e_acute_token,
                            1U),
              "NFC canonical encode");
        check(encode_equals(tokenizer,
                            special_text,
                            sizeof(special_text) - 1U,
                            special_ids,
                            sizeof(special_ids) / sizeof(special_ids[0])),
              "canonical added-token IDs 248060-248065");
        check(decode_equals(tokenizer,
                            special_ids,
                            sizeof(special_ids) / sizeof(special_ids[0]),
                            KQ_TOKENIZER_DECODE_SKIP_CANONICAL_SPECIAL,
                            (const char *)special_text),
              "FIM/repository tokens survive skip-special decode");
        status = kq_tokenizer_encode(tokenizer,
                                     special_text,
                                     sizeof(special_text) - 1U,
                                     &encode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
              "special literal rejection policy");
        status = kq_tokenizer_encode(tokenizer,
                                     invalid_utf8,
                                     sizeof(invalid_utf8),
                                     &encode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_INVALID_UTF8, "invalid UTF-8 fails closed");
        status = kq_tokenizer_encode(tokenizer,
                                     (const unsigned char *)"x",
                                     UINT64_C(16) * 1024U * 1024U + 1U,
                                     &encode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_LIMIT_EXCEEDED,
              "oversized encode input fails before dereference");
        encode_options.special_policy = (kq_tokenizer_special_policy)99;
        status = kq_tokenizer_encode(tokenizer,
                                     (const unsigned char *)"x",
                                     1U,
                                     &encode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
              "unknown encode option fails closed");
        encode_options.special_policy = KQ_TOKENIZER_SPECIAL_REJECT;
        status = kq_tokenizer_decode(tokenizer,
                                     &padded_id,
                                     1U,
                                     &decode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_INVALID_TOKEN_ID,
              "padded token ID fails closed");
        status = kq_tokenizer_decode(tokenizer,
                                     e_acute_token,
                                     1U,
                                     &decode_options,
                                     one_byte,
                                     sizeof(one_byte),
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_BUFFER_TOO_SMALL && required == 2U,
              "decode capacity failure reports exact size without truncation");
        decode_options.special_policy = (kq_tokenizer_decode_policy)99;
        status = kq_tokenizer_decode(tokenizer,
                                     e_acute_token,
                                     1U,
                                     &decode_options,
                                     NULL,
                                     0U,
                                     &required,
                                     &diagnostic);
        check(status == KQ_STATUS_UNSUPPORTED_TOKENIZER_OPTION,
              "unknown decode option fails closed");
        decode_options.special_policy = KQ_TOKENIZER_DECODE_KEEP_SPECIAL;
        test_chat(tokenizer);
        printf("tokenizer_metrics construction_ns=%llu owned_heap=%llu "
               "vocab_lookup=%llu merge_lookup=%llu\n",
               (unsigned long long)metrics->construction_nanoseconds,
               (unsigned long long)metrics->owned_heap_bytes,
               (unsigned long long)metrics->vocabulary_lookup_bytes,
               (unsigned long long)metrics->merge_lookup_bytes);
    }
    kq_tokenizer_close(tokenizer);

    tokens_metadata = kq_gguf_find_metadata(gguf, "tokenizer.ggml.tokens");
    if (kq_gguf_metadata_array_string_at(tokens_metadata,
                                         1U,
                                         &duplicate_token)) {
        memset(&mutation, 0, sizeof(mutation));
        mutation.kind = KQ_TOKENIZER_TEST_MUTATION_TOKEN_STRING;
        mutation.index = 0U;
        mutation.replacement_string = duplicate_token;
        expect_mutation_rejected(
            gguf,
            &mutation,
            "duplicate string at a different positional ID rejected");
    } else {
        check(0, "read bounded duplicate-token mutation source");
    }

    memset(&mutation, 0, sizeof(mutation));
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_TOKEN_STRING;
    mutation.index = 0U;
    mutation.replacement_string.data = changed;
    mutation.replacement_string.length = sizeof(changed) - 1U;
    expect_mutation_rejected(gguf, &mutation, "changed vocabulary string rejected");
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_MERGE_STRING;
    mutation.index = 0U;
    expect_mutation_rejected(gguf, &mutation, "changed merge/rank rejected");
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_TOKEN_TYPE;
    mutation.index = 248060U;
    mutation.replacement_i32 = 4;
    expect_mutation_rejected(gguf, &mutation, "conflicting special type rejected");
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_TOKEN_STRING;
    mutation.index = 248060U;
    expect_mutation_rejected(gguf, &mutation, "missing canonical special token rejected");
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_TOKEN_COUNT;
    expect_mutation_rejected(gguf, &mutation, "wrong vocabulary count rejected");
    mutation.kind = KQ_TOKENIZER_TEST_MUTATION_MERGE_COUNT;
    expect_mutation_rejected(gguf, &mutation, "wrong merge count rejected");

    check(payload_before == 0U && kq_gguf_payload_bytes_accessed(gguf) == 0U,
          "tokenizer integration touches zero tensor payload bytes");
    kq_gguf_close(gguf);
    kq_file_close(file);
    if (failures != 0) {
        fprintf(stderr, "%d tokenizer integration checks failed\n", failures);
        return 1;
    }
    printf("tokenizer integration: PASS; payload_bytes_accessed=0\n");
    return 0;
}
