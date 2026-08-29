#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kq_ple.h"
#include "kq_ple_internal.h"
#include "kq_status.h"

static int failures = 0;

static void check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

static int intents_equal(const kq_ple_address_intent *left,
                         const kq_ple_address_intent *right,
                         uint64_t count) {
    uint64_t index;
    for (index = 0U; index < count; ++index) {
        if (left[index].position != right[index].position ||
            left[index].global_address != right[index].global_address ||
            left[index].head_offset != right[index].head_offset ||
            left[index].head_vocab_size != right[index].head_vocab_size ||
            left[index].member_row != right[index].member_row ||
            left[index].token_id != right[index].token_id ||
            left[index].ngram_order != right[index].ngram_order ||
            left[index].local_head != right[index].local_head ||
            left[index].global_head != right[index].global_head ||
            left[index].logical_member != right[index].logical_member) {
            return 0;
        }
    }
    return 1;
}

static void build_descriptor(kq_ple_compatibility_descriptor *descriptor) {
    static const uint64_t multipliers[KQ_PLE_NGRAM_SIZE] = {
        UINT64_C(23703573157769), UINT64_C(20109073645365),
        UINT64_C(8052911324071)
    };
    static const uint64_t offsets[KQ_PLE_HEAD_COUNT] = {
        0U, 20000003U, 40000026U, 60000059U,
        80000106U, 100000165U, 120000228U, 140000297U,
        160000374U, 180000455U, 200000548U, 220000655U,
        240000802U, 260000955U, 280001114U, 300001275U
    };
    static const uint64_t vocab_sizes[KQ_PLE_HEAD_COUNT] = {
        20000003U, 20000023U, 20000033U, 20000047U,
        20000059U, 20000063U, 20000069U, 20000077U,
        20000081U, 20000093U, 20000107U, 20000147U,
        20000153U, 20000159U, 20000161U, 20000171U
    };

    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->hidden_size = 2560U;
    descriptor->vocabulary_size = 248320U;
    descriptor->context_length = 262144U;
    descriptor->layer_count = 48U;
    descriptor->gdn_layer_count = 36U;
    descriptor->qsa_layer_count = 12U;
    descriptor->ple_layer_id = 1U;
    descriptor->ple_layer_type = KQ_MODEL_LAYER_GDN;
    descriptor->ngram_size = 3U;
    descriptor->heads_per_order = 8U;
    descriptor->eos_token_id = 248044U;
    descriptor->logical_member_count = 128U;
    descriptor->member_rows = UINT64_C(2500012);
    descriptor->table_width = 160U;
    descriptor->ple_table_semantic_count = 128U;
    descriptor->ple_dense_semantic_count = 6U;
    descriptor->ple_metadata_semantic_count = 3U;
    descriptor->table_semantics_valid = 1;
    descriptor->metadata_semantics_valid = 1;
    memcpy(descriptor->multipliers, multipliers, sizeof(multipliers));
    memcpy(descriptor->head_offsets, offsets, sizeof(offsets));
    memcpy(descriptor->head_vocab_sizes, vocab_sizes, sizeof(vocab_sizes));
}

static kq_ple_config *open_test_config(void) {
    kq_ple_compatibility_descriptor descriptor;
    kq_ple_config *config = NULL;
    kq_diagnostic diagnostic;
    kq_status status;

    build_descriptor(&descriptor);
    status = kq_ple_config_open_from_descriptor_for_test(
        &descriptor, &config, &diagnostic);
    check(status == KQ_STATUS_OK && config != NULL,
          "canonical PLE descriptor opens");
    return config;
}

static void expect_descriptor_failure(
    kq_ple_compatibility_descriptor *descriptor,
    const char *message) {
    kq_ple_config *config = NULL;
    kq_diagnostic diagnostic;
    kq_status status = kq_ple_config_open_from_descriptor_for_test(
        descriptor, &config, &diagnostic);
    check(status == KQ_STATUS_INCOMPATIBLE_PLE && config == NULL, message);
    kq_ple_config_close(config);
}

static void test_descriptor_failures(void) {
    kq_ple_compatibility_descriptor descriptor;
    kq_diagnostic diagnostic;
    kq_status status;

    status = kq_ple_config_open_from_descriptor_for_test(
        NULL, NULL, &diagnostic);
    check(status == KQ_STATUS_INVALID_ARGUMENT,
          "null descriptor/output fails closed");

    build_descriptor(&descriptor);
    descriptor.hidden_size = 2559U;
    expect_descriptor_failure(&descriptor, "wrong model identity fails closed");
    build_descriptor(&descriptor);
    descriptor.ple_layer_id = 2U;
    expect_descriptor_failure(&descriptor, "wrong PLE layer fails closed");
    build_descriptor(&descriptor);
    descriptor.ple_layer_type = KQ_MODEL_LAYER_QSA;
    expect_descriptor_failure(&descriptor, "wrong PLE layer type fails closed");
    build_descriptor(&descriptor);
    descriptor.ngram_size = 4U;
    expect_descriptor_failure(&descriptor, "wrong n-gram size fails closed");
    build_descriptor(&descriptor);
    descriptor.heads_per_order = 7U;
    expect_descriptor_failure(&descriptor, "wrong head count fails closed");
    build_descriptor(&descriptor);
    descriptor.logical_member_count = 127U;
    expect_descriptor_failure(&descriptor, "wrong member count fails closed");
    build_descriptor(&descriptor);
    descriptor.member_rows -= 1U;
    expect_descriptor_failure(&descriptor, "wrong row geometry fails closed");
    build_descriptor(&descriptor);
    descriptor.ple_table_semantic_count = 127U;
    expect_descriptor_failure(&descriptor, "missing table semantic fails closed");
    build_descriptor(&descriptor);
    descriptor.metadata_semantics_valid = 0;
    expect_descriptor_failure(&descriptor, "missing PLE metadata fails closed");
    build_descriptor(&descriptor);
    descriptor.multipliers[1] += 2U;
    expect_descriptor_failure(&descriptor, "changed multiplier fails closed");
    build_descriptor(&descriptor);
    descriptor.head_offsets[8] += 1U;
    expect_descriptor_failure(&descriptor, "changed head offset fails closed");
    build_descriptor(&descriptor);
    descriptor.head_vocab_sizes[15] -= 2U;
    expect_descriptor_failure(&descriptor, "changed head modulus fails closed");
    build_descriptor(&descriptor);
    descriptor.payload_bytes_accessed = 1U;
    expect_descriptor_failure(&descriptor, "payload boundary fails closed");
}

static void test_config_and_state(kq_ple_config *config) {
    const kq_ple_config_info *info = kq_ple_config_get_info(config);
    const kq_ple_config_metrics *metrics = kq_ple_config_get_metrics(config);
    kq_ple_stream_state state;
    kq_diagnostic diagnostic;
    kq_status status;

    check(info != NULL && info->model_vocabulary_size == 248320U &&
              info->model_context_length == 262144U &&
              info->ple_layer_id == 1U && info->eos_token_id == 248044U &&
              info->ngram_size == 3U && info->heads_per_order == 8U &&
              info->head_count == 16U &&
              info->logical_member_count == 128U &&
              info->member_rows == UINT64_C(2500012) &&
              info->active_rows == UINT64_C(320001446) &&
              info->padded_rows == UINT64_C(320001536) &&
              info->addresses_per_token == 16U,
          "immutable PLE config exposes canonical geometry");
    check(metrics != NULL && metrics->owned_heap_bytes > 0U &&
              metrics->stream_state_bytes == sizeof(kq_ple_stream_state),
          "PLE config exposes bounded ownership metrics");
    status = kq_ple_state_reset(config, &state, &diagnostic);
    check(status == KQ_STATUS_OK && state.position == 0U &&
              state.history[0] == 248044U && state.history[1] == 248044U,
          "PLE reset installs EOS sentinels");
    status = kq_ple_state_validate(config, &state, &diagnostic);
    check(status == KQ_STATUS_OK, "fresh PLE state validates");
    state.position ^= UINT64_C(1);
    status = kq_ple_state_validate(config, &state, &diagnostic);
    check(status == KQ_STATUS_INVALID_PLE_STATE,
          "corrupted PLE state fails integrity validation");
    check(kq_ple_state_reset(NULL, &state, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT &&
              kq_ple_state_reset(config, NULL, &diagnostic) ==
                  KQ_STATUS_INVALID_ARGUMENT,
          "null state/config arguments fail closed");
}

static void test_prefill_and_decode(kq_ple_config *config) {
    static const uint32_t tokens[] = {1U, 2U, 3U};
    static const uint32_t eos_tokens[] = {5U, 248044U, 7U, 8U};
    kq_ple_address_intent whole[48];
    kq_ple_address_intent incremental[48];
    kq_ple_address_intent eos_intents[64];
    kq_ple_stream_state state;
    kq_ple_stream_state before;
    kq_ple_stream_state replay;
    kq_ple_run_metrics metrics;
    kq_diagnostic diagnostic;
    kq_status status;
    uint64_t required = 0U;
    uint64_t index;
    uint32_t invalid_token = 248320U;

    check(kq_ple_generate_prefill(NULL, &state, tokens, 3U,
                                  whole, 48U, &required,
                                  NULL, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT,
          "null PLE config fails closed");
    check(kq_ple_generate_prefill(config, NULL, tokens, 3U,
                                  whole, 48U, &required,
                                  NULL, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT,
          "null PLE state fails closed");
    check(kq_ple_generate_prefill(config, &state, tokens, 3U,
                                  whole, 48U, NULL,
                                  NULL, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT,
          "null required-count output fails closed");

    check(kq_ple_state_reset(config, &state, &diagnostic) == KQ_STATUS_OK,
          "reset before one-shot prefill");
    status = kq_ple_generate_prefill(config, &state, tokens, 3U,
                                     whole, 48U, &required,
                                     &metrics, &diagnostic);
    check(status == KQ_STATUS_OK && required == 48U &&
              metrics.tokens_processed == 3U &&
              metrics.addresses_emitted == 48U && state.position == 3U &&
              state.history[0] == 2U && state.history[1] == 3U,
          "one-shot prefill emits 16 addresses per token");
    check(whole[0].global_address == UINT64_C(16121432) &&
              whole[0].logical_member == 6U &&
              whole[0].member_row == UINT64_C(1121360) &&
              whole[15].global_address == UINT64_C(305959965) &&
              whole[15].logical_member == 122U &&
              whole[15].member_row == UINT64_C(958501) &&
              whole[32].global_address == UINT64_C(14605717) &&
              whole[47].global_address == UINT64_C(309645223),
          "known independent length-3 golden boundary addresses match");
    for (index = 0U; index < 48U; ++index) {
        check(whole[index].global_head == (uint32_t)(index % 16U) &&
                  whole[index].ngram_order ==
                      ((index % 16U) < 8U ? 2U : 3U),
              "canonical intent ordering is bigram then trigram");
    }

    check(kq_ple_state_reset(config, &state, &diagnostic) == KQ_STATUS_OK,
          "reset before incremental equivalence");
    for (index = 0U; index < 3U; ++index) {
        status = kq_ple_generate_decode_step(
            config, &state, tokens[index], incremental + index * 16U,
            16U, &required, NULL, &diagnostic);
        check(status == KQ_STATUS_OK && required == 16U,
              "incremental decode step succeeds");
    }
    check(intents_equal(whole, incremental, 48U),
          "incremental decode equals full recomputation");

    check(kq_ple_state_reset(config, &state, &diagnostic) == KQ_STATUS_OK,
          "reset before capacity transaction");
    before = state;
    status = kq_ple_generate_prefill(config, &state, tokens, 3U,
                                     NULL, 0U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL && required == 48U &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "size query reports exact capacity without state advance");
    status = kq_ple_generate_prefill(config, &state, tokens, 3U,
                                     whole, 47U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "short output fails transactionally");
    status = kq_ple_generate_prefill(config, &state, &invalid_token, 1U,
                                     whole, 16U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_INVALID_TOKEN_ID &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "invalid token fails transactionally");
    status = kq_ple_generate_prefill(
        config, &state, tokens,
        UINT64_MAX / KQ_PLE_ADDRESSES_PER_TOKEN + 1U,
        whole, 48U, &required, NULL, &diagnostic);
    check(status == KQ_STATUS_ARITHMETIC_OVERFLOW &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "intent-count overflow fails before input access");
    status = kq_ple_generate_prefill(config, &state, tokens, 262145U,
                                     whole, 48U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_LIMIT_EXCEEDED &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "context overflow fails before input access");
    status = kq_ple_generate_prefill(config, &state, NULL, 0U,
                                     NULL, 0U, &required,
                                     &metrics, &diagnostic);
    check(status == KQ_STATUS_OK && required == 0U &&
              metrics.tokens_processed == 0U &&
              memcmp(&state, &before, sizeof(state)) == 0,
          "empty prefill is a valid no-op");

    check(kq_ple_state_reset(config, &state, &diagnostic) == KQ_STATUS_OK,
          "reset before EOS segment case");
    status = kq_ple_generate_prefill(config, &state, eos_tokens, 4U,
                                     eos_intents, 64U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_OK &&
              eos_intents[32].token_id == 7U &&
              state.history[0] == 7U && state.history[1] == 8U,
          "EOS segment transition preserves raw bounded history");
    replay = state;
    check(kq_ple_state_reset(config, &state, &diagnostic) == KQ_STATUS_OK,
          "reset before replay");
    status = kq_ple_generate_prefill(config, &state, eos_tokens, 4U,
                                     incremental, 48U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_BUFFER_TOO_SMALL,
          "replay also enforces exact output capacity");
    status = kq_ple_generate_prefill(config, &state, eos_tokens, 4U,
                                     eos_intents, 64U, &required,
                                     NULL, &diagnostic);
    check(status == KQ_STATUS_OK && memcmp(&state, &replay, sizeof(state)) == 0,
          "reset/replay produces identical committed state");
}

int main(void) {
    kq_ple_config *config;
    kq_diagnostic diagnostic;
    kq_ple_config *invalid = NULL;

    _Static_assert(sizeof(kq_ple_stream_state) == 32U,
                   "PLE stream state must remain explicitly bounded");
    test_descriptor_failures();
    check(kq_ple_config_open_from_model(NULL, &invalid, &diagnostic) ==
              KQ_STATUS_INVALID_ARGUMENT && invalid == NULL,
          "null model construction fails closed");
    config = open_test_config();
    if (config != NULL) {
        test_config_and_state(config);
        test_prefill_and_decode(config);
    }
    kq_ple_config_close(config);
    if (failures != 0) {
        fprintf(stderr, "%d PLE assertion(s) failed\n", failures);
        return 1;
    }
    printf("PLE synthetic/fail-closed tests: PASS\n");
    return 0;
}
