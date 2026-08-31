// Research/test-only full-model oracle for Task 2.12.
// SPDX-License-Identifier: Apache-2.0

#include "ggml-backend.h"
#include "llama.h"

#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct options {
    std::string model;
    std::string output;
    std::vector<llama_token> tokens;
    uint32_t context = 8;
    uint32_t top_n = 20;
    int32_t threads = 0;
};

struct layer_summary {
    std::string name;
    int64_t elements = 0;
    double sum = 0.0;
    double squared_sum = 0.0;
    float minimum = 0.0f;
    float maximum = 0.0f;
    std::vector<float> first;
};

struct checkpoint_collector {
    std::vector<layer_summary> layers;
};

bool collect_layer_output(ggml_tensor * tensor, bool ask, void * user_data) {
    constexpr const char * prefix = "l_last";
    const size_t prefix_length = std::strlen(prefix);
    const char * suffix = tensor->name + prefix_length;
    char * suffix_end = nullptr;
    if (std::strncmp(tensor->name, prefix, prefix_length) != 0 ||
        *suffix != '-' ||
        std::strtol(suffix + 1, &suffix_end, 10) < 0 ||
        suffix_end == suffix + 1 || *suffix_end != '\0') {
        return ask ? false : true;
    }
    if (ask) return true;
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
        std::fprintf(stderr, "oracle: unexpected layer checkpoint %s type=%s\n",
                     tensor->name, ggml_type_name(tensor->type));
        return false;
    }
    auto * collector = static_cast<checkpoint_collector *>(user_data);
    if (tensor->ne[0] > std::numeric_limits<int64_t>::max() / tensor->ne[1]) {
        std::fprintf(stderr, "oracle: layer checkpoint element count overflows\n");
        return false;
    }
    const size_t count = static_cast<size_t>(tensor->ne[0] * tensor->ne[1]);
    std::vector<float> values(count);
    const size_t offset = static_cast<size_t>(tensor->ne[2] - 1) * tensor->nb[2];
    ggml_backend_tensor_get(tensor, values.data(), offset, count * sizeof(float));
    layer_summary summary;
    summary.name = tensor->name;
    summary.elements = tensor->ne[0] * tensor->ne[1];
    summary.minimum = values[0];
    summary.maximum = values[0];
    for (float value : values) {
        summary.sum += value;
        summary.squared_sum += static_cast<double>(value) * value;
        summary.minimum = std::min(summary.minimum, value);
        summary.maximum = std::max(summary.maximum, value);
    }
    summary.first.assign(values.begin(), values.begin() + std::min<size_t>(8, values.size()));
    collector->layers.push_back(std::move(summary));
    return true;
}

void usage(const char * program) {
    std::fprintf(stderr,
        "usage: %s --model FILE --tokens ID,ID,... [--context N] "
        "[--top-n N] [--threads N] [--output FILE]\n",
        program);
}

bool parse_u32(const char * text, uint32_t & value) {
    char * end = nullptr;
    unsigned long parsed;
    errno = 0;
    parsed = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_i32(const char * text, int32_t & value) {
    char * end = nullptr;
    long parsed;
    errno = 0;
    parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    value = static_cast<int32_t>(parsed);
    return true;
}

bool parse_tokens(const char * text, std::vector<llama_token> & tokens) {
    const char * cursor = text;
    tokens.clear();
    while (*cursor != '\0') {
        char * end = nullptr;
        long parsed;
        errno = 0;
        parsed = std::strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || parsed < 0 ||
            parsed > std::numeric_limits<llama_token>::max()) {
            return false;
        }
        tokens.push_back(static_cast<llama_token>(parsed));
        if (*end == '\0') {
            break;
        }
        if (*end != ',') {
            return false;
        }
        cursor = end + 1;
        if (*cursor == '\0') {
            return false;
        }
    }
    return !tokens.empty();
}

bool parse_options(int argc, char ** argv, options & out) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            out.model = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            out.output = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            if (!parse_tokens(argv[++i], out.tokens)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], out.context) || out.context == 0) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], out.top_n) || out.top_n == 0) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], out.threads)) {
                return false;
            }
        } else {
            return false;
        }
    }
    if (out.threads == 0) {
        const unsigned int detected = std::thread::hardware_concurrency();
        out.threads = detected == 0 ? 1 : static_cast<int32_t>(detected);
    }
    return !out.model.empty() && !out.tokens.empty() &&
           out.tokens.size() <= out.context;
}

bool progress(float progress, void *) {
    const unsigned int percent = static_cast<unsigned int>(progress * 100.0f);
    std::fprintf(stderr, "oracle-load: %u%%\r", percent);
    std::fflush(stderr);
    return true;
}

uint64_t elapsed_ms(const LARGE_INTEGER & start,
                    const LARGE_INTEGER & end,
                    const LARGE_INTEGER & frequency) {
    if (frequency.QuadPart <= 0 || end.QuadPart < start.QuadPart) {
        return 0;
    }
    return static_cast<uint64_t>(
        (end.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart);
}

void print_json_string(const std::string & value) {
    std::putchar('"');
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': std::fputs("\\\\", stdout); break;
            case '"':  std::fputs("\\\"", stdout); break;
            case '\b': std::fputs("\\b", stdout); break;
            case '\f': std::fputs("\\f", stdout); break;
            case '\n': std::fputs("\\n", stdout); break;
            case '\r': std::fputs("\\r", stdout); break;
            case '\t': std::fputs("\\t", stdout); break;
            default:
                if (ch < 0x20) {
                    std::fprintf(stdout, "\\u%04x", static_cast<unsigned int>(ch));
                } else {
                    std::putchar(ch);
                }
        }
    }
    std::putchar('"');
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    PROCESS_MEMORY_COUNTERS_EX memory{};
    LARGE_INTEGER frequency{};
    LARGE_INTEGER load_start{};
    LARGE_INTEGER load_end{};
    LARGE_INTEGER decode_start{};
    LARGE_INTEGER decode_end{};
    int exit_code = 1;
    checkpoint_collector checkpoints;

    if (!parse_options(argc, argv, opts)) {
        usage(argv[0]);
        return 2;
    }
    if (!opts.output.empty()) {
        FILE * output_file = nullptr;
        if (freopen_s(&output_file, opts.output.c_str(), "wb", stdout) != 0 ||
            output_file == nullptr) {
            std::fprintf(stderr, "oracle: could not open output file\n");
            return 2;
        }
    }

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&load_start);
    std::fprintf(stderr, "oracle-stage: load-backends\n");
    std::fflush(stderr);
    ggml_backend_load_all();
    std::fprintf(stderr, "oracle-stage: load-model\n");
    std::fflush(stderr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
    model_params.vocab_only = false;
    model_params.check_tensors = false;
    model_params.use_extra_bufts = false;
    model_params.no_alloc = false;
    model_params.load_mtp = false;
    model_params.progress_callback = progress;

    model = llama_model_load_from_file(opts.model.c_str(), model_params);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "oracle-stage: model-load-returned\n");
    std::fflush(stderr);
    QueryPerformanceCounter(&load_end);
    if (model == nullptr) {
        std::fprintf(stderr, "oracle: model load failed\n");
        goto cleanup;
    }

    {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocabulary = llama_vocab_n_tokens(vocab);
        if (vocabulary <= 0 || opts.top_n > static_cast<uint32_t>(vocabulary)) {
            std::fprintf(stderr, "oracle: invalid vocabulary/top-N geometry\n");
            goto cleanup;
        }
        for (llama_token token : opts.tokens) {
            if (token < 0 || token >= vocabulary) {
                std::fprintf(stderr, "oracle: input token is outside vocabulary\n");
                goto cleanup;
            }
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = opts.context;
        context_params.n_batch = static_cast<uint32_t>(opts.tokens.size());
        context_params.n_ubatch = static_cast<uint32_t>(opts.tokens.size());
        context_params.n_seq_max = 1;
        context_params.n_threads = opts.threads;
        context_params.n_threads_batch = opts.threads;
        context_params.embeddings = false;
        context_params.offload_kqv = false;
        context_params.no_perf = false;
        context_params.op_offload = false;
        context_params.cb_eval = collect_layer_output;
        context_params.cb_eval_user_data = &checkpoints;

        context = llama_init_from_model(model, context_params);
        std::fprintf(stderr, "oracle-stage: context-create-returned\n");
        std::fflush(stderr);
        if (context == nullptr) {
            std::fprintf(stderr, "oracle: context creation failed\n");
            goto cleanup;
        }

        llama_batch batch = llama_batch_get_one(
            opts.tokens.data(), static_cast<int32_t>(opts.tokens.size()));
        QueryPerformanceCounter(&decode_start);
        std::fprintf(stderr, "oracle-stage: decode-begin\n");
        std::fflush(stderr);
        const int decode_result = llama_decode(context, batch);
        std::fprintf(stderr, "oracle-stage: decode-returned\n");
        std::fflush(stderr);
        QueryPerformanceCounter(&decode_end);
        if (decode_result != 0) {
            std::fprintf(stderr, "oracle: decode failed with code %d\n", decode_result);
            goto cleanup;
        }

        float * logits = llama_get_logits_ith(context, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "oracle: final-token logits unavailable\n");
            goto cleanup;
        }

        std::vector<std::pair<float, llama_token>> ranking;
        ranking.reserve(static_cast<size_t>(vocabulary));
        llama_token greedy = 0;
        for (llama_token token = 0; token < vocabulary; ++token) {
            if (!std::isfinite(logits[token])) {
                std::fprintf(stderr, "oracle: non-finite logit at token %d\n", token);
                goto cleanup;
            }
            if (token == 0 || logits[token] > logits[greedy]) {
                greedy = token;
            }
            ranking.emplace_back(logits[token], token);
        }
        std::partial_sort(
            ranking.begin(), ranking.begin() + opts.top_n, ranking.end(),
            [](const auto & left, const auto & right) {
                if (left.first != right.first) {
                    return left.first > right.first;
                }
                return left.second < right.second;
            });

        std::string piece;
        {
            char piece_buffer[256];
            int32_t required = llama_token_to_piece(
                vocab, greedy, piece_buffer, sizeof(piece_buffer), 0, true);
            if (required >= 0) {
                piece.assign(piece_buffer, static_cast<size_t>(required));
            } else {
                std::vector<char> buffer(static_cast<size_t>(-required));
                required = llama_token_to_piece(
                    vocab, greedy, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
                if (required < 0) {
                    std::fprintf(stderr, "oracle: selected token piece decode failed\n");
                    goto cleanup;
                }
                piece.assign(buffer.data(), static_cast<size_t>(required));
            }
        }

        memory.cb = sizeof(memory);
        if (!GetProcessMemoryInfo(GetCurrentProcess(),
                                  reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&memory),
                                  sizeof(memory))) {
            std::memset(&memory, 0, sizeof(memory));
        }

        std::printf("{\n");
        std::printf("  \"schema_version\": 1,\n");
        std::printf("  \"oracle_revision\": \"90c26fcd4b2114b4aa39d09d69318cb8f438d27a\",\n");
        std::printf("  \"load_mode\": \"MMAP\",\n");
        std::printf("  \"tensor_read_lazy\": \"ON\",\n");
        std::printf("  \"gpu_layers\": 0,\n");
        std::printf("  \"context_capacity\": %u,\n", opts.context);
        std::printf("  \"threads\": %d,\n", opts.threads);
        std::printf("  \"vocabulary_size\": %d,\n", vocabulary);
        std::printf("  \"input_token_ids\": [");
        for (size_t index = 0; index < opts.tokens.size(); ++index) {
            std::printf("%s%d", index == 0 ? "" : ", ", opts.tokens[index]);
        }
        std::printf("],\n");
        std::printf("  \"selected_token_id\": %d,\n", greedy);
        std::printf("  \"selected_token_logit\": %.9g,\n", logits[greedy]);
        std::printf("  \"selected_token_piece\": ");
        print_json_string(piece);
        std::printf(",\n");
        std::printf("  \"selected_token_is_eog\": %s,\n",
                    llama_vocab_is_eog(vocab, greedy) ? "true" : "false");
        std::printf("  \"top_logits\": [\n");
        for (uint32_t index = 0; index < opts.top_n; ++index) {
            std::printf("    {\"token_id\": %d, \"logit\": %.9g}%s\n",
                        ranking[index].second, ranking[index].first,
                        index + 1 == opts.top_n ? "" : ",");
        }
        std::printf("  ],\n");
        std::printf("  \"layer_checkpoints\": [\n");
        for (size_t i = 0; i < checkpoints.layers.size(); ++i) {
            const auto & checkpoint = checkpoints.layers[i];
            std::printf("    {\"name\": ");
            print_json_string(checkpoint.name);
            std::printf(", \"elements\": %lld, \"sum\": %.17g, "
                        "\"squared_sum\": %.17g, \"minimum\": %.9g, "
                        "\"maximum\": %.9g, \"first\": [",
                        static_cast<long long>(checkpoint.elements), checkpoint.sum,
                        checkpoint.squared_sum, checkpoint.minimum, checkpoint.maximum);
            for (size_t j = 0; j < checkpoint.first.size(); ++j) {
                if (j != 0) std::printf(", ");
                std::printf("%.9g", checkpoint.first[j]);
            }
            std::printf("]}%s\n", i + 1 == checkpoints.layers.size() ? "" : ",");
        }
        std::printf("  ],\n");
        std::printf("  \"load_elapsed_ms\": %llu,\n",
                    static_cast<unsigned long long>(
                        elapsed_ms(load_start, load_end, frequency)));
        std::printf("  \"decode_elapsed_ms\": %llu,\n",
                    static_cast<unsigned long long>(
                        elapsed_ms(decode_start, decode_end, frequency)));
        std::printf("  \"peak_working_set_bytes\": %llu,\n",
                    static_cast<unsigned long long>(memory.PeakWorkingSetSize));
        std::printf("  \"peak_pagefile_bytes\": %llu,\n",
                    static_cast<unsigned long long>(memory.PeakPagefileUsage));
        std::printf("  \"private_usage_bytes\": %llu\n",
                    static_cast<unsigned long long>(memory.PrivateUsage));
        std::printf("}\n");
        std::fflush(stdout);
        exit_code = 0;
    }

cleanup:
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
