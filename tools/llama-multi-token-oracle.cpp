// Research/test-only incremental greedy oracle for Task 2.13.
// SPDX-License-Identifier: Apache-2.0

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
    uint32_t context = 16;
    uint32_t max_new_tokens = 4;
    uint32_t top_n = 10;
    int32_t threads = 0;
};

struct ranked_logit {
    llama_token token;
    float logit;
};

struct generated_step {
    uint64_t position;
    llama_token token;
    float logit;
    bool is_eog;
    std::string piece;
    std::vector<ranked_logit> top;
    uint64_t elapsed_ms;
};

bool parse_u32(const char *text, uint32_t &value, bool allow_zero) {
    char *end = nullptr;
    unsigned long parsed;
    errno = 0;
    parsed = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max() ||
        (!allow_zero && parsed == 0)) {
        return false;
    }
    value = static_cast<uint32_t>(parsed);
    return true;
}

bool parse_i32(const char *text, int32_t &value) {
    char *end = nullptr;
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

bool parse_tokens(const char *text, std::vector<llama_token> &tokens) {
    const char *cursor = text;
    tokens.clear();
    while (*cursor != '\0') {
        char *end = nullptr;
        long parsed;
        errno = 0;
        parsed = std::strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || parsed < 0 ||
            parsed > std::numeric_limits<llama_token>::max()) {
            return false;
        }
        tokens.push_back(static_cast<llama_token>(parsed));
        if (*end == '\0') break;
        if (*end != ',') return false;
        cursor = end + 1;
        if (*cursor == '\0') return false;
    }
    return !tokens.empty();
}

bool parse_options(int argc, char **argv, options &out) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            out.model = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            out.output = argv[++i];
        } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            if (!parse_tokens(argv[++i], out.tokens)) return false;
        } else if (std::strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], out.context, false)) return false;
        } else if (std::strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], out.max_new_tokens, true)) return false;
        } else if (std::strcmp(argv[i], "--top-n") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], out.top_n, false)) return false;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            if (!parse_i32(argv[++i], out.threads)) return false;
        } else {
            return false;
        }
    }
    if (out.threads == 0) {
        const unsigned int detected = std::thread::hardware_concurrency();
        out.threads = detected == 0 ? 1 : static_cast<int32_t>(detected);
    }
    return !out.model.empty() && !out.tokens.empty() &&
           out.tokens.size() <= out.context &&
           out.max_new_tokens <= out.context - out.tokens.size();
}

bool progress(float value, void *) {
    std::fprintf(stderr, "oracle-load: %u%%\r",
                 static_cast<unsigned int>(value * 100.0f));
    std::fflush(stderr);
    return true;
}

uint64_t elapsed_ms(const LARGE_INTEGER &start, const LARGE_INTEGER &end,
                    const LARGE_INTEGER &frequency) {
    if (frequency.QuadPart <= 0 || end.QuadPart < start.QuadPart) return 0;
    return static_cast<uint64_t>(
        (end.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart);
}

void print_json_string(const std::string &value) {
    std::putchar('"');
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': std::fputs("\\\\", stdout); break;
            case '"': std::fputs("\\\"", stdout); break;
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

bool decode_batch(llama_context *context, const llama_token *tokens,
                  uint32_t count, uint64_t start_position) {
    llama_batch batch = llama_batch_init(static_cast<int32_t>(count), 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr ||
        batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
        batch.logits == nullptr) {
        llama_batch_free(batch);
        return false;
    }
    batch.n_tokens = static_cast<int32_t>(count);
    for (uint32_t index = 0; index < count; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = static_cast<llama_pos>(start_position + index);
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index + 1U == count ? 1 : 0;
    }
    const int result = llama_decode(context, batch);
    llama_batch_free(batch);
    return result == 0;
}

bool token_piece(const llama_vocab *vocab, llama_token token,
                 std::string &piece) {
    char local[256];
    int32_t result = llama_token_to_piece(vocab, token, local,
                                          sizeof(local), 0, true);
    if (result >= 0) {
        piece.assign(local, static_cast<size_t>(result));
        return true;
    }
    std::vector<char> buffer(static_cast<size_t>(-result));
    result = llama_token_to_piece(vocab, token, buffer.data(),
                                  static_cast<int32_t>(buffer.size()), 0, true);
    if (result < 0) return false;
    piece.assign(buffer.data(), static_cast<size_t>(result));
    return true;
}

} // namespace

int main(int argc, char **argv) {
    options opts;
    llama_model *model = nullptr;
    llama_context *context = nullptr;
    int exit_code = 1;
    LARGE_INTEGER frequency{}, load_start{}, load_end{};
    PROCESS_MEMORY_COUNTERS_EX memory{};

    if (!parse_options(argc, argv, opts)) {
        std::fprintf(stderr, "usage: %s --model FILE --tokens ID,... "
            "[--context N] [--max-new-tokens N] [--top-n N] "
            "[--threads N] [--output FILE]\n", argv[0]);
        return 2;
    }
    if (!opts.output.empty()) {
        FILE *stream = nullptr;
        if (freopen_s(&stream, opts.output.c_str(), "wb", stdout) != 0 ||
            stream == nullptr) {
            std::fprintf(stderr, "oracle: could not open output file\n");
            return 2;
        }
    }

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&load_start);
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.tensor_read_lazy = LLAMA_TENSOR_READ_LAZY_ON;
    model_params.load_mtp = false;
    model_params.progress_callback = progress;
    model = llama_model_load_from_file(opts.model.c_str(), model_params);
    QueryPerformanceCounter(&load_end);
    std::fprintf(stderr, "\n");
    if (model == nullptr) {
        std::fprintf(stderr, "oracle: model load failed\n");
        goto cleanup;
    }

    {
        const llama_vocab *vocab = llama_model_get_vocab(model);
        const int32_t vocabulary = llama_vocab_n_tokens(vocab);
        if (vocabulary <= 0 || opts.top_n > static_cast<uint32_t>(vocabulary)) {
            std::fprintf(stderr, "oracle: invalid vocabulary/top-N geometry\n");
            goto cleanup;
        }
        for (llama_token token : opts.tokens) {
            if (token < 0 || token >= vocabulary) {
                std::fprintf(stderr, "oracle: input token outside vocabulary\n");
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
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            std::fprintf(stderr, "oracle: context creation failed\n");
            goto cleanup;
        }

        LARGE_INTEGER prompt_start{}, prompt_end{};
        QueryPerformanceCounter(&prompt_start);
        if (!decode_batch(context, opts.tokens.data(),
                          static_cast<uint32_t>(opts.tokens.size()), 0U)) {
            std::fprintf(stderr, "oracle: prompt decode failed\n");
            goto cleanup;
        }
        QueryPerformanceCounter(&prompt_end);

        std::vector<generated_step> steps;
        for (uint32_t step_index = 0; step_index < opts.max_new_tokens;
             ++step_index) {
            float *logits = llama_get_logits_ith(context, -1);
            if (logits == nullptr) {
                std::fprintf(stderr, "oracle: logits unavailable at step %u\n",
                             step_index);
                goto cleanup;
            }
            std::vector<std::pair<float, llama_token>> ranking;
            ranking.reserve(static_cast<size_t>(vocabulary));
            for (llama_token token = 0; token < vocabulary; ++token) {
                if (!std::isfinite(logits[token])) {
                    std::fprintf(stderr, "oracle: non-finite logit at token %d\n", token);
                    goto cleanup;
                }
                ranking.emplace_back(logits[token], token);
            }
            std::partial_sort(ranking.begin(), ranking.begin() + opts.top_n,
                              ranking.end(), [](const auto &left, const auto &right) {
                if (left.first != right.first) return left.first > right.first;
                return left.second < right.second;
            });
            generated_step step{};
            step.position = opts.tokens.size() + step_index;
            step.token = ranking[0].second;
            step.logit = ranking[0].first;
            step.is_eog = llama_vocab_is_eog(vocab, step.token);
            if (!token_piece(vocab, step.token, step.piece)) {
                std::fprintf(stderr, "oracle: token piece decode failed\n");
                goto cleanup;
            }
            for (uint32_t rank = 0; rank < opts.top_n; ++rank) {
                step.top.push_back({ranking[rank].second, ranking[rank].first});
            }
            steps.push_back(step);
            if (step.is_eog || step_index + 1U == opts.max_new_tokens) break;

            LARGE_INTEGER step_start{}, step_end{};
            QueryPerformanceCounter(&step_start);
            if (!decode_batch(context, &step.token, 1U, step.position)) {
                std::fprintf(stderr, "oracle: incremental decode failed at step %u\n",
                             step_index + 1U);
                goto cleanup;
            }
            QueryPerformanceCounter(&step_end);
            steps.back().elapsed_ms = elapsed_ms(step_start, step_end, frequency);
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
        std::printf("  \"execution\": \"incremental_greedy\",\n");
        std::printf("  \"load_mode\": \"MMAP\",\n");
        std::printf("  \"tensor_read_lazy\": \"ON\",\n");
        std::printf("  \"gpu_layers\": 0,\n");
        std::printf("  \"context_capacity\": %u,\n", opts.context);
        std::printf("  \"max_new_tokens\": %u,\n", opts.max_new_tokens);
        std::printf("  \"prompt_prefill_count\": 1,\n");
        std::printf("  \"incremental_decode_count\": %zu,\n",
                    steps.empty() ? 0U : steps.size() - 1U);
        std::printf("  \"input_token_ids\": [");
        for (size_t index = 0; index < opts.tokens.size(); ++index) {
            std::printf("%s%d", index == 0 ? "" : ", ", opts.tokens[index]);
        }
        std::printf("],\n");
        std::printf("  \"generated_steps\": [\n");
        for (size_t index = 0; index < steps.size(); ++index) {
            const generated_step &step = steps[index];
            std::printf("    {\"step\": %zu, \"position\": %llu, \"token_id\": %d, "
                        "\"logit\": %.9g, \"is_eog\": %s, \"piece\": ",
                        index + 1U, static_cast<unsigned long long>(step.position),
                        step.token, step.logit, step.is_eog ? "true" : "false");
            print_json_string(step.piece);
            std::printf(", \"decode_elapsed_ms\": %llu, \"top_logits\": [",
                        static_cast<unsigned long long>(step.elapsed_ms));
            for (size_t rank = 0; rank < step.top.size(); ++rank) {
                std::printf("%s{\"token_id\": %d, \"logit\": %.9g}",
                            rank == 0 ? "" : ", ", step.top[rank].token,
                            step.top[rank].logit);
            }
            std::printf("]}%s\n", index + 1U == steps.size() ? "" : ",");
        }
        std::printf("  ],\n");
        std::printf("  \"prompt_elapsed_ms\": %llu,\n",
                    static_cast<unsigned long long>(
                        elapsed_ms(prompt_start, prompt_end, frequency)));
        std::printf("  \"load_elapsed_ms\": %llu,\n",
                    static_cast<unsigned long long>(
                        elapsed_ms(load_start, load_end, frequency)));
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
