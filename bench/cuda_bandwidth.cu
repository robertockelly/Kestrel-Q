#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifndef KQ_GIT_COMMIT
#define KQ_GIT_COMMIT "unknown"
#endif

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;
constexpr unsigned char kDeviceSourcePattern = 0x5a;
constexpr unsigned char kDestinationSentinel = 0xa5;

struct Options {
    const char *machine_id = "unknown";
    int warmup_iterations = 3;
    int measured_iterations = 10;
    bool self_test = false;
    bool show_help = false;
};

struct Metadata {
    std::string timestamp_utc;
    std::string gpu_name;
    std::string compute_capability;
    int runtime_version = 0;
    int driver_version = 0;
    int async_engine_count = 0;
};

struct SetupMetric {
    const char *operation;
    size_t bytes;
    double elapsed_ms;
};

struct TransferMetric {
    const char *direction;
    const char *host_memory;
    size_t transfer_size;
    size_t payload_bytes;
    int warmup_iterations;
    int measured_iterations;
    double host_mean_ms;
    double host_median_ms;
    double host_gbps;
    double cuda_mean_ms;
    double cuda_median_ms;
    double cuda_gbps;
    const char *correctness;
    const char *status;
    const char *detail;
};

struct Resources {
    unsigned char *pageable_source = nullptr;
    unsigned char *pageable_destination = nullptr;
    unsigned char *pinned_source = nullptr;
    unsigned char *pinned_destination = nullptr;
    unsigned char *device_h2d = nullptr;
    unsigned char *device_d2h = nullptr;
    cudaStream_t h2d_stream = nullptr;
    cudaStream_t d2h_stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool cuda_call_succeeded(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return true;
    }

    const char *name = cudaGetErrorName(status);
    const char *message = cudaGetErrorString(status);
    if (name == nullptr) {
        name = "unknown CUDA error";
    }
    if (message == nullptr) {
        message = "no CUDA diagnostic available";
    }
    std::fprintf(stderr,
                 "CUDA error: %s failed: %s (%s)\n",
                 operation,
                 name,
                 message);
    return false;
}

bool parse_nonnegative_int(const char *text, int *value) {
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || parsed < 0 || parsed > 100000) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parse_options(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--self-test") == 0) {
            options->self_test = true;
            options->warmup_iterations = 1;
            options->measured_iterations = 1;
        } else if (std::strcmp(argv[index], "--machine-id") == 0) {
            if (++index >= argc || argv[index][0] == '\0') {
                std::fprintf(stderr, "--machine-id requires a value\n");
                return false;
            }
            options->machine_id = argv[index];
        } else if (std::strcmp(argv[index], "--warmup") == 0) {
            if (++index >= argc ||
                !parse_nonnegative_int(argv[index],
                                       &options->warmup_iterations)) {
                std::fprintf(stderr, "--warmup requires an integer >= 0\n");
                return false;
            }
        } else if (std::strcmp(argv[index], "--iterations") == 0) {
            if (++index >= argc ||
                !parse_nonnegative_int(argv[index],
                                       &options->measured_iterations) ||
                options->measured_iterations == 0) {
                std::fprintf(stderr, "--iterations requires an integer > 0\n");
                return false;
            }
        } else if (std::strcmp(argv[index], "--help") == 0) {
            options->show_help = true;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[index]);
            return false;
        }
    }
    return true;
}

void print_usage(const char *program) {
    std::fprintf(stderr,
                 "usage: %s [--machine-id ID] [--warmup N] "
                 "[--iterations N] [--self-test]\n",
                 program);
}

std::string utc_timestamp(void) {
    const std::time_t now = std::time(nullptr);
    std::tm utc = {};
#if defined(_WIN32)
    if (gmtime_s(&utc, &now) != 0) {
        return "unknown";
    }
#else
    if (gmtime_r(&now, &utc) == nullptr) {
        return "unknown";
    }
#endif
    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc) ==
        0) {
        return "unknown";
    }
    return buffer;
}

bool initialize_metadata(Metadata *metadata) {
    int device_count = 0;
    int device_index = 0;
    cudaDeviceProp properties = {};

    if (!cuda_call_succeeded(cudaRuntimeGetVersion(&metadata->runtime_version),
                             "cudaRuntimeGetVersion") ||
        !cuda_call_succeeded(cudaDriverGetVersion(&metadata->driver_version),
                             "cudaDriverGetVersion") ||
        !cuda_call_succeeded(cudaGetDeviceCount(&device_count),
                             "cudaGetDeviceCount")) {
        return false;
    }
    if (device_count <= 0) {
        std::fprintf(stderr, "CUDA error: no CUDA-capable devices found\n");
        return false;
    }
    if (!cuda_call_succeeded(cudaGetDevice(&device_index), "cudaGetDevice") ||
        !cuda_call_succeeded(cudaSetDevice(device_index), "cudaSetDevice") ||
        !cuda_call_succeeded(
            cudaGetDeviceProperties(&properties, device_index),
            "cudaGetDeviceProperties")) {
        return false;
    }

    metadata->timestamp_utc = utc_timestamp();
    metadata->gpu_name = properties.name;
    metadata->compute_capability = std::to_string(properties.major) + "." +
                                   std::to_string(properties.minor);
    metadata->async_engine_count = properties.asyncEngineCount;
    return true;
}

void add_setup_metric(std::vector<SetupMetric> *metrics,
                      const char *operation,
                      size_t bytes,
                      Clock::time_point start,
                      Clock::time_point stop) {
    metrics->push_back({operation, bytes, elapsed_ms(start, stop)});
}

template <typename Function>
bool time_cuda_setup(std::vector<SetupMetric> *metrics,
                     const char *operation,
                     size_t bytes,
                     Function function) {
    const Clock::time_point start = Clock::now();
    const cudaError_t status = function();
    const Clock::time_point stop = Clock::now();
    if (!cuda_call_succeeded(status, operation)) {
        return false;
    }
    add_setup_metric(metrics, operation, bytes, start, stop);
    return true;
}

bool allocate_pageable(unsigned char **pointer,
                       size_t bytes,
                       const char *operation,
                       std::vector<SetupMetric> *metrics) {
    const Clock::time_point start = Clock::now();
    *pointer = static_cast<unsigned char *>(std::malloc(bytes));
    const Clock::time_point stop = Clock::now();
    if (*pointer == nullptr) {
        std::fprintf(stderr, "%s failed for %zu bytes\n", operation, bytes);
        return false;
    }
    add_setup_metric(metrics, operation, bytes, start, stop);
    return true;
}

bool allocate_resources(Resources *resources,
                        size_t bytes,
                        std::vector<SetupMetric> *metrics) {
    if (!time_cuda_setup(metrics,
                         "cuda_context_initialize",
                         0,
                         []() { return cudaFree(nullptr); }) ||
        !allocate_pageable(&resources->pageable_source,
                           bytes,
                           "pageable_malloc_source",
                           metrics) ||
        !allocate_pageable(&resources->pageable_destination,
                           bytes,
                           "pageable_malloc_destination",
                           metrics)) {
        return false;
    }

    Clock::time_point start = Clock::now();
    std::memset(resources->pageable_source, 0, bytes);
    std::memset(resources->pageable_destination, 0, bytes);
    Clock::time_point stop = Clock::now();
    add_setup_metric(metrics,
                     "pageable_touch_two_buffers",
                     bytes * 2,
                     start,
                     stop);

    if (!time_cuda_setup(metrics,
                         "cudaMallocHost_source",
                         bytes,
                         [&]() {
                             return cudaMallocHost(
                                 reinterpret_cast<void **>(
                                     &resources->pinned_source),
                                 bytes);
                         }) ||
        !time_cuda_setup(metrics,
                         "cudaMallocHost_destination",
                         bytes,
                         [&]() {
                             return cudaMallocHost(
                                 reinterpret_cast<void **>(
                                     &resources->pinned_destination),
                                 bytes);
                         })) {
        return false;
    }

    start = Clock::now();
    std::memset(resources->pinned_source, 0, bytes);
    std::memset(resources->pinned_destination, 0, bytes);
    stop = Clock::now();
    add_setup_metric(metrics,
                     "pinned_touch_two_buffers",
                     bytes * 2,
                     start,
                     stop);

    if (!time_cuda_setup(metrics,
                         "cudaMalloc_h2d",
                         bytes,
                         [&]() {
                             return cudaMalloc(
                                 reinterpret_cast<void **>(
                                     &resources->device_h2d),
                                 bytes);
                         }) ||
        !time_cuda_setup(metrics,
                         "cudaMalloc_d2h",
                         bytes,
                         [&]() {
                             return cudaMalloc(
                                 reinterpret_cast<void **>(
                                     &resources->device_d2h),
                                 bytes);
                         }) ||
        !time_cuda_setup(metrics,
                         "cudaStreamCreate_h2d",
                         0,
                         [&]() {
                             return cudaStreamCreateWithFlags(
                                 &resources->h2d_stream,
                                 cudaStreamNonBlocking);
                         }) ||
        !time_cuda_setup(metrics,
                         "cudaStreamCreate_d2h",
                         0,
                         [&]() {
                             return cudaStreamCreateWithFlags(
                                 &resources->d2h_stream,
                                 cudaStreamNonBlocking);
                         }) ||
        !time_cuda_setup(metrics,
                         "cudaEventCreate_h2d_start",
                         0,
                         [&]() { return cudaEventCreate(&resources->h2d_start); }) ||
        !time_cuda_setup(metrics,
                         "cudaEventCreate_h2d_stop",
                         0,
                         [&]() { return cudaEventCreate(&resources->h2d_stop); }) ||
        !time_cuda_setup(metrics,
                         "cudaEventCreate_d2h_start",
                         0,
                         [&]() { return cudaEventCreate(&resources->d2h_start); }) ||
        !time_cuda_setup(metrics,
                         "cudaEventCreate_d2h_stop",
                         0,
                         [&]() { return cudaEventCreate(&resources->d2h_stop); })) {
        return false;
    }
    return true;
}

bool validate_equal(const unsigned char *data,
                    const unsigned char *expected,
                    size_t bytes,
                    const char *operation) {
    if (std::memcmp(data, expected, bytes) == 0) {
        return true;
    }
    for (size_t index = 0; index < bytes; ++index) {
        if (data[index] != expected[index]) {
            std::fprintf(stderr,
                         "%s validation failed at byte %zu: expected %u, "
                         "received %u\n",
                         operation,
                         index,
                         static_cast<unsigned int>(expected[index]),
                         static_cast<unsigned int>(data[index]));
            return false;
        }
    }
    std::fprintf(stderr, "%s validation failed\n", operation);
    return false;
}

void poison_destination(unsigned char *data, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    data[0] = kDestinationSentinel;
    data[bytes - 1] = kDestinationSentinel;
}

bool seed_device_source(Resources *resources,
                        size_t bytes,
                        std::vector<SetupMetric> *metrics) {
    std::memset(resources->pageable_source, kDeviceSourcePattern, bytes);
    if (!time_cuda_setup(metrics,
                         "device_source_seed_h2d",
                         bytes,
                         [&]() {
                             return cudaMemcpy(resources->device_d2h,
                                               resources->pageable_source,
                                               bytes,
                                               cudaMemcpyHostToDevice);
                         })) {
        return false;
    }
    std::memset(resources->pageable_destination,
                kDestinationSentinel,
                bytes);
    if (!cuda_call_succeeded(
            cudaMemcpy(resources->pageable_destination,
                       resources->device_d2h,
                       bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy device source validation D2H")) {
        return false;
    }
    return validate_equal(resources->pageable_destination,
                          resources->pageable_source,
                          bytes,
                          "device source seed");
}

bool run_single_transfer(Resources *resources,
                         bool h2d,
                         bool pinned,
                         size_t bytes,
                         unsigned char sequence_pattern,
                         bool measured,
                         double *host_ms,
                         double *cuda_ms) {
    unsigned char *host_buffer = nullptr;
    cudaMemcpyKind copy_kind;

    if (h2d) {
        host_buffer = pinned ? resources->pinned_source
                             : resources->pageable_source;
        std::memset(host_buffer, sequence_pattern, bytes);
        copy_kind = cudaMemcpyHostToDevice;
    } else {
        host_buffer = pinned ? resources->pinned_destination
                             : resources->pageable_destination;
        std::memset(resources->pageable_source, sequence_pattern, bytes);
        if (!cuda_call_succeeded(
                cudaMemsetAsync(resources->device_d2h,
                                sequence_pattern,
                                bytes,
                                resources->h2d_stream),
                "cudaMemsetAsync D2H source pattern") ||
            !cuda_call_succeeded(
                cudaStreamSynchronize(resources->h2d_stream),
                "cudaStreamSynchronize D2H source pattern")) {
            return false;
        }
        poison_destination(host_buffer, bytes);
        copy_kind = cudaMemcpyDeviceToHost;
    }

    if (measured &&
        !cuda_call_succeeded(
            cudaEventRecord(resources->h2d_start, resources->h2d_stream),
            "cudaEventRecord transfer start")) {
        return false;
    }

    const Clock::time_point host_start = Clock::now();
    const cudaError_t copy_status = h2d
                                        ? cudaMemcpyAsync(
                                              resources->device_h2d,
                                              host_buffer,
                                              bytes,
                                              copy_kind,
                                              resources->h2d_stream)
                                        : cudaMemcpyAsync(
                                              host_buffer,
                                              resources->device_d2h,
                                              bytes,
                                              copy_kind,
                                              resources->h2d_stream);
    if (!cuda_call_succeeded(copy_status, "cudaMemcpyAsync transfer")) {
        return false;
    }
    if (measured &&
        !cuda_call_succeeded(
            cudaEventRecord(resources->h2d_stop, resources->h2d_stream),
            "cudaEventRecord transfer stop")) {
        return false;
    }
    if (!cuda_call_succeeded(cudaStreamSynchronize(resources->h2d_stream),
                             "cudaStreamSynchronize transfer")) {
        return false;
    }
    const Clock::time_point host_stop = Clock::now();

    if (measured) {
        float event_ms = 0.0F;
        if (!cuda_call_succeeded(
                cudaEventElapsedTime(
                    &event_ms, resources->h2d_start, resources->h2d_stop),
                "cudaEventElapsedTime transfer")) {
            return false;
        }
        *host_ms = elapsed_ms(host_start, host_stop);
        *cuda_ms = static_cast<double>(event_ms);
    }

    if (h2d) {
        std::memset(resources->pageable_destination,
                    static_cast<unsigned char>(sequence_pattern ^ 0xffU),
                    bytes);
        if (!cuda_call_succeeded(
                cudaMemcpy(resources->pageable_destination,
                           resources->device_h2d,
                           bytes,
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy H2D validation D2H")) {
            return false;
        }
        return validate_equal(resources->pageable_destination,
                              host_buffer,
                              bytes,
                              "H2D transfer");
    }
    return validate_equal(host_buffer,
                          resources->pageable_source,
                          bytes,
                          "D2H transfer");
}

bool run_bidirectional_transfer(Resources *resources,
                                size_t bytes,
                                unsigned char sequence_pattern,
                                bool measured,
                                double *host_ms,
                                double *cuda_ms) {
    std::memset(resources->pinned_source, sequence_pattern, bytes);
    unsigned char d2h_pattern = static_cast<unsigned char>(
        sequence_pattern ^ 0x5aU);
    if (d2h_pattern == kDestinationSentinel) {
        d2h_pattern ^= 0x01U;
    }
    std::memset(resources->pageable_source, d2h_pattern, bytes);
    if (!cuda_call_succeeded(
            cudaMemsetAsync(resources->device_d2h,
                            d2h_pattern,
                            bytes,
                            resources->d2h_stream),
            "cudaMemsetAsync bidirectional D2H source pattern") ||
        !cuda_call_succeeded(
            cudaStreamSynchronize(resources->d2h_stream),
            "cudaStreamSynchronize bidirectional D2H source pattern")) {
        return false;
    }
    poison_destination(resources->pinned_destination, bytes);

    if (measured &&
        (!cuda_call_succeeded(
             cudaEventRecord(resources->h2d_start, resources->h2d_stream),
             "cudaEventRecord bidirectional H2D start") ||
         !cuda_call_succeeded(
             cudaEventRecord(resources->d2h_start, resources->d2h_stream),
             "cudaEventRecord bidirectional D2H start"))) {
        return false;
    }

    const Clock::time_point host_start = Clock::now();
    if (!cuda_call_succeeded(
            cudaMemcpyAsync(resources->device_h2d,
                            resources->pinned_source,
                            bytes,
                            cudaMemcpyHostToDevice,
                            resources->h2d_stream),
            "cudaMemcpyAsync bidirectional H2D") ||
        !cuda_call_succeeded(
            cudaMemcpyAsync(resources->pinned_destination,
                            resources->device_d2h,
                            bytes,
                            cudaMemcpyDeviceToHost,
                            resources->d2h_stream),
            "cudaMemcpyAsync bidirectional D2H")) {
        return false;
    }
    if (measured &&
        (!cuda_call_succeeded(
             cudaEventRecord(resources->h2d_stop, resources->h2d_stream),
             "cudaEventRecord bidirectional H2D stop") ||
         !cuda_call_succeeded(
             cudaEventRecord(resources->d2h_stop, resources->d2h_stream),
             "cudaEventRecord bidirectional D2H stop"))) {
        return false;
    }
    if (!cuda_call_succeeded(cudaStreamSynchronize(resources->h2d_stream),
                             "cudaStreamSynchronize bidirectional H2D") ||
        !cuda_call_succeeded(cudaStreamSynchronize(resources->d2h_stream),
                             "cudaStreamSynchronize bidirectional D2H")) {
        return false;
    }
    const Clock::time_point host_stop = Clock::now();

    if (measured) {
        float h2d_event_ms = 0.0F;
        float d2h_event_ms = 0.0F;
        if (!cuda_call_succeeded(
                cudaEventElapsedTime(&h2d_event_ms,
                                     resources->h2d_start,
                                     resources->h2d_stop),
                "cudaEventElapsedTime bidirectional H2D") ||
            !cuda_call_succeeded(
                cudaEventElapsedTime(&d2h_event_ms,
                                     resources->d2h_start,
                                     resources->d2h_stop),
                "cudaEventElapsedTime bidirectional D2H")) {
            return false;
        }
        *host_ms = elapsed_ms(host_start, host_stop);
        *cuda_ms = static_cast<double>(
            std::max(h2d_event_ms, d2h_event_ms));
    }

    std::memset(resources->pageable_destination,
                static_cast<unsigned char>(sequence_pattern ^ 0xffU),
                bytes);
    if (!cuda_call_succeeded(
            cudaMemcpy(resources->pageable_destination,
                       resources->device_h2d,
                       bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy bidirectional H2D validation D2H")) {
        return false;
    }
    return validate_equal(resources->pageable_destination,
                          resources->pinned_source,
                          bytes,
                          "bidirectional H2D") &&
           validate_equal(resources->pinned_destination,
                          resources->pageable_source,
                          bytes,
                          "bidirectional D2H");
}

double mean(const std::vector<double> &values) {
    double total = 0.0;
    for (double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() % 2) != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

double bandwidth_gbps(size_t bytes, double milliseconds) {
    return static_cast<double>(bytes) / (milliseconds / 1000.0) / 1.0e9;
}

bool collect_transfer_metric(Resources *resources,
                             const Options &options,
                             bool h2d,
                             bool pinned,
                             size_t bytes,
                             int *sequence,
                             TransferMetric *metric) {
    double ignored_host_ms = 0.0;
    double ignored_cuda_ms = 0.0;
    for (int index = 0; index < options.warmup_iterations; ++index) {
        const unsigned char pattern = static_cast<unsigned char>(
            1 + ((*sequence)++ % 250));
        if (!run_single_transfer(resources,
                                 h2d,
                                 pinned,
                                 bytes,
                                 pattern,
                                 false,
                                 &ignored_host_ms,
                                 &ignored_cuda_ms)) {
            return false;
        }
    }

    std::vector<double> host_times;
    std::vector<double> cuda_times;
    host_times.reserve(static_cast<size_t>(options.measured_iterations));
    cuda_times.reserve(static_cast<size_t>(options.measured_iterations));
    for (int index = 0; index < options.measured_iterations; ++index) {
        const unsigned char pattern = static_cast<unsigned char>(
            1 + ((*sequence)++ % 250));
        double host_ms = 0.0;
        double cuda_ms = 0.0;
        if (!run_single_transfer(resources,
                                 h2d,
                                 pinned,
                                 bytes,
                                 pattern,
                                 true,
                                 &host_ms,
                                 &cuda_ms)) {
            return false;
        }
        host_times.push_back(host_ms);
        cuda_times.push_back(cuda_ms);
    }

    const double host_mean = mean(host_times);
    const double cuda_mean = mean(cuda_times);
    *metric = {h2d ? "h2d" : "d2h",
               pinned ? "pinned" : "pageable",
               bytes,
               bytes,
               options.warmup_iterations,
               options.measured_iterations,
               host_mean,
               median(host_times),
               bandwidth_gbps(bytes, host_mean),
               cuda_mean,
               median(cuda_times),
               bandwidth_gbps(bytes, cuda_mean),
               "PASS",
               "PASS",
               "host timing includes enqueue and stream synchronization; CUDA "
               "timing is stream event elapsed time; validation is outside "
               "timed interval"};
    return true;
}

bool collect_bidirectional_metric(Resources *resources,
                                  const Options &options,
                                  size_t bytes,
                                  int *sequence,
                                  TransferMetric *metric) {
    double ignored_host_ms = 0.0;
    double ignored_cuda_ms = 0.0;
    for (int index = 0; index < options.warmup_iterations; ++index) {
        const unsigned char pattern = static_cast<unsigned char>(
            1 + ((*sequence)++ % 250));
        if (!run_bidirectional_transfer(resources,
                                        bytes,
                                        pattern,
                                        false,
                                        &ignored_host_ms,
                                        &ignored_cuda_ms)) {
            return false;
        }
    }

    std::vector<double> host_times;
    std::vector<double> cuda_times;
    host_times.reserve(static_cast<size_t>(options.measured_iterations));
    cuda_times.reserve(static_cast<size_t>(options.measured_iterations));
    for (int index = 0; index < options.measured_iterations; ++index) {
        const unsigned char pattern = static_cast<unsigned char>(
            1 + ((*sequence)++ % 250));
        double host_ms = 0.0;
        double cuda_ms = 0.0;
        if (!run_bidirectional_transfer(resources,
                                        bytes,
                                        pattern,
                                        true,
                                        &host_ms,
                                        &cuda_ms)) {
            return false;
        }
        host_times.push_back(host_ms);
        cuda_times.push_back(cuda_ms);
    }

    const size_t aggregate_payload = bytes * 2;
    const double host_mean = mean(host_times);
    const double cuda_mean = mean(cuda_times);
    *metric = {"bidirectional",
               "pinned",
               bytes,
               aggregate_payload,
               options.warmup_iterations,
               options.measured_iterations,
               host_mean,
               median(host_times),
               bandwidth_gbps(aggregate_payload, host_mean),
               cuda_mean,
               median(cuda_times),
               bandwidth_gbps(aggregate_payload, cuda_mean),
               "PASS",
               "PASS",
               "payload and GB/s are aggregate across equal H2D and D2H "
               "transfers; CUDA timing is the maximum of the two directional "
               "stream event intervals; validation is outside timed interval"};
    return true;
}

void csv_string(const char *value) {
    std::putchar('"');
    for (const char *cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor == '"') {
            std::putchar('"');
        }
        std::putchar(*cursor);
    }
    std::putchar('"');
}

void emit_header(void) {
    std::puts(
        "schema_version,record_type,timestamp_utc,commit,machine_id,gpu_name,"
        "compute_capability,driver_api_version,runtime_version,"
        "async_engine_count,direction,host_memory,"
        "transfer_size_bytes,payload_bytes_per_iteration,warmup_iterations,"
        "measured_iterations,host_mean_ms,host_median_ms,host_gbps,"
        "cuda_mean_ms,cuda_median_ms,cuda_gbps,setup_operation,setup_bytes,"
        "setup_elapsed_ms,correctness,status,detail");
}

void emit_prefix(const Metadata &metadata,
                 const Options &options,
                 const char *record_type) {
    std::printf("1,");
    csv_string(record_type);
    std::putchar(',');
    csv_string(metadata.timestamp_utc.c_str());
    std::putchar(',');
    csv_string(KQ_GIT_COMMIT);
    std::putchar(',');
    csv_string(options.machine_id);
    std::putchar(',');
    csv_string(metadata.gpu_name.c_str());
    std::putchar(',');
    csv_string(metadata.compute_capability.c_str());
    std::printf(",%d,%d,%d,",
                metadata.driver_version,
                metadata.runtime_version,
                metadata.async_engine_count);
}

void emit_setup_metric(const Metadata &metadata,
                       const Options &options,
                       const SetupMetric &metric) {
    emit_prefix(metadata, options, "setup");
    std::printf(",,,,,,,,,,,,");
    csv_string(metric.operation);
    std::printf(",%zu,%.6f,", metric.bytes, metric.elapsed_ms);
    csv_string("PASS");
    std::putchar(',');
    csv_string("PASS");
    std::putchar(',');
    csv_string("host-visible setup latency");
    std::putchar('\n');
}

void emit_transfer_metric(const Metadata &metadata,
                          const Options &options,
                          const TransferMetric &metric) {
    emit_prefix(metadata, options, "transfer");
    csv_string(metric.direction);
    std::putchar(',');
    csv_string(metric.host_memory);
    std::printf(
        ",%zu,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,,,%s",
        metric.transfer_size,
        metric.payload_bytes,
        metric.warmup_iterations,
        metric.measured_iterations,
        metric.host_mean_ms,
        metric.host_median_ms,
        metric.host_gbps,
        metric.cuda_mean_ms,
        metric.cuda_median_ms,
        metric.cuda_gbps,
        "");
    std::putchar(',');
    csv_string(metric.correctness);
    std::putchar(',');
    csv_string(metric.status);
    std::putchar(',');
    csv_string(metric.detail);
    std::putchar('\n');
}

void emit_bidirectional_skip(const Metadata &metadata,
                             const Options &options,
                             size_t bytes) {
    TransferMetric metric = {"bidirectional",
                             "pinned",
                             bytes,
                             bytes * 2,
                             options.warmup_iterations,
                             options.measured_iterations,
                             0.0,
                             0.0,
                             0.0,
                             0.0,
                             0.0,
                             0.0,
                             "SKIP",
                             "SKIP",
                             "device reports fewer than two asynchronous copy "
                             "engines"};
    emit_transfer_metric(metadata, options, metric);
}

bool cleanup_resources(Resources *resources) {
    bool success = true;
    if (resources->h2d_start != nullptr) {
        success = cuda_call_succeeded(cudaEventDestroy(resources->h2d_start),
                                      "cudaEventDestroy h2d_start") &&
                  success;
    }
    if (resources->h2d_stop != nullptr) {
        success = cuda_call_succeeded(cudaEventDestroy(resources->h2d_stop),
                                      "cudaEventDestroy h2d_stop") &&
                  success;
    }
    if (resources->d2h_start != nullptr) {
        success = cuda_call_succeeded(cudaEventDestroy(resources->d2h_start),
                                      "cudaEventDestroy d2h_start") &&
                  success;
    }
    if (resources->d2h_stop != nullptr) {
        success = cuda_call_succeeded(cudaEventDestroy(resources->d2h_stop),
                                      "cudaEventDestroy d2h_stop") &&
                  success;
    }
    if (resources->h2d_stream != nullptr) {
        success = cuda_call_succeeded(cudaStreamDestroy(resources->h2d_stream),
                                      "cudaStreamDestroy h2d") &&
                  success;
    }
    if (resources->d2h_stream != nullptr) {
        success = cuda_call_succeeded(cudaStreamDestroy(resources->d2h_stream),
                                      "cudaStreamDestroy d2h") &&
                  success;
    }
    if (resources->device_h2d != nullptr) {
        success = cuda_call_succeeded(cudaFree(resources->device_h2d),
                                      "cudaFree device_h2d") &&
                  success;
    }
    if (resources->device_d2h != nullptr) {
        success = cuda_call_succeeded(cudaFree(resources->device_d2h),
                                      "cudaFree device_d2h") &&
                  success;
    }
    if (resources->pinned_source != nullptr) {
        success = cuda_call_succeeded(cudaFreeHost(resources->pinned_source),
                                      "cudaFreeHost source") &&
                  success;
    }
    if (resources->pinned_destination != nullptr) {
        success = cuda_call_succeeded(
                      cudaFreeHost(resources->pinned_destination),
                      "cudaFreeHost destination") &&
                  success;
    }
    std::free(resources->pageable_source);
    std::free(resources->pageable_destination);
    return success;
}

}  // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return 2;
    }
    if (options.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    const std::vector<size_t> sizes = options.self_test
                                          ? std::vector<size_t>{kMiB}
                                          : std::vector<size_t>{kMiB,
                                                                4 * kMiB,
                                                                16 * kMiB,
                                                                64 * kMiB,
                                                                256 * kMiB};
    const size_t maximum_size = sizes.back();
    Metadata metadata;
    Resources resources;
    std::vector<SetupMetric> setup_metrics;
    bool success = initialize_metadata(&metadata) &&
                   allocate_resources(
                       &resources, maximum_size, &setup_metrics) &&
                   seed_device_source(
                       &resources, maximum_size, &setup_metrics);

    if (success) {
        emit_header();
        for (const SetupMetric &metric : setup_metrics) {
            emit_setup_metric(metadata, options, metric);
        }

        int sequence = 0;
        for (size_t bytes : sizes) {
            TransferMetric metric = {};
            if (!collect_transfer_metric(&resources,
                                         options,
                                         true,
                                         false,
                                         bytes,
                                         &sequence,
                                         &metric)) {
                success = false;
                break;
            }
            emit_transfer_metric(metadata, options, metric);

            if (!collect_transfer_metric(&resources,
                                         options,
                                         true,
                                         true,
                                         bytes,
                                         &sequence,
                                         &metric)) {
                success = false;
                break;
            }
            emit_transfer_metric(metadata, options, metric);

            if (!collect_transfer_metric(&resources,
                                         options,
                                         false,
                                         false,
                                         bytes,
                                         &sequence,
                                         &metric)) {
                success = false;
                break;
            }
            emit_transfer_metric(metadata, options, metric);

            if (!collect_transfer_metric(&resources,
                                         options,
                                         false,
                                         true,
                                         bytes,
                                         &sequence,
                                         &metric)) {
                success = false;
                break;
            }
            emit_transfer_metric(metadata, options, metric);

            if (metadata.async_engine_count >= 2) {
                if (!collect_bidirectional_metric(&resources,
                                                  options,
                                                  bytes,
                                                  &sequence,
                                                  &metric)) {
                    success = false;
                    break;
                }
                emit_transfer_metric(metadata, options, metric);
            } else {
                emit_bidirectional_skip(metadata, options, bytes);
            }
        }
    }

    if (!cleanup_resources(&resources)) {
        success = false;
    }
    return success ? 0 : 1;
}
