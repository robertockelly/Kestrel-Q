#include "kq_cuda.h"

#include <cuda_runtime.h>

#include <stdio.h>

namespace {

constexpr int kInputValue = 41;
constexpr int kExpectedValue = 42;

__global__ void kq_increment_kernel(int *value) {
    *value += 1;
}

bool kq_cuda_call_succeeded(cudaError_t status, const char *operation) {
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
    fprintf(stderr,
            "CUDA error: %s failed: %s (%s)\n",
            operation,
            name,
            message);
    return false;
}

void kq_print_cuda_version(const char *label, int version) {
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    printf("%s: %d.%d (raw %d)\n", label, major, minor, version);
}

}  // namespace

extern "C" int kq_cuda_smoke(void) {
    int runtime_version = 0;
    int driver_version = 0;
    int device_count = 0;
    int device_index = 0;
    cudaDeviceProp properties = {};
    size_t free_memory = 0;
    size_t total_memory = 0;
    int *device_value = nullptr;
    int host_value = kInputValue;
    int result = 1;

    if (!kq_cuda_call_succeeded(cudaRuntimeGetVersion(&runtime_version),
                                "cudaRuntimeGetVersion")) {
        return 1;
    }
    if (!kq_cuda_call_succeeded(cudaDriverGetVersion(&driver_version),
                                "cudaDriverGetVersion")) {
        return 1;
    }
    if (!kq_cuda_call_succeeded(cudaGetDeviceCount(&device_count),
                                "cudaGetDeviceCount")) {
        return 1;
    }
    if (device_count <= 0) {
        fprintf(stderr, "CUDA error: no CUDA-capable devices found\n");
        return 1;
    }
    if (!kq_cuda_call_succeeded(cudaGetDevice(&device_index),
                                "cudaGetDevice")) {
        return 1;
    }
    if (!kq_cuda_call_succeeded(
            cudaGetDeviceProperties(&properties, device_index),
            "cudaGetDeviceProperties")) {
        return 1;
    }
    if (!kq_cuda_call_succeeded(cudaMemGetInfo(&free_memory, &total_memory),
                                "cudaMemGetInfo")) {
        return 1;
    }

    kq_print_cuda_version("CUDA runtime version", runtime_version);
    kq_print_cuda_version("CUDA driver API version", driver_version);
    printf("CUDA device count: %d\n", device_count);
    printf("Selected CUDA device: %d\n", device_index);
    printf("GPU name: %s\n", properties.name);
    printf("Compute capability: %d.%d\n", properties.major, properties.minor);
    printf("Total global memory: %llu bytes (%llu MiB)\n",
           static_cast<unsigned long long>(properties.totalGlobalMem),
           static_cast<unsigned long long>(properties.totalGlobalMem /
                                           (1024ULL * 1024ULL)));
    printf("Free global memory: %llu bytes (%llu MiB)\n",
           static_cast<unsigned long long>(free_memory),
           static_cast<unsigned long long>(free_memory /
                                           (1024ULL * 1024ULL)));
    printf("Multiprocessor count: %d\n", properties.multiProcessorCount);
    printf("Warp size: %d\n", properties.warpSize);

    if (!kq_cuda_call_succeeded(
            cudaMalloc(reinterpret_cast<void **>(&device_value),
                       sizeof(*device_value)),
            "cudaMalloc")) {
        return 1;
    }

    if (!kq_cuda_call_succeeded(
            cudaMemcpy(device_value,
                       &host_value,
                       sizeof(host_value),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D")) {
        goto cleanup;
    }

    kq_increment_kernel<<<1, 1>>>(device_value);
    if (!kq_cuda_call_succeeded(cudaGetLastError(),
                                "kq_increment_kernel launch")) {
        goto cleanup;
    }
    if (!kq_cuda_call_succeeded(cudaDeviceSynchronize(),
                                "cudaDeviceSynchronize")) {
        goto cleanup;
    }

    host_value = 0;
    if (!kq_cuda_call_succeeded(
            cudaMemcpy(&host_value,
                       device_value,
                       sizeof(host_value),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H")) {
        goto cleanup;
    }

    if (host_value != kExpectedValue) {
        fprintf(stderr,
                "CUDA result validation failed: expected %d, received %d\n",
                kExpectedValue,
                host_value);
        goto cleanup;
    }

    printf("CUDA kernel result: %d (validated)\n", host_value);
    result = 0;

cleanup:
    if (!kq_cuda_call_succeeded(cudaFree(device_value), "cudaFree")) {
        result = 1;
    }
    return result;
}
