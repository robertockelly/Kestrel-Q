#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "kq_file.h"

#include <stdint.h>
#include <stdlib.h>

#include "kq_internal.h"

struct kq_file {
    HANDLE file_handle;
    HANDLE mapping_handle;
    uint64_t size;
    uint64_t allocation_granularity;
};

struct kq_file_view {
    void *mapping_base;
    const unsigned char *data;
    uint64_t logical_offset;
    uint64_t logical_length;
};

static int kq_u64_add_checked(uint64_t left,
                              uint64_t right,
                              uint64_t *result) {
    if (UINT64_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

kq_status kq_file_open_readonly(const wchar_t *path,
                                kq_file **out_file,
                                kq_diagnostic *diagnostic) {
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = NULL;
    LARGE_INTEGER size_value;
    SYSTEM_INFO system_info;
    kq_file *file = NULL;
    DWORD error_code;

    kq_diagnostic_clear(diagnostic);
    if (path == NULL || path[0] == L'\0' || out_file == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "path and output file are required");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    *out_file = NULL;

    file_handle = CreateFileW(path,
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              NULL,
                              OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL,
                              NULL);
    if (file_handle == INVALID_HANDLE_VALUE) {
        error_code = GetLastError();
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_FILE_OPEN_FAILED,
                          "CreateFileW failed (Win32 error %lu)",
                          (unsigned long)error_code);
        return KQ_STATUS_FILE_OPEN_FAILED;
    }

    if (!GetFileSizeEx(file_handle, &size_value) || size_value.QuadPart < 0) {
        error_code = GetLastError();
        CloseHandle(file_handle);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_FILE_SIZE_FAILED,
                          "GetFileSizeEx failed (Win32 error %lu)",
                          (unsigned long)error_code);
        return KQ_STATUS_FILE_SIZE_FAILED;
    }

    if (size_value.QuadPart > 0) {
        mapping_handle = CreateFileMappingW(file_handle,
                                            NULL,
                                            PAGE_READONLY,
                                            0,
                                            0,
                                            NULL);
        if (mapping_handle == NULL) {
            error_code = GetLastError();
            CloseHandle(file_handle);
            kq_diagnostic_set(diagnostic,
                              KQ_STATUS_FILE_MAP_FAILED,
                              "CreateFileMappingW failed (Win32 error %lu)",
                              (unsigned long)error_code);
            return KQ_STATUS_FILE_MAP_FAILED;
        }
    }

    file = (kq_file *)calloc(1U, sizeof(*file));
    if (file == NULL) {
        if (mapping_handle != NULL) {
            CloseHandle(mapping_handle);
        }
        CloseHandle(file_handle);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate file state");
        return KQ_STATUS_OUT_OF_MEMORY;
    }

    GetSystemInfo(&system_info);
    if (system_info.dwAllocationGranularity == 0U) {
        if (mapping_handle != NULL) {
            CloseHandle(mapping_handle);
        }
        CloseHandle(file_handle);
        free(file);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_FILE_MAP_FAILED,
                          "Windows reported zero allocation granularity");
        return KQ_STATUS_FILE_MAP_FAILED;
    }

    file->file_handle = file_handle;
    file->mapping_handle = mapping_handle;
    file->size = (uint64_t)size_value.QuadPart;
    file->allocation_granularity =
        (uint64_t)system_info.dwAllocationGranularity;
    *out_file = file;
    return KQ_STATUS_OK;
}

uint64_t kq_file_size(const kq_file *file) {
    return file == NULL ? 0U : file->size;
}

void kq_file_close(kq_file *file) {
    if (file == NULL) {
        return;
    }
    if (file->mapping_handle != NULL) {
        CloseHandle(file->mapping_handle);
    }
    if (file->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(file->file_handle);
    }
    free(file);
}

kq_status kq_file_view_open(kq_file *file,
                            uint64_t offset,
                            uint64_t length,
                            kq_file_view **out_view,
                            kq_diagnostic *diagnostic) {
    uint64_t logical_end;
    uint64_t aligned_offset;
    uint64_t alignment_delta;
    uint64_t mapped_length_u64;
    SIZE_T mapped_length;
    void *mapping_base;
    kq_file_view *view;
    DWORD offset_high;
    DWORD offset_low;
    DWORD error_code;

    kq_diagnostic_clear(diagnostic);
    if (file == NULL || out_view == NULL || length == 0U) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_INVALID_ARGUMENT,
                          "file, non-zero length and output view are required");
        return KQ_STATUS_INVALID_ARGUMENT;
    }
    *out_view = NULL;

    if (!kq_u64_add_checked(offset, length, &logical_end)) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "logical view offset plus length overflows uint64");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }
    if (logical_end > file->size || file->mapping_handle == NULL) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_SPAN_OUT_OF_RANGE,
                          "logical view [%llu, %llu) exceeds file size %llu",
                          (unsigned long long)offset,
                          (unsigned long long)logical_end,
                          (unsigned long long)file->size);
        return KQ_STATUS_SPAN_OUT_OF_RANGE;
    }

    aligned_offset =
        offset - (offset % file->allocation_granularity);
    alignment_delta = offset - aligned_offset;
    if (!kq_u64_add_checked(alignment_delta,
                            length,
                            &mapped_length_u64) ||
        mapped_length_u64 > (uint64_t)SIZE_MAX) {
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_ARITHMETIC_OVERFLOW,
                          "aligned Windows view length does not fit SIZE_T");
        return KQ_STATUS_ARITHMETIC_OVERFLOW;
    }

    mapped_length = (SIZE_T)mapped_length_u64;
    offset_high = (DWORD)(aligned_offset >> 32U);
    offset_low = (DWORD)(aligned_offset & UINT32_MAX);
    mapping_base = MapViewOfFile(file->mapping_handle,
                                 FILE_MAP_READ,
                                 offset_high,
                                 offset_low,
                                 mapped_length);
    if (mapping_base == NULL) {
        error_code = GetLastError();
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_FILE_MAP_FAILED,
                          "MapViewOfFile failed (Win32 error %lu)",
                          (unsigned long)error_code);
        return KQ_STATUS_FILE_MAP_FAILED;
    }

    view = (kq_file_view *)calloc(1U, sizeof(*view));
    if (view == NULL) {
        UnmapViewOfFile(mapping_base);
        kq_diagnostic_set(diagnostic,
                          KQ_STATUS_OUT_OF_MEMORY,
                          "could not allocate logical file view state");
        return KQ_STATUS_OUT_OF_MEMORY;
    }

    view->mapping_base = mapping_base;
    view->data = (const unsigned char *)mapping_base +
                 (size_t)alignment_delta;
    view->logical_offset = offset;
    view->logical_length = length;
    *out_view = view;
    return KQ_STATUS_OK;
}

const unsigned char *kq_file_view_data(const kq_file_view *view) {
    return view == NULL ? NULL : view->data;
}

uint64_t kq_file_view_offset(const kq_file_view *view) {
    return view == NULL ? 0U : view->logical_offset;
}

uint64_t kq_file_view_length(const kq_file_view *view) {
    return view == NULL ? 0U : view->logical_length;
}

void kq_file_view_close(kq_file_view *view) {
    if (view == NULL) {
        return;
    }
    if (view->mapping_base != NULL) {
        UnmapViewOfFile(view->mapping_base);
    }
    free(view);
}
