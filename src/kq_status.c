#include "kq_status.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "kq_internal.h"

const char *kq_status_string(kq_status status) {
    switch (status) {
        case KQ_STATUS_OK:
            return "ok";
        case KQ_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case KQ_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case KQ_STATUS_FILE_OPEN_FAILED:
            return "file open failed";
        case KQ_STATUS_FILE_SIZE_FAILED:
            return "file size failed";
        case KQ_STATUS_FILE_MAP_FAILED:
            return "file mapping failed";
        case KQ_STATUS_TRUNCATED:
            return "truncated file";
        case KQ_STATUS_BAD_MAGIC:
            return "bad GGUF magic";
        case KQ_STATUS_UNSUPPORTED_VERSION:
            return "unsupported GGUF version";
        case KQ_STATUS_MALFORMED_METADATA:
            return "malformed GGUF metadata";
        case KQ_STATUS_UNSUPPORTED_METADATA:
            return "unsupported GGUF metadata";
        case KQ_STATUS_MALFORMED_TENSOR:
            return "malformed GGUF tensor";
        case KQ_STATUS_UNSUPPORTED_TENSOR_TYPE:
            return "unsupported GGUF tensor type";
        case KQ_STATUS_ARITHMETIC_OVERFLOW:
            return "arithmetic overflow";
        case KQ_STATUS_LIMIT_EXCEEDED:
            return "defensive limit exceeded";
        case KQ_STATUS_INVALID_ALIGNMENT:
            return "invalid GGUF alignment";
        case KQ_STATUS_SPAN_OUT_OF_RANGE:
            return "tensor span out of range";
        case KQ_STATUS_DUPLICATE_METADATA:
            return "duplicate GGUF metadata key";
        case KQ_STATUS_DUPLICATE_TENSOR:
            return "duplicate GGUF tensor name";
        case KQ_STATUS_INCONSISTENT_DATA_SECTION:
            return "inconsistent GGUF data section";
        case KQ_STATUS_UNSUPPORTED_MODEL:
            return "unsupported model";
        case KQ_STATUS_MODEL_TOPOLOGY_MISMATCH:
            return "model topology mismatch";
        case KQ_STATUS_SEMANTIC_MAPPING_FAILED:
            return "semantic tensor mapping failed";
        case KQ_STATUS_UNKNOWN_PHYSICAL_TENSOR:
            return "unknown physical tensor";
        default:
            return "unknown Kestrel-Q status";
    }
}

void kq_diagnostic_clear(kq_diagnostic *diagnostic) {
    if (diagnostic == NULL) {
        return;
    }
    diagnostic->status = KQ_STATUS_OK;
    diagnostic->message[0] = '\0';
}

void kq_diagnostic_set(kq_diagnostic *diagnostic,
                       kq_status status,
                       const char *format,
                       ...) {
    va_list arguments;

    if (diagnostic == NULL) {
        return;
    }

    diagnostic->status = status;
    diagnostic->message[0] = '\0';
    if (format == NULL) {
        return;
    }

    va_start(arguments, format);
    (void)vsnprintf(diagnostic->message,
                    sizeof(diagnostic->message),
                    format,
                    arguments);
    va_end(arguments);
    diagnostic->message[sizeof(diagnostic->message) - 1U] = '\0';
}
