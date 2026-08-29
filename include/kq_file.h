#ifndef KQ_FILE_H
#define KQ_FILE_H

#include <stdint.h>
#include <wchar.h>

#include "kq_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kq_file kq_file;
typedef struct kq_file_view kq_file_view;

kq_status kq_file_open_readonly(const wchar_t *path,
                                kq_file **out_file,
                                kq_diagnostic *diagnostic);
uint64_t kq_file_size(const kq_file *file);
void kq_file_close(kq_file *file);

kq_status kq_file_view_open(kq_file *file,
                            uint64_t offset,
                            uint64_t length,
                            kq_file_view **out_view,
                            kq_diagnostic *diagnostic);
const unsigned char *kq_file_view_data(const kq_file_view *view);
uint64_t kq_file_view_offset(const kq_file_view *view);
uint64_t kq_file_view_length(const kq_file_view *view);
void kq_file_view_close(kq_file_view *view);

#ifdef __cplusplus
}
#endif

#endif
