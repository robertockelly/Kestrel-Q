#ifndef KQ_INTERNAL_H
#define KQ_INTERNAL_H

#include "kq_status.h"

void kq_diagnostic_set(kq_diagnostic *diagnostic,
                       kq_status status,
                       const char *format,
                       ...);

#endif
