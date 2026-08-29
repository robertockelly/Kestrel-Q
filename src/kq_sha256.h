#ifndef KQ_SHA256_H
#define KQ_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct kq_sha256 {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_used;
} kq_sha256;

void kq_sha256_init(kq_sha256 *context);
void kq_sha256_update(kq_sha256 *context,
                      const unsigned char *data,
                      size_t length);
void kq_sha256_final(kq_sha256 *context, unsigned char digest[32]);

#endif
