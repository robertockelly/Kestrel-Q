#include "kq_sha256.h"

#include <string.h>

static const uint32_t kq_sha256_constants[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static uint32_t kq_rotr32(uint32_t value, unsigned int count) {
    return (value >> count) | (value << (32U - count));
}

static void kq_sha256_transform(kq_sha256 *context,
                                const unsigned char block[64]) {
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    uint32_t s0;
    uint32_t s1;
    uint32_t choose;
    uint32_t majority;
    uint32_t temporary1;
    uint32_t temporary2;
    unsigned int index;

    for (index = 0U; index < 16U; ++index) {
        const unsigned char *item = block + index * 4U;
        words[index] = ((uint32_t)item[0] << 24U) |
                       ((uint32_t)item[1] << 16U) |
                       ((uint32_t)item[2] << 8U) |
                       (uint32_t)item[3];
    }
    for (index = 16U; index < 64U; ++index) {
        s0 = kq_rotr32(words[index - 15U], 7U) ^
             kq_rotr32(words[index - 15U], 18U) ^
             (words[index - 15U] >> 3U);
        s1 = kq_rotr32(words[index - 2U], 17U) ^
             kq_rotr32(words[index - 2U], 19U) ^
             (words[index - 2U] >> 10U);
        words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0U; index < 64U; ++index) {
        s1 = kq_rotr32(e, 6U) ^ kq_rotr32(e, 11U) ^ kq_rotr32(e, 25U);
        choose = (e & f) ^ ((~e) & g);
        temporary1 = h + s1 + choose + kq_sha256_constants[index] +
                     words[index];
        s0 = kq_rotr32(a, 2U) ^ kq_rotr32(a, 13U) ^ kq_rotr32(a, 22U);
        majority = (a & b) ^ (a & c) ^ (b & c);
        temporary2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void kq_sha256_init(kq_sha256 *context) {
    if (context == NULL) {
        return;
    }
    context->state[0] = 0x6a09e667U;
    context->state[1] = 0xbb67ae85U;
    context->state[2] = 0x3c6ef372U;
    context->state[3] = 0xa54ff53aU;
    context->state[4] = 0x510e527fU;
    context->state[5] = 0x9b05688cU;
    context->state[6] = 0x1f83d9abU;
    context->state[7] = 0x5be0cd19U;
    context->total_bytes = 0U;
    context->block_used = 0U;
}

void kq_sha256_update(kq_sha256 *context,
                      const unsigned char *data,
                      size_t length) {
    size_t available;
    size_t copied;
    if (context == NULL || (data == NULL && length != 0U)) {
        return;
    }
    context->total_bytes += (uint64_t)length;
    while (length != 0U) {
        available = sizeof(context->block) - context->block_used;
        copied = length < available ? length : available;
        memcpy(context->block + context->block_used, data, copied);
        context->block_used += copied;
        data += copied;
        length -= copied;
        if (context->block_used == sizeof(context->block)) {
            kq_sha256_transform(context, context->block);
            context->block_used = 0U;
        }
    }
}

void kq_sha256_final(kq_sha256 *context, unsigned char digest[32]) {
    uint64_t bits;
    unsigned int index;
    if (context == NULL || digest == NULL) {
        return;
    }
    bits = context->total_bytes * 8U;
    context->block[context->block_used++] = 0x80U;
    if (context->block_used > 56U) {
        memset(context->block + context->block_used,
               0,
               sizeof(context->block) - context->block_used);
        kq_sha256_transform(context, context->block);
        context->block_used = 0U;
    }
    memset(context->block + context->block_used, 0, 56U - context->block_used);
    for (index = 0U; index < 8U; ++index) {
        context->block[63U - index] = (unsigned char)(bits >> (index * 8U));
    }
    kq_sha256_transform(context, context->block);
    for (index = 0U; index < 8U; ++index) {
        digest[index * 4U] = (unsigned char)(context->state[index] >> 24U);
        digest[index * 4U + 1U] =
            (unsigned char)(context->state[index] >> 16U);
        digest[index * 4U + 2U] =
            (unsigned char)(context->state[index] >> 8U);
        digest[index * 4U + 3U] = (unsigned char)context->state[index];
    }
    memset(context, 0, sizeof(*context));
}
