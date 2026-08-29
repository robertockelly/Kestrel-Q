#include <stdio.h>
#include <string.h>

#include "kq_sha256.h"

static int hash_equals(const unsigned char *input,
                       size_t length,
                       const unsigned char expected[32]) {
    kq_sha256 context;
    unsigned char actual[32];
    kq_sha256_init(&context);
    kq_sha256_update(&context, input, length);
    kq_sha256_final(&context, actual);
    return memcmp(actual, expected, sizeof(actual)) == 0;
}

int main(void) {
    static const unsigned char empty_hash[32] = {
        0xe3U, 0xb0U, 0xc4U, 0x42U, 0x98U, 0xfcU, 0x1cU, 0x14U,
        0x9aU, 0xfbU, 0xf4U, 0xc8U, 0x99U, 0x6fU, 0xb9U, 0x24U,
        0x27U, 0xaeU, 0x41U, 0xe4U, 0x64U, 0x9bU, 0x93U, 0x4cU,
        0xa4U, 0x95U, 0x99U, 0x1bU, 0x78U, 0x52U, 0xb8U, 0x55U
    };
    static const unsigned char abc_hash[32] = {
        0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU,
        0x41U, 0x41U, 0x40U, 0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U,
        0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U, 0x7aU, 0x9cU,
        0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU
    };
    if (!hash_equals((const unsigned char *)"", 0U, empty_hash) ||
        !hash_equals((const unsigned char *)"abc", 3U, abc_hash)) {
        fprintf(stderr, "SHA-256 known-vector failure\n");
        return 1;
    }
    printf("SHA-256 known vectors: PASS\n");
    return 0;
}
