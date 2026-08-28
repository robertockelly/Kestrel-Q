#include <stdio.h>
#include "kq.h"

int main(void) {
    printf("Kestrel-Q %s\n", kq_version_string());
    printf("Qwen3.8-Flash-Next specialized runtime: research scaffold only.\n");
    return 0;
}
