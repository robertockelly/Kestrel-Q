#include <stdio.h>
#include <string.h>
#include "kq.h"

int main(void) {
    const char *version = kq_version_string();
    if (version == NULL || strlen(version) == 0) {
        fprintf(stderr, "invalid version string\n");
        return 1;
    }
    return 0;
}
