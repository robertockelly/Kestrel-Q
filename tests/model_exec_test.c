#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "kq_model_exec.h"

static int expect_status(kq_status actual, kq_status expected,
                         const char *name) {
    if (actual == expected) return 1;
    (void)fprintf(stderr, "%s: expected %s, got %s\n", name,
                  kq_status_string(expected), kq_status_string(actual));
    return 0;
}

int main(void) {
    kq_diagnostic diagnostic;
    uint32_t selected = UINT32_MAX;
    float tie[] = {2.0f, 3.0f, 3.0f, -4.0f};
    float negative[] = {-8.0f, -2.0f, -3.0f};
    float nonfinite[] = {1.0f, NAN};
    int passed = 1;

    passed &= expect_status(kq_model_exec_greedy_argmax_f32(
        tie, 4U, &selected, &diagnostic), KQ_STATUS_OK,
        "stable tie selection");
    if (selected != 1U) {
        (void)fprintf(stderr, "stable tie selected %u, expected 1\n", selected);
        passed = 0;
    }
    passed &= expect_status(kq_model_exec_greedy_argmax_f32(
        negative, 3U, &selected, &diagnostic), KQ_STATUS_OK,
        "negative selection");
    if (selected != 1U) passed = 0;
    selected = 77U;
    passed &= expect_status(kq_model_exec_greedy_argmax_f32(
        nonfinite, 2U, &selected, &diagnostic), KQ_STATUS_NUMERIC_DOMAIN,
        "non-finite rejection");
    if (selected != 77U) {
        (void)fprintf(stderr, "failed argmax modified its output\n");
        passed = 0;
    }
    passed &= expect_status(kq_model_exec_greedy_argmax_f32(
        NULL, 1U, &selected, &diagnostic), KQ_STATUS_INVALID_ARGUMENT,
        "null logits rejection");
    passed &= expect_status(kq_model_exec_greedy_argmax_f32(
        tie, 0U, &selected, &diagnostic), KQ_STATUS_INVALID_ARGUMENT,
        "empty logits rejection");
    return passed ? 0 : 1;
}
