// Minimal replacements for the parts of ggml.c that ggml-quants.c needs.
#include "ggml.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static const struct { int blck; size_t size; const char * name; } TYPES[] = {
    [GGML_TYPE_F32] = {1, 4, "f32"}, [GGML_TYPE_F16] = {1, 2, "f16"},
    [GGML_TYPE_Q4_0] = {32, 18, "q4_0"}, [GGML_TYPE_Q4_1] = {32, 20, "q4_1"},
    [GGML_TYPE_Q5_0] = {32, 22, "q5_0"}, [GGML_TYPE_Q5_1] = {32, 24, "q5_1"},
    [GGML_TYPE_Q8_0] = {32, 34, "q8_0"}, [GGML_TYPE_Q8_1] = {32, 36, "q8_1"},
    [GGML_TYPE_Q2_K] = {256, 84, "q2_K"}, [GGML_TYPE_Q3_K] = {256, 110, "q3_K"},
    [GGML_TYPE_Q4_K] = {256, 144, "q4_K"}, [GGML_TYPE_Q5_K] = {256, 176, "q5_K"},
    [GGML_TYPE_Q6_K] = {256, 210, "q6_K"}, [GGML_TYPE_Q8_K] = {256, 292, "q8_K"},
    [GGML_TYPE_IQ2_XXS] = {256, 66, "iq2_xxs"}, [GGML_TYPE_IQ2_XS] = {256, 74, "iq2_xs"},
    [GGML_TYPE_IQ3_XXS] = {256, 98, "iq3_xxs"}, [GGML_TYPE_IQ1_S] = {256, 50, "iq1_s"},
    [GGML_TYPE_IQ4_NL] = {32, 18, "iq4_nl"}, [GGML_TYPE_IQ3_S] = {256, 110, "iq3_s"},
    [GGML_TYPE_IQ2_S] = {256, 82, "iq2_s"}, [GGML_TYPE_IQ4_XS] = {256, 136, "iq4_xs"},
    [GGML_TYPE_IQ1_M] = {256, 56, "iq1_m"}, [GGML_TYPE_BF16] = {1, 2, "bf16"},
};

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list a; va_start(a, fmt);
    fprintf(stderr, "ggml abort %s:%d: ", file, line); vfprintf(stderr, fmt, a); fprintf(stderr, "\n");
    va_end(a); abort();
}
size_t ggml_type_size(enum ggml_type t) { return TYPES[t].size; }
size_t ggml_row_size(enum ggml_type t, int64_t ne) { return TYPES[t].size * ne / TYPES[t].blck; }
const char * ggml_type_name(enum ggml_type t) { return TYPES[t].name ? TYPES[t].name : "?"; }
