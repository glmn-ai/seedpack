// ggml-quants.c plus accessors for its IQ lookup tables (built by iq2xs_init_impl /
// iq3xs_init_impl on the host), so the GPU port can use exactly the same tables.
#include "ggml-quants.c"

void * vq_iq2_table(int i, int which) {
    return which == 0 ? (void *)iq2_data[i].grid : which == 1 ? (void *)iq2_data[i].map : (void *)iq2_data[i].neighbours;
}
void * vq_iq3_table(int i, int which) {
    return which == 0 ? (void *)iq3_data[i].grid : which == 1 ? (void *)iq3_data[i].map : (void *)iq3_data[i].neighbours;
}
