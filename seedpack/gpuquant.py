"""llama.cpp quantisers on the GPU, bit-identical to the CPU build (requant.py).

The CUDA source is generated from ggml/ggml-quants.c itself: every top-level
function becomes a __device__ function and each GPU thread runs the library's own
quantize_<type>(row, dst, 1, n_per_row, imatrix) on one row. All rows of all tensors of one
type are quantised in a single launch. Exactness comes from compiling without FMA
contraction (--fmad=false) and with IEEE division / square root, which reproduces the
scalar SSE arithmetic of the CPU build; the IQ grids, maps and neighbour lists are copied
from the CPU library after its own init functions built them.
"""
import ctypes
import hashlib
import os
import re

import cupy as cp
import cupyx
import numpy as np

from .ptxfix import link_cubin, opaque_divisions, ptx_of
from .requant import QFUN, IQ2_INIT, IQ3_INIT, lib

_DIR = os.path.join(os.path.dirname(__file__), "ggml")      # llama.cpp sources, IQ tables (read-only)


def cache_dir():
    """where compiled kernels are kept: $SEEDPACK_CACHE, else the user's cache directory"""
    d = os.environ.get("SEEDPACK_CACHE")
    if not d:
        if os.name == "nt":
            d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "seedpack", "cache")
        else:
            d = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "seedpack")
    os.makedirs(d, exist_ok=True)
    return d

# functions that only run on the host (table construction) or are not needed
_SKIP = {"iq2xs_init_impl", "iq2xs_free_impl", "iq3xs_init_impl", "iq3xs_free_impl", "iq2_compare_func",
         "iq3_compare_func", "ggml_validate_row_data", "validate_float", "validate_fp16", "validate_e_e8m0",
         "isinf_fp16", "isnan_fp16", "best_index_mxfp4", "quantize_row_mxfp4_ref", "quantize_row_nvfp4_ref",
         "dequantize_row_mxfp4", "dequantize_row_nvfp4", "quantize_mxfp4", "quantize_nvfp4"}

_PRELUDE = r"""
typedef signed char int8_t; typedef unsigned char uint8_t; typedef short int16_t; typedef unsigned short uint16_t;
typedef int int32_t; typedef unsigned int uint32_t; typedef long long int64_t; typedef unsigned long long uint64_t;
typedef decltype(sizeof(0)) size_t;      // unsigned long long on Windows, unsigned long on Linux
#define UINT32_C(x) (x ## u)
#define UINT16_C(x) (x)
#define INT8_MAX 127
#define FLT_MAX 3.402823466e+38F
#define NULL 0
#define GGML_RESTRICT __restrict__
#define GGML_UNUSED(x) (void)(x)
#define GGML_API
#undef assert
#define assert(x) ((void)0)
#define GGML_ASSERT(x) do { if (!(x)) { printf("GGML_ASSERT %s line %d\n", #x, __LINE__); __trap(); } } while (0)
#define GGML_ABORT(...) do { printf(__VA_ARGS__); __trap(); } while (0)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
typedef uint16_t ggml_fp16_t;
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q4_1 = 3, GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q8_1 = 9, GGML_TYPE_Q2_K = 10, GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12, GGML_TYPE_Q5_K = 13, GGML_TYPE_Q6_K = 14, GGML_TYPE_Q8_K = 15, GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17, GGML_TYPE_IQ3_XXS = 18, GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ4_NL = 20, GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22, GGML_TYPE_IQ4_XS = 23, GGML_TYPE_IQ1_M = 29, GGML_TYPE_BF16 = 30, GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35, GGML_TYPE_MXFP4 = 39, GGML_TYPE_NVFP4 = 40, GGML_TYPE_Q1_0 = 41, GGML_TYPE_Q2_0 = 42 };

// qsort replacement: stable insertion sort (the library sorts at most 16 small records)
__device__ void qsort(void * base, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
    unsigned char * b = (unsigned char *)base;
    unsigned char tmp[16];
    for (size_t i = 1; i < n; ++i) {
        memcpy(tmp, b + i * sz, sz);
        size_t j = i;
        while (j > 0 && cmp(b + (j - 1) * sz, tmp) > 0) { memcpy(b + j * sz, b + (j - 1) * sz, sz); --j; }
        if (j != i) memcpy(b + j * sz, tmp, sz);
    }
}
"""


def _fp16_funcs():
    src = open(os.path.join(_DIR, "ggml-impl.h"), encoding="utf-8").read()
    a = src.index("static inline float fp32_from_bits")
    b = src.index("#define GGML_FP32_TO_FP16(x)")
    body = src[a:b] + "#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)\n"
    return body.replace("static inline", "__device__ static inline")


def _common_header():
    src = open(os.path.join(_DIR, "ggml-common.h"), encoding="utf-8").read()
    src = re.sub(r"#include\s*<[^>]+>", "", src).replace("__extension__", "")
    # the CPU's 64-bit iq1s_grid, not the packed GPU variant
    src = src.replace("#if defined(GGML_COMMON_IMPL_C) || defined(GGML_COMMON_IMPL_CPP)", "#if 1")
    return "#define GGML_COMMON_DECL_C\n#define GGML_COMMON_IMPL_CUDA\n" + src


def _strip_comment(line):
    return re.sub(r"//.*", "", line)


def _quants_body(path="ggml-quants.c", select=None, suffix=""):
    """top-level items of ggml-quants.c, functions turned into __device__ functions.
    With `select`, only those functions are kept and renamed with `suffix` (a quantiser variant)."""
    lines = open(os.path.join(_DIR, path), encoding="utf-8").read().split("\n")
    out, item, depth, i = [], [], 0, 0
    pp = []                                   # preprocessor stack: True while in an #else/#elif branch
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if item and s.startswith("#"):
            # braces are counted in the first branch of #if/#else only (both branches open the same block)
            if s.startswith("#if"):
                pp.append(False)
            elif s.startswith("#el") and pp:
                pp[-1] = True
            elif s.startswith("#endif") and pp:
                pp.pop()
            item.append(ln); i += 1
            continue
        if item and any(pp):
            item.append(ln); i += 1
            continue
        if depth == 0 and not item:
            s = ln.strip()
            if s.startswith("#"):
                if s.startswith("#include") or "GGML_COMMON_IMPL_C" in s or select:
                    pass
                else:
                    out.append(ln)
                    while ln.rstrip().endswith("\\"):
                        i += 1; ln = lines[i]; out.append(ln)
                i += 1
                continue
            if not s or s.startswith("//"):
                if not select:
                    out.append(ln)
                i += 1
                continue
        item.append(ln)
        code = _strip_comment(ln)
        depth += code.count("{") - code.count("}")
        if depth == 0 and (code.rstrip().endswith("}") or code.rstrip().endswith(";")):
            text = _device_item("\n".join(item))
            if select:
                name = re.search(r"(\w+)\s*\(", text.split("{", 1)[0]) if text.startswith("__device__") else None
                text = text if name and name.group(1) in select else ""
                for f in select:
                    text = re.sub(rf"\b{f}\b", f + suffix, text)
            out.append(text)
            item = []
        i += 1
    return "\n".join(out)


def _device_item(text):
    head = text.split("{", 1)[0]
    if "(" in head and "=" not in head and head.rstrip().endswith(")"):
        name = re.search(r"(\w+)\s*\(", head).group(1)
        if name in _SKIP:
            return ""
        return "__device__ " + text
    m = re.match(r"\s*static\s+(iq[23]_entry_t)\s+(iq[23]_data)\[(\d+)\]", text)
    if m:
        return f'extern "C" {{ __device__ {m.group(1)} {m.group(2)}[{m.group(3)}]; }}'
    return text


def _cxx_fixes(src):
    """C -> C++: implicit conversions from void * that C++ rejects"""
    src = re.sub(r"(\b(?:const\s+)?(\w+)\s*\*\s*(?:GGML_RESTRICT\s+)?\w+\s*=\s*)(vy|left|right|y)\s*;",
                 lambda m: f"{m.group(1)}({'const ' if m.group(0).lstrip().startswith('const') else ''}{m.group(2)} *){m.group(3)};", src)
    src = re.sub(r"(quantize_row_(\w+)_ref\()src, dst,", r"\1src, (block_\2 *)dst,", src)
    return src


# Older llama.cpp quantisers whose arithmetic differs from the current one: ggml type ->
# [(source file in third_party/ggml/variants, functions to take from it)]. Variant 0 is the
# current ggml-quants.c, variant i the i-th entry; the archive records the variant per tensor.
VARIANTS = {
    # f95b0d9539 (2026-09) rebuilt the IQ1_M split search on prefix sums, changing its rounding
    29: [("variants/ggml-quants.pre-f95b0d9539.c", ["quantize_row_iq1_m_impl", "quantize_iq1_m"])],
}


def _variant_bodies():
    return "\n".join(_quants_body(path, set(funcs), f"_v{i + 1}")
                     for vs in VARIANTS.values() for i, (path, funcs) in enumerate(vs))


def _kernel_names():
    """(ggml type, variant) -> quantize function name"""
    names = {(t, 0): fname for t, (fname, _, _) in QFUN.items()}
    for t, vs in VARIANTS.items():
        for i in range(len(vs)):
            names[(t, i + 1)] = f"{QFUN[t][0]}_v{i + 1}"
    return names


def _kernels():
    ks = []
    for fname in _kernel_names().values():
        ks.append(f"""
extern "C" __global__ void k_{fname}(const float * x, unsigned char * y, const long long * xoff, const long long * yoff,
                                     const int * npr, const long long * woff, const float * w, int nrows) {{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    {fname}(x + xoff[r], y + yoff[r], 1, npr[r], woff[r] < 0 ? NULL : w + woff[r]);
}}""")
    for fname, _, _ in QFUN.values():
        t = fname[len("quantize_"):]
        ks.append(f"""
extern "C" __global__ void k_dq_{t}(const unsigned char * y, float * x, int nrows, int npr, int rb) {{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    dequantize_row_{t}((const block_{t} *)(y + (size_t)r * rb), x + (size_t)r * npr, npr);
}}""")
    return "\n".join(ks)


def build_source():
    rowsize = """
__device__ static size_t ggml_row_size(enum ggml_type t, int64_t ne) {
    switch (t) {""" + "".join(f"\n        case {t}: return ne / {qk} * {bs};" for t, (_, qk, bs) in QFUN.items()) + """
        default: __trap(); return 0;
    }
}
"""
    src = (_PRELUDE + _fp16_funcs() + _common_header() + rowsize + _cxx_fixes(_quants_body())
           + _cxx_fixes(_variant_bodies()) + _kernels())
    with open(os.path.join(cache_dir(), "gpuquant.cu"), "w", encoding="utf-8") as f:
        f.write(src)
    return src


_bf16_to_f32 = cp.ElementwiseKernel("uint16 a", "float32 y", "y = __uint_as_float(((unsigned int)a) << 16)",
                                    "vq_bf16_to_f32")


def _to_f32_into(a, out):
    """weights (bf16 bits as uint16 / f16 / f32, or an object with into(out)) -> float32, written
    into `out`"""
    if hasattr(a, "into"):
        a.into(out)
    elif a.dtype == cp.uint16:
        _bf16_to_f32(a, out)
    else:
        out[...] = a


PINNED = [0]      # bytes of pinned host memory handed out since the pool was last emptied


def to_host(a):
    """device array -> numpy array in pinned memory (twice the bandwidth of a pageable copy).
    The blocks return to CuPy's pinned pool and are reused; the caller empties the pool now and
    then (repo3._release), since on Windows pinned blocks count against the commit charge.
    Pinned memory is a scarce resource (WSL limits it): when it runs out, the pool's free blocks
    go back to the OS and the copy is retried, then falls back to pageable memory."""
    try:
        host = cupyx.empty_pinned(a.shape, a.dtype)
    except cp.cuda.runtime.CUDARuntimeError:
        cp.get_default_pinned_memory_pool().free_all_blocks()
        try:
            host = cupyx.empty_pinned(a.shape, a.dtype)
        except cp.cuda.runtime.CUDARuntimeError:
            return a.get()
    a.get(out=host)
    PINNED[0] += a.nbytes
    return host


class GPUQuant:
    def __init__(self, fmad=False):
        """fmad=True lets the compiler fuse a*b+c into FMA, as a CPU build with -march=native
        and floating-point contraction does (some published quantisations were made that way)"""
        src = build_source()
        # the quantisers keep block-local arrays on the stack (up to ~0.8 KB per thread in the
        # optimised build; the default limit is 1 KB). The limit is reserved for every resident
        # thread, so it is kept small.
        cp.cuda.runtime.deviceSetLimit(0, 4096)
        # fully optimised, but with divisions by literals made opaque to ptxas (see ptxfix.py);
        # the cubin is cached next to the generated source, per GPU architecture
        opts = (f"--fmad={'true' if fmad else 'false'}", "--prec-div=true", "--prec-sqrt=true", "-std=c++17", "--diag-suppress=177,550,1650")
        key = hashlib.sha1((src + repr(opts) + repr(cp.cuda.Device().compute_capability)).encode()).hexdigest()[:16]
        path = os.path.join(cache_dir(), f"gpuquant-{key}.cubin")
        if not os.path.exists(path):
            ptx, _ = opaque_divisions(ptx_of(src, opts))
            with open(path + ".tmp", "wb") as f:
                f.write(link_cubin(ptx))
            os.replace(path + ".tmp", path)
        self.mod = cp.RawModule(path=path)
        self._upload_tables()

    def _upload_tables(self):
        """IQ grids, maps and neighbour lists, as built by the CPU library's init functions
        (slow: ~20 s): shipped in ggml/iqtables.npz, else built once into the cache"""
        path = os.path.join(_DIR, "iqtables.npz")
        if not os.path.exists(path):
            path = os.path.join(cache_dir(), "iqtables.npz")
        if not os.path.exists(path):
            np.savez(path + ".tmp.npz", **self._build_tables())
            os.replace(path + ".tmp.npz", path)
        tb = np.load(path)
        self._keep = []
        for glob_name, n in (("iq2_data", 4), ("iq3_data", 2)):
            e = np.zeros((n, 3), np.uint64)
            for gi in range(n):
                e[gi] = [self._dev(tb[f"{glob_name}_{gi}_{k}"]) for k in ("grid", "map", "nb")]
            cp.cuda.runtime.memcpy(self.mod.get_global(glob_name).ptr, e.ctypes.data, e.nbytes, 1)

    def _build_tables(self):
        L = lib()
        L.vq_iq2_table.restype = ctypes.c_void_p
        L.vq_iq3_table.restype = ctypes.c_void_p
        out = {}
        for t in sorted(IQ2_INIT):
            L.iq2xs_init_impl(t)
        for gi, gsize in enumerate((256, 512, 2048, 1024)):
            out[f"iq2_data_{gi}_grid"] = np.ctypeslib.as_array(ctypes.cast(L.vq_iq2_table(gi, 0), ctypes.POINTER(ctypes.c_uint64)), (gsize,)).copy()
            out[f"iq2_data_{gi}_map"] = np.ctypeslib.as_array(ctypes.cast(L.vq_iq2_table(gi, 1), ctypes.POINTER(ctypes.c_int32)), (43692,)).copy()
            out[f"iq2_data_{gi}_nb"] = self._neighbours(L.vq_iq2_table(gi, 2), out[f"iq2_data_{gi}_map"])
        for gsize in sorted(set(IQ3_INIT.values())):
            L.iq3xs_init_impl(gsize)
        for gi, gsize in enumerate((256, 512)):
            out[f"iq3_data_{gi}_grid"] = np.ctypeslib.as_array(ctypes.cast(L.vq_iq3_table(gi, 0), ctypes.POINTER(ctypes.c_uint32)), (gsize,)).copy()
            out[f"iq3_data_{gi}_map"] = np.ctypeslib.as_array(ctypes.cast(L.vq_iq3_table(gi, 1), ctypes.POINTER(ctypes.c_int32)), (4096,)).copy()
            out[f"iq3_data_{gi}_nb"] = self._neighbours(L.vq_iq3_table(gi, 2), out[f"iq3_data_{gi}_map"])
        return out

    @staticmethod
    def _neighbours(ptr, kmap):
        """neighbour lists: for every off-grid map entry m < 0, [n, n indices] at offset -m - 1"""
        offs = -kmap[kmap < 0].astype(np.int64) - 1
        p = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint16))
        end = max(int(o) + 1 + p[int(o)] for o in offs)
        return np.ctypeslib.as_array(p, (end,)).copy()

    def _dev(self, a):
        d = cp.asarray(a)
        self._keep.append(d)
        return d.data.ptr

    def quantize_tensors(self, ttype, srcs, imatrices, variant=0, max_weights=1 << 28, on_result=None):
        """srcs: list of (device array of the weights as bf16 (uint16) / f16 / f32, n_per_row).
        Converts to float32 on the GPU in launches of at most `max_weights` weights;
        returns the quantised bytes of every tensor as host numpy arrays, or, with on_result,
        calls on_result(index, bytes) per tensor as each launch completes and returns nothing."""
        out, i = [], 0
        while i < len(srcs):
            j, n = i, 0
            while j < len(srcs) and (j == i or n + srcs[j][0].size <= max_weights):
                n += srcs[j][0].size; j += 1
            # all tensors of the launch converted straight into one input buffer (no extra copies)
            X = cp.empty(n, cp.float32)
            xs, o = [], 0
            for a, npr in srcs[i:j]:
                _to_f32_into(a, X[o:o + a.size]); xs.append(X[o:o + a.size].reshape(-1, npr)); o += a.size
            Y, spans = self._launch(ttype, xs, imatrices[i:j], variant, X)
            del X, xs
            host = to_host(Y)
            del Y
            for k, (a, b) in enumerate(spans):
                if on_result:
                    on_result(i + k, host[a:a + b])
                else:
                    out.append(host[a:a + b])
            del host
            i = j
        return None if on_result else out

    def dequantize(self, ttype, y, npr, out=None):
        """quantised bytes of a tensor (device uint8, rows of npr weights) -> float32 device array
        (or into `out`), with llama.cpp's own dequantize_row_<type>"""
        fname, qk, bs = QFUN[ttype]
        rb = npr // qk * bs
        nrows = y.size // rb
        x = cp.empty(nrows * npr, cp.float32) if out is None else out
        k = self.mod.get_function(f"k_dq_{fname[len('quantize_'):]}")
        k(((nrows + 63) // 64,), (64,), (y, x, np.int32(nrows), np.int32(npr), np.int32(rb)))
        return x

    def quantize_batch(self, ttype, xs, imatrices, variant=0):
        """quantise several tensors of one type in one launch (with quantiser `variant`, see VARIANTS).
        xs: list of float32 cupy arrays (nrows, n_per_row); imatrices: list of float32 numpy arrays or None.
        Returns list of cupy uint8 arrays with the quantised bytes."""
        X = cp.concatenate([x.reshape(-1) for x in xs]) if len(xs) > 1 else xs[0].reshape(-1)
        Y, spans = self._launch(ttype, xs, imatrices, variant, X)
        return [Y[o:o + n] for o, n in spans]

    def _launch(self, ttype, xs, imatrices, variant, X):
        """xs: the tensors' shapes (as arrays) laid out back to back in X -> (Y, [(offset, nbytes)])"""
        fname, qk, bs = QFUN[ttype]
        rows_x, rows_y, rows_n, rows_w, outs = [], [], [], [], []
        xo = yo = wo = 0
        ws = []
        for x, imx in zip(xs, imatrices):
            nr, npr = x.shape
            rb = npr // qk * bs
            rows_x.append(xo + np.arange(nr, dtype=np.int64) * npr)
            rows_y.append(yo + np.arange(nr, dtype=np.int64) * rb)
            rows_n.append(np.full(nr, npr, np.int32))
            rows_w.append(np.full(nr, -1, np.int64) if imx is None else np.full(nr, wo, np.int64))
            if imx is not None:
                ws.append(np.asarray(imx, np.float32)); wo += npr
            outs.append((yo, nr * rb))
            xo += nr * npr; yo += nr * rb
        Y = cp.empty(yo, cp.uint8)
        W = cp.asarray(np.concatenate(ws)) if ws else cp.zeros(1, cp.float32)
        nrows = sum(len(r) for r in rows_x)
        args = [cp.asarray(np.concatenate(a)) for a in (rows_x, rows_y, rows_n, rows_w)]
        k = self.mod.get_function(f"k_{_kernel_names()[(ttype, variant)]}")
        tpb = 64
        k(((nrows + tpb - 1) // tpb,), (tpb,), (X, Y, args[0], args[1], args[2], args[3], W, np.int32(nrows)))
        return Y, outs
