"""CUDA kernels on the CPU, for --cpu: the CUDA C++ source a RawModule would compile with NVRTC is
compiled as C++ for the host, behind a prelude that supplies what CUDA provides (thread and block
indices, IEEE intrinsics, atomics, dynamic shared memory), and every launch runs its blocks on
CPU threads, the threads of a block one after another.

Why the results are the GPU's, bit for bit: the kernels never use __syncthreads or warp
communication (the histograms' __match_any_sync aggregation degrades to one lane, which adds the
same counts), so no thread depends on the others' schedule; and the GPU builds use --fmad=false
with IEEE division and square root, which is plain IEEE single precision, as is C++ on x86-64
without FMA contraction or fast-math (-ffp-contract=off).

Compiler: $SEEDPACK_CXX, else the `ziglang` package (pip install "seedpack[cpu]"), else clang++ / g++ on
the PATH. The libraries are cached next to the GPU kernels (gpuquant.cache_dir()), keyed by the
source, the flags and the compiler."""
import atexit
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PRELUDE = r"""
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef _WIN32
#define SP_EXPORT extern "C" __declspec(dllexport)
#else
#define SP_EXPORT extern "C" __attribute__((visibility("default")))
#endif
using std::isnan; using std::isinf; using std::isfinite; using std::signbit;
struct dim3 { unsigned int x, y, z; };
static thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
static thread_local unsigned char* sp_shmem_;
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __constant__
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __syncthreads() ((void)0)
#define __threadfence() ((void)0)
#define __trap() __builtin_trap()
#define __ldg(p) (*(p))
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __frcp_rn(float a) { return 1.0f / a; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline float __fmaf_rn(float a, float b, float c) { return fmaf(a, b, c); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline unsigned int __float_as_uint(float f) { unsigned int u; memcpy(&u, &f, 4); return u; }
static inline float __uint_as_float(unsigned int u) { float f; memcpy(&f, &u, 4); return f; }
static inline long long __double_as_longlong(double d) { long long u; memcpy(&u, &d, 8); return u; }
static inline double __longlong_as_double(long long u) { double d; memcpy(&d, &u, 8); return d; }
static inline int __float_as_int(float f) { int u; memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int u) { float f; memcpy(&f, &u, 4); return f; }
static inline int __popc(unsigned int x) { return __builtin_popcount(x); }
static inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
static inline int __ffs(int x) { return __builtin_ffs(x); }
static inline int __ffsll(long long x) { return __builtin_ffsll(x); }
static inline int __clz(int x) { return x ? __builtin_clz((unsigned int)x) : 32; }
static inline int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
// one lane at a time: a warp of one
static inline unsigned int __activemask() { return 1u << (threadIdx.x & 31); }
template <class T> static inline unsigned int __match_any_sync(unsigned int mask, T) { return mask; }
static inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
static inline unsigned int atomicAdd(unsigned int* p, unsigned int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
static inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
template <class F> static inline F sp_atomic_add_float(F* p, F v) {
    F old, nv;
    __atomic_load(p, &old, __ATOMIC_RELAXED);
    do { nv = old + v; } while (!__atomic_compare_exchange(p, &old, &nv, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    return old;
}
static inline float atomicAdd(float* p, float v) { return sp_atomic_add_float(p, v); }
static inline double atomicAdd(double* p, double v) { return sp_atomic_add_float(p, v); }
template <class T> static inline T atomicMax(T* p, T v) { T o = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (o < v && !__atomic_compare_exchange_n(p, &o, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {} return o; }
template <class T> static inline T atomicMin(T* p, T v) { T o = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v < o && !__atomic_compare_exchange_n(p, &o, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {} return o; }
template <class T> static inline T atomicOr(T* p, T v) { return __atomic_fetch_or(p, v, __ATOMIC_RELAXED); }
template <class T> static inline T atomicExch(T* p, T v) { return __atomic_exchange_n(p, v, __ATOMIC_RELAXED); }
template <class T> static inline T atomicCAS(T* p, T c, T v) {
    __atomic_compare_exchange_n(p, &c, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED); return c; }
// CUDA's min / max: overloads for every arithmetic pair; floats as fminf / fmaxf
static inline float min(float a, float b) { return fminf(a, b); }
static inline float max(float a, float b) { return fmaxf(a, b); }
static inline double min(double a, double b) { return fmin(a, b); }
static inline double max(double a, double b) { return fmax(a, b); }
template <class A, class B> static inline typename std::common_type<A, B>::type min(A a, B b) {
    typedef typename std::common_type<A, B>::type C; return (C)b < (C)a ? (C)b : (C)a; }
template <class A, class B> static inline typename std::common_type<A, B>::type max(A a, B b) {
    typedef typename std::common_type<A, B>::type C; return (C)a < (C)b ? (C)b : (C)a; }
// rounding, exactly as CUDA does it, whatever the C library (MinGW's roundf(0.49999997f) is 1)
static inline float sp_truncf(float x) { return fabsf(x) < 8388608.0f ? copysignf((float)(int)x, x) : x; }
static inline float sp_floorf(float x) { float t = sp_truncf(x); return t > x ? t - 1.0f : t; }
static inline float sp_ceilf(float x) { float t = sp_truncf(x); return t < x ? t + 1.0f : t; }
static inline float sp_roundf(float x) { float t = sp_truncf(x); return fabsf(x - t) >= 0.5f ? t + copysignf(1.0f, x) : t; }
static inline float sp_rintf(float x) {           // half to even: the addition rounds
    if (!(fabsf(x) < 8388608.0f)) return x;
    float y = fabsf(x) + 8388608.0f; y = y - 8388608.0f; return copysignf(y, x); }
#define truncf sp_truncf
#define floorf sp_floorf
#define ceilf sp_ceilf
#define roundf sp_roundf
#define rintf sp_rintf
#define nearbyintf sp_rintf
#define lrintf(x) ((long)sp_rintf(x))
#define lroundf(x) ((long)sp_roundf(x))
template <class T> static int sp_kind() {
    return std::is_pointer<T>::value ? 1 : std::is_floating_point<T>::value ? 4
         : std::is_integral<T>::value ? (std::is_signed<T>::value ? 2 : 3) : 5; }
#line 1 "kernels"
"""

_SHARED = re.compile(r"extern\s+__shared__\s+([\w\s]+?)\s+(\w+)\s*\[\s*\]\s*;")


_PROFILE = {} if os.environ.get("SEEDPACK_CPU_PROFILE") else None   # kernel -> seconds, printed at exit


def _report():
    if _PROFILE:
        tot = sum(_PROFILE.values())
        print(f"CPU kernels: {tot:.1f} s", file=sys.stderr)
        for k, v in sorted(_PROFILE.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {k:28s} {v:8.2f} s", file=sys.stderr)


if _PROFILE is not None:
    atexit.register(_report)
    _PROFILE["(start)"] = 0.0


def threads():
    """CPU threads a launch uses: $SEEDPACK_THREADS, else all"""
    return max(1, int(os.environ.get("SEEDPACK_THREADS") or os.cpu_count() or 1))


_pool = None
_pool_lock = threading.Lock()


def _workers():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(threads(), thread_name_prefix="seedpack-cpu")
    return _pool


def compiler():
    """the C++ compiler command (a list), or RuntimeError with what to install"""
    env = os.environ.get("SEEDPACK_CXX")
    if env:
        return env.split()
    try:
        import ziglang  # noqa: F401
        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        pass
    for c in ("clang++", "g++", "c++"):           # MinGW g++: thread_local is emulated, kernels run ~2x slower
        p = shutil.which(c)
        if p:
            return [p]
    raise RuntimeError("--cpu compiles SeedPack's kernels for the CPU and needs a C++ compiler: "
                       "pip install \"seedpack[cpu]\" (ziglang; or put clang++ / g++ on the PATH, or set SEEDPACK_CXX)")


_ident = {}


def _compiler_id(cc):
    key = tuple(cc)
    if key not in _ident:
        try:
            out = subprocess.run(cc + ["--version"], capture_output=True, text=True, timeout=120).stdout
            _ident[key] = out.strip().splitlines()[0] if out.strip() else " ".join(cc)
        except (OSError, subprocess.TimeoutExpired):
            _ident[key] = " ".join(cc)
    return _ident[key]


def _flags(cc):
    f = ["-O2", "-std=c++17", "-ffp-contract=off", "-fno-fast-math", "-fno-math-errno", "-fno-strict-aliasing",
         "-fwrapv", "-w", "-shared"]
    if os.name != "nt":
        f += ["-fPIC"]
    elif "zig" not in " ".join(cc).lower():
        f += ["-static"]                     # MinGW: no runtime DLLs (libstdc++, winpthread) to find at load time
    return f


def _params(plist):
    """'const uchar* F, int n' -> [(type, name)]"""
    out = []
    for p in (x.strip() for x in plist.split(",")):
        if not p:
            continue
        m = re.match(r"(.*?)(\w+)\s*$", p, re.S)
        typ = re.sub(r"\b__restrict(__)?\b", "", m.group(1)).strip()
        out.append((" ".join(typ.split()), m.group(2)))
    return out


def _globals(src):
    """names of the file-scope __device__ / __constant__ variables (for get_global)"""
    names = []
    for ln in src.split("\n"):
        if ln.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\b(?:__device__|__constant__)\b", ln):
            rest = ln[m.end():]
            cut = min((i for i in (rest.find(c) for c in "(;={") if i >= 0), default=-1)
            if cut < 0 or rest[cut] == "(":
                continue
            decl = rest[:cut].split("[")[0]
            ids = re.findall(r"\w+", decl)
            if ids and ids[-1] not in ("static", "const", "inline") and len(ids) >= 2:
                names.append(ids[-1])
    return sorted(set(names))


def _signatures(code, cc):
    """[(kernel, [(type, name)])] from the preprocessed source (parameter lists may be macros)"""
    from .gpuquant import cache_dir
    tmp = os.path.join(cache_dir(), f"sig-{os.getpid()}-{threading.get_ident()}.cpp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        r = subprocess.run(cc + ["-E", "-P", "-x", "c++", "-D__global__=SP_KERNEL_MARK", tmp],
                           capture_output=True, text=True)
    finally:
        os.remove(tmp)
    if r.returncode != 0:
        raise RuntimeError(f"preprocessing the CPU kernels failed ({' '.join(cc)}):\n{r.stderr[-4000:]}")
    mark = re.compile(r'extern\s+"C"\s+SP_KERNEL_MARK\s+void\s+(\w+)\s*\(([^)]*)\)', re.S)
    return [(m.group(1), _params(m.group(2))) for m in mark.finditer(r.stdout)]


_LITDIV = re.compile(r"/\s*((?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF])\b")


def translate(code, kernels, exact_division=False):
    """CUDA source and its kernels' signatures -> C++ source with launch / argspec / global accessors.
    A float division by a literal becomes a multiplication by its reciprocal, which is what ptxas
    makes of it (x / 0.9875f == x * (1 / 0.9875f) in float, on every input); with exact_division
    (sources whose PTX went through ptxfix.opaque_divisions) it stays a division."""
    body = _SHARED.sub(lambda m: f"#define {m.group(2)} (({m.group(1).strip()}*)sp_shmem_)", code)
    if not exact_division:
        body = _LITDIV.sub(lambda m: f"* (1.0f / {m.group(1)})", body)
    tail = []
    for name, ps in kernels:
        decl = "".join(f"    {t} a{i} = *({t} *)a[{i}];\n" for i, (t, _) in enumerate(ps))
        call = ", ".join(f"a{i}" for i in range(len(ps)))
        spec = "".join(f"    kind[{i}] = sp_kind<{t}>(); size[{i}] = (int)sizeof({t});\n" for i, (t, _) in enumerate(ps))
        tail.append(f"""
SP_EXPORT int sp_argspec_{name}(int* kind, int* size) {{
{spec}    return {len(ps)};
}}
SP_EXPORT void sp_launch_{name}(const unsigned int* g, const unsigned int* b, unsigned long long shmem, void** a,
                                long long t0, long long t1) {{
    // threads t0 .. t1 - 1 of the launch (numbered block by block); a thread uses only its own
    // slots of the block's shared memory, so this worker's copy of it serves all of them
{decl}    gridDim.x = g[0]; gridDim.y = g[1]; gridDim.z = g[2];
    blockDim.x = b[0]; blockDim.y = b[1]; blockDim.z = b[2];
    unsigned char* sh = shmem ? new unsigned char[shmem] : nullptr;
    sp_shmem_ = sh;
    const long long tpb = (long long)b[0] * b[1] * b[2], bxy = (long long)b[0] * b[1], gxy = (long long)g[0] * g[1];
    const long long k = t0 / tpb, r = t0 % tpb;
    dim3 bi = {{(unsigned int)(k % g[0]), (unsigned int)((k / g[0]) % g[1]), (unsigned int)(k / gxy)}};
    dim3 ti = {{(unsigned int)(r % b[0]), (unsigned int)((r / b[0]) % b[1]), (unsigned int)(r / bxy)}};
    blockIdx = bi;
    for (long long t = t0; t < t1; t++) {{
        threadIdx = ti;
        {name}({call});
        if (++ti.x == b[0]) {{                   // the next thread: indices counted, not divided
            ti.x = 0;
            if (++ti.y == b[1]) {{
                ti.y = 0;
                if (++ti.z == b[2]) {{
                    ti.z = 0;
                    if (++bi.x == g[0]) {{ bi.x = 0; if (++bi.y == g[1]) {{ bi.y = 0; ++bi.z; }} }}
                    blockIdx = bi;
                }}
            }}
        }}
    }}
    delete[] sh;
}}""")
    for g in _globals(body):
        tail.append(f"SP_EXPORT void* sp_global_{g}() {{ return (void*)&{g}; }}")
    return PRELUDE + body + "\n" + "\n".join(tail) + "\n"


_built = {}
_build_lock = threading.Lock()


def build(code, exact_division=False):
    """compile (once; cached on disk, keyed by the CUDA source, this module's prelude and wrappers,
    the flags and the compiler) -> ctypes library"""
    cc = compiler()
    flags = _flags(cc)
    with open(__file__, "rb") as f:
        me = f.read()
    key = hashlib.sha1(code.encode() + me + (" ".join(flags) + _compiler_id(cc) + str(exact_division)).encode()).hexdigest()[:16]
    with _build_lock:
        if key in _built:
            return _built[key]
        from .gpuquant import cache_dir
        path = os.path.join(cache_dir(), f"cpu-{key}" + (".dll" if os.name == "nt" else ".so"))
        if not os.path.exists(path):
            kernels = _signatures(code, cc)
            src = translate(code, kernels, exact_division)
            cpp = path.rsplit(".", 1)[0] + ".cpp"
            with open(cpp, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"  compiling {len(kernels)} kernel(s) for the CPU (once, cached in {os.path.dirname(path)}) ...",
                  file=sys.stderr, flush=True)
            tmp = path + f".{os.getpid()}.tmp"
            r = subprocess.run(cc + flags + ["-o", tmp, cpp], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(tmp):
                raise RuntimeError(f"compiling the CPU kernels failed ({' '.join(cc)}):\n{(r.stderr or r.stdout)[-4000:]}")
            os.replace(tmp, path)
        _built[key] = ctypes.CDLL(path)
        return _built[key]


_CT = {(2, 1): ctypes.c_int8, (2, 2): ctypes.c_int16, (2, 4): ctypes.c_int32, (2, 8): ctypes.c_int64,
       (3, 1): ctypes.c_uint8, (3, 2): ctypes.c_uint16, (3, 4): ctypes.c_uint32, (3, 8): ctypes.c_uint64,
       (4, 4): ctypes.c_float, (4, 8): ctypes.c_double}


def address(a):
    """a kernel's pointer argument: an array's data, a raw address, or None"""
    if a is None:
        return 0
    if isinstance(a, int):
        return a
    if hasattr(a, "ctypes"):
        return a.ctypes.data
    raise TypeError(f"cannot pass {type(a).__name__} as a pointer to a CPU kernel")


def _dim(d):
    d = (d,) if isinstance(d, int) else tuple(int(x) for x in d)
    return (d + (1, 1, 1))[:3]


class Kernel:
    def __init__(self, lib, name):
        self.name = name
        self.fn = getattr(lib, f"sp_launch_{name}")
        self.fn.restype = None
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p,
                            ctypes.c_longlong, ctypes.c_longlong]
        spec = getattr(lib, f"sp_argspec_{name}")
        kind, size = (ctypes.c_int * 64)(), (ctypes.c_int * 64)()
        n = spec(kind, size)
        self.spec = [(kind[i], size[i]) for i in range(n)]

    def _box(self, i, value):
        k, s = self.spec[i]
        if k == 1:
            return ctypes.c_void_p(address(value))
        if k in (2, 3, 4):
            v = float(value) if k == 4 else int(value)
            if k == 3:
                v &= (1 << (8 * s)) - 1
            return _CT[(k, s)](v)
        data = value.tobytes() if hasattr(value, "tobytes") else bytes(value)
        if len(data) != s:
            raise TypeError(f"{self.name}: argument {i} is a {s}-byte struct, got {len(data)} bytes")
        return ctypes.create_string_buffer(data, s)

    def __call__(self, grid, block, args, shared_mem=0, stream=None, **kw):
        if len(args) != len(self.spec):
            raise TypeError(f"{self.name}: {len(self.spec)} arguments expected, got {len(args)}")
        g, b = _dim(grid), _dim(block)
        boxes = [self._box(i, v) for i, v in enumerate(args)]
        argv = (ctypes.c_void_p * max(1, len(boxes)))(*[ctypes.addressof(x) for x in boxes])
        gp, bp = (ctypes.c_uint * 3)(*g), (ctypes.c_uint * 3)(*b)
        if _PROFILE:
            t = time.perf_counter()
            try:
                return self._run(g, b, gp, bp, argv, args, shared_mem)
            finally:
                _PROFILE[self.name] = _PROFILE.get(self.name, 0.0) + time.perf_counter() - t
        return self._run(g, b, gp, bp, argv, args, shared_mem)

    def _run(self, g, b, gp, bp, argv, args, shared_mem):
        # split by threads, not blocks: a coding launch has few blocks of long-running threads
        nt = g[0] * g[1] * g[2] * b[0] * b[1] * b[2]
        parts = 1 if nt < 64 or threads() == 1 else min(nt, 4 * threads())
        try:
            if parts <= 1:
                self.fn(gp, bp, int(shared_mem), argv, 0, nt)
                return
            edges = [nt * i // parts for i in range(parts + 1)]
            jobs = [_workers().submit(self.fn, gp, bp, int(shared_mem), argv, edges[i], edges[i + 1])
                    for i in range(parts) if edges[i + 1] > edges[i]]
            for j in jobs:
                j.result()
        except OSError as e:                   # an access violation: which argument's memory was it next to
            spans = [f"arg {i}: {address(v):#x}..{address(v) + v.nbytes:#x}" for i, v in enumerate(args)
                     if self.spec[i][0] == 1 and hasattr(v, "nbytes")]
            raise OSError(f"CPU kernel {self.name} (grid {g}, block {b}): {e}; " + ", ".join(spans)) from None


class Global:
    def __init__(self, ptr):
        self.ptr = ptr


class RawModule:
    """cupy.RawModule's interface over a CPU build of the same source"""

    def __init__(self, code=None, path=None, options=(), name_expressions=None, exact_division=False, **kw):
        """exact_division: the GPU build's divisions by literals are exact (ptxfix.opaque_divisions)"""
        if code is None:
            raise RuntimeError(f"a precompiled GPU module ({path}) cannot run on the CPU")
        self.lib = build(code, exact_division)
        self._k = {}

    def get_function(self, name):
        if name not in self._k:
            self._k[name] = Kernel(self.lib, name)
        return self._k[name]

    def get_global(self, name):
        f = getattr(self.lib, f"sp_global_{name}")
        f.restype = ctypes.c_void_p
        return Global(f())
