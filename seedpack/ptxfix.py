"""Compile CUDA source with NVRTC to PTX, make float divisions by literal constants opaque,
and link the PTX to a cubin with nvJitLink.

Why: ptxas rewrites `div.rn.f32 x, <literal>` (e.g. x / 127) into a multiplication by the
reciprocal, which is off by one ulp in ~4% of cases, while the C quantisers divide exactly.
Loading the divisor from a (writable) global array keeps the exact IEEE division and lets
everything else be fully optimised."""
import ctypes
import glob
import os
import re

import cupy as cp
from cupy_backends.cuda.libs import nvrtc

_DIV = re.compile(r"div\.rn\.f32\s+(%\w+),\s*(%\w+),\s*0[fF]([0-9A-Fa-f]{8});")


def _nvjitlink():
    import nvidia
    name = "nvJitLink*.dll" if os.name == "nt" else "libnvJitLink.so*"
    pats = [os.path.join(p, "**", name) for p in nvidia.__path__]
    dll = next((f for p in pats for f in glob.glob(p, recursive=True)), None)
    if dll is None:
        raise RuntimeError("nvJitLink was not found: pip install nvidia-nvjitlink")
    return ctypes.CDLL(dll)


def ptx_of(src, options):
    cc = cp.cuda.Device().compute_capability
    prog = nvrtc.createProgram(src, "k.cu", [], [])
    try:
        nvrtc.compileProgram(prog, [f"--gpu-architecture=compute_{cc}"] + list(options))
    except Exception:
        raise RuntimeError(nvrtc.getProgramLog(prog))
    ptx = nvrtc.getPTX(prog)
    return ptx.decode() if isinstance(ptx, bytes) else ptx


def opaque_divisions(ptx):
    consts = []

    def fix(m):
        consts.append(m.group(3))
        k = len(consts) - 1
        return (f"{{ .reg .b32 %vqd; ld.global.b32 %vqd, [vq_divc+{4 * k}]; "
                f"div.rn.f32 {m.group(1)}, {m.group(2)}, %vqd; }}")

    ptx = _DIV.sub(fix, ptx)
    if consts:
        decl = ".global .align 4 .b32 vq_divc[%d] = {%s};\n" % (len(consts), ", ".join("0x" + c for c in consts))
        ptx = re.sub(r"(\.address_size \d+\s*\n)", lambda m: m.group(1) + decl, ptx, count=1)
    return ptx, len(consts)


def link_cubin(ptx, ptx_version="9.0"):
    """PTX -> cubin with nvJitLink (its PTX ISA may be older than NVRTC's: the version is lowered)."""
    ptx = re.sub(r"^\.version \d+\.\d+", f".version {ptx_version}", ptx, count=1, flags=re.M)
    L = _nvjitlink()
    cc = cp.cuda.Device().compute_capability
    h = ctypes.c_void_p()
    opts = (ctypes.c_char_p * 2)(f"-arch=sm_{cc}".encode(), b"-O3")
    f = lambda n: getattr(L, next(a for a in (f"__nvJitLink{n}_13_0", f"nvJitLink{n}") if hasattr(L, a)))
    assert f("Create")(ctypes.byref(h), 2, opts) == 0
    data = ptx.encode()
    r = f("AddData")(h, 2, data, ctypes.c_size_t(len(data)), b"k.ptx")
    r2 = f("Complete")(h) if r == 0 else r
    if r2 != 0:
        n = ctypes.c_size_t()
        f("GetErrorLogSize")(h, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value + 1)
        f("GetErrorLog")(h, log)
        raise RuntimeError(f"nvJitLink error {r2}: {log.value.decode(errors='replace')[:3000]}")
    n = ctypes.c_size_t()
    f("GetLinkedCubinSize")(h, ctypes.byref(n))
    cubin = ctypes.create_string_buffer(n.value)
    f("GetLinkedCubin")(h, cubin)
    return cubin.raw
