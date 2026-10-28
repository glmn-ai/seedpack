"""The array module of SeedPack's GPU code: CuPy, or with --cpu (SEEDPACK_CPU=1, set by the CLI
before the codecs are imported) the NumPy stand-in of cpu.py, which runs the same kernels on
the CPU with bit-identical results."""
import os

CPU = os.environ.get("SEEDPACK_CPU") == "1"

if CPU:
    from . import cpu as cp
    cupyx = cp.cupyx
else:
    import cupy as cp
    import cupyx


def ptr(a):
    """address of a device array's data"""
    return a.ctypes.data if CPU else a.data.ptr
