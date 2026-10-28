"""NumPy stand-in for the parts of CuPy SeedPack uses, for --cpu (xp.py picks it instead of cupy).

Device arrays are numpy arrays of a subclass (DArray) with CuPy's transfer methods (.get, .set),
whose element access returns 0-d arrays as CuPy's does; array functions are numpy's, their array
results made DArrays; RawModule compiles the same CUDA source for the CPU (cpukernels.py); the
runtime, devices and memory pools are stand-ins. "Device" memory is host memory, as much as the
machine has: free is half of what memory.py leaves this process at that moment (the other half is
for the host-side buffers, which follow the same room), total that plus the large arrays alive;
$SEEDPACK_CPU_MEM (MB) fixes the total instead. The archive does not depend on it."""
import ctypes
import os
import threading
import types
import weakref

import numpy as np

from . import cpukernels, memory

CPU = True


class DArray(np.ndarray):
    """an array in "device" memory (host memory), with CuPy's transfer methods"""

    def get(self, stream=None, order="C", out=None, blocking=True):
        if out is None:
            return np.array(self, order=order, copy=True)
        if out.shape != self.shape:
            out = out.reshape(self.shape)
        np.copyto(out, self, casting="no")
        return out

    def set(self, arr, stream=None):
        self[...] = np.asarray(arr).reshape(self.shape) if np.size(arr) == self.size else arr

    def __getitem__(self, key):
        r = super().__getitem__(key)
        return r if isinstance(r, np.ndarray) else np.asarray(r).view(DArray)

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        return arr.view(DArray) if isinstance(arr, np.ndarray) else arr

    def __array_function__(self, func, types, args, kwargs):
        return _dev(super().__array_function__(func, types, args, kwargs))

    # CuPy sorts stably (CUB radix sort): ties keep their order, which numpy's default does not
    def argsort(self, axis=-1, kind=None, order=None, stable=None):
        return np.ndarray.argsort(self, axis=axis, kind="stable", order=order)

    def sort(self, axis=-1, kind=None, order=None, stable=None):
        return np.ndarray.sort(self, axis=axis, kind="stable", order=order)


ndarray = DArray

_live = [0]                                 # bytes of large device arrays alive
_live_lock = threading.Lock()
_TRACK = 1 << 20


def _forget(n):
    with _live_lock:
        _live[0] -= n


def _dev(x):
    if type(x) is np.ndarray:
        return _track(x.view(DArray))
    if isinstance(x, tuple):
        return tuple(_dev(v) for v in x)
    if isinstance(x, list):
        return [_dev(v) for v in x]
    return x


def _track(a):
    """a DArray viewing a fresh numpy array: counted against the "device" memory (if large) until
    that array is freed; views of other arrays are not counted again"""
    owner = a.base
    if type(owner) is np.ndarray and owner.base is None and owner.nbytes == a.nbytes and a.nbytes >= _TRACK:
        with _live_lock:
            _live[0] += a.nbytes
        weakref.finalize(owner, _forget, a.nbytes)
    return a


def asarray(a, dtype=None, order=None, blocking=False):
    if isinstance(a, DArray):
        return a if dtype is None or a.dtype == np.dtype(dtype) else a.astype(dtype)
    return _track(np.array(a, dtype=dtype, order=order or "K", copy=True).view(DArray))


array = asarray


def asnumpy(a, stream=None, order="C", out=None, blocking=True):
    if isinstance(a, DArray):
        return a.get(order=order, out=out)
    if out is not None:
        np.copyto(out, a)
        return out
    return np.asarray(a, order=order)


def ascontiguousarray(a, dtype=None):
    return _dev(np.ascontiguousarray(a, dtype=dtype)) if isinstance(a, DArray) else asarray(a, dtype, "C")


def _wrap(f):
    def g(*args, **kw):
        return _dev(f(*args, **kw))
    g.__name__ = getattr(f, "__name__", "f")
    return g


def argsort(a, axis=-1, kind=None):
    return _dev(np.argsort(a, axis=axis, kind="stable"))


def sort(a, axis=-1, kind=None):
    return _dev(np.sort(a, axis=axis, kind="stable"))


for _n in ("zeros", "empty", "ones", "full", "arange", "eye", "zeros_like", "empty_like", "ones_like",
           "full_like", "concatenate", "stack", "linspace", "frombuffer", "tile", "repeat"):
    globals()[_n] = _wrap(getattr(np, _n))


def __getattr__(name):
    """everything else: numpy's (types and ufuncs as they are, functions with device results)"""
    a = getattr(np, name)
    if isinstance(a, (type, np.ufunc)) or not callable(a):
        return a
    return _wrap(a)


class _NS(types.SimpleNamespace):
    pass


def _wrap_ns(mod, names):
    return _NS(**{n: _wrap(getattr(mod, n)) for n in names})


linalg = _wrap_ns(np.linalg, ("solve", "norm", "eigh", "inv", "lstsq", "cholesky", "qr", "svd", "det"))


class _RandomState:
    def __init__(self, seed=None):
        self._r = np.random.RandomState(seed)

    def __getattr__(self, name):
        return _wrap(getattr(self._r, name))


random = _NS(RandomState=_RandomState, seed=np.random.seed)


# ---------------------------------------------------------------------------------- runtime, memory
class CUDARuntimeError(RuntimeError):
    pass


class OutOfMemoryError(MemoryError):
    pass


def _mem_get_info():
    """("device" bytes free, total), read at every call"""
    with _live_lock:
        used = _live[0]
    mb = os.environ.get("SEEDPACK_CPU_MEM")
    if mb:
        total = int(mb) << 20
        return max(0, total - used), total
    free = max(0, memory.room() // 2)
    return free, free + used


runtime = _NS(memGetInfo=_mem_get_info, CUDARuntimeError=CUDARuntimeError, getDeviceCount=lambda: 1,
              memcpy=lambda dst, src, size, kind=None: ctypes.memmove(dst, src, size),
              deviceSetLimit=lambda *a: None, deviceSynchronize=lambda: None)


class Device:
    compute_capability = "cpu"

    def __init__(self, device=0):
        self.id = 0

    def synchronize(self):
        pass

    @property
    def mem_info(self):
        return _mem_get_info()

    def use(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Stream:
    null = None

    def __init__(self, null=False, non_blocking=False, ptds=False):
        pass

    def synchronize(self):
        pass

    def use(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


Stream.null = Stream()


class Event:
    def __init__(self, *a, **kw):
        pass

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


cuda = _NS(runtime=runtime, Device=Device, Stream=Stream, Event=Event, get_current_stream=lambda: Stream.null,
           memory=_NS(OutOfMemoryError=OutOfMemoryError))


class _Pool:
    def free_all_blocks(self, stream=None):
        pass

    def free_bytes(self):
        return 0

    def used_bytes(self):
        return _live[0]

    def total_bytes(self):
        return _live[0]

    def n_free_blocks(self):
        return 0

    def set_limit(self, size=None, fraction=None):
        pass


_POOL, _PINNED = _Pool(), _Pool()


def get_default_memory_pool():
    return _POOL


def get_default_pinned_memory_pool():
    return _PINNED


RawModule = cpukernels.RawModule


# ---------------------------------------------------------------------------------- elementwise kernels
def _bf16_to_f32(a, y=None):
    r = (np.asarray(a).astype(np.uint32) << 16).view(np.float32)
    if y is None:
        return asarray(r)
    y[...] = r
    return y


_ELEMENTWISE = {"vq_bf16_to_f32": _bf16_to_f32}


def ElementwiseKernel(in_params, out_params, operation, name="kernel", **kw):
    if name not in _ELEMENTWISE:
        raise NotImplementedError(f"ElementwiseKernel {name} has no CPU version")
    return _ELEMENTWISE[name]


cupyx = _NS(empty_pinned=lambda shape, dtype=float, order="C": np.empty(shape, dtype, order),
            zeros_pinned=lambda shape, dtype=float, order="C": np.zeros(shape, dtype, order))
