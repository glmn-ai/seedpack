"""Re-run llama.cpp's own quantisers (ggml-quants.c built as ggmlq.dll) on the BF16 weights.

The quantised files of a repository were produced from the same BF16 weights and the same
importance matrix (imatrix_unsloth.dat, shipped in the repository), so re-quantising each
tensor with its own type predicts the stored blocks, scales included, often bit-exactly.
"""
import ctypes
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_DLL = os.path.join(os.path.dirname(__file__), "..", "third_party", "ggml", "ggmlq.dll")
_lib = None
_lock = threading.Lock()
_inited = set()

# ggml type -> (function name, block elements, block bytes)
QFUN = {2: ("quantize_q4_0", 32, 18), 3: ("quantize_q4_1", 32, 20), 6: ("quantize_q5_0", 32, 22),
        7: ("quantize_q5_1", 32, 24), 8: ("quantize_q8_0", 32, 34), 10: ("quantize_q2_K", 256, 84),
        11: ("quantize_q3_K", 256, 110), 12: ("quantize_q4_K", 256, 144), 13: ("quantize_q5_K", 256, 176),
        14: ("quantize_q6_K", 256, 210), 16: ("quantize_iq2_xxs", 256, 66), 17: ("quantize_iq2_xs", 256, 74),
        18: ("quantize_iq3_xxs", 256, 98), 19: ("quantize_iq1_s", 256, 50), 20: ("quantize_iq4_nl", 32, 18),
        21: ("quantize_iq3_s", 256, 110), 22: ("quantize_iq2_s", 256, 82), 23: ("quantize_iq4_xs", 256, 136),
        29: ("quantize_iq1_m", 256, 56)}
IQ2_INIT = {16, 17, 19, 22, 29}            # iq2xs_init_impl(type)
IQ3_INIT = {18: 256, 21: 512}              # iq3xs_init_impl(grid size)


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.abspath(_DLL))
        for name, _, _ in QFUN.values():
            f = getattr(_lib, name)
            f.restype = ctypes.c_size_t
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _lib.iq2xs_init_impl.argtypes = [ctypes.c_int]
        _lib.iq3xs_init_impl.argtypes = [ctypes.c_int]
    return _lib


def _init(t):
    with _lock:
        if t in _inited:
            return
        if t in IQ2_INIT:
            lib().iq2xs_init_impl(t)
        if t in IQ3_INIT:
            lib().iq3xs_init_impl(IQ3_INIT[t])
        _inited.add(t)


def load_imatrix(path_or_bytes, divide=True):
    """llama.cpp imatrix (legacy .dat or GGUF) -> {tensor name: float32 array}, normalised as
    llama-quantize does"""
    buf = open(path_or_bytes, "rb").read() if isinstance(path_or_bytes, str) else path_or_bytes
    if buf[:4] == b"GGUF":
        return _load_imatrix_gguf(buf)
    (n,), pos, out = struct.unpack_from("<i", buf), 4, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<i", buf, pos); pos += 4
        name = buf[pos:pos + ln].decode(); pos += ln
        ncall, nval = struct.unpack_from("<ii", buf, pos); pos += 8
        v = np.frombuffer(buf, np.float32, nval, pos).copy(); pos += 4 * nval
        if divide and ncall > 0:
            v /= np.float32(ncall)
        out[name] = v
    return out


def _load_imatrix_gguf(buf):
    """GGUF imatrix: per tensor '<name>.in_sum2' (n_mat x ne0) and '<name>.counts' (n_mat);
    value = in_sum2 / count per matrix, 1 where the matrix never saw a token"""
    from gguf_reader import GGUF
    g = GGUF.from_bytes(buf)
    ts = {t.name: t for t in g.tensors}
    out = {}
    for name, t in ts.items():
        if not name.endswith(".in_sum2"):
            continue
        base = name[:-len(".in_sum2")]
        sums = np.frombuffer(buf, np.float32, t.nbytes // 4, t.offset)
        ct = ts[base + ".counts"]
        counts = np.frombuffer(buf, np.float32, ct.nbytes // 4, ct.offset)
        with np.errstate(divide="ignore", invalid="ignore"):
            e = sums.reshape(len(counts), -1) / counts[:, None]
        e[counts <= 0] = 1
        out[base] = e.reshape(-1).astype(np.float32)
    return out


def quantize(ttype, x, imatrix=None, threads=12):
    """x: float32 (nrows, n_per_row) -> quantised bytes (numpy uint8) of the whole tensor"""
    fname, qk, bs = QFUN[ttype]
    _init(ttype)
    x = np.ascontiguousarray(x, np.float32)
    nrows, npr = x.shape
    out = np.empty(nrows * npr // qk * bs, np.uint8)
    rb = npr // qk * bs
    imp = None if imatrix is None else np.ascontiguousarray(imatrix, np.float32)
    f = getattr(lib(), fname)
    chunk = max(1, (nrows + threads * 4 - 1) // (threads * 4))

    def run(r0):
        r1 = min(nrows, r0 + chunk)
        f(x[r0:r1].ctypes.data, out[r0 * rb:r1 * rb].ctypes.data, r1 - r0, npr,
          None if imp is None else imp.ctypes.data)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(run, range(0, nrows, chunk)))
    return out
