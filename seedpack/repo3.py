"""Method 28: M27 (whole model repository in one archive) with the re-quantisation on the GPU.

Same idea as M27: the archive holds the BF16 model (M24) and the repository's small files; every
quantised GGUF is stored as the XOR of its tensors with what llama.cpp's quantiser produces from
the BF16 weights and the repository's imatrix. Differences from M27:
  * the quantisers run on the GPU (gpuquant.py, bit-identical to the CPU build), all tensors of
    one type of a file in one launch;
  * quantiser variants: for types whose llama.cpp code changed (gpuquant.VARIANTS) the encoder
    keeps, per tensor, the variant that reproduces the file best and records it in the manifest;
  * predictions are cached across files (most tensors repeat from file to file), so no content
    hashing / dedup pass is needed: a repeated tensor is predicted exactly and costs nothing;
  * file reading, zstd and the GPU work are pipelined.
"""
import json
import lzma
import os
import threading
import time
from collections import Counter, deque, namedtuple
from concurrent.futures import Future, ThreadPoolExecutor

import cupy as cp
import blake3
import numpy as np
import zstandard

from .codecs_gguf import nbytes, pack, pack_index, pack_parts, unpack, zstd_c, zstd_d
from .gguf_reader import GGUF
from .gpu_bf16 import M24BF16GPU
from . import gpu2, gpuquant
from . import qr
from .gpuquant import GPUQuant, VARIANTS, to_host
from .repo2 import F32, F16, BF16, base_float_np, float_as_np
from .requant import QFUN, load_imatrix
from .seed import SeedCodec, bits_per_weight


_POOL = ThreadPoolExecutor(8)
SPARSE = 0.02                             # residuals with fewer non-zero bytes get zstd level >= 9
_IO = ThreadPoolExecutor(8)
_READ_CHUNK = 32 << 20


def _zstd_into(dec, src, size):
    """zstd frame -> writable buffer of `size` bytes (anonymous memory, no extra copy)"""
    import mmap
    if size == 0:
        return bytearray()
    buf = mmap.mmap(-1, size)
    mv = memoryview(buf)
    with dec.stream_reader(src, read_size=1 << 22) as r:
        o = 0
        while o < size:
            k = r.readinto(mv[o:min(o + (64 << 20), size)])
            if not k:
                break
            o += k
    del mv
    if o != size:
        raise ValueError(f"residual decodes to {o} bytes, {size} expected")
    return buf


def _read_file(path, once=False):
    """the whole file as a writable buffer (anonymous memory: no zero-filling first), read in
    parallel pieces (see _read_range)"""
    import mmap
    n = os.path.getsize(path)
    if n == 0:
        return bytearray()
    buf = mmap.mmap(-1, n)
    _read_range(path, 0, n, np.frombuffer(buf, np.uint8), once)
    return buf


_ONCE = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_SEQUENTIAL", 0)


def _read_range(path, off, n, out=None, once=False):
    """bytes [off, off + n) of a file into `out` (default: a new uint8 array), read in parallel
    pieces (an SSD reaches its full speed only with several requests in flight). once: data read
    only once (a derived GGUF), kept out of the OS cache as far as possible so that the base
    model, read again and again, stays in it (Windows: FILE_FLAG_SEQUENTIAL_SCAN, whose pages
    are reused first; elsewhere: dropped after the read)"""
    out = np.empty(n, np.uint8) if out is None else out
    mv = memoryview(out).cast("B")

    def part(o):
        with (os.fdopen(os.open(path, _ONCE), "rb", buffering=0) if once else open(path, "rb", buffering=0)) as f:
            f.seek(off + o)
            want = min(_READ_CHUNK, n - o)
            got = f.readinto(mv[o:o + want])
            while got < want:
                k = f.readinto(mv[o + got:o + want])
                if not k:
                    raise IOError(f"{path}: file shrank while being read")
                got += k
            if once and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), off + o, want, os.POSIX_FADV_DONTNEED)

    for f in [_IO.submit(part, o) for o in range(0, n, _READ_CHUNK)]:
        f.result()
    return out


class _HashingReader:
    """read(a, b) for the base coder of a file too big for the GPU: explicit parallel reads; the
    first pass over the file (ranges in increasing order) also feeds `update` (the file's hash),
    the bytes between the ranges included; finish() hashes the rest"""

    def __init__(self, path, update=None):
        self.path, self.update, self.pos, self.size = path, update, 0, os.path.getsize(path)

    def __call__(self, a, b):
        buf = _read_range(self.path, a, b - a)
        if self.update is not None and a >= self.pos:
            if a > self.pos:
                self.update(_read_range(self.path, self.pos, a - self.pos))
            self.update(buf)
            self.pos = b
        return buf

    def finish(self):
        if self.update is not None:
            for o in range(self.pos, self.size, 1 << 30):
                self.update(_read_range(self.path, o, min(1 << 30, self.size - o)))
            self.pos = self.size


class _Spill:
    """a growing byte string (a compressed residual coded in pieces), in memory up to MEM bytes,
    beyond that in a temporary file in `where`; data() -> its buffers"""
    MEM = 64 << 20

    def __init__(self, where=None):
        self.where, self.parts, self.n, self.f = where, [], 0, None

    def write(self, b):
        if not len(b):
            return
        if self.f is None and self.n + len(b) > self.MEM:
            import tempfile
            self.f = tempfile.TemporaryFile(dir=self.where)
            for p in self.parts:
                self.f.write(p)
            self.parts = []
        if self.f is None:
            self.parts.append(b)
        else:
            self.f.write(b)
        self.n += len(b)

    def data(self):
        if self.f is None:
            return self.parts
        import mmap
        self.f.flush()
        return [mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)]


_T0 = __import__("time").perf_counter()


def _mem():
    """process commit charge, when SEEDPACK_MEMDEBUG is set (debugging aid)"""
    if not os.environ.get("SEEDPACK_MEMDEBUG"):
        return ""
    import psutil
    import time
    free, total = cp.cuda.runtime.memGetInfo()
    return (f"  [{time.perf_counter() - _T0:.0f} s, private {psutil.Process().memory_info().private / 1e9:.1f} GB,"
            f" GPU used {(total - free) / 1e9:.1f} GB,"
            f" pool {cp.get_default_memory_pool().used_bytes() / 1e9:.1f}/{cp.get_default_memory_pool().total_bytes() / 1e9:.1f} GB]")


def _map_file(path):
    """the file's read-only mapping: its pages come from the OS cache, not the commit charge"""
    import mmap
    with open(path, "rb") as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


GPU_RESERVE = 5 << 30                     # device bytes the bases leave free for the rest of the work
RESIDENT_HEADROOM = 8 << 30               # a base in host memory keeps at most (card - this) on the GPU


def _dev_free():
    """device bytes free for new arrays (CuPy's cached free blocks included)"""
    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


def _fits_gpu(n):
    """whether a base of n bytes is kept whole on the GPU (SEEDPACK_BASE_GPU_MAX_MB caps it, for tests)"""
    cap = os.environ.get("SEEDPACK_BASE_GPU_MAX_MB")
    return n + GPU_RESERVE <= _dev_free() and (cap is None or n <= int(cap) << 20)


class _HostBase:
    """a base too large for the GPU, sliced as the GPU code slices a device copy (base[a:b] ->
    device array): kept in host memory (when packing, the file's mapping) and uploaded per tensor.
    Uploaded tensors stay on the GPU while it has room, from the first used on: every file is
    predicted in the same tensor order, so a fixed set is reused where a least recently used one
    would always miss"""

    def __init__(self, host):
        self.host = np.frombuffer(host, np.uint8)
        self.dev, self.resident = {}, 0
        # GPU_RESERVE of free memory is not enough: the QR batches and the pool's cached blocks
        # come on top, and an over-full card has WDDM page device memory out (10-100x slower)
        self.limit = max(0, cp.cuda.runtime.memGetInfo()[1] - RESIDENT_HEADROOM)

    def __len__(self):
        return len(self.host)

    def __getitem__(self, s):
        a = self.dev.get((s.start, s.stop))
        if a is None:
            a = gpu2.h2d(self.host[s])
            if self.resident + a.nbytes <= self.limit and _dev_free() - a.nbytes > GPU_RESERVE:
                self.dev[(s.start, s.stop)] = a
                self.resident += a.nbytes
        return a


def _trim_cached(keep=1 << 30):
    """return CuPy's cached free device blocks once they exceed `keep`: with several threads on the
    GPU (and WDDM letting allocations exceed the card's memory, then paging it out: 10-100x
    slower) the pool must not hoard the blocks of their differently sized temporaries"""
    pool = cp.get_default_memory_pool()
    if pool.free_bytes() > keep:
        pool.free_all_blocks()


class _DevBase:
    """the base tensors of one window on the GPU, sliced as a device copy of the whole file is
    (base[offset:offset + nbytes] -> that tensor's device array); most are views of one buffer
    `buf` (pos: their byte positions in it), which the QR coder takes as its seed without a copy"""

    def __init__(self, buf=None):
        self.dev, self.buf, self.pos = {}, buf, {}

    def add(self, key, p):
        self.dev[key] = self.buf[p:p + key[1] - key[0]]
        self.pos[key] = p

    def __getitem__(self, s):
        return self.dev[(s.start, s.stop)]


class _Residual:
    """a file's residual, its zstd stream read forward into fresh buffers, span by span"""

    def __init__(self, raw, zdict, size):
        dec = zstandard.ZstdDecompressor(dict_data=zdict) if zdict else zstandard.ZstdDecompressor()
        self.r = dec.stream_reader(raw, read_size=1 << 22)
        self.pos, self.size, self.ahead = 0, size, np.zeros(0, np.uint8)

    def _fill(self, buf):
        mv, o = memoryview(buf), 0
        while o < len(buf):
            k = self.r.readinto(mv[o:o + (64 << 20)])
            if not k:
                raise ValueError(f"residual ends at {self.pos + o} bytes, {self.size} expected")
            o += k

    def peek(self, n):
        """the next n bytes (or fewer at the end), not consumed"""
        n = min(n, self.size - self.pos)
        if len(self.ahead) < n:
            more = np.empty(n - len(self.ahead), np.uint8)
            self._fill(more)
            self.ahead = np.concatenate([self.ahead, more])
        return self.ahead[:n]

    def take(self, end):
        """bytes [pos, end) as a new writable array"""
        n = end - self.pos
        buf = np.empty(n, np.uint8)
        k = min(len(self.ahead), n)
        buf[:k] = self.ahead[:k]
        self.ahead = self.ahead[k:]
        self._fill(buf[k:])
        self.pos = end
        return buf


class _Collect:
    """sink.stream() for a sink that takes only whole files: the file is assembled in memory"""

    def __init__(self, sink, name, size):
        self.sink, self.name, self.buf, self.o = sink, name, np.empty(size, np.uint8), 0

    def write(self, b):
        b = np.frombuffer(b, np.uint8) if not isinstance(b, np.ndarray) else b.view(np.uint8).reshape(-1)
        self.buf[self.o:self.o + len(b)] = b; self.o += len(b)

    def close(self):
        self.sink(self.name, self.buf)


def _out_stream(sink, name, size):
    return sink.stream(name, size) if hasattr(sink, "stream") else _Collect(sink, name, size)


class _Upload:
    """a tensor of a base kept in host memory, as a quantiser input: uploaded only when its launch
    converts it to float32, so a launch's tensors are not all on the GPU twice"""

    def __init__(self, get, size):
        self.get, self.size = get, size

    def into(self, out):
        gpuquant._to_f32_into(self.get(), out)


def _r2_tensors(reader, chunk):
    """R2Tensors (random access) of base chunk `chunk` of a container.Reader, None when the base is
    not coded with R2: only the chunk's small parts are read here"""
    from .bf16v2 import R2Tensors
    read = lambda o, n: reader.read_at(chunk, o, n)
    outer = pack_index(read)
    if "r2" not in outer:
        return None
    parts = pack_index(read, outer["r2"][0])
    so = parts["stream"][0]
    return R2Tensors(lambda name: read(*parts[name]), lambda o, n, out: reader.read_at(chunk, so + o, n, out))


def _scratch_array(n, where=None):
    """a writable uint8 array of n bytes backed by a temporary file in `where` (deleted once the
    array is gone): its pages are written back to that file, so it costs no commit charge"""
    import mmap
    import tempfile
    with tempfile.TemporaryFile(dir=where) as f:
        if os.name != "nt":
            f.truncate(n)                   # (on Windows the mapping extends the file itself; truncate
                                            # would write n zero bytes first)
        mm = mmap.mmap(f.fileno(), n)       # keeps its own handle: the file lives on with it
    return np.frombuffer(mm, np.uint8)


def _keep_base(data, chunk=256 << 20):
    """(header bytes, the file as the GPU code reads it): on the GPU, uploaded in chunks (no
    whole-file staging copy), or a _HostBase when the GPU has no room for it"""
    mv = memoryview(data).cast("B")
    head = bytes(mv[:GGUF.from_bytes(mv, light=True).data_start])
    del mv
    if not _fits_gpu(len(data)):
        return head, _HostBase(data)
    dev = cp.empty(len(data), cp.uint8)
    host = np.frombuffer(data, np.uint8)
    for o in range(0, len(data), chunk):
        dev[o:o + chunk].set(host[o:o + chunk])
    return head, dev


PINNED_KEEP = 2 << 30                     # pinned bytes handed out before the pinned pool is emptied
DEVICE_KEEP = 1 << 30                     # cached free device bytes kept between files


def _release(pinned=False, device=True):
    """return CuPy's cached device blocks once they total DEVICE_KEEP, and the pinned ones once
    many were handed out (freeing per file is slow: cudaFree waits for all GPU work, and pinned
    blocks are slow to re-allocate). On Windows (WDDM) both count against the process commit
    charge, which otherwise grows with every file until allocations fail"""
    pool = cp.get_default_memory_pool()
    if device and (pinned or pool.free_bytes() > DEVICE_KEEP):
        pool.free_all_blocks()
    if pinned or gpuquant.PINNED[0] > PINNED_KEEP:
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gpuquant.PINNED[0] = 0
_CHUNK = 2 << 20                          # small enough that the zero test reads the chunk from cache


def _xor_chunk(r, p, check):
    np.bitwise_xor(r, p, out=r)
    return check and bool(r.any())


def _xor_async(r, p, check=True):
    """r ^= p in parallel chunks, without waiting (numpy releases the GIL, so the GPU work goes on)
    -> futures, each telling whether its chunk of the residual is non-zero (False if not check)"""
    return [_POOL.submit(_xor_chunk, r[o:o + _CHUNK], p[o:o + _CHUNK], check) for o in range(0, max(len(r), 1), _CHUNK)]


def _copy_chunk(dst, src):
    np.copyto(dst, src)


def _copy_async(dst, src):
    """dst[:] = src in parallel chunks -> futures"""
    return [_POOL.submit(_copy_chunk, dst[o:o + _CHUNK], src[o:o + _CHUNK]) for o in range(0, len(dst), _CHUNK)]


def _diff_chunk(r, p):
    return not np.array_equal(r, p)


def _diff_async(r, p):
    """futures telling, per chunk, whether r and p differ"""
    return [_POOL.submit(_diff_chunk, r[o:o + _CHUNK], p[o:o + _CHUNK]) for o in range(0, max(len(r), 1), _CHUNK)]


def _nonzero(futures):
    return any([f.result() for f in futures])


def _ndiff(a, b):
    return int(np.count_nonzero(a != b)) if b is not None else int(np.count_nonzero(a))


def _count_diff(a, b=None):
    """bytes where a and b differ (b None: non-zero bytes of a), counted in parallel chunks"""
    fs = [_POOL.submit(_ndiff, a[o:o + _CHUNK], None if b is None else b[o:o + _CHUNK])
          for o in range(0, len(a), _CHUNK)]
    return sum(f.result() for f in fs)


def _xor3(a, b, c):
    """(a ^ b ^ c as a new array, its number of non-zero bytes), in parallel chunks"""
    out = np.empty_like(a)

    def one(o):
        x = out[o:o + _CHUNK]
        np.bitwise_xor(a[o:o + _CHUNK], b[o:o + _CHUNK], out=x)
        np.bitwise_xor(x, c[o:o + _CHUNK], out=x)
        return int(np.count_nonzero(x))

    return out, sum(f.result() for f in [_POOL.submit(one, o) for o in range(0, len(a), _CHUNK)])


def _pcopy(a):
    """a copy of a (big) array, made in parallel chunks"""
    out = np.empty_like(a)
    for f in _copy_async(out, a):
        f.result()
    return out


def _digest(b):
    return blake3.blake3(b, max_threads=blake3.blake3.AUTO if len(b) >= (64 << 20) else 1).digest()


def _qr_job(T, args, seed, names):
    """one QR batch coded (qr.encode); its tensors' regions (args[i][0]) zeroed after"""
    part = qr.encode(T, args, seed)
    for a in args:
        a[0][:] = 0
    return [T, names], part


_T = namedtuple("_T", "name type shape")    # what _predict needs of a tensor
UNIT_W = 100                              # plan variants from here on: unit importance weights


class _Dequant:
    """a quantised tensor of a seed, as a quantiser input: dequantised (llama.cpp's
    dequantize_row_*) straight into the launch's float32 buffer, never held as a whole"""

    def __init__(self, gq, bt, raw):
        self.gq, self.bt, self.raw = gq, bt, raw
        self.size = int(np.prod(bt.shape))

    def into(self, out):
        self.gq.dequantize(self.bt.type, self.raw, self.bt.shape[0], out=out)

    def f32(self):
        return self.gq.dequantize(self.bt.type, self.raw, self.bt.shape[0])
LOW_COMMIT = 4 << 30                      # the prediction cache stops growing below this much free commit


def _commit_free():
    """bytes the system can still commit (Windows: RAM + page file left), None if unknown"""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                    ("virt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
    ms = MS(); ms.len = ctypes.sizeof(MS)
    return ms.apf if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None


def _gguf_tensors(path):
    """the tensor table of a GGUF file (header only), [] if it is not one"""
    import mmap
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return GGUF(data=mm, light=True).tensors
    except Exception:
        return []


def _quant_tensors(path):
    """the quantised and float tensors of a GGUF file (header only)"""
    return [t for t in _gguf_tensors(path) if t.type in QFUN or t.type in (F32, F16, BF16)]


class M28RepoGPU:
    name = "M28 model repo: BF16 + quants re-derived on the GPU with llama.cpp quantisers + imatrix"

    def __init__(self, level=7, cache_bytes=6 << 30, inflight=4 << 30):
        self.level = level
        self.inflight = inflight
        self.cache_bytes = cache_bytes
        self.base_codec = M24BF16GPU()
        self.gq = None
        self.sizes = {}
        self.q8_files = set()                 # derived GGUFs stored with M21 (no base)
        self.seed_files = set()               # derived GGUFs that serve as a base (seed.py)
        self.stats = {"requant_exact": 0, "requant_diff": 0, "variant": 0, "xor": 0, "raw": 0}
        self.timing = {}
        self.verbose = False

    # ------------------------------------------------------------------ shared by both sides
    def _setup(self, bases, imatrix):
        """bases: list of (GGUF header bytes, whole file as a device array) of the float models
        (BF16 model, mmproj, ...); only the GPU copy of the weights is kept"""
        if self.gq is None:
            self.gq = GPUQuant()
        self.bases = []
        for item in bases:
            if item is None:                    # not needed for this extraction
                self.bases.append(None)
                continue
            hdr, dev = item
            bg = GGUF.from_bytes(hdr, light=True)
            zd = zstandard.ZstdCompressionDict(bytes(hdr[:bg.data_start]), dict_type=zstandard.DICT_TYPE_RAWCONTENT)
            self.bases.append((None, dev, {t.name: t for t in bg.tensors}, zd))
        self.imatrix = imatrix
        self.cache, self.cached = {}, 0
        self._lock = threading.Lock()             # the cache is filled from two threads (streamed windows)
        self.uses = None
        _release()
        self._use(next((i for i, b in enumerate(self.bases) if b is not None), None))

    def _use(self, b):
        """make base b the one quantised files are predicted from"""
        self.b = b
        if b is None:
            self.base = self.base_dev = self.zdict = None
            self.by_name = {}
        else:
            self.base, self.base_dev, self.by_name, self.zdict = self.bases[b]

    def _best_base(self, qg):
        """index of the base sharing most tensor names with qg, or None (nothing in common)"""
        names = {t.name for t in qg.tensors}
        common = [len(names & b[2].keys()) if b else 0 for b in self.bases]
        return max(range(len(common)), key=common.__getitem__) if common and max(common) else None

    def _src(self, bt, lazy=False, dev=None):
        """base tensor bt as the quantisers take it; lazy, from a base in host memory: a _Upload.
        dev: the base to take it from (default: the current one)"""
        dev = self.base_dev if dev is None else dev
        if lazy and isinstance(dev, _HostBase):
            return _Upload(lambda: self._view(dev, bt), int(np.prod(bt.shape)))
        return self._view(dev, bt)

    def _view(self, dev, bt):
        a = dev[bt.offset:bt.offset + bt.nbytes]
        if bt.type in QFUN:                   # a seed (quantised base): dequantised when used
            return _Dequant(self.gq, bt, a)
        return a.view(cp.uint16) if bt.type == BF16 else a.view(cp.float16) if bt.type == F16 else a.view(cp.float32)

    def _float_dev(self, bt, ttype, dev=None, lo=0, hi=None):
        """base tensor bt (its weights lo:hi) converted to float type ttype on the GPU (same rounding
        as float_as_np), as bytes"""
        a = self._src(bt, dev=dev)
        if isinstance(a, _Dequant):
            x = a.f32()[lo:hi]
        else:
            a = a[lo:hi]
            x = (a.astype(cp.uint32) << 16).view(cp.float32) if bt.type == BF16 else a.astype(cp.float32)
        if ttype == F32:
            y = x.view(cp.uint8)
        elif ttype == F16:
            y = x.astype(cp.float16).view(cp.uint8)
        else:
            u = x.view(cp.uint32)
            y = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(cp.uint16).view(cp.uint8)
        return y

    def _kind(self, t):
        """how tensor t of a quantised file is predicted: 'rq', 'xor' or None"""
        bt = self.by_name.get(t.name)
        if bt is None or not (bt.type in (F32, F16, BF16) or bt.type in QFUN) or                 int(np.prod(bt.shape)) != int(np.prod(t.shape)):
            return None
        if t.type in (F32, F16, BF16):
            return "xor"
        if t.type in QFUN and t.nbytes == int(np.prod(t.shape)) // QFUN[t.type][1] * QFUN[t.type][2]:
            return "rq"
        return None

    def _room(self, nbytes):
        """whether a prediction of nbytes may be cached: within cache_bytes, and the system is not
        short of memory (then predictions are recomputed rather than the process failing)"""
        if self.cached + nbytes > self.cache_bytes and not self.streaming:
            self.stats["cache_full"] = self.stats.get("cache_full", 0) + 1
            return False
        free = _commit_free()
        ok = free is None or free - nbytes > LOW_COMMIT
        if not ok:
            self.stats["cache_low_memory"] = self.stats.get("cache_low_memory", 0) + 1
        return ok

    def _predict_into(self, b, wanted, out):
        self._use(b)
        self._predict(wanted, lambda key, p: out.__setitem__(key[0], p))

    def _predict_xor(self, b, plan, tens, view, qr_buf):
        """decoding a file (`view`, its residual) without holding its predictions: each is XOR-ed
        in as soon as it is ready; the QR tensors are predicted and decoded batch by batch
        (qr_buf() -> the file's QR chunk)"""
        self._use(b)
        isqr = {e[0] for e in plan if e[1] == "qr"}
        jobs = []

        def xor(n, p):
            t = tens[n]
            jobs.extend(_xor_async(view[t.offset:t.offset + t.nbytes], p, check=False))

        self._predict([(tens[e[0]], e[2]) for e in plan if e[0] not in isqr], lambda key, p: xor(key[0], p))
        if isqr:
            ch = unpack(qr_buf())
            var = {e[0]: e[2] for e in plan}
            for k, (T, names) in enumerate(json.loads(zstd_d(ch["meta"]))):
                held = {}
                self._predict([(tens[n], var[n]) for n in names], lambda key, p: held.__setitem__(key[0], p))
                self._qr_decode_batch(ch, k, T, names, held)
                for n in names:
                    xor(n, held[n])
                for f in jobs:                  # the batch's memory is released before the next
                    f.result()
                jobs.clear()
                del held
        for f in jobs:
            f.result()

    def _expect(self, keys):
        """keys: every (name, type, variant) the coming files will predict, repeats included. The
        cache then keeps a prediction only while it has a later use, dropping it after the last"""
        self.uses = Counter(keys)

    def _used(self, key):
        """one use of key done -> whether a later one is expected (True without _expect)"""
        if self.uses is None:
            return True
        self.uses[key] -= 1
        if self.uses[key] > 0:
            return True
        with self._lock:
            hit = self.cache.pop((self.b,) + key, None)
            if hit is not None:
                self.cached -= hit.nbytes
        return False

    def _keep(self, key, p):
        """cache prediction p of key (a copy: p may be a view of a launch's whole output)"""
        p = _pcopy(p)
        with self._lock:
            self.cache[(self.b,) + key] = p
            self.cached += p.nbytes

    FLOAT_BATCH = 256 << 20                   # base bytes of float tensors converted in one go

    def _predict(self, wanted, use, dev=None, device=False):
        """wanted: list of (tensor, variant); use((name, type, variant), bytes as numpy uint8) is
        called as soon as each prediction is ready, so a file's predictions are never all held.
        dev: the base predicted from (default: the current one). device: quantised predictions
        handed over as device arrays, not cached"""
        todo, floats = {}, []
        for t, v in wanted:
            key = (t.name, t.type, v)
            p = self.cache.get((self.b,) + key)
            if p is not None:
                self.stats["cache_hit"] = self.stats.get("cache_hit", 0) + 1
                self._used(key)
                use(key, p)
            elif t.type in (F32, F16, BF16):
                floats.append(key)
            else:
                self.stats["cache_miss"] = self.stats.get("cache_miss", 0) + 1
                todo.setdefault((t.type, v), {})[key] = t
        for key in [k for k in floats if self.by_name[k[0]].nbytes > self.FLOAT_BATCH]:
            # a big float tensor (an F16 output / token embedding): converted and downloaded in
            # pieces, never whole in float32 on the GPU
            bt = self.by_name[key[0]]
            n = int(np.prod(bt.shape))
            isz = 4 if key[1] == F32 else 2
            p = np.empty(n * isz, np.uint8)
            step = self.FLOAT_BATCH // 4
            for lo in range(0, n, step):
                hi = min(n, lo + step)
                self._float_dev(bt, key[1], dev, lo, hi).get(out=p[lo * isz:hi * isz])
            if self._used(key) and self._room(p.nbytes):
                with self._lock:
                    self.cache[(self.b,) + key] = p
                    self.cached += p.nbytes
            use(key, p)
            del p
        floats = [k for k in floats if self.by_name[k[0]].nbytes <= self.FLOAT_BATCH]
        groups, nb = [[]], 0
        for key in floats:
            # float tensors (norms, ...) are small and many: converted together, one download per
            # group of at most 512 tensors and FLOAT_BATCH base bytes (an F16/F32 file has big ones)
            b = self.by_name[key[0]].nbytes
            if groups[-1] and (len(groups[-1]) == 512 or nb + b > self.FLOAT_BATCH):
                groups.append([]); nb = 0
            groups[-1].append(key); nb += b
        for keys in groups if floats else []:
            ys = [self._float_dev(self.by_name[n], ty, dev) for n, ty, _ in keys]
            host = to_host(cp.concatenate(ys)) if len(ys) > 1 else to_host(ys[0])
            o = 0
            for key, y in zip(keys, ys):
                p = host[o:o + y.size]; o += y.size
                if self._used(key) and self._room(p.nbytes):
                    self._keep(key, p)
                use(key, p)
            del ys, host
        for (ty, v), ts in todo.items():
            keys = list(ts)

            def got(i, o):
                if not device and self._used(keys[i]) and self._room(o.nbytes):
                    self._keep(keys[i], o)
                use(keys[i], o)

            srcs = [(self._src(self.by_name[n], lazy=True, dev=dev), ts[(n, ty, v)].shape[0]) for n, _, _ in keys]
            # variant v: the quantiser's code variant v % UNIT_W, with unit weights from UNIT_W
            ims = [self.imatrix.get(n) if v < UNIT_W else np.ones(ts[(n, ty, v)].shape[0], np.float32)
                   for n, _, _ in keys]
            self.gq.quantize_tensors(ty, srcs, ims, v % UNIT_W, on_result=got, device=device,
                                     max_weights=1 << 27 if self.streaming else 1 << 28)
            if self.streaming:
                _trim_cached()

    # ------------------------------------------------------------------ compression
    # ------------------------------------------------------------------ QR coding (qr.py)
    USE_QR = True

    def _qr_ok(self, t):
        bt = self.by_name.get(t.name)
        return self.USE_QR and t.type in qr.QR_TYPES and bt is not None and bt.type in qr.SEED_TYPES \
            and int(np.prod(bt.shape)) == int(np.prod(t.shape))

    def _seed_dev(self, names):
        """(device array, {name: byte offset}) holding base tensors `names` for the QR coder: the
        base itself when it is on the GPU, else just those tensors, gathered"""
        if isinstance(self.base_dev, cp.ndarray):
            return self.base_dev, {n: self.by_name[n].offset for n in names}
        if isinstance(self.base_dev, _DevBase):
            keys = [(self.by_name[n].offset, self.by_name[n].offset + self.by_name[n].nbytes) for n in names]
            if all(k in self.base_dev.pos for k in keys):
                return self.base_dev.buf, {n: self.base_dev.pos[k] for n, k in zip(names, keys)}
        off, o = {}, 0
        for n in names:
            off[n] = o
            o += -(-self.by_name[n].nbytes // 256) * 256
        seed = cp.empty(max(o, 1), cp.uint8)
        for n in names:
            bt = self.by_name[n]
            seed[off[n]:off[n] + bt.nbytes] = self.base_dev[bt.offset:bt.offset + bt.nbytes]
        return seed, off

    def _again(self, wanted):
        """{name: prediction} of tensors this file already predicted once ([(tensor, variant)]):
        their expected uses are counted again first, so the cache keeps what later files need"""
        if self.uses is not None:
            for t, v in wanted:
                self.uses[(t.name, t.type, v)] += 1
        out = {}
        self._predict(wanted, lambda key, p: out.__setitem__(key[0], p))
        return out

    def _qr_encode(self, items, choice):
        """[(tensor, file bytes, prediction or None)] -> QR batches (meta [[type, names]], coded
        parts); the regions are zeroed. A missing prediction is made again, batch by batch, with
        variant choice[name]"""
        by = {}
        for t, f, p in items:
            by.setdefault(t.type, []).append((t, f, p))
        meta, parts = [], []
        for T, ts in by.items():
            for bi in qr.batches(T, [t.nbytes for t, _, _ in ts]):
                its = [ts[i] for i in bi]
                pred = self._again([(t, choice[t.name]) for t, _, p in its if p is None])
                seed, off = self._seed_dev([t.name for t, _, _ in its])
                parts.append(qr.encode(
                    T, [(f, pred[t.name] if p is None else p, off[t.name], self.by_name[t.name].type)
                        for t, f, p in its], seed))
                meta.append([T, [t.name for t, _, _ in its]])
                del seed, pred
                _release()                      # a batch's temporaries must not pile up in the pool
        for _, f, _ in items:
            f[:] = 0
        self.stats["qr"] = self.stats.get("qr", 0) + len(items)
        _release()
        return meta, parts

    def _qr_submit(self, pool, items, choice, hashes=None, memo=None):
        """_qr_encode with the coding itself done in `pool`, overlapping the caller's next work: the
        predictions and seeds are prepared here -> futures of ([type, names], coded part); each job
        zeroes its tensors' regions when done. hashes, memo: the tensors' hashes and the batches
        coded before ({batch key: future}): a batch of the same tensors with the same variants
        (another file of the repository) gets that batch's coded part"""
        by = {}
        for t, f, p in items:
            by.setdefault(t.type, []).append((t, f, p))
        futs = []
        for T, ts in by.items():
            for bi in qr.batches(T, [t.nbytes for t, _, _ in ts]):
                its = [ts[i] for i in bi]
                names = [t.name for t, _, _ in its]
                key = None
                if memo is not None and hashes and all(n in hashes for n in names):
                    key = tuple((t.name, t.type, hashes[t.name], choice[t.name]) for t, _, _ in its)
                    if key in memo:
                        for _, f, _ in its:
                            f[:] = 0
                        src = memo[key]
                        futs.append(pool.submit(lambda src=src, m=[T, names]: (m, src.result()[1])))
                        self.stats["qr_memo"] = self.stats.get("qr_memo", 0) + len(its)
                        continue
                pred = self._again([(t, choice[t.name]) for t, _, p in its if p is None])
                seed, off = self._seed_dev([t.name for t, _, _ in its])
                args = [(f, pred[t.name] if p is None else p, off[t.name], self.by_name[t.name].type) for t, f, p in its]
                futs.append(pool.submit(_qr_job, T, args, seed, names))
                if key is not None:
                    memo[key] = futs[-1]
        self.stats["qr"] = self.stats.get("qr", 0) + len(items)
        return futs

    @staticmethod
    def _qr_pack(meta, parts):
        """a file's QR chunk"""
        return pack({"meta": zstd_c(json.dumps(meta).encode()), **{str(k): p for k, p in enumerate(parts)}})

    def _qr_decode(self, buf, pred):
        """the file's QR chunk + predictions {name: bytes} -> the QR tensors' bytes put in `pred`
        (their residual regions are zero, so the caller's XOR writes them)"""
        ch = unpack(buf)
        for k, (T, names) in enumerate(json.loads(zstd_d(ch["meta"]))):
            self._qr_decode_batch(ch, k, T, names, pred)

    def _qr_decode_batch(self, ch, k, T, names, pred, release=True):
        """batch k of an unpacked QR chunk: pred[n] (the predictions of `names`, host or device
        arrays) -> the tensors (host arrays)"""
        seed, off = self._seed_dev(names)
        out = qr.decode(T, ch[str(k)], [(pred[n], off[n], self.by_name[n].type) for n in names], seed)
        del seed
        host = to_host(out)
        o = 0
        for n in names:
            sz = len(pred[n])
            pred[n] = host[o:o + sz]; o += sz
        del out
        if release:
            _release()

    # ------------------------------------------------------------------ duplicate tensors
    DUP_MIN = 64 << 10                        # smaller tensors are not worth a reference
    DUP_KEEP = 2 << 30                        # bytes the decoder may hold for later references

    def _dedup(self, fi, view, tensors):
        """tensors of file fi equal to one of an earlier file -> {name: [source file, source name]};
        their bytes are zeroed (the decoder copies the source's). A reference is made only while
        the sources the decoder must keep stay within DUP_KEEP bytes at every point."""
        big = [t for t in tensors if t.nbytes >= self.DUP_MIN]
        hs = list(_POOL.map(lambda t: blake3.blake3(view[t.offset:t.offset + t.nbytes]).digest(), big))
        while len(self.live) <= fi:
            self.live.append(0)
        refs = {}
        for t, h in zip(big, hs):
            src = self.seen_t.get(h)
            if src is not None and src[2] == t.nbytes:
                key = (src[0], src[1])
                start = self.keep_until.get(key, src[0]) + 1
                if max(self.live[start:fi + 1]) + t.nbytes <= self.DUP_KEEP:
                    for j in range(start, fi + 1):
                        self.live[j] += t.nbytes
                    self.keep_until[key] = fi
                    refs[t.name] = [src[0], src[1]]
                    view[t.offset:t.offset + t.nbytes] = 0
            self.seen_t[h] = (fi, t.name, t.nbytes)     # the latest copy: the shortest to keep
        return refs

    def _residual(self, data, fi=None):
        """XOR the predictions into the file in place -> (plan [[name, kind, variant, type], ...], base index).
        Tensors equal to one of an earlier file (fi: this file's index) become [name, "dup", [file, name], type]."""
        view = np.frombuffer(data, np.uint8)
        qg = GGUF.from_bytes(data, light=True)
        self._use(self._best_base(qg))
        self._qr_chunk = None
        if self.b is None:
            self.stats["raw"] += len(qg.tensors)
            return [], None
        refs = self._dedup(fi, view, qg.tensors) if fi is not None else {}
        self.stats["dup"] = self.stats.get("dup", 0) + len(refs)
        ts = [(t, self._kind(t)) for t in qg.tensors if t.name not in refs]
        self.stats["raw"] += sum(k is None for _, k in ts)
        plan, qr_items, choice = self._residual_tensors(view, 0, [(t, k) for t, k in ts if k], len(data))
        if qr_items:
            self._qr_chunk = self._qr_pack(*self._qr_encode(qr_items, choice))
        plan += [[t.name, "dup", refs[t.name], t.type] for t in qg.tensors if t.name in refs]
        return plan, self.b

    def _residual_tensors(self, view, at, ts, held, memo=None, hashes=None):
        """_residual for the tensors ts [(tensor, kind)] of a file, `view` holding its bytes from
        offset `at` on: the predictions are XOR-ed in place, variants tried where they leave a
        residual -> (plan entries, QR items [(tensor, its bytes, prediction or None)], variant per
        tensor). held: bytes of the file the caller holds (the QR coder's predictions are kept
        while the system can commit that much more). memo, hashes: outcomes of tensors already
        coded ({(name, type, hash): (kind, variant, residual left)}) and the hashes of these
        tensors' bytes: a tensor equal to one coded before gets its outcome without being
        predicted and compared again (the files of a repository share many tensors)"""
        hit = {}
        if memo is not None and hashes:
            for t, k in ts:
                r = memo.get((t.name, t.type, hashes.get(t.name)))
                if r is not None and not (r[2] and r[0] != "qr"):   # a residual to XOR: coded again
                    hit[t.name] = r
        todo = [(t, k) for t, k in ts if t.name not in hit]
        tens = {t.name: t for t, _ in ts}
        regions = {t.name: view[t.offset - at:t.offset - at + t.nbytes] for t, _ in ts}
        # QR tensors (qr.py) are not XOR-ed: their file bytes and prediction go to the QR coder
        isqr = {t.name for t, k in todo if k == "rq" and self._qr_ok(t)}
        nonzero, p0 = {}, {}
        # without the tensor's importance matrix (quants packed without imatrix file), unit
        # weights first: the quantisers' weighted path, the one imatrix quants were made with
        v0 = {t.name: self._v0(t, k) for t, k in todo}

        def cands(t):
            """the other predictions worth trying when v0's leaves a residual: the quantiser's code
            variants and, for unit weights, the unweighted path (IQ types have none)"""
            nv = len(VARIANTS.get(t.type, ()))
            out = [v0[t.name] + j for j in range(1, nv + 1)]
            if v0[t.name] == UNIT_W and t.type < 16:
                out += list(range(nv + 1))
            return out

        # the QR coder needs the QR tensors' predictions: kept when the system can commit about
        # the file's size once more, else predicted again batch by batch (_qr_encode)
        free = _commit_free()
        keep_qr = free is None or free - held > LOW_COMMIT

        def xor0(key, p):
            if key[0] in isqr:
                nonzero[key[0]] = _diff_async(regions[key[0]], p)
            else:
                nonzero[key[0]] = _xor_async(regions[key[0]], p)
            if cands(tens[key[0]]) or (keep_qr and key[0] in isqr):
                p0[key[0]] = p                  # kept for the alternatives below / the QR coder

        self._predict([(t, v0[t.name]) for t, _ in todo], xor0)
        nonzero = {n: _nonzero(f) for n, f in nonzero.items()}
        p0 = {n: p for n, p in p0.items() if nonzero[n]}
        choice = dict(v0)
        retry = [t for t, k in todo if k == "rq" and nonzero[t.name] and cands(t)]
        if retry:
            # XOR tensors: the region now holds file ^ p0, and prediction pv leaves
            # file ^ pv = region ^ p0 ^ pv; QR tensors: the region is the file, pv is kept
            alts = {t.name: [(v0[t.name], None if t.name in isqr else regions[t.name],
                              _count_diff(regions[t.name], p0[t.name]) if t.name in isqr
                              else _count_diff(regions[t.name]))]
                    for t in retry}

            def alt(key, p):
                n = key[0]
                if n in isqr:
                    alts[n].append((key[2], p, _count_diff(regions[n], p)))
                else:
                    alts[n].append((key[2], *_xor3(regions[n], p0[n], p)))

            for v in sorted({c for t in retry for c in cands(t)}):
                self._predict([(t, v) for t in retry if v in cands(t)], alt)
            for t in retry:
                a = alts[t.name]
                errs = [e for _, _, e in a]
                j = int(np.argmin(errs))
                choice[t.name] = a[j][0]
                if t.name in isqr:
                    if j:
                        p0[t.name] = a[j][1]
                elif j:
                    regions[t.name][:] = a[j][1]
                nonzero[t.name] = errs[j] > 0
            del alts
        # exact predictions stay XOR (an all-zero residual costs nothing with zstd)
        for n in [n for n in isqr if not nonzero[n]]:
            isqr.discard(n)
            regions[n][:] = 0                   # file == prediction: the XOR residual
        if memo is not None and hashes:
            for t, k in todo:
                if t.name in hashes:
                    memo[(t.name, t.type, hashes[t.name])] = ("qr" if t.name in isqr else k, choice[t.name],
                                                              nonzero[t.name])
        for n, (k, v, nz) in hit.items():         # equal to a tensor coded before: its outcome
            t = tens[n]
            k0 = (n, t.type, self._v0(t, "rq" if t.type in QFUN else "xor"))
            if k == "qr" and v == k0[2]:
                p = self.cache.get((self.b,) + k0)
                if p is not None:
                    p0[n] = p                   # the QR coder's prediction, cached (its use counted here)
            self._used(k0)
            choice[n], nonzero[n] = v, nz
            if k == "qr":
                isqr.add(n)
            else:
                regions[n][:] = 0
        qr_items = [(t, regions[t.name], p0.get(t.name)) for t, _ in ts if t.name in isqr]   # file order
        plan = []
        for t, k in ts:
            v = choice[t.name]
            if k == "xor":
                self.stats["xor"] += 1
            else:
                self.stats["requant_diff" if nonzero[t.name] else "requant_exact"] += 1
                self.stats["variant"] += v % UNIT_W > 0
                self.stats["unit_w"] = self.stats.get("unit_w", 0) + (v >= UNIT_W)
            plan.append([t.name, "qr" if t.name in isqr else k, v, t.type])
        self.stats["memo"] = self.stats.get("memo", 0) + len(hit)
        return plan, qr_items, choice

    # ------------------------------------------------------------------ files without a base
    Q8_MIN = 0.5                              # share of Q8_0 bytes from which M21 is tried

    def _q8_codec(self):
        if getattr(self, "_q8", None) is None:
            from .gpu3 import Q8Parts
            self._q8 = Q8Parts(int(os.environ.get("SEEDPACK_Q8_PART_MB", 0)) << 20 or None)
        return self._q8

    def _no_base(self, data):
        """a GGUF that no base predicts: M21 (context-modelled Q8_0 on the GPU) when the file is
        mostly Q8_0 (coded in parts that fit in GPU memory); else None (zstd)"""
        from .gguf_reader import GGML_Q8_0
        try:
            g = GGUF.from_bytes(data, light=True)
        except Exception:
            return None
        q8 = sum(t.nbytes for t in g.tensors if t.type == GGML_Q8_0)
        _release()
        if q8 < self.Q8_MIN * len(data):
            return None
        try:
            c = self._q8_codec().compress(data)
        except (cp.cuda.memory.OutOfMemoryError, MemoryError):
            c = None
        _release()
        return c

    @staticmethod
    def _base_header(buf, is_seed):
        """GGUF header bytes of a base from its archived chunk (its 'rest' part only), or None"""
        try:
            ch = unpack(memoryview(buf))        # views: the gigabytes of the stream are not copied
            if is_seed:
                from .gpu2 import GPU2
                rest = GPU2._header_d(ch["rest"])
            elif "r2" in ch:
                from .gpu_bf16 import M24BF16GPU
                rest = M24BF16GPU._header_d(unpack(ch["r2"])["rest"])
            else:
                return None
            rest = bytes(rest)
            return rest[:GGUF.from_bytes(rest, light=True).data_start]
        except Exception:
            return None

    @staticmethod
    def _imatrix(ex):
        name = next((n for n in ex if n.startswith("imatrix")), None)
        return load_imatrix(ex[name]) if name else {}

    BUNDLE_MAX = 64 << 20                     # extra files up to this size go into the LZMA bundle

    @staticmethod
    def _pick_seeds(bases, files):
        """derived files that share no tensor with a base are grouped by model (tensor names); the
        senior quant of each group (most bits per weight) becomes its seed (seed.py)"""
        base_names = [{t.name for t in _gguf_tensors(p)} for p in bases.values()]
        heads = {n: _gguf_tensors(p) for n, p in files.items()}
        names = {n: {t.name for t in ts} for n, ts in heads.items()}
        rest = [n for n in files if names[n] and not any(names[n] & b for b in base_names)]
        seeds = []
        while rest:
            s = max(rest, key=lambda n: (bits_per_weight(heads[n]), n))
            group = {n for n in rest if 2 * len(names[n] & names[s]) >= len(names[n])}
            seeds.append(s)
            rest = [n for n in rest if n not in group]
        return seeds

    def compress(self, bases: dict, files: dict, extras: dict, put=None, on_data=None, on_stream=None,
                 put_stream=None):
        """bases: float GGUFs (BF16 model, mmproj, ...) name -> path; files: GGUFs to derive from
        them, name -> path; extras: any other files, name -> path.
        put(chunk name, bytes) receives the archive's chunks as they are produced; without it they
        are collected and the packed archive is returned. on_data(name, bytes) sees every input
        file's original content once (e.g. to hash it). on_stream(name) -> update(bytes): the
        files derived from a base too big for the GPU are read in pieces, which go (in order) to
        that file's update instead; without on_stream (but with on_data) they are not streamed.
        put_stream(chunk name) -> writer of a chunk in pieces (container.Writer.put_stream): such a
        base's coded form is then written while it is produced, during the derived files' pass."""
        ch = {}
        collect = put is None
        put = put or ch.__setitem__
        seen = on_data or (lambda n, d: None)
        # quants without their BF16 model: the senior quant is coded like a base and the others
        # are derived from it
        seeds = self._pick_seeds(bases, files)
        self.seed_files = set(seeds)
        bases = {**bases, **{n: files[n] for n in seeds}}
        files = {n: p for n, p in files.items() if n not in self.seed_files}

        def read(name):
            # in the reader thread: the content is seen (hashed) before it is XOR-ed in place,
            # overlapping with the GPU work on the current file
            data = _read_file(files[name], once=True)   # the buffer that is XOR-ed in place
            seen(name, data)
            return data

        names = list(files)
        fsize = [os.path.getsize(files[n]) for n in names]
        rd = ThreadPoolExecutor(1)
        # a base too big for the GPU is a big model: its coding (output and buffers of gigabytes)
        # is finished before the derived files are read, not overlapped with them
        huge = any(not _fits_gpu(os.path.getsize(p)) for p in bases.values())
        # the first derived file is read while the bases are coded
        reads = {0: rd.submit(read, names[0])} if names and not huge else {}
        kept = []
        # the bases are coded in the background: the derived files only need their GPU copy
        bp = ThreadPoolExecutor(1)
        base_jobs = []

        import types
        # an R2 base too big for the GPU is written while it is coded (code_big)
        big1 = types.SimpleNamespace(index=None, analysed=threading.Event(), go=threading.Event(),
                                     opened=threading.Event())

        def code_big(name, data, read, chunk):
            """an R2 base too big for the GPU, its chunk written while it is coded: the first pass
            over the file (which leaves it in the OS cache) first, then, once the derived files'
            pass starts (go), the second, overlapping with it"""
            ws, n = None, 0
            try:
                for kind, x in self.base_codec.c.compress_stream(data, read=read):
                    if kind == "analysed":
                        big1.analysed.set()
                        big1.go.wait()
                        ws = put_stream(chunk)
                        big1.opened.set()
                    elif kind == "head":
                        ws.head(x); n += len(x)
                    elif kind == "data":
                        ws.write(x); n += len(x)
                    else:
                        ws.close(x)
                read.finish()
            except BaseException:
                if ws is not None:
                    ws.abort()
                big1.analysed.set()
                raise
            if self.verbose:
                print(f"  {name}: {len(data):,} -> {n:,} B (base, written while coded){_mem()}", flush=True)
            return n

        def code_base(name, data, read=None):
            if name in self.seed_files:
                c = SeedCodec().compress(data)
            elif read is not None and isinstance(self.base_codec, R2Base):
                c = pack_parts(self.base_codec.compress_dev(data, read=read))
            else:
                c = pack_parts(self.base_codec.compress_dev(data))   # pieces: never joined
            if read is not None:
                read.finish()
            if self.verbose:
                kind = "seed" if name in self.seed_files else "base"
                print(f"  {name}: {len(data):,} -> {nbytes(c):,} B ({kind}){_mem()}", flush=True)
            return c

        for i, (name, path) in enumerate(bases.items()):
            rd_ = None
            if _fits_gpu(os.path.getsize(path)):
                data = _read_file(path)
                seen(name, data)
            else:
                # a base the GPU has no room for: mapped (its header and gaps are read from the
                # mapping), its tensors read explicitly by the coder (and hashed on the way)
                data = _map_file(path)
                rd_ = _HashingReader(path, on_stream(name) if on_stream else None)
                if on_stream is None:
                    seen(name, data)
            kept.append(_keep_base(data))
            if rd_ is not None and put_stream is not None and isinstance(self.base_codec, R2Base) \
                    and name not in self.seed_files and big1.index is None:
                big1.index = i
                base_jobs.append(bp.submit(code_big, name, data, rd_, f"b{i}"))
            else:
                base_jobs.append(bp.submit(code_base, name, data, rd_))
            del data
        small = {n: p for n, p in extras.items() if os.path.getsize(p) <= self.BUNDLE_MAX}
        ex = {}
        for n, p in small.items():
            with open(p, "rb") as f:
                ex[n] = f.read()
            seen(n, ex[n])
        c = lzma.compress(pack(ex), preset=9)
        put("extras", c)
        self.sizes["other files"] = len(c)
        big = {}
        zbig = zstandard.ZstdCompressionParameters.from_level(self.level, threads=8, enable_ldm=True)
        for j, (n, p) in enumerate((n, p) for n, p in extras.items() if n not in small):
            with open(p, "rb") as f:
                data = f.read()
            seen(n, data)
            c = bytes(memoryview(zstandard.ZstdCompressor(compression_params=zbig).compress(data)))
            put(f"x{j}", c)
            self.sizes[n] = len(c)
            big[n] = [f"x{j}", len(data)]
            if self.verbose:
                print(f"  {n}: {len(data):,} -> {len(c):,} B (zstd){_mem()}", flush=True)
            del data, c
        self._setup(kept, self._imatrix(ex))
        self._expect((t.name, t.type, 0) for p in files.values() for t in _quant_tensors(p))
        self.seen_t, self.keep_until, self.live = {}, {}, []      # duplicate tensors (_dedup)
        if self.verbose:
            print(f"  bases coded{_mem()}", flush=True)
        params = zstandard.ZstdCompressionParameters.from_level(self.level, window_log=25, threads=4)
        sparse = zstandard.ZstdCompressionParameters.from_level(max(self.level, 9), window_log=25, threads=4)

        def zc(data, zdict):
            # an almost all-zero residual compresses fast at any level: the higher one there
            v = np.frombuffer(data, np.uint8)
            prm = sparse if np.count_nonzero(v[::61]) * 61 < SPARSE * len(v) else params
            z = zstandard.ZstdCompressor(compression_params=prm, dict_data=zdict) if zdict else \
                zstandard.ZstdCompressor(compression_params=prm)
            # copy: compress() allocates a buffer the size of the input and only shrinks it, and on
            # Windows the shrunk block keeps the whole allocation committed
            return bytes(memoryview(z.compress(data)))

        manifest = {"method": self.name, "bases": list(bases), "seeds": seeds, "files": [None] * len(names),
                    "extras": {"bundle": list(ex), "single": big}}
        pending = deque()                     # (file index, zstd future, bytes held)

        def drain(limit):
            """wait for zstd jobs until the files they hold total at most `limit` bytes"""
            while pending and sum(s for _, _, s in pending) > limit:
                k, f, _ = pending.popleft()
                c = f.result()
                put(f"f{k}", c)
                self.sizes[names[k]] = len(c)

        def put_bases(wait=False):
            """the coded bases out as soon as they are ready (a big model's is gigabytes)"""
            for i, name in enumerate(bases):
                if base_jobs[i] is not None and (wait or base_jobs[i].done()):
                    c = base_jobs[i].result()
                    base_jobs[i] = None
                    if big1.index == i:
                        self.sizes[name] = c    # already written (code_big): its size
                        continue
                    put(f"b{i}", c)
                    self.sizes[name] = nbytes(c)
                    del c

        if big1.index is not None:
            # the big base's first pass done (the file is now in the OS cache): its second pass and
            # chunk go on while the derived files are coded
            big1.analysed.wait()
            big1.go.set()
            job = base_jobs[big1.index]
            while not big1.opened.wait(0.1):    # its chunk takes its place in the archive first
                if job.done():
                    job.result()                # (raises its error)
                    break
        elif huge:
            put_bases(wait=True)                # its coded form out of memory before the files

        # the files derived from a base too big for the GPU: all together, window by window
        large = [k for k, (n, p) in enumerate(bases.items())
                 if n not in self.seed_files and not _fits_gpu(os.path.getsize(p))]
        streamed = {}
        if large and (on_data is None or on_stream is not None):
            per = {}
            for i, n in enumerate(names):
                bb = self._best_base(types.SimpleNamespace(tensors=_gguf_tensors(files[n])))
                if bb in large:
                    per.setdefault(bb, []).append((i, n, files[n]))
            single = (zstandard.ZstdCompressionParameters.from_level(self.level, window_log=25),
                      zstandard.ZstdCompressionParameters.from_level(max(self.level, 9), window_log=25))
            for bb, its in per.items():
                for i, (entry, c, q) in self._stream_encode(bb, list(bases.values())[bb], its, single, on_stream).items():
                    put(f"f{i}", c)
                    self.sizes[names[i]] = nbytes(c)
                    if q is not None:
                        put(f"q{i}", q)
                        self.sizes[names[i] + " (QR)"] = len(q)
                    manifest["files"][i] = entry
                    streamed[i] = True
                    if self.verbose:
                        print(f"  {names[i]}: {fsize[i]:,} B (from {list(bases)[bb]}, streamed){_mem()}", flush=True)
        todo = [i for i in range(len(names)) if i not in streamed]

        # pipelined: the next file is read and older residuals are compressed while the GPU
        # predicts the current one, as long as the files held stay within self.inflight bytes
        # and the system can commit them
        with ThreadPoolExecutor(3) as zp:
            for j, i in enumerate(todo):
                name = names[i]
                if i not in reads:              # not read ahead
                    reads[i] = rd.submit(read, name)
                data = reads.pop(i).result()
                if j + 1 < len(todo):
                    # the next file is read while this one is predicted and XOR-ed
                    nx = todo[j + 1]
                    drain(max(0, self.inflight - fsize[i] - fsize[nx]))
                    free = _commit_free()
                    if free is None or free - fsize[i] - fsize[nx] > LOW_COMMIT:
                        reads[nx] = rd.submit(read, names[nx])
                plan, b = self._residual(data, fi=i)
                if self._qr_chunk is not None:
                    put(f"q{i}", self._qr_chunk)
                    self.sizes[name + " (QR)"] = len(self._qr_chunk)
                    self._qr_chunk = None
                _release()
                entry = {"name": name, "size": len(data), "base": b, "plan": plan}
                c = self._no_base(data) if b is None else None
                if c is not None:
                    entry["codec"] = "q8"
                    self.q8_files.add(name)
                    job = Future(); job.set_result(c)
                else:
                    job = zp.submit(zc, data, self.zdict if b is not None else None)
                manifest["files"][i] = entry
                pending.append((i, job, len(data)))
                del data, c
                if self.verbose:
                    src = list(bases)[b] if b is not None else "no base: " + ("M21" if "codec" in entry else "zstd only")
                    print(f"  {name}: {fsize[i]:,} B (from {src}){_mem()}", flush=True)
            drain(0)
        rd.shutdown()
        put_bases(wait=True)
        bp.shutdown()
        put("manifest", zstd_c(json.dumps(manifest).encode()))
        self.bases = self.base_dev = None
        _release(pinned=True)
        return pack(ch) if collect else None

    # ------------------------------------------------------------------ big models, streamed
    # A base too big for the GPU is never held whole: the files derived from it are processed
    # together, a window of tensors at a time (in the files' own tensor order), with only that
    # window's base tensors on the GPU. Each base tensor is then read / decoded once for all files,
    # and every file is read / written front to back.
    WINDOW = 1 << 30                          # base bytes per window

    streaming = False                         # in a streamed pass: the cache is bounded by memory only
    QPRED_GPU = 1 << 30                       # decoder: QR predictions kept on the GPU per window (bytes)
    PRED_WINDOW = 3 << 30                     # first predictions of a window's files (bytes) at most

    def _window_bytes(self):
        """the window size: WINDOW, less on a GPU with less free memory (the quantisers' float32
        copy of a launch and the QR coder's buffers come on top of the window)"""
        return int(max(64 << 20, min(self.WINDOW, (_dev_free() - GPU_RESERVE) // 4)))

    @staticmethod
    def _windows(nbytes, keep=(), window=WINDOW, pred=None, pred_window=PRED_WINDOW):
        """positions 0..len(nbytes)-1 (tensors in file order, nbytes: their base bytes) -> windows
        [(start, end)] of about `window` base bytes and `pred_window` bytes of predictions (pred:
        per position); keep: (lo, hi) position ranges that must stay in one window (QR batches)"""
        n = len(nbytes)
        pred = pred or [0] * n
        blocked = np.zeros(n + 1, bool)       # blocked[p]: no window may start at p
        for lo, hi in keep:
            blocked[lo + 1:hi + 1] = True
        out, s, acc, accp = [], 0, 0, 0
        for p in range(n):
            if p > s and (acc + nbytes[p] > window or accp + pred[p] > pred_window) and not blocked[p]:
                out.append((s, p)); s, acc, accp = p, 0, 0
            acc += nbytes[p]; accp += pred[p]
        if n:
            out.append((s, n))
        return out

    def _stream_plan(self, reader, b, files, zdict):
        """decoder side: the files (i, manifest entry) of big base b that can be streamed, grouped by
        tensor order -> [(order, windows, [(i, m, tensors by name, _Residual, QR chunk, QR batches)])]
        plus the indices of the files that cannot (archives of older versions: duplicate tensors,
        QR batches spanning the whole file)"""
        by_name = self.bases[b][2]
        groups, rejected = {}, set()
        limit = 4 * self._window_bytes()
        for i, m in files:
            plan = m["plan"]
            if not plan or m.get("codec") or any(len(e) < 4 or e[1] == "dup" for e in plan):
                rejected.add(i)
                continue
            res = _Residual(reader.open_chunk(f"f{i}"), zdict, m["size"])
            n = 32 << 20
            while True:
                try:
                    g = GGUF.from_bytes(res.peek(n).tobytes(), light=True)
                    break
                except Exception:
                    if n >= m["size"]:
                        raise
                    n *= 4
            tens = {t.name: t for t in g.tensors}
            order = tuple(sorted((e[0] for e in plan), key=lambda n: tens[n].offset))
            qch, qmeta = None, []
            if any(e[1] == "qr" for e in plan):
                qch = unpack(reader.get(f"q{i}"))
                qmeta = json.loads(zstd_d(qch["meta"]))
            groups.setdefault(order, []).append((i, m, tens, res, qch, qmeta))
        out = []
        for order, gf in groups.items():
            pos = {n: p for p, n in enumerate(order)}
            keep = [(min(pos[n] for n in names), max(pos[n] for n in names)) for f in gf for _, names in f[5]]
            nbytes = [by_name[n].nbytes for n in order]
            plans = [{e[0]: e for e in f[1]["plan"]} for f in gf]
            pred = [sum({(f[2][n].type, pl[n][2]): f[2][n].nbytes for f, pl in zip(gf, plans)}.values()) for n in order]
            wins = self._windows(nbytes, keep, self._window_bytes(), pred)
            if any(sum(nbytes[a:z]) > limit for a, z in wins):
                rejected.update(f[0] for f in gf)
                continue
            out.append((order, wins, gf))
        return out, rejected

    def _window_fetch(self, rt, names):
        """the coded stream of the base tensors `names`, read (in a background thread)"""
        return rt.fetch([rt.index[o] for o in {self.by_name[n].offset for n in names} if o in rt.index])

    def _window_base(self, rt, names, fetched=None):
        """_DevBase holding the base tensors `names` (R2Tensors rt: coded ones decoded on the GPU,
        the others uploaded from the base's uncoded part); fetched: _window_fetch's result"""
        bts = {self.by_name[n].offset: self.by_name[n] for n in names}
        coded = [rt.index[o] for o in bts if o in rt.index]
        dev, outs = rt.decode(fetched if fetched is not None else coded)
        coded = fetched[0] if fetched is not None else coded
        win = _DevBase(dev)
        for k, i in enumerate(coded):
            off, nb = rt.layout["t"][i][0], rt.layout["t"][i][1]
            win.add((off, off + nb), outs[k])
        others = [bt for o, bt in bts.items() if o not in rt.index]
        if others:                                # the uncoded ones (norms, ...): one upload
            at, n = [], 0
            for bt in others:
                at.append(n); n += -(-bt.nbytes // 256) * 256
            host = np.empty(n, np.uint8)
            for bt, p in zip(others, at):
                host[p:p + bt.nbytes] = rt.outside(bt.offset, bt.offset + bt.nbytes)
            dev2 = gpu2.h2d(host)
            for bt, p in zip(others, at):
                win.dev[(bt.offset, bt.offset + bt.nbytes)] = dev2[p:p + bt.nbytes]
        return win

    @staticmethod
    def _window_read(path, by_name, names):
        """the base tensors `names` read from the base file -> (host bytes, {(offset, end): position})"""
        bts = sorted({by_name[n].offset: by_name[n] for n in names}.values(), key=lambda t: t.offset)
        ranges = []                               # neighbouring tensors read in one go
        for bt in bts:
            if ranges and ranges[-1][1] == bt.offset:
                ranges[-1][1] = bt.offset + bt.nbytes
            else:
                ranges.append([bt.offset, bt.offset + bt.nbytes])
        host = np.empty(sum(z - a for a, z in ranges), np.uint8)
        where, o = {}, 0
        for a, z in ranges:
            _read_range(path, a, z - a, host[o:o + z - a])
            where[a] = o; o += z - a
        pos, k = {}, 0
        for bt in bts:
            while k + 1 < len(ranges) and ranges[k + 1][0] <= bt.offset:
                k += 1
            pos[(bt.offset, bt.offset + bt.nbytes)] = where[ranges[k][0]] + bt.offset - ranges[k][0]
        return host, pos

    def _v0(self, t, kind):
        """the variant a tensor is first predicted with (see _residual_tensors)"""
        return UNIT_W if kind == "rq" and self.imatrix.get(t.name) is None else 0

    def _prewarm(self, wanted, dev=None):
        """the predictions [(tensor, variant)] a window's files will ask for, made at once (big
        launches keep the GPU busy, where one file's few tensors per type would not) and cached
        until their last use by these files"""
        if self.uses is None:
            self.uses = Counter()
        want = {}
        for t, v in wanted:
            key = (t.name, t.type, v)
            if key not in want:
                want[key] = [t, 0]
            want[key][1] += 1
        for key, (t, c) in want.items():
            self.uses[key] = c + 1                # this call's use, then each file's
        self._predict([(t, key[2]) for key, (t, _) in want.items()], lambda key, p: None, dev)

    AHEAD = 2 << 30                           # bytes of the files read ahead of their coding (at most)
    QR_BACKLOG = 4                            # QR batches waiting for the coder
    FEED_BYTES = 2 << 30                      # coded spans waiting for zstd (at most)

    def _stream_encode(self, b, path, items, params, on_stream=None):
        """the derived files items [(index, name, path)] of big base b (the base file `path`),
        packed together window by window -> {index: (manifest entry, coded residual (buffers),
        QR chunk or None)}. params: zstd parameters (normal, sparse residual)"""
        import types
        self._use(b)
        by_name, zdict = self.by_name, self.zdict
        groups = {}
        for i, name, fpath in items:
            ts = [(t, self._kind(t)) for t in _gguf_tensors(fpath)]
            self.stats["raw"] += sum(k is None for _, k in ts)
            ts = sorted(((t, k) for t, k in ts if k), key=lambda x: x[0].offset)
            if not ts:                          # nothing to predict: coded the usual way
                continue
            st = types.SimpleNamespace(
                i=i, name=name, path=fpath, size=os.path.getsize(fpath), tens={t.name: t for t, _ in ts},
                kind={t.name: k for t, k in ts}, start=ts[0][0].offset, pos=0, plan=[], qmeta=[], qparts=[],
                comp=None, spill=_Spill(self.scratch), lane=ThreadPoolExecutor(1), qjobs=[],
                hlane=ThreadPoolExecutor(1),
                hash=on_stream(name) if on_stream else None)
            groups.setdefault(tuple(t.name for t, _ in ts), []).append(st)
        out, feeds = {}, deque()
        qpool = ThreadPoolExecutor(1)             # the QR coder, overlapping the next file's work

        def span_read(st, s0, s1, names):
            """a span read (in the reader thread); its hashing started: the file's (in order, in
            the file's own lane) and its predicted tensors' (tensors equal to one of an earlier
            file are not coded again)"""
            buf = _read_range(st.path, s0, s1 - s0, once=True)
            hf = st.hlane.submit(st.hash, buf) if st.hash else None
            tf = {n: _POOL.submit(_digest, buf[st.tens[n].offset - s0:st.tens[n].offset - s0 + st.tens[n].nbytes])
                  for n in names}
            return buf, hf, tf

        def compress(st, span, wait):
            for f in wait:                      # the QR jobs zero their tensors' regions first
                f.result()
            st.spill.write(st.comp.compress(span))

        def feed(st, span, wait=()):
            if st.comp is None:
                # the level as zc() picks it for a whole file, judged from the first span's tensors
                v = span[max(0, st.start - st.pos):]
                prm = params[1] if len(v) and np.count_nonzero(v[::61]) * 61 < SPARSE * len(v) else params[0]
                z = zstandard.ZstdCompressor(compression_params=prm, dict_data=zdict) if zdict else \
                    zstandard.ZstdCompressor(compression_params=prm)
                st.comp = z.compressobj(size=st.size)
            feeds.append((st.lane.submit(compress, st, span, wait), len(span)))
            while len(feeds) > 1 and sum(n for _, n in feeds) > self.FEED_BYTES:
                feeds.popleft()[0].result()

        home = self.bases[b]
        self.streaming = True
        for order, sts in groups.items():
            # the window's distinct first predictions (each file's tensors, repeats once)
            pred = [sum({(st.tens[n].type, self._v0(st.tens[n], st.kind[n])): st.tens[n].nbytes for st in sts}.values())
                    for n in order]
            wlim = self._window_bytes()
            wins = self._windows([by_name[n].nbytes for n in order], (), wlim, pred)
            wb = [sum(by_name[n].nbytes for n in order[a:z]) for a, z in wins]
            jobs = []                             # (window, file, start, end) in processing order
            pos = {st.i: 0 for st in sts}
            for w, (a, z) in enumerate(wins):
                for st in sts:
                    end = st.size if w == len(wins) - 1 else \
                        max(st.tens[n].offset + st.tens[n].nbytes for n in order[a:z])
                    jobs.append((w, st, pos[st.i], end)); pos[st.i] = end
            def prepare(w, read):
                """window w on the GPU, its files' first predictions made: in the GPU thread, while
                the previous window's files are coded"""
                host, where = read.result()
                win = _DevBase(gpu2.h2d(host))
                del host
                for key, p in where.items():
                    win.add(key, p)
                names = order[wins[w][0]:wins[w][1]]
                self._prewarm([(st.tens[n], self._v0(st.tens[n], st.kind[n])) for st in sts for n in names], win)
                _release(device=True)           # (three threads use the GPU: its pool must not hoard)
                return win

            def ahead(w):
                return gpu.submit(prepare, w, bread.submit(self._window_read, path, by_name,
                                                           order[wins[w][0]:wins[w][1]]))

            with ThreadPoolExecutor(1) as bread, ThreadPoolExecutor(1) as sread, ThreadPoolExecutor(1) as gpu:
                prep = {0: ahead(0)}
                wq_prev = []
                sf = deque()
                nxt = 0
                for w, (a, z) in enumerate(wins):
                    tw = {"predict wait": -time.perf_counter()}
                    win = prep.pop(w).result()
                    tw["predict wait"] += time.perf_counter()
                    if w + 1 < len(wins) and wb[w] + wb[w + 1] <= 2 * wlim:
                        prep[w + 1] = ahead(w + 1)
                    dev = win.buf
                    self.bases[b] = self.bases[b][:1] + (win,) + self.bases[b][2:]
                    self._use(b)
                    names = order[a:z]
                    memo, qmemo, wq = {}, {}, []
                    tw["files"], tw["span wait"] = -time.perf_counter(), 0.0
                    free = _commit_free()
                    limit = self.AHEAD if free is None else min(self.AHEAD, max(1 << 30, (free - LOW_COMMIT) // 3))
                    for st in sts:
                        # spans read ahead: at least the next one, at most `limit` bytes
                        while nxt < len(jobs) and (len(sf) < 2 or sum(n for _, n in sf) < limit):
                            ww, sst, s0, s1 = jobs[nxt]
                            sf.append((sread.submit(span_read, sst, s0, s1, order[wins[ww][0]:wins[ww][1]]), s1 - s0))
                            nxt += 1
                        tw["span wait"] -= time.perf_counter()
                        span, hf, tf = sf.popleft()[0].result()
                        if hf is not None:
                            hf.result()             # hashed before it is modified
                        hashes = {n: f.result() for n, f in tf.items()}
                        tw["span wait"] += time.perf_counter()
                        plan, qitems, choice = self._residual_tensors(
                            span, st.pos, [(st.tens[n], st.kind[n]) for n in names], len(span), memo, hashes)
                        qf = self._qr_submit(qpool, qitems, choice, hashes, qmemo) if qitems else []
                        st.qjobs += qf; wq += qf
                        while sum(not f.done() for f in wq_prev + wq) > self.QR_BACKLOG:
                            next(f for f in wq_prev + wq if not f.done()).result()
                        st.plan += plan
                        feed(st, span, qf)
                        st.pos += len(span)
                        del span
                    tw["files"] += time.perf_counter()
                    tw["QR wait"] = -time.perf_counter()
                    for f in wq_prev:             # the previous window's (its buffer is their seed)
                        f.result()
                    tw["QR wait"] += time.perf_counter()
                    wq_prev = wq
                    del win, dev, wq
                    if w + 1 < len(wins) and w + 1 not in prep:
                        for f in wq_prev:         # (a big window next: this one's buffer released first)
                            f.result()
                        wq_prev = []
                        _release()
                        prep[w + 1] = ahead(w + 1)
                    self.bases[b] = home
                    self._use(b)
                    _release()
                    if self.verbose:
                        print(f"  window {w + 1}/{len(wins)}: {z - a} tensors x {len(sts)} files, "
                              f"{sum(by_name[n].nbytes for n in names) / 1e9:.2f} GB of the base, "
                              f"{sum(st.pos for st in sts) / 1e9:.1f} GB of the files read{_mem()}"
                              + "".join(f", {k} {v:.1f} s" for k, v in tw.items())
                              + (f", cache {self.stats.get('cache_hit', 0)} hit / {self.stats.get('cache_miss', 0)} miss"
                                 f" / {self.stats.get('cache_low_memory', 0)} no memory, {self.cached / 1e9:.1f} GB"
                                 if os.environ.get("SEEDPACK_MEMDEBUG") else ""), flush=True)
            for f in wq_prev:
                f.result()
            for st in sts:
                st.lane.submit(lambda st=st: st.spill.write(st.comp.flush())).result()
                st.lane.shutdown(); st.hlane.shutdown()
                for f in st.qjobs:
                    m, part = f.result()
                    st.qmeta.append(m); st.qparts.append(part)
                q = self._qr_pack(st.qmeta, st.qparts) if st.qmeta else None
                out[st.i] = ({"name": st.name, "size": st.size, "base": b, "plan": st.plan}, st.spill.data(), q)
        while feeds:
            feeds.popleft()[0].result()
        self.streaming = False
        return out

    def _stream_decode(self, b, rt, order, wins, gf, sink, want):
        """decode the files gf (from _stream_plan) of big base b window by window. The next
        window's coded tensors and the files' next residual spans are read in the background"""
        outs = [_out_stream(sink, m["name"], m["size"]) if want(m["name"]) else None for _, m, *_ in gf]
        plans = [{e[0]: e for e in m["plan"]} for _, m, *_ in gf]
        jobs, pos = [], [0] * len(gf)         # (file, span start, span end) in processing order
        for w, (a, z) in enumerate(wins):
            for k, (i, m, tens, *_) in enumerate(gf):
                end = m["size"] if w == len(wins) - 1 else max(tens[n].offset + tens[n].nbytes for n in order[a:z])
                jobs.append((k, pos[k], end)); pos[k] = end
        home = self.bases[b]
        self.streaming = True

        def batch_keys(m, qch, qmeta):
            """per QR batch: its type, tensors, their variants and its coded bytes' hash"""
            var = {e[0]: e[2] for e in m["plan"]}
            return [(T, tuple(qn), tuple(var[n] for n in qn), blake3.blake3(qch[str(q)]).digest())
                    for q, (T, qn) in enumerate(qmeta)]

        bkeys = {i: batch_keys(m, qch, qmeta) for i, m, tens, res, qch, qmeta in gf}

        def prepare(w, fetch):
            """window w decoded on the GPU and its files' predictions made: in the GPU thread, while
            the previous window's files are finished"""
            names = order[wins[w][0]:wins[w][1]]
            win = self._window_base(rt, names, fetch.result())
            # the QR tensors' predictions stay on the GPU (the QR decoder's input), the others
            # come to the host (XOR-ed there)
            qkeys, host, room = {}, [], min(self.QPRED_GPU, cp.cuda.runtime.memGetInfo()[1] // 24)
            for k, f in enumerate(gf):
                for n in names:
                    e, t = plans[k][n], f[2][n]
                    key = (n, t.type, e[2])
                    if e[1] == "qr" and (key in qkeys or t.nbytes <= room):
                        if key not in qkeys:
                            qkeys[key] = t; room -= t.nbytes
                    else:                       # (beyond the GPU budget: through the host cache)
                        host.append((t, e[2]))
            self._prewarm(host, win)
            win.qpred = {}
            self._predict([(t, key[2]) for key, t in qkeys.items()],
                          lambda key, p: win.qpred.__setitem__(key, p), win, device=True)
            _release()
            return win

        fetched = {}                          # the coded windows read ahead (two: the drive never waits)

        def fetch(w):
            if w < len(wins) and w not in fetched:
                fetched[w] = bread.submit(self._window_fetch, rt, order[wins[w][0]:wins[w][1]])

        def ahead(w):
            fetch(w)
            fetch(w + 1)
            return gpu.submit(prepare, w, fetched.pop(w))

        wb = [sum(self.by_name[n].nbytes for n in order[a:z]) for a, z in wins]
        limit = 2 * self._window_bytes()

        def both_fit(w):
            """whether windows w and w + 1 may be on the GPU together"""
            return wb[w] + wb[w + 1] <= limit

        with ThreadPoolExecutor(1) as bread, ThreadPoolExecutor(1) as sread, ThreadPoolExecutor(1) as gpu:
            prep = {0: ahead(0)}
            sf, nxt, jn = deque(), 0, 0
            for w, (a, z) in enumerate(wins):
                names = order[a:z]
                win = prep.pop(w).result()
                if w + 1 < len(wins) and both_fit(w):
                    prep[w + 1] = ahead(w + 1)
                self.bases[b] = home[:1] + (win,) + home[2:]
                self._use(b)
                inwin = set(names)
                # QR batches equal to one of an earlier file (same tensors, variants, coded bytes:
                # the encoder codes such a batch once) are decoded once
                need, qdmemo = Counter(), {}
                for i, m, tens, res, qch, qmeta in gf:
                    for q, (T, qn) in enumerate(qmeta):
                        if qn[0] in inwin:
                            need[bkeys[i][q]] += 1
                for _ in gf:
                    while nxt < len(jobs) and len(sf) < 3:
                        k, _, end = jobs[nxt]
                        sf.append(sread.submit(gf[k][3].take, end)); nxt += 1
                    k, p0, _ = jobs[jn]; jn += 1
                    span = sf.popleft().result()
                    i, m, tens, res, qch, qmeta = gf[k]
                    plan, xj = plans[k], []

                    def xor(key, p):
                        t = tens[key[0]]
                        xj.extend(_xor_async(span[t.offset - p0:t.offset - p0 + t.nbytes], p, check=False))

                    self._predict([(tens[n], plan[n][2]) for n in names if plan[n][1] != "qr"], xor)
                    for q, (T, qn) in enumerate(qmeta):
                        if qn[0] in inwin:
                            bk = bkeys[i][q]
                            held = qdmemo.get(bk)
                            if held is None:
                                held = {n: win.qpred.get((n, tens[n].type, plan[n][2])) for n in qn}
                                miss = [n for n in qn if held[n] is None]
                                if miss:
                                    self._predict([(tens[n], plan[n][2]) for n in miss],
                                                  lambda key, p: held.__setitem__(key[0], p))
                                self._qr_decode_batch(qch, q, T, qn, held, release=False)
                                _trim_cached()      # (the batches' temporaries must not pile up)
                            need[bk] -= 1
                            if need[bk] > 0:
                                qdmemo[bk] = held       # the same batch (same bytes) in a later file
                            else:
                                qdmemo.pop(bk, None)
                            for n in qn:
                                xor((n,), held[n])
                    for f in xj:
                        f.result()
                    if outs[k] is not None:
                        outs[k].write(span)
                    span = xj = None
                del win
                self.bases[b] = home
                self._use(b)
                _release()
                if w + 1 < len(wins) and w + 1 not in prep:
                    prep[w + 1] = ahead(w + 1)
                if os.environ.get("SEEDPACK_MEMDEBUG"):
                    print(f"  window {w + 1}/{len(wins)}: {z - a} tensors x {len(gf)} files{_mem()}", flush=True)
        self.streaming = False
        for out in outs:
            if out is not None:
                out.close()

    # ------------------------------------------------------------------ decompression
    scratch = None                            # directory for _scratch_array (None: the system's temp)

    def _base_alloc(self, n):
        """output buffer of a decoded base: memory when it fits on the GPU, else a temporary
        file's mapping (a big model's base would otherwise take its size in RAM + page file)"""
        return np.empty(n, np.uint8) if _fits_gpu(n) else _scratch_array(n, self.scratch)

    @staticmethod
    def manifest_of(get):
        return json.loads(zstd_d(get("manifest")))

    @staticmethod
    def _rt_header(rt):
        rest = rt.rest.tobytes() if len(rt.rest) < (256 << 20) else bytes(rt.rest[:256 << 20])
        return rest[:GGUF.from_bytes(rest, light=True).data_start]

    def decompress(self, src, sink, select=None):
        """src: the packed archive (bytes), a chunk getter get(name) -> bytes, or a container.Reader
        (then a base too big for the GPU is decoded tensor by tensor, never held whole).
        sink(name, bytes-like) receives every file, or only those in `select` (a set of names);
        only the bases those files need are decoded. A sink with a stream(name, size) method
        (-> object with write(bytes-like) and close()) receives big files in pieces instead."""
        reader = src if hasattr(src, "read_at") else None
        get = reader.get if reader is not None else src if callable(src) else unpack(src).__getitem__
        manifest = self.manifest_of(get)
        want = (lambda n: True) if select is None else (lambda n: n in select)
        mf = manifest["files"]
        sel = {i for i, m in enumerate(mf) if want(m["name"])}
        todo = list(sel)
        while todo:                           # files holding the sources of duplicate tensors
            for e in mf[todo.pop()]["plan"]:
                if e[1] == "dup" and e[2][0] not in sel:
                    sel.add(e[2][0]); todo.append(e[2][0])
        files = [(i, mf[i]) for i in sorted(sel)]
        last_use = {}                         # (file, tensor) -> last file that copies it
        for i, m in files:
            for e in m["plan"]:
                if e[1] == "dup":
                    last_use[tuple(e[2])] = max(last_use.get(tuple(e[2]), -1), i)
        need = {i for i, n in enumerate(manifest["bases"]) if want(n)} | \
               {m["base"] for _, m in files if m["base"] is not None}
        bases = []
        seeds = set(manifest.get("seeds", []))
        big = {}                              # R2 bases too big for the GPU -> R2Tensors
        if reader is not None:
            for i in sorted(need):
                if manifest["bases"][i] not in seeds:
                    rt = _r2_tensors(reader, f"b{i}")
                    if rt is not None and not _fits_gpu(rt.size):
                        big[i] = rt
        # the residuals' zstd dictionaries are the bases' headers, decoded first (cheap), so the
        # residuals decompress on the CPU while the bases decode on the GPU
        base_buf = {i: get(f"b{i}") for i in sorted(need) if i not in big}
        zdicts = {}
        for i in need:
            h = self._rt_header(big[i]) if i in big else self._base_header(base_buf[i], manifest["bases"][i] in seeds)
            if h is not None:
                zdicts[i] = zstandard.ZstdCompressionDict(h, dict_type=zstandard.DICT_TYPE_RAWCONTENT)

        def zd(i):
            m = manifest["files"][i]
            if m.get("codec") == "q8":
                return get(f"f{i}")             # decoded on the GPU in the main loop
            b = m["base"]
            zdict = None if b is None else zdicts[b] if b in zdicts else self.bases[b][3]
            dec = zstandard.ZstdDecompressor(dict_data=zdict) if zdict else zstandard.ZstdDecompressor()
            return _zstd_into(dec, get(f"f{i}"), m["size"])

        zp = ThreadPoolExecutor(2)
        nxt, ahead = {}, 0                    # files decompressed ahead, within self.inflight bytes
        if len(zdicts) == len({m["base"] for _, m in files if m["base"] is not None}) and not big:
            while ahead < len(files) and ahead < 2 and sum(
                    files[q][1]["size"] for q in range(0, ahead + 1)) <= self.inflight:
                nxt[ahead] = zp.submit(zd, files[ahead][0])
                ahead += 1
        for i, name in enumerate(manifest["bases"]):
            if i not in need:
                bases.append(None)
                continue
            predicts = any(m["base"] == i for _, m in files)
            if i in big:                      # streamed: decoded window by window, here or below
                if want(name):
                    out = _out_stream(sink, name, big[i].size)
                    big[i].write_to(out.write)
                    out.close()
                bases.append((self._rt_header(big[i]), big[i]) if predicts else None)
                _release()
                continue
            buf = base_buf.pop(i)
            if name in seeds:
                data = SeedCodec().decompress(buf)
                kept = _keep_base(data) if predicts else None
            elif hasattr(self.base_codec, "decompress_host"):
                data = self.base_codec.decompress_host(unpack(memoryview(buf)), alloc=self._base_alloc)
                kept = _keep_base(data) if predicts else None
            else:
                dev = self.base_codec.decompress_dev(unpack(buf))
                data = cp.asnumpy(dev)
                kept = (data[:GGUF.from_bytes(memoryview(data), light=True).data_start].tobytes(), dev) if predicts else None
            if want(name):
                sink(name, data)
            bases.append(kept)
            del data, buf
            _release()
        exm = manifest.get("extras", {"bundle": None, "single": {}})
        ex = {}
        if exm["bundle"] is None or any(want(n) for n in exm["bundle"]) or files:
            ex = dict(unpack(lzma.decompress(get("extras"))))
            for n, b in ex.items():
                if want(n):
                    sink(n, b)
        for n, (c, size) in exm["single"].items():
            if want(n):
                sink(n, zstandard.ZstdDecompressor().decompress(get(c), max_output_size=size))
        if not files:
            zp.shutdown()
            return
        if any(m["base"] is not None for _, m in files):
            self._setup(bases, self._imatrix(ex))
            if all(len(e) > 3 for _, m in files for e in m["plan"]):     # archives that record the types
                self._expect((e[0], e[3], e[2]) for _, m in files for e in m["plan"] if e[1] != "dup")
        sources = {k[0] for k in last_use}      # files holding tensors that others copy: not streamed
        for b, rt in big.items():
            mine = [(i, m) for i, m in files if m["base"] == b and i not in sources]
            groups, rejected = self._stream_plan(reader, b, mine, zdicts.get(b))
            for order, wins, gf in groups:
                self._stream_decode(b, rt, order, wins, gf, sink, want)
            done = {f[0] for _, _, gf in groups for f in gf}
            files = [(i, m) for i, m in files if i not in done]
            if os.environ.get("SEEDPACK_DEBUG"):
                print(f"  base {b}: {len(done)} file(s) streamed in {len(groups)} group(s), "
                      f"{sum(m['base'] == b for _, m in files)} the old way", flush=True)
            if any(m["base"] == b and m["plan"] for _, m in files):
                # files of an older archive that cannot be streamed: the whole base, decoded into a
                # temporary file, as before
                data = self.base_codec.decompress_host(unpack(memoryview(get(f"b{b}"))), alloc=self._base_alloc)
                hdr, dev = _keep_base(data)
                self.bases[b] = self.bases[b][:1] + (dev,) + self.bases[b][2:]
                del data
        if not files:
            zp.shutdown()
            self.bases = self.base_dev = None
            _release(pinned=True)
            return

        typed = all(len(e) > 3 for _, m in files for e in m["plan"])
        # bases in host memory (a big model): a file's predictions are XOR-ed in as they come,
        # not held whole one file ahead
        stream = any(b is not None and isinstance(b[1], _HostBase) for b in self.bases or [])

        def preds(i):
            """the predictions of file i, from the manifest alone (no need for its residual), so the
            GPU works one file ahead of the zstd / XOR / output of the current one"""
            m = manifest["files"][i]
            if not m["plan"] or not typed:
                return None
            self._use(m["base"])
            out = {}
            self._predict([(_T(e[0], e[3], self.by_name[e[0]].shape), e[2]) for e in m["plan"] if e[1] != "dup"],
                          lambda key, p: out.__setitem__(key[0], p))
            if any(e[1] == "qr" for e in m["plan"]):
                self._qr_decode(get(f"q{i}"), out)
            _release()                        # here: cudaFree waits for the GPU, which is this thread's
            return out

        with zp, ThreadPoolExecutor(1) as gp:
            pf = {}                           # predictions: the current file and the next
            kept_t = {}                       # (file, tensor) -> bytes, kept for later duplicates
            for k, (i, m) in enumerate(files):
                while ahead < len(files) and (ahead <= k or (ahead - k < 3 and sum(
                        files[q][1]["size"] for q in range(k, ahead + 1)) <= self.inflight)):
                    nxt[ahead] = zp.submit(zd, files[ahead][0])
                    ahead += 1
                for q in (k, k + 1):
                    if q < len(files) and q not in pf and not stream:
                        pf[q] = gp.submit(preds, files[q][0])
                data = nxt.pop(k).result()
                if m.get("codec") == "q8":
                    data = self._q8_codec().decompress(data)
                if m["plan"]:
                    view = np.frombuffer(data, np.uint8)
                    tens = {t.name: t for t in GGUF.from_bytes(data, light=True).tensors}
                    by_name = self.bases[m["base"]][2]
                    pred = pf.pop(k).result() if not stream else None
                    plan = [e for e in m["plan"] if e[1] != "dup"]
                    if stream:
                        gp.submit(self._predict_xor, m["base"], plan, tens, view, lambda: get(f"q{i}")).result()
                        pred = {}
                    elif pred is None or any(tens[e[0]].shape[0] != by_name[e[0]].shape[0] for e in plan):
                        # old archive (no types in the plan) or rows unlike the base's: predicted
                        # here from the file's own header
                        pred = {}
                        gp.submit(self._predict_into, m["base"], [(tens[e[0]], e[2]) for e in plan], pred).result()
                        if any(e[1] == "qr" for e in plan):
                            gp.submit(self._qr_decode, get(f"q{i}"), pred).result()
                    jobs = []
                    for n, p in pred.items():
                        t = tens[n]
                        jobs.extend(_xor_async(view[t.offset:t.offset + t.nbytes], p, check=False))
                    for f in jobs:
                        f.result()
                    jobs = []
                    for e in m["plan"]:
                        if e[1] == "dup":             # a copy of an earlier file's tensor
                            t = tens[e[0]]
                            jobs += _copy_async(view[t.offset:t.offset + t.nbytes], kept_t[tuple(e[2])])
                    for f in jobs:
                        f.result()
                    for key in [key for key in kept_t if last_use[key] <= i]:
                        del kept_t[key]
                    jobs = []
                    for n, t in tens.items():
                        if (i, n) in last_use:
                            kept_t[(i, n)] = np.empty(t.nbytes, np.uint8)
                            jobs += _copy_async(kept_t[(i, n)], view[t.offset:t.offset + t.nbytes])
                    for f in jobs:
                        f.result()
                    del view, jobs, pred
                elif not stream:
                    pf.pop(k).result()
                if want(m["name"]):
                    sink(m["name"], data)
                del data
                _release(device=False)
        self.bases = self.base_dev = None
        _release(pinned=True)


class R2Base:
    """compress_dev / decompress_dev interface (as M24's) around the BF16 v2 codec (bf16v2.py:
    finer contexts, role-shared tables, duplicate rows, bounded GPU memory)"""

    def __init__(self, budget=1_500_000_000):
        from .bf16v2 import BF16v2
        self.c = BF16v2(budget=budget)

    def compress_dev(self, data, dev=None, read=None):
        return {"r2": self.c.compress_parts(data, read=read)}

    def decompress_dev(self, ch):
        return cp.asarray(self.c.decompress(ch["r2"]))

    def decompress_host(self, ch, alloc=None):
        return self.c.decompress(ch["r2"], alloc=alloc)


class M29RepoGPU(M28RepoGPU):
    name = "M29 model repo: M28 with the BF16 v2 base codec (R2)"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.base_codec = R2Base()
