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
from collections import Counter, deque, namedtuple
from concurrent.futures import Future, ThreadPoolExecutor

import cupy as cp
import blake3
import numpy as np
import zstandard

from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gguf_reader import GGUF
from .gpu_bf16 import M24BF16GPU
from . import gpuquant
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


def _read_file(path):
    """the whole file as a writable buffer, read in parallel chunks (an NVMe drive delivers ~30%
    more with several requests in flight) into anonymous memory (no zero-filling first)"""
    import mmap
    n = os.path.getsize(path)
    if n == 0:
        return bytearray()
    buf = mmap.mmap(-1, n)
    mv = memoryview(buf)

    def part(o):
        with open(path, "rb", buffering=0) as f:
            f.seek(o)
            want = min(_READ_CHUNK, n - o)
            got = f.readinto(mv[o:o + want])
            while got < want:                 # readinto may return less (rare)
                k = f.readinto(mv[o + got:o + want])
                if not k:
                    raise IOError(f"{path}: file shrank while being read")
                got += k

    for f in [_IO.submit(part, o) for o in range(0, n, _READ_CHUNK)]:
        f.result()
    del mv
    return buf


def _mem():
    """process commit charge, when VARC_MEMDEBUG is set (debugging aid)"""
    if not os.environ.get("VARC_MEMDEBUG"):
        return ""
    import psutil
    free, total = cp.cuda.runtime.memGetInfo()
    return (f"  [private {psutil.Process().memory_info().private / 1e9:.1f} GB, GPU used {(total - free) / 1e9:.1f} GB,"
            f" pool {cp.get_default_memory_pool().used_bytes() / 1e9:.1f}/{cp.get_default_memory_pool().total_bytes() / 1e9:.1f} GB]")


def _map_file(path):
    """the file's read-only mapping: its pages come from the OS cache, not the commit charge"""
    import mmap
    with open(path, "rb") as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


GPU_RESERVE = 5 << 30                     # device bytes the bases leave free for the rest of the work


def _dev_free():
    """device bytes free for new arrays (CuPy's cached free blocks included)"""
    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


def _fits_gpu(n):
    """whether a base of n bytes is kept whole on the GPU (VARC_BASE_GPU_MAX_MB caps it, for tests)"""
    cap = os.environ.get("VARC_BASE_GPU_MAX_MB")
    return n + GPU_RESERVE <= _dev_free() and (cap is None or n <= int(cap) << 20)


class _HostBase:
    """a base too large for the GPU, sliced as the GPU code slices a device copy (base[a:b] ->
    device array): kept in host memory (when packing, the file's mapping) and uploaded per tensor.
    Uploaded tensors stay on the GPU while it has room, from the first used on: every file is
    predicted in the same tensor order, so a fixed set is reused where a least recently used one
    would always miss"""

    def __init__(self, host):
        self.host = np.frombuffer(host, np.uint8)
        self.dev = {}

    def __len__(self):
        return len(self.host)

    def __getitem__(self, s):
        a = self.dev.get((s.start, s.stop))
        if a is None:
            a = cp.asarray(self.host[s])
            if _dev_free() - a.nbytes > GPU_RESERVE:
                self.dev[(s.start, s.stop)] = a
        return a


class _Upload:
    """a tensor of a base kept in host memory, as a quantiser input: uploaded only when its launch
    converts it to float32, so a launch's tensors are not all on the GPU twice"""

    def __init__(self, get, size):
        self.get, self.size = get, size

    def into(self, out):
        gpuquant._to_f32_into(self.get(), out)


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
        dev[o:o + chunk] = cp.asarray(host[o:o + chunk])
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

    def _src(self, bt, lazy=False):
        """base tensor bt as the quantisers take it; lazy, from a base in host memory: a _Upload"""
        dev = self.base_dev
        if lazy and isinstance(dev, _HostBase):
            return _Upload(lambda: self._view(dev, bt), int(np.prod(bt.shape)))
        return self._view(dev, bt)

    def _view(self, dev, bt):
        a = dev[bt.offset:bt.offset + bt.nbytes]
        if bt.type in QFUN:                   # a seed (quantised base): dequantised when used
            return _Dequant(self.gq, bt, a)
        return a.view(cp.uint16) if bt.type == BF16 else a.view(cp.float16) if bt.type == F16 else a.view(cp.float32)

    def _float_dev(self, bt, ttype):
        """base tensor bt converted to float type ttype on the GPU (same rounding as float_as_np), as bytes"""
        a = self._src(bt)
        if isinstance(a, _Dequant):
            x = a.f32()
        else:
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
        if self.cached + nbytes > self.cache_bytes:
            return False
        free = _commit_free()
        return free is None or free - nbytes > LOW_COMMIT

    def _predict_into(self, b, wanted, out):
        self._use(b)
        self._predict(wanted, lambda key, p: out.__setitem__(key[0], p))

    def _predict_xor(self, b, plan, tens, view, qr_buf):
        """decoding a file (`view`, its residual) without holding its predictions: each is XOR-ed
        in as soon as it is ready; only the QR tensors' are kept, for their decoder (qr_buf() ->
        the file's QR chunk)"""
        self._use(b)
        isqr = {e[0] for e in plan if e[1] == "qr"}
        held, jobs = {}, []

        def xor(n, p):
            t = tens[n]
            jobs.extend(_xor_async(view[t.offset:t.offset + t.nbytes], p, check=False))

        self._predict([(tens[e[0]], e[2]) for e in plan],
                      lambda key, p: held.__setitem__(key[0], p) if key[0] in isqr else xor(key[0], p))
        if held:
            self._qr_decode(qr_buf(), held)
            for n, p in held.items():
                xor(n, p)
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
        hit = self.cache.pop((self.b,) + key, None)
        if hit is not None:
            self.cached -= hit.nbytes
        return False

    FLOAT_BATCH = 256 << 20                   # base bytes of float tensors converted in one go

    def _predict(self, wanted, use):
        """wanted: list of (tensor, variant); use((name, type, variant), bytes as numpy uint8) is
        called as soon as each prediction is ready, so a file's predictions are never all held"""
        todo, floats = {}, []
        for t, v in wanted:
            key = (t.name, t.type, v)
            if (self.b,) + key in self.cache:
                p = self.cache[(self.b,) + key]
                self._used(key)
                use(key, p)
            elif t.type in (F32, F16, BF16):
                floats.append(key)
            else:
                todo.setdefault((t.type, v), {})[key] = t
        groups, nb = [[]], 0
        for key in floats:
            # float tensors (norms, ...) are small and many: converted together, one download per
            # group of at most 512 tensors and FLOAT_BATCH base bytes (an F16/F32 file has big ones)
            b = self.by_name[key[0]].nbytes
            if groups[-1] and (len(groups[-1]) == 512 or nb + b > self.FLOAT_BATCH):
                groups.append([]); nb = 0
            groups[-1].append(key); nb += b
        for keys in groups if floats else []:
            ys = [self._float_dev(self.by_name[n], ty) for n, ty, _ in keys]
            host = to_host(cp.concatenate(ys)) if len(ys) > 1 else to_host(ys[0])
            o = 0
            for key, y in zip(keys, ys):
                p = host[o:o + y.size]; o += y.size
                if self._used(key) and self._room(p.nbytes):
                    self.cache[(self.b,) + key] = p.copy(); self.cached += p.nbytes
                use(key, p)
            del ys, host
        for (ty, v), ts in todo.items():
            keys = list(ts)

            def got(i, o):
                if self._used(keys[i]) and self._room(o.nbytes):
                    # a copy: o is a view of the whole batch output, which it would keep alive
                    self.cache[(self.b,) + keys[i]] = o.copy(); self.cached += o.nbytes
                use(keys[i], o)

            srcs = [(self._src(self.by_name[n], lazy=True), ts[(n, ty, v)].shape[0]) for n, _, _ in keys]
            # variant v: the quantiser's code variant v % UNIT_W, with unit weights from UNIT_W
            ims = [self.imatrix.get(n) if v < UNIT_W else np.ones(ts[(n, ty, v)].shape[0], np.float32)
                   for n, _, _ in keys]
            self.gq.quantize_tensors(ty, srcs, ims, v % UNIT_W, on_result=got)

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
        if not isinstance(self.base_dev, _HostBase):
            return self.base_dev, {n: self.by_name[n].offset for n in names}
        off, o = {}, 0
        for n in names:
            off[n] = o
            o += -(-self.by_name[n].nbytes // 256) * 256
        seed = cp.empty(max(o, 1), cp.uint8)
        for n in names:
            bt = self.by_name[n]
            seed[off[n]:off[n] + bt.nbytes] = self.base_dev[bt.offset:bt.offset + bt.nbytes]
        return seed, off

    def _qr_encode(self, items):
        """[(tensor, file bytes, prediction)] -> the file's QR chunk; the regions are zeroed"""
        by = {}
        for t, f, p in items:
            by.setdefault(t.type, []).append((t, f, p))
        meta, parts = [], {}
        for T, ts in by.items():
            for bi in qr.batches(T, [t.nbytes for t, _, _ in ts]):
                its = [ts[i] for i in bi]
                seed, off = self._seed_dev([t.name for t, _, _ in its])
                parts[str(len(meta))] = qr.encode(
                    T, [(f, p, off[t.name], self.by_name[t.name].type) for t, f, p in its], seed)
                meta.append([T, [t.name for t, _, _ in its]])
                del seed
        for _, f, _ in items:
            f[:] = 0
        self.stats["qr"] = self.stats.get("qr", 0) + len(items)
        _release()
        return pack({"meta": zstd_c(json.dumps(meta).encode()), **parts})

    def _qr_decode(self, buf, pred):
        """the file's QR chunk + predictions {name: bytes} -> the QR tensors' bytes put in `pred`
        (their residual regions are zero, so the caller's XOR writes them)"""
        ch = unpack(buf)
        for k, (T, names) in enumerate(json.loads(zstd_d(ch["meta"]))):
            seed, off = self._seed_dev(names)
            out = qr.decode(T, ch[str(k)], [(pred[n], off[n], self.by_name[n].type) for n in names], seed)
            del seed
            host = to_host(out)
            o = 0
            for n in names:
                sz = len(pred[n])
                pred[n] = host[o:o + sz]; o += sz
            del out

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
        ts = [(t, k) for t, k in ts if k]
        tens = {t.name: t for t, _ in ts}
        regions = {t.name: view[t.offset:t.offset + t.nbytes] for t, _ in ts}
        # QR tensors (qr.py) are not XOR-ed: their file bytes and prediction go to the QR coder
        isqr = {t.name for t, k in ts if k == "rq" and self._qr_ok(t)}
        nonzero, p0 = {}, {}
        # without the tensor's importance matrix (quants packed without imatrix file), unit
        # weights first: the quantisers' weighted path, the one imatrix quants were made with
        v0 = {t.name: UNIT_W if k == "rq" and self.imatrix.get(t.name) is None else 0 for t, k in ts}

        def cands(t):
            """the other predictions worth trying when v0's leaves a residual: the quantiser's code
            variants and, for unit weights, the unweighted path (IQ types have none)"""
            nv = len(VARIANTS.get(t.type, ()))
            out = [v0[t.name] + j for j in range(1, nv + 1)]
            if v0[t.name] == UNIT_W and t.type < 16:
                out += list(range(nv + 1))
            return out

        def xor0(key, p):
            if key[0] in isqr:
                nonzero[key[0]] = _diff_async(regions[key[0]], p)
                p0[key[0]] = p
                return
            nonzero[key[0]] = _xor_async(regions[key[0]], p)
            if cands(tens[key[0]]):
                p0[key[0]] = p                  # kept to try the alternatives below

        self._predict([(t, v0[t.name]) for t, _ in ts], xor0)
        nonzero = {n: _nonzero(f) for n, f in nonzero.items()}
        p0 = {n: p for n, p in p0.items() if nonzero[n] or n in isqr}
        choice = dict(v0)
        retry = [t for t, k in ts if k == "rq" and nonzero[t.name] and cands(t)]
        if retry:
            # XOR tensors: the region now holds file ^ p0, and prediction pv leaves
            # file ^ pv = region ^ p0 ^ pv; QR tensors: the region is the file, pv is kept
            alts = {t.name: [(v0[t.name], None if t.name in isqr else regions[t.name],
                              int(np.count_nonzero(regions[t.name] != p0[t.name])) if t.name in isqr else -1)]
                    for t in retry}

            def alt(key, p):
                n = key[0]
                if n in isqr:
                    alts[n].append((key[2], p, int(np.count_nonzero(regions[n] != p))))
                else:
                    alts[n].append((key[2], regions[n] ^ p0[n] ^ p, -1))

            for v in sorted({c for t in retry for c in cands(t)}):
                self._predict([(t, v) for t in retry if v in cands(t)], alt)
            for t in retry:
                a = alts[t.name]
                errs = [e if e >= 0 else int(np.count_nonzero(r)) for _, r, e in a]
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
        self._qr_chunk = self._qr_encode([(tens[n], regions[n], p0[n]) for n in isqr]) if isqr else None
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
        plan += [[t.name, "dup", refs[t.name], t.type] for t in qg.tensors if t.name in refs]
        return plan, self.b

    # ------------------------------------------------------------------ files without a base
    Q8_MIN = 0.5                              # share of Q8_0 bytes from which M21 is tried

    def _q8_codec(self):
        if getattr(self, "_q8", None) is None:
            from .gpu3 import Q8Parts
            self._q8 = Q8Parts(int(os.environ.get("VARC_Q8_PART_MB", 0)) << 20 or None)
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
            ch = unpack(buf)
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

    def compress(self, bases: dict, files: dict, extras: dict, put=None, on_data=None):
        """bases: float GGUFs (BF16 model, mmproj, ...) name -> path; files: GGUFs to derive from
        them, name -> path; extras: any other files, name -> path.
        put(chunk name, bytes) receives the archive's chunks as they are produced; without it they
        are collected and the packed archive is returned. on_data(name, bytes) sees every input
        file's original content once (e.g. to hash it)."""
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
            data = _read_file(files[name])   # the buffer that is XOR-ed in place
            seen(name, data)
            return data

        names = list(files)
        fsize = [os.path.getsize(files[n]) for n in names]
        rd = ThreadPoolExecutor(1)
        # the first derived file is read while the bases are coded
        reads = {0: rd.submit(read, names[0])} if names else {}
        kept = []
        # the bases are coded in the background: the derived files only need their GPU copy
        bp = ThreadPoolExecutor(1)
        base_jobs = []

        def code_base(name, data):
            if name in self.seed_files:
                c = SeedCodec().compress(data)
            else:
                c = pack(self.base_codec.compress_dev(data))
            if self.verbose:
                kind = "seed" if name in self.seed_files else "base"
                print(f"  {name}: {len(data):,} -> {len(c):,} B ({kind}){_mem()}", flush=True)
            return c

        for i, (name, path) in enumerate(bases.items()):
            # a base the GPU has no room for is read through the file's mapping: the per-tensor
            # uploads then come from the page cache, at no commit charge
            data = _read_file(path) if _fits_gpu(os.path.getsize(path)) else _map_file(path)
            seen(name, data)
            kept.append(_keep_base(data))
            base_jobs.append(bp.submit(code_base, name, data))
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

        manifest = {"method": self.name, "bases": list(bases), "seeds": seeds, "files": [],
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
                    put(f"b{i}", c)
                    self.sizes[name] = len(c)
                    del c

        # pipelined: the next file is read and older residuals are compressed while the GPU
        # predicts the current one, as long as the files held stay within self.inflight bytes
        with ThreadPoolExecutor(3) as zp:
            for i, name in enumerate(names):
                data = reads.pop(i).result()
                if i + 1 < len(names):
                    # the next file is read while this one is predicted and XOR-ed
                    drain(max(0, self.inflight - fsize[i] - fsize[i + 1]))
                    reads[i + 1] = rd.submit(read, names[i + 1])
                plan, b = self._residual(data, fi=len(manifest["files"]))
                if self._qr_chunk is not None:
                    put(f"q{len(manifest['files'])}", self._qr_chunk)
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
                manifest["files"].append(entry)
                pending.append((i, job, len(data)))
                del data, c
                put_bases()
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

    # ------------------------------------------------------------------ decompression
    @staticmethod
    def manifest_of(get):
        return json.loads(zstd_d(get("manifest")))

    def decompress(self, src, sink, select=None):
        """src: the packed archive (bytes) or a chunk getter get(name) -> bytes.
        sink(name, bytes-like) receives every file, or only those in `select` (a set of names);
        only the bases those files need are decoded."""
        get = src if callable(src) else unpack(src).__getitem__
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
        # the residuals' zstd dictionaries are the bases' headers, decoded first (cheap), so the
        # residuals decompress on the CPU while the bases decode on the GPU
        base_buf = {i: get(f"b{i}") for i in sorted(need)}
        zdicts = {}
        for i in need:
            h = self._base_header(base_buf[i], manifest["bases"][i] in seeds)
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
        if len(zdicts) == len({m["base"] for _, m in files if m["base"] is not None}):
            while ahead < len(files) and ahead < 2 and sum(
                    files[q][1]["size"] for q in range(0, ahead + 1)) <= self.inflight:
                nxt[ahead] = zp.submit(zd, files[ahead][0])
                ahead += 1
        for i, name in enumerate(manifest["bases"]):
            if i not in need:
                bases.append(None)
                continue
            predicts = any(m["base"] == i for _, m in files)
            buf = base_buf.pop(i)
            if name in seeds:
                data = SeedCodec().decompress(buf)
                kept = _keep_base(data) if predicts else None
            elif hasattr(self.base_codec, "decompress_host"):
                data = self.base_codec.decompress_host(unpack(buf))
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

    def compress_dev(self, data, dev=None):
        return {"r2": self.c.compress(data)}

    def decompress_dev(self, ch):
        return cp.asarray(self.c.decompress(ch["r2"]))

    def decompress_host(self, ch):
        return self.c.decompress(ch["r2"])


class M29RepoGPU(M28RepoGPU):
    name = "M29 model repo: M28 with the BF16 v2 base codec (R2)"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.base_codec = R2Base()
