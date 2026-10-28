"""Method 30: sibling quantisations of one model, without the model itself.

Some repositories ship several sizes of one model but not its BF16 weights: ISTA-DASLab's GSQ-RCO
GGUFs, for instance, quantise every tensor separately for each size, with types chosen per tensor.
There every weight tensor is there several times, in different types or in the same type with
different codes. M30 codes the archive tensor by tensor instead of file by file:

  * identical files, and identical tensors in different files, are stored once;
  * the versions of a tensor (same name and shape) are coded in a chain, most precise (most bytes)
    first. The first is coded on its own: QR without a seed (qr.self_encode) where its type has
    one, else zstd. Every later version is coded with QR, its codes predicted from the
    precision-weighted mean of the versions before it (each version's value of a weight and half
    its code step: x = sum(x_i / h_i^2) / sum(1 / h_i^2), h = sum(1 / h_i^2)^-1/2), its scales from
    llama.cpp's quantiser run on that mean;
  * tensors are coded in row slices of at most SLICE weights, so a 28.8 GB lookup table needs no
    more memory than any other tensor;
  * GGUF headers and padding go into one zstd chunk, small tensors into another, other files
    into the LZMA bundle as in M28.

The decoder writes all the files it is asked for at once, each in order, tensor by tensor, slice by
slice; a version is decoded with the versions before it in its chain (the same slice of each), and
a chain once for all the files that hold its versions: the sizes of one model keep their tensors in
the same order, and a version needed by a file further behind waits for it in host memory. Everything the predictions depend
on runs in kernels compiled without FMA contraction, so any GPU decodes what another coded.
"""
import json
import lzma
import os
import time
from concurrent.futures import ThreadPoolExecutor

import blake3
from .xp import cp
import numpy as np
import zstandard

from . import memory, qr
from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gguf_reader import BLOCK, GGUF
from .gpuquant import GPUQuant
from .repo3 import _read_range, _mem
from .requant import QFUN

F32, F16, BF16 = 0, 1, 30
FLOATS = {F32: np.float32, F16: np.float16}
SLICE = 1 << 26                          # weights per coding slice (its seed estimate: 512 MB on the GPU)
SMALL = 1 << 20                          # versions smaller than this go into the "small" chunk, raw
NEED_IMX = {16, 17, 19}                  # llama.cpp's quantisers that assert an importance matrix
_PIECE = 64 << 20

_EST = r"""
extern "C" __global__ void half_step(const float* x, const float* lo, const float* hi, long long n, float* h) {
    // half the interval a weight's code stands for; one infinite side: the finite one; none: -1 (no information)
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float l = lo[i], u = hi[i], v = x[i];
    bool fl = isfinite(l), fu = isfinite(u);
    h[i] = fl && fu ? __fmul_rn(0.5f, __fsub_rn(u, l)) : fu ? __fsub_rn(u, v) : fl ? __fsub_rn(v, l) : -1.f;
}
extern "C" __global__ void est_add(const float* x, const float* h, long long n, float* ws, float* wx) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float hv = h[i];
    if (hv < 0.f) return;
    float w = __fdiv_rn(1.f, fmaxf(__fmul_rn(hv, hv), 1e-30f));
    ws[i] = __fadd_rn(ws[i], w);
    wx[i] = __fadd_rn(wx[i], __fmul_rn(x[i], w));
}
extern "C" __global__ void est_seed(const float* ws, const float* wx, const float* x0, long long n, float* out) {
    // explicit QR seed (x, h); a weight no version told anything about: the first version's value, h 0
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = ws[i];
    if (s > 0.f) { out[2 * i] = __fdiv_rn(wx[i], s); out[2 * i + 1] = __fdiv_rn(1.f, __fsqrt_rn(s)); }
    else { out[2 * i] = x0[i]; out[2 * i + 1] = 0.f; }
}
"""
_mod = None


def _k(name):
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=_EST, options=("--fmad=false",))
    return _mod.get_function(name)


def _launch(name, n, *args):
    _k(name)(((n + 255) // 256,), (256,), args)


def _row_bytes(T, cols):
    be, bb = BLOCK[T]
    return cols // be * bb


class _Est:
    """the precision-weighted mean of a slice's versions so far (device float32)"""

    def __init__(self, gq, nw):
        self.gq, self.nw = gq, nw
        self.ws = cp.zeros(nw, cp.float32)
        self.wx = cp.zeros(nw, cp.float32)
        self.x0 = None

    def values(self, T, blocks, cols):
        """a version's (x, h) per weight, or None when its type gives none"""
        if T in FLOATS:
            x = cp.asarray(blocks).view(FLOATS[T]).astype(cp.float32)
            return x, cp.zeros(self.nw, cp.float32)
        if T == BF16:
            x = (cp.asarray(blocks).view(cp.uint16).astype(cp.uint32) << 16).view(cp.float32)
            return x, cp.zeros(self.nw, cp.float32)
        if T not in QFUN:
            return None
        x = self.gq.dequantize(T, blocks, cols)
        if T in qr.GRID_TYPES:
            h = qr.explicit_seed(T, blocks, x).reshape(-1, 2)[:, 1].copy()
        elif T in qr.QR_TYPES:
            lo, hi = qr.intervals(T, blocks, self.nw)
            h = cp.empty(self.nw, cp.float32)
            _launch("half_step", self.nw, x, lo, hi, np.int64(self.nw), h)
            del lo, hi
        else:
            return None
        return x, h

    def add(self, T, blocks, cols):
        v = self.values(T, blocks, cols)
        if v is None:
            return False
        x, h = v
        _launch("est_add", self.nw, x, h, np.int64(self.nw), self.ws, self.wx)
        if self.x0 is None:
            self.x0 = x
        return True

    def seed(self):
        out = cp.empty(2 * self.nw, cp.float32)
        _launch("est_seed", self.nw, self.ws, self.wx, self.x0, np.int64(self.nw), out)
        return out


def _predicted(gq, T, x, cols):
    """llama.cpp's quantisation of the estimate: the prediction of a version's blocks (its scales)"""
    imx = [np.ones(cols, np.float32)] if T in NEED_IMX else [None]
    return gq.quantize_batch(T, [x.reshape(-1, cols)], imx)[0]


def _throttled(put, limit=48):
    """put(name, data) that waits while the archive writer (container.Writer) has more than `limit`
    chunks queued: the coder can outrun the disk, and the queued chunks are held in memory"""
    w = getattr(put, "__self__", None)
    if w is None or not hasattr(w, "jobs"):
        return put

    def put2(name, data):
        put(name, data)
        while len(w.jobs) > limit:
            w.jobs.popleft().result()
    return put2


def _codable(T, cols):
    return T in qr.QR_TYPES and T in QFUN and cols % qr.QK[T] == 0


class M30Siblings:
    name = "M30 sibling quants: tensor versions coded in chains, no base"

    def __init__(self, level=7, cache_bytes=None):
        self.level = level
        self.cache_bytes = cache_bytes           # unpacking: versions kept for files further behind
        self.verbose = True
        self.sizes, self.q8_files = {}, set()
        self.scratch = None
        self._gq = None
        self.slice = int(os.environ.get("SEEDPACK_M30_SLICE", SLICE))     # smaller: tests

    @property
    def gq(self):
        if self._gq is None:
            self._gq = GPUQuant()
        return self._gq

    # ------------------------------------------------------------------ scan
    @staticmethod
    def _header(path):
        n = 16 << 20
        size = os.path.getsize(path)
        while True:
            buf = _read_range(path, 0, min(n, size))
            try:
                return GGUF(data=memoryview(buf), light=True)
            except Exception:
                if n >= size:
                    raise
                n *= 4

    def _scan(self, names, paths, on_stream):
        """read every GGUF once: the hashes the catalog needs (on_stream), each file's blake3 and
        each tensor's -> per file {path, header bytes, gaps, tensors}"""
        phys = {}                              # (device, inode, size) -> names (hardlinks are read once)
        for n in names:
            st = os.stat(paths[n])
            key = (st.st_dev, st.st_ino, st.st_size) if st.st_ino else (n,)
            phys.setdefault(key, []).append(n)
        info = {}
        rd = ThreadPoolExecutor(1)
        for key, group in phys.items():
            path = paths[group[0]]
            g = self._header(path)
            size = os.path.getsize(path)
            ts = sorted(g.tensors, key=lambda t: t.offset)
            ups = [on_stream(n) for n in group] if on_stream else []
            fh = blake3.blake3(max_threads=blake3.blake3.AUTO)
            th = [None] * len(ts)
            spans = [(t.offset, t.offset + t.nbytes) for t in ts]
            gaps, head = [], None
            cur, ti = 0, 0                     # file position, current tensor
            nxt = rd.submit(_read_range, path, 0, min(_PIECE, size), None, True)
            pos = 0
            while pos < size:
                buf = nxt.result()
                end = pos + len(buf)
                if end < size:
                    nxt = rd.submit(_read_range, path, end, min(_PIECE, size - end), None, True)
                for u in ups:
                    u(buf)
                fh.update(buf)
                if head is None:
                    head = bytes(buf[:min(g.data_start, len(buf))])
                    assert len(head) == g.data_start, "GGUF header longer than one read"
                    cur = g.data_start
                # walk tensors and the gaps between them within [pos, end)
                while cur < end:
                    if ti < len(ts) and cur >= spans[ti][0]:
                        a, b = spans[ti]
                        if th[ti] is None:
                            th[ti] = blake3.blake3(max_threads=blake3.blake3.AUTO) if b - a > (8 << 20) else blake3.blake3()
                        e = min(b, end)
                        th[ti].update(buf[cur - pos:e - pos])
                        cur = e
                        if cur == b:
                            th[ti] = th[ti].hexdigest()
                            ti += 1
                    else:
                        stop = min(end, spans[ti][0] if ti < len(ts) else size)
                        if gaps and gaps[-1][0] + len(gaps[-1][1]) == cur:
                            gaps[-1][1] += bytes(buf[cur - pos:stop - pos])
                        else:
                            gaps.append([cur, bytearray(buf[cur - pos:stop - pos])])
                        cur = stop
                pos = end
            assert ti == len(ts), f"{path}: tensors past the end of the file"
            fhash = fh.hexdigest()
            for n in group:
                info[n] = {"path": path, "size": size, "hash": fhash, "head": head,
                           "gaps": [(o, bytes(b)) for o, b in gaps],
                           "tensors": [(t, h) for t, h in zip(ts, th)]}
            if self.verbose:
                print(f"  scanned {', '.join(group)} ({size / 1e9:.2f} GB){_mem()}", flush=True)
        rd.shutdown()
        return info

    # ------------------------------------------------------------------ pack
    def compress(self, bases, files, extras, put=None, on_data=None, on_stream=None, put_stream=None):
        t0 = time.perf_counter()
        put = _throttled(put)
        paths = {**bases, **files}
        names = sorted(paths)
        self._extras(extras, put, on_data)
        info = self._scan(names, paths, on_stream)
        # identical files: the first of them is coded, the others point at it
        first_of, fman = {}, []
        for i, n in enumerate(names):
            f = info[n]
            same = first_of.setdefault((f["hash"], f["size"]), i)
            fman.append({"name": n, "size": f["size"], "same": None if same == i else same})
        uniq = [i for i, m in enumerate(fman) if m["same"] is None]
        # tensor groups: same name and shape; versions: distinct (type, bytes)
        groups, gid_of = [], {}
        for i in uniq:
            for t, h in info[names[i]]["tensors"]:
                key = (t.name, tuple(t.shape))
                if key not in gid_of:
                    gid_of[key] = len(groups)
                    groups.append({"name": t.name, "shape": list(t.shape), "cols": int(t.shape[0]), "vers": [], "at": []})
                g = groups[gid_of[key]]
                vk = (t.type, h)
                if vk not in [(v["type"], v["hash"]) for v in g["vers"]]:
                    g["vers"].append({"type": t.type, "hash": h, "nbytes": t.nbytes})
                    g["at"].append((i, t.offset))
        for g in groups:                       # chain order: most bytes first (ties: first seen)
            order = sorted(range(len(g["vers"])), key=lambda k: -g["vers"][k]["nbytes"])
            g["vers"] = [g["vers"][k] for k in order]
            g["at"] = [g["at"][k] for k in order]
        heads = bytearray()
        for i in uniq:
            f = info[names[i]]
            fman[i]["head"] = [len(heads), len(f["head"])]
            heads += f["head"]
            fman[i]["gaps"] = []
            for o, b in f["gaps"]:
                fman[i]["gaps"].append([o, len(heads), len(b)])
                heads += b
            vidx = {}
            for t, h in f["tensors"]:
                g = gid_of[(t.name, tuple(t.shape))]
                k = [(v["type"], v["hash"]) for v in groups[g]["vers"]].index((t.type, h))
                vidx[t.offset] = [g, k, t.offset, t.nbytes]
            fman[i]["tensors"] = [vidx[o] for o in sorted(vidx)]
        z = zstandard.ZstdCompressor(compression_params=zstandard.ZstdCompressionParameters.from_level(
            19, window_log=27, enable_ldm=True, threads=8))
        put("heads", z.compress(bytes(heads)))
        self.sizes["headers"] = len(heads)
        nv = sum(len(g["vers"]) for g in groups)
        total = sum(v["nbytes"] for g in groups for v in g["vers"])
        if self.verbose:
            print(f"  {len(uniq)} distinct files of {len(names)}, {len(groups)} tensors in {nv} distinct versions, "
                  f"{total / 1e9:.2f} GB{_mem()}", flush=True)
        small = bytearray()
        stored = {}
        for gi, g in enumerate(groups):
            self._code_group(gi, g, names, info, put, small, stored)
            if self.verbose and (gi % 50 == 0 or gi == len(groups) - 1):
                done = sum(v["nbytes"] for gg in groups[:gi + 1] for v in gg["vers"])
                cod = sum(stored.values())
                print(f"  {gi + 1}/{len(groups)} tensors, {done / 1e9:.2f} of {total / 1e9:.2f} GB -> {cod / 1e9:.2f} GB "
                      f"({time.perf_counter() - t0:.0f} s){_mem()}", flush=True)
        put("small", zstd_c(bytes(small), 19))
        for g in groups:
            del g["at"]
            for v in g["vers"]:
                del v["hash"]
        # stored bytes per file: each version counted for the first file that holds it
        seen = set()
        for i in uniq:
            s = 0
            for g, k, _, _ in fman[i]["tensors"]:
                if (g, k) not in seen:
                    seen.add((g, k))
                    s += stored.get((g, k), 0)
            self.sizes[names[i]] = s
        for i, m in enumerate(fman):
            if m["same"] is not None:
                self.sizes[names[i]] = 0
        manifest = {"method": self.name, "needs": "0.4.0", "slice": self.slice, "files": fman, "groups": groups,
                    "extras": self._exm}
        put("manifest", zstd_c(json.dumps(manifest).encode(), 19))

    def _extras(self, extras, put, on_data):
        small = {n: p for n, p in extras.items() if os.path.getsize(p) <= (64 << 20)}
        ex = {}
        for n, p in small.items():
            with open(p, "rb") as f:
                ex[n] = f.read()
            if on_data:
                on_data(n, ex[n])
        c = lzma.compress(pack(ex), preset=9)
        put("extras", c)
        self.sizes["other files"] = len(c)
        big = {}
        zp = zstandard.ZstdCompressionParameters.from_level(self.level, threads=8, enable_ldm=True)
        for j, (n, p) in enumerate((n, p) for n, p in extras.items() if n not in small):
            with open(p, "rb") as f:
                data = f.read()
            if on_data:
                on_data(n, data)
            c = zstandard.ZstdCompressor(compression_params=zp).compress(data)
            put(f"x{j}", c)
            self.sizes[n] = len(c)
            big[n] = [f"x{j}", len(data)]
        self._exm = {"bundle": list(ex), "single": big}

    def _slices(self, g):
        rows = int(np.prod(g["shape"])) // g["cols"]
        per = max(1, self.slice // g["cols"])
        return [(r, min(per, rows - r)) for r in range(0, rows, per)]

    def _code_group(self, gi, g, names, info, put, small, stored):
        cols = g["cols"]
        vers = g["vers"]
        for k, v in enumerate(vers):
            v["kinds"] = []
        if all(v["nbytes"] < SMALL for v in vers):
            for k, v in enumerate(vers):
                i, off = g["at"][k]
                b = _read_range(info[names[i]]["path"], off, v["nbytes"])
                v["small"] = [len(small), v["nbytes"]]
                small += b.tobytes()
                del v["kinds"]
            return
        pool = ThreadPoolExecutor(len(vers))
        sl = self._slices(g)

        def read(j):
            r0, nr = sl[j]
            return [pool.submit(_read_range, info[names[g["at"][k][0]]]["path"],
                                g["at"][k][1] + r0 * _row_bytes(v["type"], cols), nr * _row_bytes(v["type"], cols))
                    for k, v in enumerate(vers)]

        nxt = read(0)
        for j, (r0, nr) in enumerate(sl):
            cur = [f.result() for f in nxt]
            if j + 1 < len(sl):
                nxt = read(j + 1)
            nw = nr * cols
            est = _Est(self.gq, nw) if len(vers) > 1 else None
            for k, v in enumerate(vers):
                T, raw = v["type"], cur[k]
                name = f"g{gi}.{k}.{j}"
                if k == 0:
                    kind, data = self._first(T, raw, cols)
                else:
                    kind, data = "z", None
                    if est is not None and est.x0 is not None and _codable(T, cols):
                        dev = cp.asarray(raw)
                        seed = est.seed()
                        P = _predicted(self.gq, T, seed.reshape(-1, 2)[:, 0].copy(), cols)
                        data = qr.encode(T, [(dev, P, 0, qr.EXPLICIT, cols)], seed)
                        kind = "qr"
                        del dev, seed, P
                        if len(data) >= len(raw):
                            kind, data = "z", None
                    if data is None:
                        data = zstd_c(raw.tobytes(), 3)
                put(name, data)
                stored[(gi, k)] = stored.get((gi, k), 0) + len(data)
                v["kinds"].append(kind)
                if est is not None and k + 1 < len(vers):
                    est.add(T, cp.asarray(raw), cols)
            del est, cur
            cp.get_default_memory_pool().free_all_blocks()
        pool.shutdown()

    def _first(self, T, raw, cols):
        # QR without a seed: the grid types too (their points' frequencies, ~2%)
        if T in qr.QR_TYPES and cols % qr.QK[T] == 0:
            c = qr.self_encode(T, [(cp.asarray(raw), cols)])
            if len(c) < len(raw):
                return "self", c
        c = zstd_c(raw.tobytes(), 3)
        return ("z", c) if len(c) < len(raw) else ("raw", raw.tobytes())

    # ------------------------------------------------------------------ unpack
    @staticmethod
    def manifest_of(get):
        return json.loads(zstd_d(get("manifest")))

    def decompress(self, src, sink, select=None):
        reader = src if hasattr(src, "read_at") else None
        get = reader.get if reader is not None else src if callable(src) else unpack(src).__getitem__
        man = self.manifest_of(get)
        self.slice = man["slice"]
        want = (lambda n: True) if select is None else (lambda n: n in select)
        exm = man["extras"]
        if any(want(n) for n in exm["bundle"]):
            for n, b in unpack(lzma.decompress(get("extras"))).items():
                if want(n):
                    sink(n, b)
        for n, (c, size) in exm["single"].items():
            if want(n):
                sink(n, zstandard.ZstdDecompressor().decompress(get(c), max_output_size=size))
        fm = man["files"]
        targets = {}                           # coded file -> selected files it gives (itself, copies)
        for i, m in enumerate(fm):
            if want(m["name"]):
                targets.setdefault(i if m["same"] is None else m["same"], []).append(i)
        if not targets:
            return
        heads = zstd_d(get("heads"))
        smallb = zstd_d(get("small")) if any("small" in v for g in man["groups"] for v in g["vers"]) else b""
        groups = man["groups"]
        # all selected files are written at once, tensor by tensor: a chain is decoded once for every
        # file that holds one of its versions (the sizes of one model keep their tensors in the same
        # order); a version a file further behind needs is kept for it, within the cache budget
        files = [_Cursor(fm[i], [_Out(sink, fm[t]["name"], fm[i]["size"]) for t in targets[i]], heads)
                 for i in sorted(targets)]
        need = {}                              # group -> {version: files that still need it}
        for f in files:
            for gi, k, _, _ in f.m["tensors"]:
                need.setdefault(gi, {})
                need[gi][k] = need[gi].get(k, 0) + 1
        cache = _Cache(need, self.cache_bytes if self.cache_bytes is not None else memory.budget(16 << 30))
        for f in files:
            f.flush(cache)
        while True:
            live = [f for f in files if f.head() is not None]
            if not live:
                break
            # the group to decode: one at the head of some file; the fewest files needing it later first
            at = {}
            for f in live:
                at[f.head()[0]] = at.get(f.head()[0], 0) + 1
            gi = min(at, key=lambda c: (sum(need[c].values()) - at[c], -at[c]))
            now = [f for f in live if f.head()[0] == gi]
            ks = sorted(k for k, n in need[gi].items() if n > 0)
            keep = {k: [] for k in ks
                    if (gi, k) not in cache.v and need[gi][k] > sum(f.head()[1] == k for f in now)}
            for f in now:
                f.begin()
            for k, piece in self._chain(get, groups[gi], gi, ks, smallb):
                for f in now:
                    if f.head()[1] == k:
                        f.emit(piece)
                if k in keep and not cache.hold(keep[k], piece):
                    cache.drop(keep.pop(k))        # over budget: decoded again when it is needed
            for f in now:
                cache.done(f.head())
                f.end()
            for k, pieces in keep.items():
                cache.v[gi, k] = pieces
            for f in files:
                f.flush(cache)

    def _chain(self, get, g, gi, ks, smallb):
        """versions ks (ascending) of group g, slice by slice: (k, bytes of that slice of version k);
        the versions before each are decoded once per slice, for all of them"""
        vers = g["vers"]
        if "small" in vers[0]:
            for k in ks:
                a, n = vers[k]["small"]
                yield k, np.frombuffer(smallb, np.uint8, n, a)
            return
        cols, top, want = g["cols"], ks[-1], set(ks)
        for j, (r0, nr) in enumerate(self._slices(g)):
            nw = nr * cols
            est = _Est(self.gq, nw) if top > 0 else None
            for kk in range(top + 1):
                vv = vers[kk]
                T = vv["type"]
                kind = vv["kinds"][j]
                buf = get(f"g{gi}.{kk}.{j}")
                n = nr * _row_bytes(T, cols)
                if kind == "self":
                    dev = qr.self_decode(T, buf)[0]
                elif kind == "qr":
                    seed = est.seed()
                    P = _predicted(self.gq, T, seed.reshape(-1, 2)[:, 0].copy(), cols)
                    dev = qr.decode(T, buf, [(P, 0, qr.EXPLICIT, cols)], seed)
                    del seed, P
                elif kind == "z":
                    dev = np.frombuffer(zstandard.ZstdDecompressor().decompress(buf, max_output_size=n), np.uint8)
                else:
                    dev = np.frombuffer(buf, np.uint8)
                if kk in want:
                    yield kk, dev.get() if isinstance(dev, cp.ndarray) else dev
                if kk < top:
                    est.add(T, cp.asarray(dev), cols)
                del dev
            del est
            cp.get_default_memory_pool().free_all_blocks()


class _Cursor:
    """one coded file being written (to itself and its copies): header, then tensor by tensor with
    the padding between them"""

    def __init__(self, m, outs, heads):
        self.m, self.outs, self.heads = m, outs, heads
        self.gaps = {o: (s, ln) for o, s, ln in m["gaps"]}
        self.i = 0
        h0, hn = m["head"]
        self.emit(heads[h0:h0 + hn])
        self.pos = hn
        if not m["tensors"]:
            self._close()

    def emit(self, b):
        for o in self.outs:
            o.write(b)

    def head(self):
        """(group, version) of the next tensor, None when the file is done"""
        t = self.m["tensors"]
        return (t[self.i][0], t[self.i][1]) if self.i < len(t) else None

    def _pad(self, upto):
        while self.pos < upto:
            s, ln = self.gaps[self.pos]
            self.emit(self.heads[s:s + ln])
            self.pos += ln

    def begin(self):
        self._pad(self.m["tensors"][self.i][2])

    def end(self):
        _, _, off, nb = self.m["tensors"][self.i]
        self.pos = off + nb
        self.i += 1
        if self.i == len(self.m["tensors"]):
            self._close()

    def _close(self):
        self._pad(self.m["size"])
        for o in self.outs:
            o.close()

    def flush(self, cache):
        """write the tensors at the head that were decoded earlier for another file"""
        while (h := self.head()) is not None and h in cache.v:
            self.begin()
            for piece in cache.v[h]:
                self.emit(piece)
            cache.done(h)
            self.end()


class _Cache:
    """versions decoded for one file and kept for the files further behind, in host memory"""

    def __init__(self, need, cap):
        self.need, self.cap = need, cap
        self.v, self.bytes = {}, 0

    def hold(self, pieces, piece):
        if self.bytes + piece.nbytes > self.cap or not memory.fits(piece.nbytes):
            return False
        pieces.append(piece)
        self.bytes += piece.nbytes
        return True

    def drop(self, pieces):
        self.bytes -= sum(p.nbytes for p in pieces)

    def done(self, h):
        """one file has written version h"""
        gi, k = h
        self.need[gi][k] -= 1
        if self.need[gi][k] == 0 and h in self.v:
            self.drop(self.v.pop(h))


class _Out:
    """one output file: streamed to the sink when it takes streams, else collected"""

    def __init__(self, sink, name, size):
        self.sink, self.name = sink, name
        self.s = sink.stream(name, size) if hasattr(sink, "stream") else None
        self.parts = [] if self.s is None else None

    def write(self, b):
        if self.s is not None:
            self.s.write(b)
        else:
            self.parts.append(bytes(b))

    def close(self):
        if self.s is not None:
            self.s.close()
        else:
            self.sink(self.name, b"".join(self.parts))
