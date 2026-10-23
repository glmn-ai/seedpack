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
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import zstandard

from codecs_gguf import pack, unpack, zstd_c, zstd_d
from gguf_reader import GGUF
from gpu_bf16 import M24BF16GPU
from gpuquant import GPUQuant, VARIANTS
from repo2 import F32, F16, BF16, base_float_np, float_as_np
from requant import QFUN, load_imatrix


_POOL = ThreadPoolExecutor(8)


def _mem():
    """process commit charge, when VARC_MEMDEBUG is set (debugging aid)"""
    if not os.environ.get("VARC_MEMDEBUG"):
        return ""
    import psutil
    free, total = cp.cuda.runtime.memGetInfo()
    return (f"  [private {psutil.Process().memory_info().private / 1e9:.1f} GB, GPU used {(total - free) / 1e9:.1f} GB,"
            f" pool {cp.get_default_memory_pool().used_bytes() / 1e9:.1f}/{cp.get_default_memory_pool().total_bytes() / 1e9:.1f} GB]")


def _keep_base(data, chunk=256 << 20):
    """(header bytes, file on the GPU): uploaded in chunks, so no whole-file staging copy"""
    dev = cp.empty(len(data), cp.uint8)
    host = np.frombuffer(data, np.uint8)
    for o in range(0, len(data), chunk):
        dev[o:o + chunk] = cp.asarray(host[o:o + chunk])
    mv = memoryview(data).cast("B")
    hdr = GGUF.from_bytes(mv, light=True).data_start
    return bytes(mv[:hdr]), dev


def _release():
    """return CuPy's cached device and pinned blocks: on Windows (WDDM) they count against the
    process commit charge, which otherwise grows with every file until allocations fail"""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
_CHUNK = 8 << 20


def _par_xor(pairs):
    """region ^= prediction for every (region, prediction), in parallel chunks (numpy releases
    the GIL) -> per pair: whether the residual has any non-zero byte"""
    jobs = []
    for i, (r, p) in enumerate(pairs):
        for o in range(0, max(len(r), 1), _CHUNK):
            jobs.append((i, r[o:o + _CHUNK], p[o:o + _CHUNK]))

    def run(j):
        i, r, p = j
        np.bitwise_xor(r, p, out=r)
        return i, bool(r.any())

    nz = [False] * len(pairs)
    for i, a in _POOL.map(run, jobs):
        nz[i] |= a
    return nz


class M28RepoGPU:
    name = "M28 model repo: BF16 + quants re-derived on the GPU with llama.cpp quantisers + imatrix"

    def __init__(self, level=9, cache_bytes=8 << 30, inflight=3 << 30):
        self.level = level
        self.inflight = inflight
        self.cache_bytes = cache_bytes
        self.base_codec = M24BF16GPU()
        self.gq = None
        self.sizes = {}
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

    def _src(self, bt):
        a = self.base_dev[bt.offset:bt.offset + bt.nbytes]
        return a.view(cp.uint16) if bt.type == BF16 else a.view(cp.float16) if bt.type == F16 else a.view(cp.float32)

    def _float_pred(self, bt, ttype):
        """base tensor bt converted to float type ttype on the GPU (same rounding as float_as_np)"""
        a = self._src(bt)
        x = (a.astype(cp.uint32) << 16).view(cp.float32) if bt.type == BF16 else a.astype(cp.float32)
        if ttype == F32:
            y = x.view(cp.uint8)
        elif ttype == F16:
            y = x.astype(cp.float16).view(cp.uint8)
        else:
            u = x.view(cp.uint32)
            y = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(cp.uint16).view(cp.uint8)
        return cp.asnumpy(y)

    def _kind(self, t):
        """how tensor t of a quantised file is predicted: 'rq', 'xor' or None"""
        bt = self.by_name.get(t.name)
        if bt is None or bt.type not in (F32, F16, BF16) or int(np.prod(bt.shape)) != int(np.prod(t.shape)):
            return None
        if t.type in (F32, F16, BF16):
            return "xor"
        if t.type in QFUN and t.nbytes == int(np.prod(t.shape)) // QFUN[t.type][1] * QFUN[t.type][2]:
            if t.type >= 16 and self.imatrix.get(t.name) is None:
                return None                       # IQ quantisers need an importance matrix
            return "rq"
        return None

    def _predict(self, wanted, use):
        """wanted: list of (tensor, variant); use((name, type, variant), bytes as numpy uint8) is
        called as soon as each prediction is ready, so a file's predictions are never all held"""
        todo = {}
        for t, v in wanted:
            key = (t.name, t.type, v)
            if (self.b,) + key in self.cache:
                use(key, self.cache[(self.b,) + key])
            elif t.type in (F32, F16, BF16):
                use(key, self._float_pred(self.by_name[t.name], t.type))
            else:
                todo.setdefault((t.type, v), {})[key] = t
        for (ty, v), ts in todo.items():
            keys = list(ts)

            def got(i, o):
                if self.cached + o.nbytes <= self.cache_bytes:
                    # a copy: o is a view of the whole batch output, which it would keep alive
                    self.cache[(self.b,) + keys[i]] = o.copy(); self.cached += o.nbytes
                use(keys[i], o)

            srcs = [(self._src(self.by_name[n]), ts[(n, ty, v)].shape[0]) for n, _, _ in keys]
            self.gq.quantize_tensors(ty, srcs, [self.imatrix.get(n) for n, _, _ in keys], v, on_result=got)

    # ------------------------------------------------------------------ compression
    def _residual(self, data):
        """XOR the predictions into the file in place -> (plan [[name, kind, variant], ...], base index)"""
        view = np.frombuffer(data, np.uint8)
        qg = GGUF.from_bytes(data, light=True)
        self._use(self._best_base(qg))
        if self.b is None:
            self.stats["raw"] += len(qg.tensors)
            return [], None
        ts = [(t, self._kind(t)) for t in qg.tensors]
        self.stats["raw"] += sum(k is None for _, k in ts)
        ts = [(t, k) for t, k in ts if k]
        tens = {t.name: t for t, _ in ts}
        regions = {t.name: view[t.offset:t.offset + t.nbytes] for t, _ in ts}
        nonzero, p0 = {}, {}

        def xor0(key, p):
            nonzero[key[0]] = _par_xor([(regions[key[0]], p)])[0]
            if nonzero[key[0]] and key[1] in VARIANTS:
                p0[key[0]] = p                  # kept to try the quantiser variants below

        self._predict([(t, 0) for t, _ in ts], xor0)
        choice = {t.name: 0 for t, _ in ts}
        retry = [t for t, k in ts if k == "rq" and t.type in VARIANTS and nonzero[t.name]]
        if retry:
            # the region now holds file ^ p0; variant v leaves file ^ pv = region ^ p0 ^ pv
            alts = {t.name: [regions[t.name]] for t in retry}

            def alt(key, p):
                alts[key[0]].append(regions[key[0]] ^ p0[key[0]] ^ p)

            for v in range(1, max(len(VARIANTS[t.type]) for t in retry) + 1):
                self._predict([(t, v) for t in retry if v <= len(VARIANTS[t.type])], alt)
            for t in retry:
                errs = [int(np.count_nonzero(a)) for a in alts[t.name]]
                v = choice[t.name] = int(np.argmin(errs))
                if v:
                    regions[t.name][:] = alts[t.name][v]
                nonzero[t.name] = errs[v] > 0
        plan = []
        for t, k in ts:
            v = choice[t.name]
            if k == "xor":
                self.stats["xor"] += 1
            else:
                self.stats["requant_diff" if nonzero[t.name] else "requant_exact"] += 1
                self.stats["variant"] += v > 0
            plan.append([t.name, k, v])
        return plan, self.b

    @staticmethod
    def _imatrix(ex):
        name = next((n for n in ex if n.startswith("imatrix")), None)
        return load_imatrix(ex[name]) if name else {}

    BUNDLE_MAX = 64 << 20                     # extra files up to this size go into the LZMA bundle

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
        kept = []
        for i, (name, path) in enumerate(bases.items()):
            with open(path, "rb") as f:
                data = f.read()
            seen(name, data)
            c = pack(self.base_codec.compress_dev(data))
            put(f"b{i}", c)
            self.sizes[name] = len(c)
            if self.verbose:
                print(f"  {name}: {len(data):,} -> {len(c):,} B (base){_mem()}", flush=True)
            del c
            _release()
            kept.append(_keep_base(data))
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
        if self.verbose:
            print(f"  bases coded{_mem()}", flush=True)
        params = zstandard.ZstdCompressionParameters.from_level(self.level, window_log=25, threads=4)

        def read(name):
            # in the reader thread: the content is seen (hashed) before it is XOR-ed in place,
            # overlapping with the GPU work on the previous file
            with open(files[name], "rb") as f:
                data = bytearray(f.read())
            seen(name, data)
            return data

        def zc(data, zdict):
            z = zstandard.ZstdCompressor(compression_params=params, dict_data=zdict) if zdict else \
                zstandard.ZstdCompressor(compression_params=params)
            # copy: compress() allocates a buffer the size of the input and only shrinks it, and on
            # Windows the shrunk block keeps the whole allocation committed
            return bytes(memoryview(z.compress(data)))

        names = list(files)
        manifest = {"method": self.name, "bases": list(bases), "files": [],
                    "extras": {"bundle": list(ex), "single": big}}
        fsize = [os.path.getsize(files[n]) for n in names]
        pending = deque()                     # (file index, zstd future, bytes held)

        def drain(limit):
            """wait for zstd jobs until the files they hold total at most `limit` bytes"""
            while pending and sum(s for _, _, s in pending) > limit:
                k, f, _ = pending.popleft()
                c = f.result()
                put(f"f{k}", c)
                self.sizes[names[k]] = len(c)

        # pipelined: the next file is read and older residuals are compressed while the GPU
        # predicts the current one, as long as the files held stay within self.inflight bytes
        with ThreadPoolExecutor(1) as rd, ThreadPoolExecutor(2) as zp:
            nxt = rd.submit(read, names[0]) if names else None
            for i, name in enumerate(names):
                data = nxt.result()
                plan, b = self._residual(data)
                _release()
                manifest["files"].append({"name": name, "size": len(data), "base": b, "plan": plan})
                pending.append((i, zp.submit(zc, data, self.zdict if b is not None else None), len(data)))
                del data
                if self.verbose:
                    src = list(bases)[b] if b is not None else "no base: zstd only"
                    print(f"  {name}: {fsize[i]:,} B (from {src}){_mem()}", flush=True)
                if i + 1 < len(names):
                    drain(max(0, self.inflight - fsize[i + 1]))
                    nxt = rd.submit(read, names[i + 1])
            drain(0)
        put("manifest", zstd_c(json.dumps(manifest).encode()))
        self.bases = self.base_dev = None
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
        files = [(i, m) for i, m in enumerate(manifest["files"]) if want(m["name"])]
        need = {i for i, n in enumerate(manifest["bases"]) if want(n)} | \
               {m["base"] for _, m in files if m["base"] is not None}
        bases = []
        for i, name in enumerate(manifest["bases"]):
            if i not in need:
                bases.append(None)
                continue
            predicts = any(m["base"] == i for _, m in files)
            if hasattr(self.base_codec, "decompress_host"):
                data = self.base_codec.decompress_host(unpack(get(f"b{i}")))
                kept = _keep_base(data) if predicts else None
            else:
                dev = self.base_codec.decompress_dev(unpack(get(f"b{i}")))
                data = cp.asnumpy(dev)
                kept = (data[:GGUF.from_bytes(memoryview(data), light=True).data_start].tobytes(), dev) if predicts else None
            if want(name):
                sink(name, data)
            bases.append(kept)
            del data
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
            return
        self._setup(bases, self._imatrix(ex))

        def zd(i):
            m = manifest["files"][i]
            zdict = self.bases[m["base"]][3] if m["base"] is not None else None
            dec = zstandard.ZstdDecompressor(dict_data=zdict) if zdict else zstandard.ZstdDecompressor()
            return bytearray(dec.decompress(get(f"f{i}"), max_output_size=m["size"]))

        with ThreadPoolExecutor(2) as zp:
            nxt, ahead = {}, 0                # files decompressed ahead, within self.inflight bytes
            for k, (i, m) in enumerate(files):
                while ahead < len(files) and (ahead <= k or (ahead - k < 3 and sum(
                        files[q][1]["size"] for q in range(k, ahead + 1)) <= self.inflight)):
                    nxt[ahead] = zp.submit(zd, files[ahead][0])
                    ahead += 1
                data = nxt.pop(k).result()
                if m["plan"]:
                    self._use(m["base"])
                    view = np.frombuffer(data, np.uint8)
                    tens = {t.name: t for t in GGUF.from_bytes(data, light=True).tensors}

                    def xor(key, p, view=view, tens=tens):
                        t = tens[key[0]]
                        _par_xor([(view[t.offset:t.offset + t.nbytes], p)])

                    self._predict([(tens[n], v) for n, _, v in m["plan"]], xor)
                    del view
                sink(m["name"], data)
                del data
                _release()
        self.bases = self.base_dev = None


class R2Base:
    """compress_dev / decompress_dev interface (as M24's) around the BF16 v2 codec (bf16v2.py:
    finer contexts, role-shared tables, duplicate rows, bounded GPU memory)"""

    def __init__(self, budget=1_500_000_000):
        from bf16v2 import BF16v2
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
