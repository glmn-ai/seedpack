"""R3: M28 with a sparse residual + R2 base + R5 extras.

M28 stores every quantised GGUF as zstd(file XOR prediction) -- ~10 GB of mostly zero bytes go
through zstd on both sides and the decoder parses every GGUF header again. R3 stores per file
  * 'gaps': every byte outside the predicted tensors (GGUF header, tensors that are not predicted)
            -> zstd with the BF16 header as dictionary;
  * the XOR residual only of predicted tensors that are NOT reproduced exactly (zstd each);
  * the tensor table (name, type, row length, offset, size) in the manifest, so the decoder never
    parses a GGUF: it allocates the file, fills the gaps, and copies the predictions straight in.
Base: R2 (bf16v2.BF16v2) through an adapter; extras: R5 (imatrix byte planes).
"""
import json, lzma, os, sys
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
import zstandard

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codecs_gguf import pack, unpack, zstd_c, zstd_d
from gguf_reader import GGUF, Tensor
from gpuquant import VARIANTS
from repo3 import M28RepoGPU, _par_xor
from bf16v2 import BF16v2
from extras_codec import enc_imatrix, dec_imatrix


class R2Base:
    """M24-compatible interface (compress_dev / decompress_dev) around R2."""
    def __init__(self, budget=1_500_000_000):
        self.c = BF16v2(budget=budget)

    def compress_dev(self, data, dev=None):
        return {"r2": self.c.compress(data)}

    def decompress_dev(self, ch):
        return cp.asarray(self.c.decompress(ch["r2"]))


def extras_c(ex):
    out = {}
    for n, b in ex.items():
        if n.startswith("imatrix") and b[:4] != b"GGUF":
            try:
                z = enc_imatrix(b)
                if dec_imatrix(z) == b:
                    out["I" + n] = z; continue
            except Exception:
                pass
        out["P" + n] = b
    plain = {n[1:]: b for n, b in out.items() if n[0] == "P"}
    return pack({"plain": lzma.compress(pack(plain), preset=9 | lzma.PRESET_EXTREME),
                 **{n: b for n, b in out.items() if n[0] == "I"}})


def extras_d(buf):
    ch = unpack(buf)
    ex = dict(unpack(lzma.decompress(ch.pop("plain"))))
    for n, z in ch.items():
        ex[n[1:]] = dec_imatrix(z)
    return ex


class R3RepoSparse(M28RepoGPU):
    name = "R3 repo: M28 prediction + sparse residual (no zstd of zeros, no GGUF parse on decode) + R2 base"

    def __init__(self, base_codec=None, sparse=True, extras_r5=True, **kw):
        super().__init__(**kw)
        if base_codec is not None:
            self.base_codec = base_codec
        self.sparse, self.extras_r5 = sparse, extras_r5

    # same as M28._residual, but also returns which tensors still differ after prediction
    def _residual2(self, data):
        view = np.frombuffer(data, np.uint8)
        qg = GGUF.from_bytes(data, light=True)
        self._use(self._best_base(qg))
        ts = [(t, self._kind(t)) for t in qg.tensors]
        self.stats["raw"] += sum(k is None for _, k in ts)
        ts = [(t, k) for t, k in ts if k]
        pred = self._predict([(t, 0) for t, _ in ts])
        regions = {t.name: view[t.offset:t.offset + t.nbytes] for t, _ in ts}
        nonzero = dict(zip((t.name for t, _ in ts),
                           _par_xor([(regions[t.name], pred[(t.name, t.type, 0)]) for t, _ in ts])))
        choice = {t.name: 0 for t, _ in ts}
        retry = [t for t, k in ts if k == "rq" and t.type in VARIANTS and nonzero[t.name]]
        if retry:
            pred.update(self._predict([(t, v) for t in retry for v in range(1, len(VARIANTS[t.type]) + 1)]))
            for t in retry:
                r, p0 = regions[t.name], pred[(t.name, t.type, 0)]
                alts = [r] + [r ^ p0 ^ pred[(t.name, t.type, v)] for v in range(1, len(VARIANTS[t.type]) + 1)]
                errs = [int(np.count_nonzero(a)) for a in alts]
                v = choice[t.name] = int(np.argmin(errs))
                if v:
                    r[:] = alts[v]
                nonzero[t.name] = errs[v] > 0
        plan = []
        for t, k in sorted(ts, key=lambda x: x[0].offset):
            v = choice[t.name]
            if k == "xor":
                self.stats["xor"] += 1
            else:
                self.stats["requant_diff" if nonzero[t.name] else "requant_exact"] += 1
                self.stats["variant"] += v > 0
            plan.append([t.name, k, v, t.type, list(t.shape), t.offset, t.nbytes, int(nonzero[t.name])])
        return plan, self.b

    def compress(self, bases, files, extras):
        if not self.sparse:
            return super().compress(bases, files, extras)
        ch, devs = {}, []
        for i, (name, data) in enumerate(bases.items()):
            dev = cp.asarray(np.frombuffer(data, np.uint8))
            ch[f"b{i}"] = pack(self.base_codec.compress_dev(data, dev))
            cp.get_default_memory_pool().free_all_blocks()
            self.sizes[name] = len(ch[f"b{i}"])
            devs.append(dev)
        ex = {n: open(p, "rb").read() for n, p in extras.items()}
        ch["extras"] = extras_c(ex) if self.extras_r5 else lzma.compress(pack(ex), preset=9)
        self.sizes["other files"] = len(ch["extras"])
        self._setup(list(zip(bases.values(), devs)), self._imatrix(ex))
        params = zstandard.ZstdCompressionParameters.from_level(9, window_log=25)

        def read(path):
            with open(path, "rb") as f:
                return bytearray(f.read())

        def zc(data, plan, zdict):
            view = memoryview(data)
            gaps, pos, res = [], 0, []
            for p in plan:
                off, nb = p[5], p[6]
                gaps.append(view[pos:off]); pos = off + nb
                if p[7]:
                    res.append(bytes(view[off:off + nb]))
            gaps.append(view[pos:])
            z = zstandard.ZstdCompressor(compression_params=params, dict_data=zdict)
            out = {"gaps": z.compress(b"".join(gaps))}
            if res:
                out["res"] = zstandard.ZstdCompressor(level=9, threads=4).compress(b"".join(res))
            return pack(out)

        names = list(files)
        manifest = {"bases": list(bases), "files": []}
        with ThreadPoolExecutor(1) as rd, ThreadPoolExecutor(2) as zp:
            nxt = rd.submit(read, files[names[0]]) if names else None
            jobs = []
            for i, name in enumerate(names):
                data = nxt.result()
                nxt = rd.submit(read, files[names[i + 1]]) if i + 1 < len(names) else None
                plan, b = self._residual2(data)
                manifest["files"].append({"name": name, "size": len(data), "base": b, "plan": plan})
                jobs.append(zp.submit(zc, data, plan, self.zdict))
                del data
            for i, (name, j) in enumerate(zip(names, jobs)):
                ch[f"f{i}"] = j.result()
                self.sizes[name] = len(ch[f"f{i}"])
        ch["manifest"] = zstd_c(json.dumps(manifest).encode())
        self.bases = self.base_dev = None
        return pack(ch)

    def decompress(self, buf, sink):
        if not self.sparse:
            return super().decompress(buf, sink)
        ch = unpack(buf)
        manifest = json.loads(zstd_d(ch["manifest"]))
        bases = []
        for i, name in enumerate(manifest["bases"]):
            dev = self.base_codec.decompress_dev(unpack(ch[f"b{i}"]))
            data = cp.asnumpy(dev).tobytes()
            sink(name, data)
            bases.append((data, dev))
        ex = extras_d(ch["extras"]) if self.extras_r5 else unpack(lzma.decompress(ch["extras"]))
        for n, b in ex.items():
            sink(n, b)
        self._setup(bases, self._imatrix(ex))
        files = manifest["files"]

        def zd(i):
            m = files[i]
            fc = unpack(ch[f"f{i}"])
            gaps = zstandard.ZstdDecompressor(dict_data=self.bases[m["base"]][3]).decompress(fc["gaps"], max_output_size=m["size"])
            res = zstandard.ZstdDecompressor().decompress(fc["res"], max_output_size=m["size"]) if "res" in fc else b""
            out = np.empty(m["size"], np.uint8)
            g = np.frombuffer(gaps, np.uint8)
            pos = gpos = 0
            for p in m["plan"]:
                off, nb = p[5], p[6]
                out[pos:off] = g[gpos:gpos + off - pos]; gpos += off - pos; pos = off + nb
            out[pos:] = g[gpos:]
            return out, np.frombuffer(res, np.uint8)

        with ThreadPoolExecutor(2) as zp:
            nxt = [zp.submit(zd, i) for i in range(min(2, len(files)))]
            for i, m in enumerate(files):
                out, res = nxt.pop(0).result()
                if i + 2 < len(files):
                    nxt.append(zp.submit(zd, i + 2))
                self._use(m["base"])
                tens = {p[0]: Tensor(p[0], tuple(p[4]), p[3], p[5], p[6]) for p in m["plan"]}
                pred = self._predict([(tens[p[0]], p[2]) for p in m["plan"]])
                jobs, rpos = [], 0
                for p in m["plan"]:
                    region = out[p[5]:p[5] + p[6]]
                    jobs.append((region, pred[(p[0], p[3], p[2])]))
                with ThreadPoolExecutor(8) as cpy:
                    list(cpy.map(lambda a: np.copyto(a[0], a[1]), jobs))
                for p in m["plan"]:
                    if p[7]:
                        region = out[p[5]:p[5] + p[6]]
                        np.bitwise_xor(region, res[rpos:rpos + p[6]], out=region); rpos += p[6]
                sink(m["name"], out)
                del out
        self.bases = self.base_dev = None
