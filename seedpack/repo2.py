"""Method 27: whole model repository in one archive via exact re-quantisation.

Contents of the archive:
  * the BF16 model, coded with M24 (GPU);
  * the repository's other small files (imatrix, README, config, ...), xz;
  * every quantised GGUF as a "difference file": each tensor is XOR-ed with what llama.cpp's
    quantiser (third_party/ggml/ggmlq.dll) produces from the BF16 weights and the repository's
    importance matrix; float tensors are XOR-ed with the BF16 tensor converted to their type;
    tensors already seen in an earlier file are referenced. For the unsloth repository these
    differences are all zeros, so a quantised file costs little more than its GGUF header.
The XOR transform is self-inverse given the BF16 weights, so any mismatch (another
quantiser version, a hand-edited tensor, ...) only costs size, never correctness.
"""
import hashlib
import json
import lzma

import cupy as cp
import numpy as np
import zstandard

from .codecs_gguf import pack, unpack, zstd_c, zstd_d
from .gguf_reader import GGUF
from .gpu_bf16 import M24BF16GPU
from .requant import QFUN, quantize, load_imatrix

F32, F16, BF16 = 0, 1, 30


def base_float_np(base, t):
    raw = np.frombuffer(base, np.uint8, t.nbytes, t.offset)
    if t.type == BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if t.type == F16:
        return raw.view(np.float16).astype(np.float32)
    if t.type == F32:
        return raw.view(np.float32)
    return None


def float_as_np(x, ttype):
    if ttype == F32:
        return x.astype(np.float32).view(np.uint8)
    if ttype == F16:
        return x.astype(np.float16).view(np.uint8)
    if ttype == BF16:
        u = x.astype(np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.uint8)
    return None


class M27RepoRequant:
    name = "M27 model repo: BF16 + quants re-derived with llama.cpp quantisers + imatrix"

    def __init__(self, level=9):
        self.level = level
        self.base_codec = M24BF16GPU()
        self.sizes = {}
        self.stats = {"requant_exact": 0, "requant_diff": 0, "xor": 0, "dup": 0, "raw": 0}
        self.verbose = False

    # ---------------------------------------------------------- predictions shared by both sides
    def _prediction(self, t, base, by_name, imatrix):
        """bytes the tensor is expected to have, or None"""
        bt = by_name.get(t.name)
        if bt is None:
            return None, None
        x = base_float_np(base, bt)
        if x is None or x.size != int(np.prod(t.shape)):
            return None, None
        if t.type in (F32, F16, BF16):
            return float_as_np(x, t.type), "xor"
        if t.type in QFUN and t.nbytes == x.size // QFUN[t.type][1] * QFUN[t.type][2]:
            return quantize(t.type, x.reshape(-1, t.shape[0]), imatrix.get(t.name)), "rq"
        return None, None

    def compress(self, base: bytes, files: dict, extras: dict) -> bytes:
        """files: quantised GGUFs name -> path; extras: other repository files name -> path."""
        base_dev = cp.asarray(np.frombuffer(base, np.uint8))
        ch = {"base": pack(self.base_codec.compress_dev(base, base_dev))}
        del base_dev
        cp.get_default_memory_pool().free_all_blocks()
        self.sizes["BF16 (base)"] = len(ch["base"])
        ex = {n: open(p, "rb").read() for n, p in extras.items()}
        ch["extras"] = lzma.compress(pack(ex), preset=9)
        self.sizes["other files"] = len(ch["extras"])
        imx_name = next((n for n in ex if n.startswith("imatrix") and n.endswith(".dat")), None)
        imatrix = load_imatrix(ex[imx_name]) if imx_name else {}
        bg = GGUF.from_bytes(base)
        by_name = {t.name: t for t in bg.tensors}
        zdict = zstandard.ZstdCompressionDict(base[:bg.data_start], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        # window larger than the 6 MB header dictionary, so the tokenizer is matched in full
        params = zstandard.ZstdCompressionParameters.from_level(self.level, window_log=25, threads=4)
        seen, manifest = {}, []
        for i, (name, path) in enumerate(files.items()):
            data = bytearray(open(path, "rb").read())
            size = len(data)
            qg = GGUF.from_bytes(bytes(data[:64 << 20]))
            plan = []
            view = np.frombuffer(data, np.uint8)
            for t in qg.tensors:
                region = view[t.offset:t.offset + t.nbytes]
                key = (t.name, t.type, hashlib.sha1(region).digest())
                if key in seen:
                    plan.append([t.name, ["dup"] + list(seen[key])]); region[:] = 0
                    self.stats["dup"] += 1
                    continue
                seen[key] = (i, t.name)
                pred, kind = self._prediction(t, base, by_name, imatrix)
                if pred is None:
                    self.stats["raw"] += 1
                    continue
                exact = bool((pred == region).all())
                region ^= pred
                plan.append([t.name, kind])
                self.stats["xor" if kind == "xor" else ("requant_exact" if exact else "requant_diff")] += 1
            ch[f"f{i}"] = zstandard.ZstdCompressor(compression_params=params, dict_data=zdict).compress(bytes(data))
            self.sizes[name] = len(ch[f"f{i}"])
            manifest.append({"name": name, "size": size, "plan": plan})
            if self.verbose:
                print(f"  {name}: {size:,} -> {self.sizes[name]:,} B", flush=True)
            del data, view
        ch["manifest"] = zstd_c(json.dumps(manifest).encode())
        return pack(ch)

    def decompress(self, buf: bytes, sink):
        """sink(name, bytes-like) receives the BF16 file, the extra files and every quantised file."""
        ch = unpack(buf)
        base = self.base_codec.decompress(ch["base"]).tobytes()
        cp.get_default_memory_pool().free_all_blocks()
        bg = GGUF.from_bytes(base)
        by_name = {t.name: t for t in bg.tensors}
        sink("BF16", base)
        ex = unpack(lzma.decompress(ch["extras"]))
        for n, b in ex.items():
            sink(n, b)
        imx_name = next((n for n in ex if n.startswith("imatrix") and n.endswith(".dat")), None)
        imatrix = load_imatrix(ex[imx_name]) if imx_name else {}
        zdict = zstandard.ZstdCompressionDict(base[:bg.data_start], dict_type=zstandard.DICT_TYPE_RAWCONTENT)
        manifest = json.loads(zstd_d(ch["manifest"]))
        needed = {tuple(k[1:]) for m in manifest for _, k in m["plan"] if isinstance(k, list)}
        cache = {}
        for i, m in enumerate(manifest):
            data = bytearray(zstandard.ZstdDecompressor(dict_data=zdict).decompress(ch[f"f{i}"], max_output_size=m["size"]))
            view = np.frombuffer(data, np.uint8)
            qg = GGUF.from_bytes(bytes(data[:64 << 20]))
            tens = {t.name: t for t in qg.tensors}
            for tname, kind in m["plan"]:
                t = tens[tname]
                region = view[t.offset:t.offset + t.nbytes]
                if isinstance(kind, list):
                    region[:] = cache[(kind[1], kind[2])]
                else:
                    pred, _ = self._prediction(t, base, by_name, imatrix)
                    region ^= pred
            for t in qg.tensors:
                if (i, t.name) in needed:
                    cache[(i, t.name)] = view[t.offset:t.offset + t.nbytes].copy()
            sink(m["name"], data)
            del data, view
