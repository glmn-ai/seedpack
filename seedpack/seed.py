"""Seeds: when quantised GGUFs come without their BF16 model, the most precise of them (the
"senior" quant: most bits per weight, e.g. UD-Q8_K_XL > Q8_0 > Q6_K > Q4_K_M) takes the BF16
model's place. Its tensors are dequantised (llama.cpp's dequantize_row_*) and the other files are
re-quantised from them and stored as XOR residuals, exactly as with a BF16 base (repo3.py).

The seed file itself is coded by parts (SeedCodec):
  * Q8_0 tensors with M21 (gpu3.Q8Parts);
  * F16 / BF16 matrices with the BF16 codec R2 (bf16v2.py): F16 weights converted from BF16 hold
    BF16 values, so they convert back exactly; the few that do not are listed;
  * every other byte (metadata, tokenizer, small tensors) with the header coder of M11/M20.
"""
import json
import struct
from concurrent.futures import ThreadPoolExecutor

from .xp import cp
import numpy as np

from .codecs_gguf import pack, unpack, zstd_c, zstd_d, group_of
from .gguf_reader import GGUF, BLOCK, GGML_F16, GGML_BF16, GGML_Q8_0
from .gpu2 import fill_rest, _trim
from . import qr


OTHER_MIN = 1 << 16       # smaller tensors stay with the header bytes


def bits_per_weight(tensors):
    """mean bits per weight of a GGUF's tensors (the seed is the file with the most)"""
    n = sum(int(np.prod(t.shape)) for t in tensors if t.type in BLOCK)
    return 8 * sum(t.nbytes for t in tensors if t.type in BLOCK) / max(n, 1)


def gguf_bytes(tensors, align=32):
    """a minimal GGUF v3 (no metadata) holding tensors [(name, shape, type, payload bytes)]"""
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 0))
    offs, off = [], 0
    for name, shape, ty, payload in tensors:
        nb = name.encode()
        head += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(shape))
        head += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", ty, off)
        offs.append(off)
        off += (len(payload) + align - 1) // align * align
    start = (len(head) + align - 1) // align * align
    out = bytearray(start + off)
    out[:len(head)] = head
    for (_, _, _, payload), o in zip(tensors, offs):
        out[start + o:start + o + len(payload)] = payload
    return bytes(out)


def _f16_from_bf16(u):
    """BF16 bits -> F16 bits (round to nearest; exact for values F16 can hold)"""
    return (u.astype(cp.uint32) << 16).view(cp.float32).astype(cp.float16).view(cp.uint16)


class SeedCodec:
    """a quantised GGUF coded by parts (see the module docstring); decompress -> numpy uint8"""

    def __init__(self):
        from .gpu3 import Q8Parts
        from .bf16v2 import BF16v2
        self.q8 = Q8Parts()
        self.r2 = BF16v2(budget=1_500_000_000)

    def compress(self, data):
        g = GGUF.from_bytes(data)                       # full parse: the header coder needs the tokenizer
        q8 = [t for t in g.tensors if t.type == GGML_Q8_0]
        fl = [t for t in g.tensors if t.type in (GGML_F16, GGML_BF16) and len(t.shape) == 2]
        other = [t for t in g.tensors if t not in q8 and t not in fl and t.nbytes >= OTHER_MIN]
        lay = {"size": len(data), "t": sorted([t.offset, t.nbytes, t.shape[0], group_of(t.name)] for t in q8 + fl + other)}
        view = np.frombuffer(data, np.uint8)
        with ThreadPoolExecutor(1) as ex:
            rest = ex.submit(self.q8.c._header_c, data, g, lay)
            ch = {"layout": zstd_c(json.dumps(lay).encode())}
            if q8:
                ch["q8"] = self.q8.compress(data, with_rest=False)
                _trim()
            if fl:
                parts, meta, idx, vals = [], [], [], []
                for t in fl:
                    u = cp.asarray(view[t.offset:t.offset + t.nbytes]).view(cp.uint16)
                    bad = np.zeros(0, np.uint32)
                    if t.type == GGML_F16:
                        b = (u.view(cp.float16).astype(cp.float32).view(cp.uint32) >> 16).astype(cp.uint16)
                        bad_g = cp.flatnonzero(_f16_from_bf16(b) != u)
                        bad = bad_g.astype(cp.uint32).get()
                        idx.append(bad); vals.append(u[bad_g].get())
                        u = b
                    parts.append((t.name, t.shape, GGML_BF16, u.get().tobytes()))
                    meta.append([t.name, t.type, t.offset, t.nbytes, len(bad)])
                    del u
                ch["f"] = self.r2.compress(gguf_bytes(parts))
                del parts
                ch["fmeta"] = zstd_c(json.dumps(meta).encode())
                ch["fexc"] = zstd_c(b"".join(a.tobytes() for a in idx) + b"".join(a.tobytes() for a in vals))
                _trim()
            # other quant types (a Q4_K_M seed, ...): QR without a seed (qr.self_encode) per type
            # where it beats zstd, the rest zstd
            selfq = {}
            for t in other:
                if t.type in qr.SELF_TYPES and t.shape[0] % qr.QK[t.type] == 0:
                    selfq.setdefault(t.type, []).append(t)
            sq = []
            for T, ts in selfq.items():
                blobs = [view[t.offset:t.offset + t.nbytes] for t in ts]
                c = qr.self_encode(T, [(b, t.shape[0]) for b, t in zip(blobs, ts)])
                if len(c) < len(zstd_c(b"".join(b.tobytes() for b in blobs), 9)):
                    ch[f"s{len(sq)}"] = c
                    sq.append([T, [[t.offset, t.nbytes] for t in ts]])
                _trim()
            done = {o for _, lst in sq for o, _ in lst}
            other = [t for t in other if t.offset not in done]
            if sq:
                ch["smeta"] = zstd_c(json.dumps(sq).encode())
            if other:
                ch["o"] = zstd_c(b"".join(view[t.offset:t.offset + t.nbytes] for t in other), 9)
                ch["ometa"] = zstd_c(json.dumps([[t.offset, t.nbytes] for t in other]).encode())
            ch["rest"] = rest.result()
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        lay = json.loads(zstd_d(ch["layout"]))
        out = np.empty(lay["size"], np.uint8)
        with ThreadPoolExecutor(1) as ex:
            rest = ex.submit(self.q8.c._header_d, ch["rest"])
            if "q8" in ch:
                self.q8.decompress(ch["q8"], out=out)
                _trim()
            if "f" in ch:
                syn = self.r2.decompress(ch["f"])
                gs = GGUF.from_bytes(memoryview(syn), light=True)
                meta = json.loads(zstd_d(ch["fmeta"]))
                exc = zstd_d(ch["fexc"])
                nexc = sum(m[4] for m in meta)
                idx = np.frombuffer(exc, np.uint32, nexc)
                vals = np.frombuffer(exc, np.uint16, nexc, 4 * nexc)
                e = 0
                for (name, ty, off, nb, ne), st in zip(meta, gs.tensors):
                    u = cp.asarray(syn[st.offset:st.offset + st.nbytes]).view(cp.uint16)
                    if ty == GGML_F16:
                        u = _f16_from_bf16(u)
                        if ne:
                            u[cp.asarray(idx[e:e + ne])] = cp.asarray(vals[e:e + ne])
                        e += ne
                    out[off:off + nb] = u.view(cp.uint8).get()
                del syn
                _trim()
            if "smeta" in ch:
                for k, (T, lst) in enumerate(json.loads(zstd_d(ch["smeta"]))):
                    for (off, nb), arr in zip(lst, qr.self_decode(T, ch[f"s{k}"])):
                        arr.get(out=out[off:off + nb])
                    _trim()
            if "o" in ch:
                o, pos = zstd_d(ch["o"]), 0
                for off, nb in json.loads(zstd_d(ch["ometa"])):
                    out[off:off + nb] = np.frombuffer(o, np.uint8, nb, pos); pos += nb
            fill_rest(out, rest.result(), lay)
        return out
