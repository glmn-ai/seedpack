"""Method 22: lossless BF16 GGUF codec (no reference model).

A bf16 weight = sign(1) | exponent(8) | mantissa(7). Trained weights use ~2.6 of the 8
exponent bits (the DFloat11 / ZipNN observation); on top of that this codec models
  * exponent | predicted exponent  (rank-1 variance model: row factor + column factor,
                                    quarter-octave side info per row and per column)
  * mantissa | exponent - predicted exponent  (density slope inside an octave)
  * sign     | column mean-bias class
Each stream is coded with static rANS, tables per tensor. Everything outside 2-D bf16
tensors (header, norms, padding) goes through the tokenizer-aware header codec.
"""
import json
import struct

import numpy as np

import rans
from codecs_gguf import pack, unpack, zstd_c, zstd_d
from gguf_reader import GGUF, GGML_BF16
from gpu2 import header_pairs_c, header_pairs_d, xz_chunks_c, xz_chunks_d, rest_of


def _contexts(rfi, cfi, ex=None):
    pred4 = rfi[:, None].astype(np.int32) + cfi[None, :].astype(np.int32)       # log2 sigma in 1/4 octave
    eb = np.clip((pred4 + 4 * 27) // 2, 0, 127).astype(np.int16)                # half-octave bins
    if ex is None:
        return eb, pred4
    rel = np.clip(ex.astype(np.int32) - (pred4 + 508) // 4 + 8, 0, 15).astype(np.int16)
    return eb, rel


class M22BF16:
    name = "M22 BF16: exponent/mantissa/sign context rANS"

    def compress(self, data):
        g = GGUF.from_bytes(data)
        ts = sorted((t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2), key=lambda t: t.offset)
        layout = {"size": len(data), "t": [[t.offset, t.nbytes, t.shape[0], "bf16"] for t in ts]}
        bufs, tabs, side = [], [], []
        for t in ts:
            rowlen = t.shape[0]
            u = np.frombuffer(data, np.uint16, t.nbytes // 2, t.offset).reshape(-1, rowlen)
            w = (u.astype(np.uint32) << 16).view(np.float32)
            rf = np.log2(np.sqrt((w.astype(np.float64) ** 2).mean(1)) + 1e-30)
            cf = np.log2(np.sqrt(((w / np.exp2(rf)[:, None]).astype(np.float64) ** 2).mean(0)) + 1e-30)
            rfi = np.clip(np.rint(rf * 4), -32768, 32767).astype(np.int16)
            cfi = np.clip(np.rint(cf * 4), -32768, 32767).astype(np.int16)
            bias = (w / np.exp2(rf)[:, None]).mean(0) / np.exp2(cfi / 4.0)
            sc = np.clip(np.floor(bias * 16) + 8, 0, 15).astype(np.int16)
            ex = ((u >> 7) & 255).astype(np.uint8); man = (u & 127).astype(np.uint8); sg = (u >> 15).astype(np.uint8)
            eb, rel = _contexts(rfi, cfi, ex)
            sctx = np.broadcast_to(sc[None, :], u.shape)
            for sym, ctx, nctx in ((ex, eb, 128), (man, rel, 16), (sg, sctx, 16)):
                sym = sym.ravel(); ctx = np.ascontiguousarray(ctx.ravel())
                f = rans.build_freqs(sym, ctx, nctx)
                tabs.append(f); bufs.append(rans.encode(sym, ctx, f))
            side.append(rfi.tobytes() + cfi.tobytes() + sc.astype(np.int8).tobytes())
        h = header_pairs_c(data, g, layout)
        rest = b"P" + h if h is not None else b"X" + xz_chunks_c(rest_of(data, layout))
        ch = {
            "layout": zstd_c(json.dumps(layout).encode()),
            "side": zstd_c(b"".join(side), 19),
            "tables": zstd_c(b"".join(f.astype(np.uint16).tobytes() for f in tabs), 19),
            "streams": b"".join(struct.pack("<Q", len(b)) + b for b in bufs),
            "rest": rest,
        }
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        layout = json.loads(zstd_d(ch["layout"]))
        side = zstd_d(ch["side"]); tabs = np.frombuffer(zstd_d(ch["tables"]), np.uint16)
        sbuf, spos, tpos, dpos = ch["streams"], 0, 0, 0
        r = ch["rest"]
        rest = header_pairs_d(r[1:]) if r[:1] == b"P" else xz_chunks_d(r[1:])
        out = np.empty(layout["size"], np.uint8)
        pos = rpos = 0

        def next_stream():
            nonlocal spos
            (ln,) = struct.unpack_from("<Q", sbuf, spos); spos += 8
            b = sbuf[spos:spos + ln]; spos += ln
            return b

        for off, nbytes, rowlen, _ in layout["t"]:
            n = nbytes // 2; nrows = n // rowlen
            rfi = np.frombuffer(side, np.int16, nrows, dpos); dpos += 2 * nrows
            cfi = np.frombuffer(side, np.int16, rowlen, dpos); dpos += 2 * rowlen
            sc = np.frombuffer(side, np.int8, rowlen, dpos).astype(np.int16); dpos += rowlen
            res = []
            for nctx in (128, 16, 16):
                res.append(tabs[tpos:tpos + nctx * 256].reshape(nctx, 256)); tpos += nctx * 256
            eb, pred4 = _contexts(rfi, cfi)
            ex = rans.decode(next_stream(), eb.ravel(), res[0]).reshape(nrows, rowlen)
            _, rel = _contexts(rfi, cfi, ex)
            man = rans.decode(next_stream(), np.ascontiguousarray(rel.ravel()), res[1]).reshape(nrows, rowlen)
            sg = rans.decode(next_stream(), np.ascontiguousarray(np.broadcast_to(sc[None, :], (nrows, rowlen)).ravel()), res[2])
            u = (sg.reshape(nrows, rowlen).astype(np.uint16) << 15) | (ex.astype(np.uint16) << 7) | man.astype(np.uint16)
            gap = off - pos
            out[pos:off] = np.frombuffer(rest, np.uint8, gap, rpos); rpos += gap
            out[off:off + nbytes] = u.view(np.uint8).ravel(); pos = off + nbytes
        out[pos:] = np.frombuffer(rest, np.uint8, len(rest) - rpos, rpos)
        return out
