"""Methods 12-16: lossless compression of a Q8_0 GGUF *relative to a reference model*
that the receiver already has (another quant of the same model, the base model, ...).

The reference is identified by SHA-256 stored in the archive; decompression refuses a
different file. All predictions use numpy float32 on the CPU (IEEE, deterministic), so
encoder and decoder compute bit-identical predictions.

Per Q8_0 block of the target (d, q[32]) with reference weights w_ref[32]:
  scale  : d_pred = fp16(max|w_ref| / 127);  code delta = bits(d) - bits(d_pred)
  weights: pred_j = clip(round(w_ref_j / d), -127, 127);  code r_j = (q_j - pred_j) mod 256
The residual is coded with rANS; tables per tensor, context = how precise the reference
is at that place (reference quantisation step / d) x |w_ref|/d bucket.
"""
import hashlib
import json
import struct

import numpy as np

from . import rans
from .codecs_gguf import pack, unpack, zstd_c, zstd_d, xz_c, xz_d
from .gguf_reader import GGUF, GGML_Q8_0, GGML_F32, GGML_F16, GGML_BF16

GGML_Q4_K, GGML_Q6_K = 12, 14
N_E, N_M = 8, 4                     # context: reference precision bins x magnitude bins


# ---------------------------------------------------------------- reference dequantisation
def _f16(u8pair):
    return u8pair.copy().view(np.float16).astype(np.float32).ravel()


def dequant(g: GGUF, t):
    """-> (weights float32 flat, per-weight reference step float32 flat)."""
    raw = g.raw(t)
    n = int(np.prod(t.shape))
    if t.type == GGML_F32:
        w = raw.view(np.float32).copy(); return w, np.abs(w) * np.float32(2 ** -23)
    if t.type == GGML_F16:
        w = raw.view(np.float16).astype(np.float32); return w, np.abs(w) * np.float32(2 ** -10)
    if t.type == GGML_BF16:
        w = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32); return w, np.abs(w) * np.float32(2 ** -7)
    if t.type == GGML_Q8_0:
        b = raw.reshape(-1, 34)
        d = _f16(b[:, :2])
        w = (b[:, 2:].view(np.int8).astype(np.float32) * d[:, None]).ravel()
        return w, np.repeat(d, 32)
    if t.type == GGML_Q6_K:
        b = raw.reshape(-1, 210)
        ql, qh, sc = b[:, :128], b[:, 128:192], b[:, 192:208].view(np.int8).astype(np.float32)
        d = _f16(b[:, 208:210])
        y = np.empty((len(b), 256), np.float32); st = np.empty_like(y)
        for half in range(2):
            l_ = ql[:, 64 * half:64 * half + 64]; h_ = qh[:, 32 * half:32 * half + 32]
            s_ = sc[:, 8 * half:8 * half + 8]
            q1 = (l_[:, :32] & 15) | (((h_ >> 0) & 3) << 4)
            q2 = (l_[:, 32:] & 15) | (((h_ >> 2) & 3) << 4)
            q3 = (l_[:, :32] >> 4) | (((h_ >> 4) & 3) << 4)
            q4 = (l_[:, 32:] >> 4) | (((h_ >> 6) & 3) << 4)
            for k, qq in enumerate((q1, q2, q3, q4)):
                for sub in range(2):                       # 16 weights share one scale
                    scale = d * s_[:, 2 * k + sub]
                    sl = slice(128 * half + 32 * k + 16 * sub, 128 * half + 32 * k + 16 * sub + 16)
                    y[:, sl] = scale[:, None] * (qq[:, 16 * sub:16 * sub + 16].astype(np.float32) - 32)
                    st[:, sl] = np.abs(scale)[:, None]
        return y.ravel(), st.ravel()
    if t.type == GGML_Q4_K:
        b = raw.reshape(-1, 144)
        d, dmin = _f16(b[:, 0:2]), _f16(b[:, 2:4])
        s = b[:, 4:16].astype(np.int32); qs = b[:, 16:144]
        sc = np.empty((len(b), 8), np.int32); mn = np.empty_like(sc)
        sc[:, :4] = s[:, :4] & 63; mn[:, :4] = s[:, 4:8] & 63
        sc[:, 4:] = (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)
        mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
        y = np.empty((len(b), 256), np.float32); st = np.empty_like(y)
        for c in range(4):
            q = qs[:, 32 * c:32 * c + 32]
            for half, qq in enumerate((q & 15, q >> 4)):
                j = 2 * c + half
                dd = d * sc[:, j]; mm = dmin * mn[:, j]
                y[:, 64 * c + 32 * half:64 * c + 32 * half + 32] = dd[:, None] * qq.astype(np.float32) - mm[:, None]
                st[:, 64 * c + 32 * half:64 * c + 32 * half + 32] = np.abs(dd)[:, None]
        return y.ravel(), st.ravel()
    raise NotImplementedError(f"reference type {t.type_name}")


# ---------------------------------------------------------------- predictor
def predict(w_ref, step, d_bits_actual=None, coef=None):
    """Block-wise prediction. w_ref, step: (nb, 32). Returns d_pred bits (nb,) and helpers."""
    if coef is not None:
        w_ref = w_ref * np.float32(coef)
    amax = np.abs(w_ref).max(1)
    d_pred = (amax / np.float32(127)).astype(np.float16).view(np.uint16).astype(np.int32)
    return w_ref, d_pred


def weight_pred(w_ref, step, dbits):
    d = dbits.astype(np.uint16).view(np.float16).astype(np.float32)
    inv = np.where(d > 0, np.float32(1) / np.maximum(d, np.float32(1e-30)), np.float32(0))
    pred = np.clip(np.rint(w_ref * inv[:, None]), -127, 127).astype(np.int32)
    # reference precision relative to the target step: log2(step/d) in half-octaves
    e = np.log2(np.maximum(step * inv[:, None], 1e-6))
    ebin = np.clip(np.floor(e * 2) + 6, 0, N_E - 1).astype(np.int32)
    mbin = np.minimum(np.abs(pred) // 32, N_M - 1)
    return pred, ebin * N_M + mbin


def sha(b):
    return hashlib.sha256(b).hexdigest()


class RefCodec:
    """Target Q8_0 tensors coded against one or more reference GGUFs."""
    name = "REF"
    MIX = False            # least-squares mix of two references (M16)

    def __init__(self, ref_paths):
        self.ref_paths = ref_paths if isinstance(ref_paths, list) else [ref_paths]
        self._refs = None

    def refs(self):
        if self._refs is None:
            out = []
            for p in self.ref_paths:
                raw = open(p, "rb").read()
                out.append((GGUF.from_bytes(raw), sha(raw)))
            self._refs = out
        return self._refs

    def _ref_weights(self, name, n):
        """Reference (w, step) for tensor `name`, combined over references; None if missing."""
        found = []
        for g, _ in self.refs():
            t = next((x for x in g.tensors if x.name == name), None)
            if t is not None and int(np.prod(t.shape)) == n:
                found.append(dequant(g, t))
        if not found:
            return None
        return found

    # ------------------------------------------------------------
    def _tensor_streams(self, refs_w, b, coef):
        nb = len(b)
        if self.MIX and len(refs_w) == 2:
            w = (refs_w[0][0] * coef[0] + refs_w[1][0] * coef[1]).reshape(nb, 32)
            step = np.minimum(refs_w[0][1], refs_w[1][1]).reshape(nb, 32)
        else:
            w = refs_w[0][0].reshape(nb, 32); step = refs_w[0][1].reshape(nb, 32)
        _, d_pred = predict(w, step)
        return w, step, d_pred

    def compress(self, data):
        g = GGUF.from_bytes(data)
        ch = {"refs": json.dumps([s for _, s in self.refs()]).encode()}
        layout, dsyms, desc, rsyms, rctx, tabs_meta = [], [], [], [], [], []
        enc_d, enc_r, tables = [], [], []
        pos = 0
        rest = []
        coefs = []
        for t in sorted(g.tensors, key=lambda t: t.offset):
            if t.type != GGML_Q8_0:
                continue
            n = int(np.prod(t.shape))
            refs_w = self._ref_weights(t.name, n)
            if refs_w is None:
                continue
            rest.append(data[pos:t.offset]); pos = t.offset + t.nbytes
            b = g.raw(t).reshape(-1, 34)
            q = b[:, 2:].view(np.int8).astype(np.int32)
            dbits = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8))
            coef = None
            if self.MIX and len(refs_w) == 2:
                tgt = (q.astype(np.float32) * dbits.astype(np.uint16).view(np.float16).astype(np.float32)[:, None]).ravel()
                a0, a1 = refs_w[0][0].astype(np.float64), refs_w[1][0].astype(np.float64)
                ata = np.array([[a0 @ a0, a0 @ a1], [a0 @ a1, a1 @ a1]])        # normal equations
                aty = np.array([a0 @ tgt, a1 @ tgt])
                coef = np.linalg.solve(ata, aty).astype(np.float32)
            coefs.append(None if coef is None else coef.tolist())
            w, step, d_pred = self._tensor_streams(refs_w, b, coef)
            # scale residual
            delta = dbits - d_pred
            esc = np.abs(delta) >= 127
            dsym = np.where(esc, 0, delta + 128).astype(np.uint8)
            f = rans.build_freqs(dsym, np.zeros(len(dsym), np.int32), 1)
            enc_d.append(rans.encode(dsym, np.zeros(len(dsym), np.int32), f)); tables.append(f)
            desc.append(dbits[esc].astype(np.uint16).tobytes())
            # weight residual
            pred, ctx = weight_pred(w, step, dbits)
            r = ((q - pred) & 255).astype(np.uint8).ravel()
            ctx = ctx.ravel()
            f = rans.build_freqs(r, ctx, N_E * N_M)
            enc_r.append(rans.encode(r, ctx, f)); tables.append(f)
            layout.append([t.name, t.offset, t.nbytes])
        rest.append(data[pos:])
        ref_hdr = b"".join(GGUF_header(gg) for gg, _ in self.refs()[:1])
        ch["layout"] = zstd_c(json.dumps({"size": len(data), "t": layout, "coef": coefs}).encode())
        ch["rest"] = rest_c(b"".join(rest), ref_hdr)
        ch["tables"] = zstd_c(b"".join(f.astype(np.uint16).tobytes() for f in tables), 19)
        ch["d"] = pack_list_(enc_d)
        ch["desc"] = zstd_c(pack_list_(desc))
        ch["r"] = pack_list_(enc_r)
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        if json.loads(ch["refs"]) != [s for _, s in self.refs()]:
            raise ValueError("reference model differs from the one used for compression")
        meta = json.loads(zstd_d(ch["layout"]))
        ref_hdr = b"".join(GGUF_header(gg) for gg, _ in self.refs()[:1])
        rest = rest_d(ch["rest"], ref_hdr)
        tabs = np.frombuffer(zstd_d(ch["tables"]), np.uint16)
        enc_d, desc, enc_r = unpack_list_(ch["d"]), unpack_list_(zstd_d(ch["desc"])), unpack_list_(ch["r"])
        out = np.empty(meta["size"], np.uint8)
        tp = 0; pos = rpos = 0
        for i, (name, off, nbytes) in enumerate(meta["t"]):
            nb = nbytes // 34
            refs_w = self._ref_weights(name, nb * 32)
            coef = None if meta["coef"][i] is None else np.array(meta["coef"][i], np.float32)
            w, step, d_pred = self._tensor_streams(refs_w, np.empty((nb, 34), np.uint8), coef)
            f = tabs[tp:tp + 256].reshape(1, 256); tp += 256
            dsym = rans.decode(enc_d[i], np.zeros(nb, np.int32), f).astype(np.int32)
            dbits = d_pred + dsym - 128
            esc = dsym == 0
            dbits[esc] = np.frombuffer(desc[i], np.uint16)
            pred, ctx = weight_pred(w, step, dbits)
            f = tabs[tp:tp + 256 * N_E * N_M].reshape(N_E * N_M, 256); tp += 256 * N_E * N_M
            r = rans.decode(enc_r[i], ctx.ravel(), f).astype(np.int32).reshape(nb, 32)
            q = ((pred + r + 128) & 255) - 128
            b = np.empty((nb, 34), np.uint8)
            b[:, :2] = dbits.astype(np.uint16).view(np.uint8).reshape(nb, 2)
            b[:, 2:] = q.astype(np.int8).view(np.uint8)
            gap = off - pos
            out[pos:off] = np.frombuffer(rest, np.uint8, gap, rpos); rpos += gap
            out[off:off + nbytes] = b.ravel(); pos = off + nbytes
        out[pos:] = np.frombuffer(rest, np.uint8, len(rest) - rpos, rpos)
        return out


# ---------------------------------------------------------------- header / rest vs reference
def GGUF_header(g):
    return g.data[:g.data_start]


def rest_c(rest, ref_hdr):
    """zstd with the reference header as raw-content dictionary (metadata is mostly shared)."""
    import zstandard
    d = zstandard.ZstdCompressionDict(ref_hdr, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    a = zstandard.ZstdCompressor(level=19, dict_data=d).compress(rest)
    b = xz_c(rest)
    return b"d" + a if len(a) < len(b) else b"x" + b


def rest_d(buf, ref_hdr):
    import zstandard
    if buf[:1] == b"x":
        return xz_d(buf[1:])
    d = zstandard.ZstdCompressionDict(ref_hdr, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    return zstandard.ZstdDecompressor(dict_data=d).decompress(buf[1:], max_output_size=1 << 30)


def pack_list_(bufs):
    return struct.pack("<I", len(bufs)) + struct.pack(f"<{len(bufs)}Q", *map(len, bufs)) + b"".join(bufs)


def unpack_list_(buf):
    (n,) = struct.unpack_from("<I", buf)
    lens = struct.unpack_from(f"<{n}Q", buf, 4)
    pos, out = 4 + 8 * n, []
    for ln in lens:
        out.append(buf[pos:pos + ln]); pos += ln
    return out


class M12RefXor(RefCodec):
    """Generic binary delta (what rsync/xdelta-style tools do): tensor bytes XOR the reference
    tensor with the same name, then zstd -19. No knowledge of the Q8_0 format."""
    name = "M12 ref=Base Q8_0, XOR delta + zstd"

    def _xor(self, data):
        g = GGUF.from_bytes(data)
        rg = self.refs()[0][0]
        out = np.frombuffer(data, np.uint8).copy()
        for t in g.tensors:
            r = next((x for x in rg.tensors if x.name == t.name), None)
            if r is not None and r.type == t.type and r.nbytes == t.nbytes:
                out[t.offset:t.offset + t.nbytes] ^= rg.raw(r)
        return out

    def compress(self, data):
        return pack({"refs": json.dumps([s for _, s in self.refs()]).encode(),
                     "x": zstd_c(self._xor(data).tobytes())})

    def decompress(self, buf):
        ch = unpack(buf)
        if json.loads(ch["refs"]) != [s for _, s in self.refs()]:
            raise ValueError("reference model differs from the one used for compression")
        return self._xor(zstd_d(ch["x"]))      # header is untouched, so tensor offsets parse the same


class M13RefBase(RefCodec):
    name = "M13 ref=Base Q8_0, predictive residual + context"


class M14RefQ4(RefCodec):
    name = "M14 ref=Q4_K_M same model, predictive residual + context"


class M15RefBF16(RefCodec):
    name = "M15 ref=BF16 original, exact re-quantisation"


class M16RefMix(RefCodec):
    name = "M16 ref=Base Q8_0 + Q4_K_M, least-squares mix"
    MIX = True
