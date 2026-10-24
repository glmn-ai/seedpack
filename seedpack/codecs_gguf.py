"""Five GGUF compression methods. Each method: compress(bytes) -> bytes, decompress(bytes) -> bytes.

Shared container: a flat list of named chunks. For the structure-aware methods (2-5)
the file is split into
  * layout  - positions of Q8_0 tensors (json, zstd)
  * rest    - every byte outside Q8_0 tensor data: header, metadata, F32 tensors, padding (xz)
  * Q8_0 tensor data, coded by the method itself.
A Q8_0 block is 34 bytes: fp16 scale d (lo byte, hi byte) + 32 x int8 qs.
"""
import json
import lzma
import struct
import subprocess
from collections import defaultdict

import numpy as np
import zstandard

from . import rans
from .gguf_reader import GGUF, GGML_Q8_0

XZ_EXE = "xz"


# ---------------------------------------------------------------- container / generic codecs
def nbytes(data) -> int:
    """size of a bytes-like object or of a list of them (pack_parts)"""
    if isinstance(data, list):
        return sum(nbytes(d) for d in data)
    return memoryview(data).nbytes


def pack_parts(chunks: dict) -> list:
    """pack() as a list of buffers, never joined: a value may itself be such a list (a big
    stream in pieces), which is spliced in, so gigabytes are not copied to be framed"""
    out = [struct.pack("<I", len(chunks))]
    for name, data in chunks.items():
        nb = name.encode()
        out += [struct.pack("<H", len(nb)), nb, struct.pack("<Q", nbytes(data))]
        out += data if isinstance(data, list) else [data]
    return out


def pack(chunks: dict) -> bytes:
    return b"".join(pack_parts(chunks))


def unpack(buf: bytes) -> dict:
    """-> {name: slice of buf}; with a memoryview the slices share its memory (no copies)"""
    (n,), pos, chunks = struct.unpack_from("<I", buf), 4, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", buf, pos); pos += 2
        name = bytes(buf[pos:pos + ln]).decode(); pos += ln
        (dl,) = struct.unpack_from("<Q", buf, pos); pos += 8
        chunks[name] = buf[pos:pos + dl]; pos += dl
    return chunks


def zstd_c(data: bytes, level=19) -> bytes:
    params = zstandard.ZstdCompressionParameters.from_level(level, window_log=27, enable_ldm=True, threads=-1)
    return zstandard.ZstdCompressor(compression_params=params).compress(data)


def zstd_d(data: bytes) -> bytes:
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 34)


def xz_c(data: bytes) -> bytes:
    if len(data) < (8 << 20):
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    return subprocess.run([XZ_EXE, "-9e", "-T0", "-c"], input=data, capture_output=True, check=True).stdout


def xz_d(data: bytes) -> bytes:
    if len(data) < (8 << 20):
        return lzma.decompress(data)
    return subprocess.run([XZ_EXE, "-d", "-T0", "-c"], input=data, capture_output=True, check=True).stdout


def best_of(data: bytes) -> bytes:
    """xz vs zstd, first byte tags the winner."""
    a, b = xz_c(data), zstd_c(data)
    return b"x" + a if len(a) <= len(b) else b"z" + b


def best_of_d(data: bytes) -> bytes:
    return xz_d(data[1:]) if data[:1] == b"x" else zstd_d(data[1:])


def pack_list(bufs) -> bytes:
    return struct.pack("<I", len(bufs)) + struct.pack(f"<{len(bufs)}Q", *map(len, bufs)) + b"".join(bufs)


def unpack_list(buf: bytes):
    (n,) = struct.unpack_from("<I", buf)
    lens = struct.unpack_from(f"<{n}Q", buf, 4)
    pos, out = 4 + 8 * n, []
    for ln in lens:
        out.append(buf[pos:pos + ln]); pos += ln
    return out


def tables_c(tables) -> bytes:
    """Serialise a list of uint16 frequency tables (shapes are stored too)."""
    shapes = np.array([t.shape for t in tables], np.uint32).ravel().tobytes()
    body = b"".join(t.astype(np.uint16).tobytes() for t in tables)
    return zstd_c(struct.pack("<I", len(tables)) + shapes + body, 22)


def tables_d(buf: bytes):
    raw = zstd_d(buf)
    (n,) = struct.unpack_from("<I", raw)
    shapes = np.frombuffer(raw, np.uint32, 2 * n, 4).reshape(n, 2)
    pos, out = 4 + 8 * n, []
    for r, c in shapes:
        out.append(np.frombuffer(raw, np.uint16, r * c, pos).reshape(r, c)); pos += 2 * r * c
    return out


# ---------------------------------------------------------------- layout split shared by methods 2-5
def split(data: bytes):
    g = GGUF.from_bytes(data)
    q = [t for t in g.tensors if t.type == GGML_Q8_0]
    q.sort(key=lambda t: t.offset)
    layout = {"size": len(data), "t": [[t.offset, t.nbytes, t.shape[0], group_of(t.name)] for t in q]}
    rest, pos = [], 0
    for t in q:
        rest.append(data[pos:t.offset]); pos = t.offset + t.nbytes
    rest.append(data[pos:])
    blocks = [np.frombuffer(data, np.uint8, t.nbytes, t.offset).reshape(-1, 34) for t in q]
    return layout, b"".join(rest), blocks


def join(layout, rest: bytes, blocks) -> bytes:
    out = bytearray(layout["size"])
    pos = rpos = 0
    for (off, nbytes, _, _), b in zip(layout["t"], blocks):
        gap = off - pos
        out[pos:off] = rest[rpos:rpos + gap]; rpos += gap
        out[off:off + nbytes] = b.tobytes(); pos = off + nbytes
    out[pos:] = rest[rpos:]
    return bytes(out)


def group_of(name: str) -> str:
    # blk.13.ffn_up.weight -> ffn_up ; token_embd.weight -> token_embd
    parts = name.split(".")
    return parts[2] if parts[0] == "blk" else parts[0]


def common_c(layout, rest) -> dict:
    return {"layout": zstd_c(json.dumps(layout).encode()), "rest": xz_c(rest)}


def common_d(ch):
    return json.loads(zstd_d(ch["layout"])), xz_d(ch["rest"])


# ================================================================ Method 1: generic whole-file
class M1Generic:
    name = "M1 generic xz -9e (whole file)"

    def compress(self, data):
        return xz_c(data)

    def decompress(self, buf):
        return xz_d(buf)


class M1Zstd:
    name = "M1b generic zstd -19 --long (whole file)"

    def compress(self, data):
        return zstd_c(data)

    def decompress(self, buf):
        return zstd_d(buf)


# ================================================================ Method 2: stream split + LZ
class M2Split:
    name = "M2 stream split (scale lo/hi | qs) + xz/zstd"

    def compress(self, data):
        layout, rest, blocks = split(data)
        allb = np.concatenate(blocks)
        ch = common_c(layout, rest)
        ch["d_lo"] = best_of(allb[:, 0].tobytes())
        ch["d_hi"] = best_of(allb[:, 1].tobytes())
        ch["qs"] = best_of(allb[:, 2:].tobytes())
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        layout, rest = common_d(ch)
        n = sum(t[1] for t in layout["t"]) // 34
        allb = np.empty((n, 34), np.uint8)
        allb[:, 0] = np.frombuffer(best_of_d(ch["d_lo"]), np.uint8)
        allb[:, 1] = np.frombuffer(best_of_d(ch["d_hi"]), np.uint8)
        allb[:, 2:] = np.frombuffer(best_of_d(ch["qs"]), np.uint8).reshape(n, 32)
        return join(layout, rest, split_blocks(allb, layout))


def split_blocks(allb, layout):
    out, pos = [], 0
    for t in layout["t"]:
        nb = t[1] // 34
        out.append(allb[pos:pos + nb]); pos += nb
    return out


# ================================================================ Method 3: order-0 rANS per tensor
class M3Order0:
    name = "M3 split + static order-0 rANS per tensor"

    def compress(self, data):
        layout, rest, blocks = split(data)
        ch = common_c(layout, rest)
        tabs, bufs = [], []
        for b in blocks:
            lo, hi, qs = b[:, 0], b[:, 1], b[:, 2:].ravel()
            z = np.zeros(len(qs), np.uint8)
            for sym, ctx, nctx in ((hi, z[:len(hi)], 1), (lo, hi, 256), (qs, z, 1)):
                f = rans.build_freqs(sym, ctx, nctx)
                tabs.append(f); bufs.append(rans.encode(sym, ctx, f))
        ch["tables"] = tables_c(tabs)
        ch["streams"] = pack_list(bufs)
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        layout, rest = common_d(ch)
        tabs, bufs = tables_d(ch["tables"]), unpack_list(ch["streams"])
        blocks = []
        for i, t in enumerate(layout["t"]):
            nb = t[1] // 34
            b = np.empty((nb, 34), np.uint8)
            b[:, 1] = rans.decode(bufs[3 * i], np.zeros(nb, np.uint8), tabs[3 * i])
            b[:, 0] = rans.decode(bufs[3 * i + 1], b[:, 1].copy(), tabs[3 * i + 1])
            b[:, 2:] = rans.decode(bufs[3 * i + 2], np.zeros(nb * 32, np.uint8), tabs[3 * i + 2]).reshape(nb, 32)
            blocks.append(b)
        return join(layout, rest, blocks)


# ================================================================ Method 4: context-modelling rANS
N_COL = 16        # column (input-channel) classes
N_SB = 16         # scale buckets (quarter octave around tensor median)


def scale_bucket(lo, hi):
    bits = lo.astype(np.int32) | (hi.astype(np.int32) << 8)        # positive fp16: monotonic in value
    med = int(np.median(bits))
    return np.clip((bits - med) // 256 + N_SB // 2, 0, N_SB - 1).astype(np.uint16)


def column_classes(qs_rows):
    """Class of each input channel by RMS of its quantised values (side info, stored)."""
    rms = np.sqrt((qs_rows.astype(np.float32) ** 2).mean(0))
    edges = np.quantile(rms, np.linspace(0, 1, N_COL + 1)[1:-1])
    return np.searchsorted(edges, rms).astype(np.uint8)


class M4Context:
    """Scales: hi byte order-1 (previous block in row), lo byte | hi.
    Weights: context = scale bucket x column class; tables shared per tensor group."""
    name = "M4 context rANS (scale-bucket x channel-class, order-1 scales)"
    maxtrick = False

    def _streams(self, b, rowlen, colmap):
        """Yield (stream id, sym, ctx) for one tensor, in decode order."""
        nb_row = rowlen // 32
        lo, hi = b[:, 0], b[:, 1]
        yield "hi", hi, rans.o1_context(hi, nb_row)
        yield "lo", lo, hi.astype(np.uint16)
        sb = scale_bucket(lo, hi)
        qs = b[:, 2:]
        col = colmap.reshape(-1, 32)                       # (nb_row, 32)
        colctx = np.tile(col, (len(b) // nb_row, 1))       # (nblocks, 32)
        ctx = sb[:, None] * N_COL + colctx
        if not self.maxtrick:
            yield "qs", qs.ravel(), ctx.ravel()
            return
        q = qs.view(np.int8)
        pos = np.argmax(np.abs(q.astype(np.int16)), 1)
        idx = np.arange(len(b))
        yield "pos", pos.astype(np.uint8), np.zeros(len(b), np.uint16)
        yield "mx", qs[idx, pos], np.zeros(len(b), np.uint16)
        keep = np.ones(qs.shape, bool); keep[idx, pos] = False
        yield "qs", qs[keep], ctx[keep]

    NCTX = {"hi": 256, "lo": 256, "qs": N_SB * N_COL, "pos": 1, "mx": 1}

    def compress(self, data):
        layout, rest, blocks = split(data)
        ch = common_c(layout, rest)
        colmaps = [column_classes(b[:, 2:].view(np.int8).reshape(-1, t[2])) for b, t in zip(blocks, layout["t"])]
        # pass 1: counts per (group, stream)
        counts = defaultdict(int)
        for b, t, cm in zip(blocks, layout["t"], colmaps):
            for sid, sym, ctx in self._streams(b, t[2], cm):
                n = self.NCTX[sid]
                counts[(t[3], sid)] = counts[(t[3], sid)] + np.bincount(
                    ctx.astype(np.int64) * 256 + sym, minlength=n * 256).reshape(n, 256)
        keys = sorted(counts)
        freqs = {k: rans.normalize(counts[k]) for k in keys}
        # pass 2: encode
        bufs = []
        for b, t, cm in zip(blocks, layout["t"], colmaps):
            for sid, sym, ctx in self._streams(b, t[2], cm):
                bufs.append(rans.encode(sym, ctx, freqs[(t[3], sid)]))
        ch["tables"] = tables_c([freqs[k] for k in keys])
        ch["tkeys"] = zstd_c(json.dumps(keys).encode())
        ch["colmap"] = zstd_c(np.concatenate(colmaps).tobytes())
        ch["streams"] = pack_list(bufs)
        return pack(ch)

    def decompress(self, buf):
        ch = unpack(buf)
        layout, rest = common_d(ch)
        keys = [tuple(k) for k in json.loads(zstd_d(ch["tkeys"]))]
        freqs = dict(zip(keys, tables_d(ch["tables"])))
        bufs = iter(unpack_list(ch["streams"]))
        cmall = np.frombuffer(zstd_d(ch["colmap"]), np.uint8)
        blocks, cpos = [], 0
        for off, nbytes, rowlen, grp in layout["t"]:
            nb, nb_row = nbytes // 34, rowlen // 32
            colmap = cmall[cpos:cpos + rowlen]; cpos += rowlen
            b = np.empty((nb, 34), np.uint8)
            b[:, 1] = rans.decode_o1(next(bufs), nb, freqs[(grp, "hi")], nb_row)
            b[:, 0] = rans.decode(next(bufs), b[:, 1].astype(np.uint16), freqs[(grp, "lo")])
            sb = scale_bucket(b[:, 0], b[:, 1])
            ctx = sb[:, None] * N_COL + np.tile(colmap.reshape(-1, 32), (nb // nb_row, 1))
            if not self.maxtrick:
                b[:, 2:] = rans.decode(next(bufs), ctx.ravel(), freqs[(grp, "qs")]).reshape(nb, 32)
            else:
                z = np.zeros(nb, np.uint16)
                pos = rans.decode(next(bufs), z, freqs[(grp, "pos")])
                mx = rans.decode(next(bufs), z, freqs[(grp, "mx")])
                idx = np.arange(nb)
                keep = np.ones((nb, 32), bool); keep[idx, pos] = False
                qs = np.empty((nb, 32), np.uint8)
                qs[keep] = rans.decode(next(bufs), ctx[keep], freqs[(grp, "qs")])
                qs[idx, pos] = mx
                b[:, 2:] = qs
            blocks.append(b)
        return join(layout, rest, blocks)


# ================================================================ Method 5: max-element extraction + context rANS
class M5MaxTrick(M4Context):
    """Q8_0 quantiser maps the block's largest |x| to exactly +-127, so every block contains
    one 'forced' value. It is coded separately (position + value) and the remaining
    31 values are coded with the M4 context model, whose tables no longer waste
    probability mass on +-127."""
    name = "M5 max-element extraction + context rANS"
    maxtrick = True


METHODS = [M1Generic(), M1Zstd(), M2Split(), M3Order0(), M4Context(), M5MaxTrick()]
