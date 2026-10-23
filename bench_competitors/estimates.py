"""Size estimates for systems that cannot be run on GGUF files directly.

dfloat11 FILE...   DFloat11 format: sign + mantissa kept (8 bits / weight), exponent Huffman-coded
                   with one optimal code per tensor (optimistic: real DF11 adds lookup tables and
                   padding); non-BF16 bytes stored as is.
xet DIR|FILE...    Hugging Face Xet storage: content-defined chunks (gear hash, ~64 KiB target,
                   8-128 KiB), deduplicated across all files, every unique chunk compressed
                   with the better of LZ4 and byte-grouping-4 + LZ4, as Xet does.
"""
import heapq
import json
import os
import sys
import time

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gguf_reader import GGUF, GGML_BF16  # noqa: E402


def huffman_bits(counts):
    """total bits of an optimal prefix code for the histogram"""
    c = [int(x) for x in counts if x]
    if len(c) == 1:
        return c[0]
    heapq.heapify(c)
    total = 0
    while len(c) > 1:
        a, b = heapq.heappop(c), heapq.heappop(c)
        total += a + b
        heapq.heappush(c, a + b)
    return total


def dfloat11(path):
    g = GGUF(path, light=True)
    raw = np.frombuffer(g.data, np.uint8)
    bits, bf16_bytes = 0, 0
    for t in g.tensors:
        if t.type != GGML_BF16:
            continue
        u = raw[t.offset:t.offset + t.nbytes].view(np.uint16)
        exp = (u >> 7) & 0xFF
        bits += 8 * u.size + huffman_bits(np.bincount(exp, minlength=256))
        bf16_bytes += t.nbytes
    size = bits // 8 + (len(g.data) - bf16_bytes)
    return {"file": os.path.basename(path), "size": len(g.data), "dfloat11_est": size, "ratio": size / len(g.data)}


# ---------------------------------------------------------------------------------- Xet
_GEAR = np.random.default_rng(0x5A17).integers(0, 2 ** 63, 256, dtype=np.uint64)
MIN_C, AVG_C, MAX_C = 8 << 10, 64 << 10, 128 << 10
_MASK = np.uint64((AVG_C - 1) << 48)        # 16 bits -> 64 KiB average


@numba.njit(cache=True)
def _cuts(buf, gear, mn, mx, mask):
    out = []
    n = len(buf)
    start = 0
    h = np.uint64(0)
    i = 0
    while i < n:
        h = (h << np.uint64(1)) + gear[buf[i]]
        ln = i - start + 1
        if (ln >= mn and (h & mask) == 0) or ln >= mx:
            out.append(i + 1)
            start = i + 1
            h = np.uint64(0)
        i += 1
    if start < n:
        out.append(n)
    return out


def xet(paths):
    import hashlib
    try:
        import lz4.block as lz4b

        def comp(b):
            # Xet picks per block between LZ4 and "byte grouping 4" + LZ4 (for float data)
            a = np.frombuffer(b, np.uint8)
            n4 = len(a) // 4 * 4
            g = np.concatenate([a[:n4].reshape(-1, 4).T.reshape(-1), a[n4:]]).tobytes()
            return min(len(lz4b.compress(b, store_size=False)), len(lz4b.compress(g, store_size=False)))
        cname = "lz4 / bg4+lz4"
    except ImportError:
        import zstandard
        z = zstandard.ZstdCompressor(level=1)
        comp, cname = z.compress, "zstd-1"
    seen, total, unique, stored = set(), 0, 0, 0
    t0 = time.perf_counter()
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        buf = np.frombuffer(data, np.uint8)
        prev = 0
        for c in _cuts(buf, _GEAR, MIN_C, MAX_C, _MASK):
            chunk = data[prev:c]
            prev = c
            h = hashlib.blake2b(chunk, digest_size=16).digest()
            total += len(chunk)
            if h in seen:
                continue
            seen.add(h)
            unique += len(chunk)
            c = comp(chunk)
            stored += min(c if isinstance(c, int) else len(c), len(chunk))
    return {"files": len(paths), "size": total, "after_dedup": unique, "xet_est": stored,
            "ratio": stored / total, "chunk_codec": cname, "seconds": time.perf_counter() - t0}


def expand(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            for root, dirs, fs in os.walk(a):
                dirs[:] = [d for d in dirs if d not in (".git", ".cache")]
                out += [os.path.join(root, f) for f in sorted(fs)]
        else:
            out.append(a)
    return out


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "dfloat11":
        print(json.dumps([dfloat11(p) for p in args], indent=1))
    else:
        print(json.dumps(xet(expand(args)), indent=1))
