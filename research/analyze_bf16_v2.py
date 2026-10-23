"""Table sharing study: coded size (cross entropy with 12-bit normalised tables) + table cost
(zstd-19 of the used uint16 table rows, as M24 stores them) for different context resolutions
and table groupings (per tensor / per tensor role / whole model).

usage: analyze_bf16_v2.py [gguf]
"""
import sys, os, time, re
import numpy as np
import zstandard
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gguf_reader import GGUF, GGML_BF16

path = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth/Qwen3-0.6B-BF16.gguf"
g = GGUF(path)
ts = sorted((t for t in g.tensors if t.type == GGML_BF16 and len(t.shape) == 2), key=lambda t: t.offset)
PB = 12


def normalize(c):
    """numpy port of gpu2.normalize_gpu semantics (approx.: largest-entry correction)"""
    tot = c.sum(1, keepdims=True)
    f = np.where(c > 0, np.maximum(1, (c * (1 << PB) + tot // 2) // np.maximum(tot, 1)), 0)
    for r in range(len(f)):
        d = (1 << PB) - f[r].sum()
        while d != 0:
            if d > 0:
                f[r, np.argmax(f[r])] += d; d = 0
            else:
                order = np.argsort(-f[r])
                for j in order:
                    if d == 0: break
                    if f[r, j] > 1: f[r, j] -= 1; d += 1
    return f


def cost(counts):
    """counts (nctx, nsym) -> (coded bits, table bytes)"""
    used = counts.sum(1) > 0
    c = counts[used].astype(np.int64)
    f = normalize(c)
    nz = c > 0
    bits = -(c[nz] * np.log2(f[nz] / (1 << PB))).sum()
    tab = np.zeros((len(f), 256), np.uint16); tab[:, :f.shape[1]] = f
    raw = np.flatnonzero(used).astype(np.uint32).tobytes() + tab.tobytes()
    return bits, len(zstandard.ZstdCompressor(level=19).compress(raw))


def role(name):
    return re.sub(r"blk\.\d+\.", "", name)


CONF = {}          # name -> list of per-tensor dict(stream -> counts)
for K in (4, 8):
    CONF[K] = []
t0 = time.time()
for t in ts:
    rowlen = t.shape[0]
    u = np.frombuffer(g.data, np.uint16, t.nbytes // 2, t.offset).reshape(-1, rowlen)
    w = (u.astype(np.uint32) << 16).view(np.float32)
    rf = np.log2(np.sqrt((w.astype(np.float64) ** 2).mean(1)) + 1e-30)
    wn = w / np.exp2(rf)[:, None]
    cf = np.log2(np.sqrt((wn.astype(np.float64) ** 2).mean(0)) + 1e-30)
    for K in (4, 8):
        rfi_all = np.rint(rf * K).astype(np.int64); cfi = np.rint(cf * K).astype(np.int64)
        bias = wn.mean(0) / np.exp2(cfi / K)
        sc = np.clip(np.floor(bias * 16) + 8, 0, 15).astype(np.int64)
        d = {}
        step = max(1, (8 << 20) // rowlen)
        for r0 in range(0, u.shape[0], step):
            uu = u[r0:r0 + step]
            ex = ((uu >> 7) & 255).astype(np.int64); man = (uu & 127).astype(np.int64); sg = (uu >> 15).astype(np.int64)
            pred = rfi_all[r0:r0 + step, None] + cfi[None, :]
            part = {}
            if K == 4:   # M24 contexts
                eb = np.clip((pred + 108) >> 1, 0, 127)
                rel = np.clip(ex - ((pred + 508) >> 2) + 8, 0, 15)
                part["E24"] = np.bincount((eb * 256 + ex).ravel(), minlength=128 * 256).reshape(128, 256)
                part["M24"] = np.bincount((rel * 128 + man).ravel(), minlength=16 * 128).reshape(16, 128)
            pe = np.clip(pred + 32 * K, 0, 64 * K - 1)
            part["E"] = np.bincount((pe * 256 + ex).ravel(), minlength=64 * K * 256).reshape(64 * K, 256)
            rl = np.clip(K * ex - pred - 127 * K + 8 * K, 0, 16 * K - 1)
            part["M"] = np.bincount((rl * 128 + man).ravel(), minlength=16 * K * 128).reshape(16 * K, 128)
            part["S"] = np.bincount((np.broadcast_to(sc[None, :], uu.shape) * 2 + sg).ravel(), minlength=32).reshape(16, 2)
            for k2, v in part.items():
                d[k2] = d.get(k2, 0) + v
        CONF[K].append(d)
    del w, wn
    print(f"{t.name:32s} {time.time() - t0:.0f}s", flush=True)

N = sum(t.nbytes // 2 for t in ts)
roles = [role(t.name) for t in ts]
print(f"\n{N:,} weights. coded MB + table MB per stream and grouping")
for K in (4, 8):
    for stream in (["E24", "M24"] if K == 4 else []) + ["E", "M", "S"]:
        for grp in ("tensor", "role", "model"):
            keys = list(range(len(ts))) if grp == "tensor" else roles if grp == "role" else [0] * len(ts)
            agg = {}
            for k, d in zip(keys, CONF[K]):
                agg[k] = agg.get(k, 0) + d[stream]
            b = tb = 0
            for c in agg.values():
                x, y = cost(c); b += x; tb += y
            print(f"K={K} {stream:4s} {grp:7s} coded {b / 8e6:9.3f} MB  tables {tb / 1e6:7.3f} MB  total {b / 8e6 + tb / 1e6:9.3f}", flush=True)
