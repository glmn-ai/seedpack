"""Quantizer-aware residual model, size estimate (no coder yet).

Model: the block's scale fields are coded as XOR with the prediction's (zstd here, as a proxy);
given the TRUE scales, each code is predicted as round(u), u = the seed's position on the code grid,
and the delta (true code - round(u)) is coded with context = (distance of u to the nearest rounding
boundary, seed uncertainty in code steps). Cost = empirical conditional entropy (an adaptive coder
gets within ~1% of it).

usage: qr_model.py QUANT_DIR FILE_PATTERN... [--imatrix FILE]
"""
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cupy as cp
import numpy as np
import zstandard

from seedpack.gguf_reader import GGUF, GGML_Q8_0, GGML_F16
from seedpack.repo3 import M29RepoGPU, _keep_base, _read_file, _gguf_tensors
from seedpack.requant import load_imatrix
from seedpack.seed import bits_per_weight


from qunpack import f16, scale_min_k4, q4_0, q4_k, q6_k  # noqa: E402


TYPES = {2: ("Q4_0", 18, 32, q4_0), 12: ("Q4_K", 144, 256, q4_k), 14: ("Q6_K", 210, 256, q6_k)}

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
imf = sys.argv[sys.argv.index("--imatrix") + 1] if "--imatrix" in sys.argv else None
d, pats = args[0], args[1:]
paths = sorted(glob.glob(os.path.join(d, "*.gguf")))
seed = max(paths, key=lambda p: bits_per_weight(_gguf_tensors(p)))
im = load_imatrix(open(imf, "rb").read()) if imf else {}
m = M29RepoGPU()
m._setup([_keep_base(_read_file(seed))], im)
gq = m.gq
z = zstandard.ZstdCompressor(level=7)
st = defaultdict(lambda: {"bytes": 0, "xor_zstd": 0, "scale_zstd": 0, "ctx": defaultdict(lambda: defaultdict(int))})
for p in [x for x in paths if any(s in os.path.basename(x) for s in pats) and x != seed]:
    g = GGUF(p, light=True)
    for t in g.tensors:
        if t.type not in TYPES:
            continue
        bt = m.by_name.get(t.name)
        if bt is None or bt.type not in (GGML_Q8_0, GGML_F16):
            continue
        name, bs, qk, unpack = TYPES[t.type]
        raw = m.base_dev[bt.offset:bt.offset + bt.nbytes]
        if bt.type == GGML_Q8_0:
            bb = raw.reshape(-1, 34)
            dd = (bb[:, 0].astype(cp.uint16) | (bb[:, 1].astype(cp.uint16) << 8)).view(cp.float16).astype(cp.float32)
            x = (bb[:, 2:].view(cp.int8).astype(cp.float32) * dd[:, None]).ravel()
            h = cp.repeat(cp.abs(dd) / 2, 32).get()
        else:
            x = raw.view(cp.float16).astype(cp.float32)
            h = np.zeros(x.size, np.float32)
        w = im.get(t.name)
        if w is None:
            w = np.ones(t.shape[0], np.float32)
        pred = gq.quantize_tensors(t.type, [(x, t.shape[0])], [w])[0]
        fb = np.frombuffer(g.data, np.uint8, t.nbytes, t.offset).reshape(-1, bs)
        pb = pred.reshape(-1, bs)
        s = st[name]
        s["bytes"] += t.nbytes
        s["xor_zstd"] += len(z.compress(np.bitwise_xor(fb, pb).tobytes()))
        q, a, c, sf = unpack(fb)
        _, _, _, sp = unpack(pb)
        s["scale_zstd"] += len(z.compress(np.bitwise_xor(sf, sp).tobytes()))
        xs = x.get().reshape(-1, qk)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (xs - c) / a
            hu = h.reshape(-1, qk) / np.abs(a)
        u = np.where(np.isfinite(u), u, 0)
        r = np.rint(u)
        delta = np.clip(q - r, -3, 3).astype(np.int64) + 3
        exact = (q - r == delta - 3)
        delta = np.where(exact, delta, 7)                       # escape: |delta| > 3
        marg = np.minimum((np.abs(u - r) * 16).astype(np.int64), 7)        # 0..7: distance to centre
        hub = np.where(np.isfinite(hu), np.minimum(np.ceil(np.log2(np.maximum(hu, 1e-9) * 64)).clip(0, 7), 7), 7).astype(np.int64)
        key = (hub * 8 + marg).ravel()
        cnt = np.bincount(key * 8 + delta.ravel(), minlength=64 * 8).reshape(64, 8)
        for k in range(64):
            for v in range(8):
                if cnt[k, v]:
                    s["ctx"][k][v] += int(cnt[k, v])


def H(ctx):
    bits = 0.0
    for tab in ctx.values():
        n = sum(tab.values())
        bits -= sum(v * np.log2(v / n) for v in tab.values() if v)
    return bits / 8


print(f"seed {os.path.basename(seed)}, imatrix {'yes' if im else 'no'}")
print(f"{'type':6s} {'MB':>7s} {'XOR+zstd':>9s} {'scales':>8s} {'codes H':>8s} {'model':>8s} {'gain':>6s}")
for k, s in st.items():
    hc = H(s["ctx"])
    tot = s["scale_zstd"] + hc
    print(f"{k:6s} {s['bytes']/1e6:7.1f} {s['xor_zstd']/1e6:9.2f} {s['scale_zstd']/1e6:8.2f} {hc/1e6:8.2f} {tot/1e6:8.2f} {s['xor_zstd']/max(tot,1):5.1f}x")
