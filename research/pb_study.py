"""Cost of the 12-bit probability resolution of the rANS tables (R2 counts): coded size for PB=12..16 vs ideal."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from bf16v2 import BF16v2
c = BF16v2(budget=1.5e9); c.compress(open("models/unsloth/Qwen3-0.6B-BF16.gguf", "rb").read())
C = c.last_counts.astype(np.int64); C = C[C.sum(1) > 0]
def norm(c, pb):
    tot = c.sum(1, keepdims=True); S = 1 << pb
    f = np.where(c > 0, np.maximum(1, (c * S + tot // 2) // tot), 0)
    for r in range(len(f)):
        d = S - f[r].sum()
        while d:
            if d > 0: f[r, np.argmax(f[r])] += d; d = 0
            else:
                j = np.argmax(f[r]); f[r, j] -= 1; d += 1
    return f
nz = C > 0
ideal = -(C[nz] * np.log2((C / C.sum(1, keepdims=True))[nz])).sum() / 8e6
print(f"ideal {ideal:.3f} MB")
for pb in (12, 13, 14, 15, 16):
    f = norm(C, pb)
    print(f"PB={pb}: {-(C[nz] * np.log2(f[nz] / (1 << pb))).sum() / 8e6:.3f} MB")
