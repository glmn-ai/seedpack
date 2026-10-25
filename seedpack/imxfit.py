"""Recovering the importance matrix a quant was made with, when the repository holds another one.

Unsloth, for one, makes its plain quants (Q4_0, Q4_1, Q8_0) with the importance matrix of a short
calibration run that it does not publish (their GGUF says: 45 chunks), and publishes the one of its
Dynamic quants (1251 chunks). Predicted with the published matrix, such a quant gets the codes
right but half the block scales wrong (the scale search weighs the weights by importance), and
the scales then cost a few % of the tensor.

The importance enters only through the block scales, one column's importance is shared by every
row (thousands of blocks), and the legacy quantisers' scale is a weighted least-squares fit: so it
is fitted back from the quant itself (ls_gram, ls_solve) and checked on other rows (equal_blocks).
Tensors with the same input (gate and up, q / k / v) share their importance: one fit serves them
all, in every file made with that matrix. The decoder only reads the fitted vectors back.
"""
import cupy as cp
import numpy as np

from .requant import QFUN

# Least squares. The legacy quantisers' block scale is the weighted least-squares fit of the
# block to its codes l (Q4_1 / Q5_1: scale and min, unless the min was clamped to 0): with the
# file's codes and fp16 scales, the residual x - v of every block is weighted-orthogonal to l
# (and to 1), weights qw_i sqrt(sigma2 + x_i^2) - one or two linear conditions on the block's
# importances qw per row. Their least-squares solution over many rows (and every tensor with that
# input) is the smallest eigenvector of a 32 x 32 matrix per block of columns: the importance up to
# a factor per block, which the quantisers do not see. The fp16 rounding of the scales is the noise.
LS_TYPES = {2, 3, 6, 7}


def ls_gram(gq, T, X, F):
    """the conditions of rows X (float32 device, rows x cols) and their blocks F (uint8 device) ->
    float64 device (blocks of columns, 32, 32), to be summed over tensors with the same input"""
    _, qk, bs = QFUN[T]
    R, cols = X.shape
    nb = cols // qk
    F = cp.ascontiguousarray(F).reshape(-1)
    v = gq.dequantize(T, F, cols).reshape(R, nb, qk)
    Fb = F.reshape(R, nb, bs)
    d = Fb[:, :, 0:2].copy().view(cp.float16).astype(cp.float32)
    two = T in (3, 7)
    m = Fb[:, :, 2:4].copy().view(cp.float16).astype(cp.float32) if two else cp.zeros_like(d)
    l = cp.rint((v - m) / cp.where(d == 0, 1, d))
    xb = X.reshape(R, nb, qk)
    c = cp.sqrt(((X * X).sum(1) / cols)[:, None, None] + xb * xb)
    res = xb - v
    G = cp.zeros((nb, qk, qk), cp.float64)
    for e in ([c * res * l, c * res * (m != 0)] if two else [c * res * l]):
        e = e / cp.maximum(cp.sqrt((e * e).sum(2, keepdims=True)), 1e-30)
        G += cp.einsum("rbi,rbj->bij", e, e).astype(cp.float64)
    return G


def ls_solve(G, e0):
    """importance (float32 numpy) from summed conditions G, each block of columns scaled to the sum
    of e0's there"""
    nb, qk = G.shape[0], G.shape[1]
    _, V = cp.linalg.eigh(G)
    q = V[:, :, 0]
    q = q * cp.sign(q.sum(1, keepdims=True))
    q = cp.maximum(q, 1e-6 * cp.abs(q).max(1, keepdims=True))
    e0 = cp.asarray(e0, cp.float64).reshape(nb, qk)
    q = q / q.sum(1, keepdims=True) * e0.sum(1, keepdims=True)
    return cp.asnumpy(q.reshape(-1)).astype(np.float32)


def equal_blocks(gq, T, X, F, e):
    """share of the blocks of rows X (float32 device, rows x cols) that importance e (float32 numpy)
    reproduces; F: the file's blocks of those rows (uint8 device)"""
    _, qk, bs = QFUN[T]
    Y = gq.quantize_batch(T, [cp.ascontiguousarray(X)], [np.asarray(e, np.float32)])[0]
    return float((Y.reshape(-1, bs) == cp.ascontiguousarray(F).reshape(-1, bs)).all(-1).mean())
