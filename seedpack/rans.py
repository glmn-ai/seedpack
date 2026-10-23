"""Static context-mixed rANS coder (numba).

Every symbol i is coded with the frequency table of its context ctx[i].
Tables are built in a first pass (two-pass / semi-static coding) and stored
in the archive, so the decoder needs the same ctx sequence: contexts must be
computable from already decoded data.
"""
import numpy as np
from numba import njit

PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 23


def build_freqs(sym: np.ndarray, ctx: np.ndarray, n_ctx: int, n_sym: int = 256) -> np.ndarray:
    """Counts per (ctx, sym) normalised so that each row sums to PROB_SCALE."""
    counts = np.bincount(ctx.astype(np.int64) * n_sym + sym, minlength=n_ctx * n_sym)
    return normalize(counts.reshape(n_ctx, n_sym))


def normalize(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    freqs = np.zeros(counts.shape, np.int64)
    for r in range(counts.shape[0]):
        c = counts[r]
        tot = c.sum()
        if tot == 0:
            continue
        f = np.where(c > 0, np.maximum(1, np.round(c * PROB_SCALE / tot)), 0).astype(np.int64)
        diff = PROB_SCALE - f.sum()
        # Fix rounding drift, taking from / giving to the largest entries.
        order = np.argsort(-f)
        i = 0
        while diff != 0:
            j = order[i % len(order)]
            if diff > 0:
                f[j] += 1
                diff -= 1
            elif f[j] > 1:
                f[j] -= 1
                diff += 1
            i += 1
        freqs[r] = f
    return freqs.astype(np.uint16 if PROB_BITS < 16 else np.uint32)


def freq_cost_bits(sym, ctx, freqs) -> float:
    """Ideal code length for the data under the given tables."""
    p = freqs[ctx, sym].astype(np.float64) / PROB_SCALE
    return float(-np.log2(p).sum())


@njit(cache=True)
def _encode(sym, ctx, freqs, out):
    n_ctx, n_sym = freqs.shape
    cum = np.zeros((n_ctx, n_sym + 1), np.uint32)
    for c in range(n_ctx):
        for s in range(n_sym):
            cum[c, s + 1] = cum[c, s] + freqs[c, s]
    x = np.uint32(RANS_L)
    pos = out.shape[0]
    for i in range(sym.shape[0] - 1, -1, -1):
        c = ctx[i]
        s = sym[i]
        f = np.uint32(freqs[c, s])
        x_max = np.uint32(((RANS_L >> PROB_BITS) << 8) * f)
        while x >= x_max:
            pos -= 1
            out[pos] = np.uint8(x & 0xFF)
            x >>= np.uint32(8)
        x = np.uint32(((x // f) << PROB_BITS) + (x % f) + cum[c, s])
    for _ in range(4):
        pos -= 1
        out[pos] = np.uint8(x & 0xFF)
        x >>= np.uint32(8)
    return pos


@njit(cache=True)
def _decode(buf, ctx, freqs, out):
    n_ctx, n_sym = freqs.shape
    cum = np.zeros((n_ctx, n_sym + 1), np.uint32)
    lut = np.zeros((n_ctx, PROB_SCALE), np.uint8)
    for c in range(n_ctx):
        for s in range(n_sym):
            cum[c, s + 1] = cum[c, s] + freqs[c, s]
            for k in range(cum[c, s], cum[c, s + 1]):
                lut[c, k] = s
    pos = 0
    x = np.uint32(0)
    for _ in range(4):
        x = (x << np.uint32(8)) | np.uint32(buf[pos])
        pos += 1
    mask = np.uint32(PROB_SCALE - 1)
    for i in range(out.shape[0]):
        c = ctx[i]
        slot = x & mask
        s = lut[c, slot]
        out[i] = s
        x = np.uint32(freqs[c, s]) * (x >> np.uint32(PROB_BITS)) + slot - cum[c, s]
        while x < RANS_L:
            x = (x << np.uint32(8)) | np.uint32(buf[pos])
            pos += 1
    return pos


@njit(cache=True)
def _decode_o1(buf, freqs, period, out):
    """Decoder for order-1 streams: ctx = previous symbol, reset to 0 every `period` symbols."""
    n_ctx, n_sym = freqs.shape
    cum = np.zeros((n_ctx, n_sym + 1), np.uint32)
    lut = np.zeros((n_ctx, PROB_SCALE), np.uint8)
    for c in range(n_ctx):
        for s in range(n_sym):
            cum[c, s + 1] = cum[c, s] + freqs[c, s]
            for k in range(cum[c, s], cum[c, s + 1]):
                lut[c, k] = s
    pos = 0
    x = np.uint32(0)
    for _ in range(4):
        x = (x << np.uint32(8)) | np.uint32(buf[pos])
        pos += 1
    mask = np.uint32(PROB_SCALE - 1)
    prev = 0
    for i in range(out.shape[0]):
        if i % period == 0:
            prev = 0
        c = prev
        slot = x & mask
        s = lut[c, slot]
        out[i] = s
        prev = s
        x = np.uint32(freqs[c, s]) * (x >> np.uint32(PROB_BITS)) + slot - cum[c, s]
        while x < RANS_L:
            x = (x << np.uint32(8)) | np.uint32(buf[pos])
            pos += 1
    return pos


def o1_context(sym: np.ndarray, period: int) -> np.ndarray:
    ctx = np.empty(sym.shape[0], np.int32)
    ctx[0] = 0
    ctx[1:] = sym[:-1]
    ctx[::period] = 0
    return ctx


def decode_o1(buf: bytes, n: int, freqs: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(n, np.uint8)
    _decode_o1(np.frombuffer(buf, np.uint8), freqs, period, out)
    return out


def encode(sym: np.ndarray, ctx: np.ndarray, freqs: np.ndarray) -> bytes:
    out = np.empty(sym.shape[0] * 2 + 16, np.uint8)
    pos = _encode(np.ascontiguousarray(sym, np.uint8), np.ascontiguousarray(ctx), freqs, out)
    return out[pos:].tobytes()


def decode(buf: bytes, ctx: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    out = np.empty(ctx.shape[0], np.uint8)
    _decode(np.frombuffer(buf, np.uint8), np.ascontiguousarray(ctx), freqs, out)
    return out
