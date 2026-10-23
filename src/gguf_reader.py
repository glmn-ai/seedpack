"""Minimal GGUF parser: header, metadata, tensor table and raw tensor slices."""
import struct
from dataclasses import dataclass

import numba
import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              29: "IQ1_M", 30: "BF16"}
# (block elements, block bytes)
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2), GGML_Q8_0: (32, 34),
         2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 9: (32, 36),          # Q4_0 Q4_1 Q5_0 Q5_1 Q8_1
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
         16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
         29: (256, 56)}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass
class Tensor:
    name: str
    shape: tuple      # ggml order: shape[0] is the contiguous (row) dimension
    type: int
    offset: int       # absolute file offset
    nbytes: int

    @property
    def type_name(self):
        return TYPE_NAMES.get(self.type, str(self.type))


@numba.njit(cache=True)
def _skip_strings(buf, pos, n):
    for _ in range(n):
        ln = 0
        for k in range(8):
            ln |= np.int64(buf[pos + k]) << (8 * k)
        pos += 8 + ln
    return pos


class GGUF:
    def __init__(self, path=None, data=None, light=False):
        """light: large string arrays (tokenizer vocabulary, merges) are skipped, meta value None"""
        if data is None:
            with open(path, "rb") as f:
                data = f.read()
        self.data = data
        self.light = light
        assert self.data[:4] == b"GGUF", "not a GGUF file"
        self.pos = 4
        self.version = self._u("<I")
        n_tensors = self._u("<Q")
        n_kv = self._u("<Q")
        self.meta = {}
        self.spans = {}           # key -> (start, end) byte span of the value, type tag excluded
        for _ in range(n_kv):
            key = self._str()
            vtype = self._u("<I")
            start = self.pos
            self.meta[key] = self._value(vtype)
            self.spans[key] = (start, self.pos)
        infos = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u("<I")
            shape = tuple(self._u("<Q") for _ in range(nd))
            infos.append((name, shape, self._u("<I"), self._u("<Q")))
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.pos + align - 1) // align * align
        self.tensors = []
        for name, shape, t, off in infos:
            n = int(np.prod(shape))
            be, bb = BLOCK[t]
            self.tensors.append(Tensor(name, shape, t, self.data_start + off, n // be * bb))

    @classmethod
    def from_bytes(cls, data, light=False):
        return cls(data=data, light=light)

    def _u(self, fmt):
        v = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def _str(self):
        n = self._u("<Q")
        s = bytes(self.data[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def _value(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            it, n = self._u("<I"), self._u("<Q")
            if it in _SCALAR:                      # numeric array: one numpy read
                dt = np.dtype(_SCALAR[it])
                v = np.frombuffer(self.data, dt, n, self.pos).tolist()
                self.pos += n * dt.itemsize
                return v
            if it == 8 and self.light and n > 64:
                self.pos = int(_skip_strings(np.frombuffer(self.data, np.uint8), self.pos, n))
                return None
            if it == 8:                            # string array (tokenizer vocabulary): tight loop
                data, pos, out, unpack = self.data, self.pos, [], struct.Struct("<Q").unpack_from
                for _ in range(n):
                    (ln,) = unpack(data, pos); pos += 8
                    out.append(bytes(data[pos:pos + ln]).decode("utf-8", "replace")); pos += ln
                self.pos = pos
                return out
            return [self._value(it) for _ in range(n)]
        return self._u(_SCALAR[t])

    def raw(self, t: Tensor) -> np.ndarray:
        return np.frombuffer(self.data, np.uint8, t.nbytes, t.offset)
