"""Method 23: self-contained archive of a model *repository* (several files of one model).

The BF16 original is coded with M22; every quantised file of the same model is then coded
relative to the BF16 weights that are already inside the same archive (refcodec predictor:
re-quantise + context-coded correction). The receiver needs nothing but the archive.
"""
import struct

from .codecs_gguf import pack, unpack
from .bf16codec import M22BF16
from .gguf_reader import GGUF
from .refcodec import RefCodec, sha


class _InArchiveRef(RefCodec):
    def __init__(self, ref_bytes):
        super().__init__([])
        self._refs = [(GGUF.from_bytes(ref_bytes), sha(ref_bytes))]


class M23Bundle:
    name = "M23 model-repo bundle: BF16 (M22) + quants coded against it"

    def compress_files(self, base: bytes, others: list) -> bytes:
        ch = {"base": M22BF16().compress(base)}
        rc = _InArchiveRef(base)
        for i, d in enumerate(others):
            ch[f"q{i}"] = rc.compress(d)
        return pack(ch)

    def decompress_files(self, buf: bytes):
        ch = unpack(buf)
        base = M22BF16().decompress(ch["base"]).tobytes()
        rc = _InArchiveRef(base)
        others = [rc.decompress(ch[k]) for k in sorted(k for k in ch if k.startswith("q"))]
        return base, others
