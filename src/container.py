"""The .varc container: a sequence of named chunks with an index at the end, so any chunk can be
read without touching the others (selective extraction) and chunks can be written as soon as
they are produced (the archive is never held in memory as a whole).

Layout:
    "VARC" | u32 format version
    chunk bytes, back to back
    index: zstd(JSON {"chunks": {name: [offset, length, crc32]}, "meta": {...}})
    footer: u64 index offset | u64 index length | b"VARCEND\\0"
"""
import json
import os
import struct
import threading
import zlib

import zstandard

MAGIC = b"VARC"
VERSION = 1
_FOOTER = struct.Struct("<QQ8s")
_END = b"VARCEND\0"


class FormatError(Exception):
    pass


class Writer:
    """Writes to `path`.part and renames it to `path` on close(), so an interrupted run never
    leaves something that looks like a complete archive."""

    def __init__(self, path):
        self.path = path
        self.f = open(path + ".part", "wb")
        self.f.write(MAGIC + struct.pack("<I", VERSION))
        self.chunks = {}

    def put(self, name, data):
        if name in self.chunks:
            raise ValueError(f"duplicate chunk {name}")
        self.chunks[name] = [self.f.tell(), len(data), zlib.crc32(data)]
        self.f.write(data)

    def close(self, meta):
        idx = zstandard.ZstdCompressor(level=19).compress(json.dumps({"chunks": self.chunks, "meta": meta}).encode())
        off = self.f.tell()
        self.f.write(idx)
        self.f.write(_FOOTER.pack(off, len(idx), _END))
        self.f.close()
        os.replace(self.path + ".part", self.path)

    def abort(self):
        self.f.close()
        try:
            os.remove(self.path + ".part")
        except OSError:
            pass


class Reader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        head = self.f.read(8)
        if head[:4] != MAGIC:
            raise FormatError(f"{path}: not a .varc archive")
        if struct.unpack("<I", head[4:])[0] > VERSION:
            raise FormatError(f"{path}: format version {struct.unpack('<I', head[4:])[0]} is newer than this tool")
        self.f.seek(-_FOOTER.size, os.SEEK_END)
        off, n, end = _FOOTER.unpack(self.f.read(_FOOTER.size))
        if end != _END:
            raise FormatError(f"{path}: truncated or damaged archive (no index)")
        self.f.seek(off)
        idx = json.loads(zstandard.ZstdDecompressor().decompress(self.f.read(n)))
        self.chunks, self.meta = idx["chunks"], idx["meta"]
        self.size = os.path.getsize(path)
        self.lock = threading.Lock()

    def get(self, name):
        off, n, crc = self.chunks[name]
        with self.lock:                       # chunks are read from several threads
            self.f.seek(off)
            data = self.f.read(n)
        if len(data) != n or zlib.crc32(data) != crc:
            raise FormatError(f"{self.path}: chunk {name} is damaged (CRC mismatch)")
        return data

    def close(self):
        self.f.close()
