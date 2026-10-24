"""The .seedpack container: a sequence of named chunks with an index at the end, so any chunk can be
read without touching the others (selective extraction) and chunks can be written as soon as
they are produced (the archive is never held in memory as a whole).

Layout:
    "SEEDPACK" | u32 format version
    chunk bytes, back to back
    index: zstd(JSON {"chunks": {name: [offset, length, crc32]}, "meta": {...}})
    footer: u64 index offset | u64 index length | b"SEEDEND\\0"
"""
import json
import os
import struct
import threading
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import zstandard

MAGIC = b"SEEDPACK"
VERSION = 1
_FOOTER = struct.Struct("<QQ8s")
_END = b"SEEDEND\0"


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
        self.io = ThreadPoolExecutor(1)       # CRC + write in the background, in put() order
        self.jobs = deque()

    def put(self, name, data):
        if name in self.chunks:
            raise ValueError(f"duplicate chunk {name}")
        self.chunks[name] = None
        self.jobs.append(self.io.submit(self._write, name, data))
        while self.jobs and self.jobs[0].done():
            self.jobs.popleft().result()          # raises a write error early

    def _write(self, name, data):
        self.chunks[name] = [self.f.tell(), len(data), zlib.crc32(data)]
        self.f.write(data)

    def close(self, meta):
        while self.jobs:
            self.jobs.popleft().result()
        self.io.shutdown()
        idx = zstandard.ZstdCompressor(level=19).compress(json.dumps({"chunks": self.chunks, "meta": meta}).encode())
        off = self.f.tell()
        self.f.write(idx)
        self.f.write(_FOOTER.pack(off, len(idx), _END))
        self.f.close()
        os.replace(self.path + ".part", self.path)

    def abort(self):
        self.io.shutdown(cancel_futures=True)
        self.f.close()
        try:
            os.remove(self.path + ".part")
        except OSError:
            pass


class Reader:
    def __init__(self, path, fileobj=None, size=None):
        """path: the archive file; or a name for messages plus an open seekable binary `fileobj`
        of `size` bytes (e.g. a remote file)"""
        self.path = path
        self.f = fileobj if fileobj is not None else open(path, "rb")
        head = self.f.read(len(MAGIC) + 4)
        if head[:len(MAGIC)] != MAGIC:
            raise FormatError(f"{path}: not a .seedpack archive")
        version = struct.unpack("<I", head[len(MAGIC):])[0]
        if version > VERSION:
            raise FormatError(f"{path}: format version {version} is newer than this tool")
        self.f.seek(-_FOOTER.size, os.SEEK_END)
        off, n, end = _FOOTER.unpack(self.f.read(_FOOTER.size))
        if end != _END:
            raise FormatError(f"{path}: truncated or damaged archive (no index)")
        self.f.seek(off)
        idx = json.loads(zstandard.ZstdDecompressor().decompress(self.f.read(n)))
        self.chunks, self.meta = idx["chunks"], idx["meta"]
        self.size = size if size is not None else os.path.getsize(path)
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
