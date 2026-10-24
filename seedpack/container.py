"""The .seedpack container: a sequence of named chunks with an index at the end, so any chunk can be
read without touching the others (selective extraction) and chunks can be written as soon as
they are produced (the archive is never held in memory as a whole).

Layout:
    "SEEDPACK" | u32 format version
    chunk bytes, back to back
    index: zstd(JSON {"chunks": {name: [offset, length, crc32]}, "meta": {...}})
    footer: u64 index offset | u64 index length | b"SEEDEND\\0"
"""
import functools
import io
import json
import os
import queue
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
        self.streams = []

    def put(self, name, data):
        if name in self.chunks:
            raise ValueError(f"duplicate chunk {name}")
        self.chunks[name] = None
        self.jobs.append(self.io.submit(self._write, name, data))
        while self.jobs and self.jobs[0].done():
            self.jobs.popleft().result()          # raises a write error early

    def _write(self, name, data):
        """data: bytes-like, or a list of them written back to back (a chunk in pieces)"""
        off, n, crc = self.f.tell(), 0, 0
        for part in data if isinstance(data, list) else [data]:
            crc = zlib.crc32(part, crc)
            n += memoryview(part).nbytes
            self.f.write(part)
        self.chunks[name] = [off, n, crc]

    def put_stream(self, name):
        """a chunk written while it is produced (it is the next one in the archive) -> object with
        head(bytes) (its first bytes, which may be rewritten at the end), write(bytes) for the
        rest, close(final head of the same length or None) and abort()"""
        if name in self.chunks:
            raise ValueError(f"duplicate chunk {name}")
        self.chunks[name] = None
        q = queue.Queue(maxsize=4)            # pieces waiting: bounded (a big model's are gigabytes)
        self.streams.append(q)
        self.jobs.append(self.io.submit(self._write_stream, name, q))
        return _PutStream(q)

    def _write_stream(self, name, q):
        off, head, n, crc = self.f.tell(), b"", 0, 0
        while True:
            kind, data = q.get()
            if kind == "head":
                head = bytes(data)
                self.f.write(head)
                n += len(head)
            elif kind == "data":
                crc = zlib.crc32(data, crc)
                self.f.write(data)
                n += memoryview(data).nbytes
            elif kind == "end":
                if data is not None:
                    if len(data) != len(head):
                        raise ValueError(f"chunk {name}: the final head differs in length")
                    end = self.f.tell()
                    self.f.seek(off)
                    self.f.write(data)
                    self.f.seek(end)
                    head = bytes(data)
                self.chunks[name] = [off, n, crc32_combine(zlib.crc32(head), crc, n - len(head))]
                return
            else:
                raise IOError(f"chunk {name}: its producer failed")

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
        for q in self.streams:                # a chunk still being written: its writer stops
            try:
                q.put_nowait(("abort", None))
            except queue.Full:
                pass
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
        self.fileobj = fileobj
        self._local, self._handles = threading.local(), []
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
        if n >= 2 * _PIECE and self.fileobj is None:
            data = bytearray(n)
            self.read_at(name, 0, n, data)
            if self._crc(data) != crc:
                raise FormatError(f"{self.path}: chunk {name} is damaged (CRC mismatch)")
            return data
        with self.lock:                       # chunks are read from several threads
            self.f.seek(off)
            data = self.f.read(n)
        if len(data) != n or zlib.crc32(data) != crc:
            raise FormatError(f"{self.path}: chunk {name} is damaged (CRC mismatch)")
        return data

    @staticmethod
    def _crc(buf):
        """CRC32 of a large buffer: pieces in parallel, combined (zlib releases the GIL)"""
        mv = memoryview(buf).cast("B")
        parts = [mv[o:o + _PIECE] for o in range(0, len(mv), _PIECE)]
        crcs = list(_IO.map(zlib.crc32, parts))
        crc = 0
        for p, c in zip(parts, crcs):
            crc = crc32_combine(crc, c, len(p))
        return crc

    def _handle(self):
        """this thread's own handle on the archive (reads at any offset without a lock)"""
        f = getattr(self._local, "f", None)
        if f is None:
            f = self._local.f = open(self.path, "rb", buffering=0)
            self._handles.append(f)
        return f

    def _read_into(self, pos, mv):
        f = self._handle()
        f.seek(pos)
        got = 0
        while got < len(mv):
            k = f.readinto(mv[got:])
            if not k:
                raise FormatError(f"{self.path}: truncated archive")
            got += k

    def read_at(self, name, rel, n, out=None):
        """n bytes of chunk `name` from byte `rel` of it (no CRC check: a part of a chunk), read
        in parallel pieces (an SSD delivers its full speed only with several requests in
        flight); into `out` (a writable buffer of n bytes) when given"""
        off, size, _ = self.chunks[name]
        if rel < 0 or rel + n > size:
            raise ValueError(f"{name}: bytes {rel}..{rel + n} out of the chunk ({size})")
        out = bytearray(n) if out is None else out
        mv = memoryview(out).cast("B")
        if self.fileobj is not None:              # a remote file: one request
            with self.lock:
                self.f.seek(off + rel)
                mv[:] = self.f.read(n)
            return out
        for f in [_IO.submit(self._read_into, off + rel + o, mv[o:o + _PIECE]) for o in range(0, n, _PIECE)]:
            f.result()
        return out

    def open_chunk(self, name):
        """chunk `name` as a sequential binary stream, its CRC checked when it has been read to the end"""
        return _ChunkStream(self, name)

    def close(self):
        self.f.close()
        for f in self._handles:
            f.close()


class _PutStream:
    def __init__(self, q):
        self.q = q

    def head(self, b):
        self.q.put(("head", b))

    def write(self, b):
        self.q.put(("data", b))

    def close(self, head=None):
        self.q.put(("end", head))

    def abort(self):
        self.q.put(("abort", None))


class _ChunkStream(io.RawIOBase):
    def __init__(self, reader, name):
        self.r, self.name = reader, name
        self.size, self.crc_want = reader.chunks[name][1:]
        self.pos, self.crc = 0, 0

    def readable(self):
        return True

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        mv = memoryview(b).cast("B")[:n]
        self.r.read_at(self.name, self.pos, n, mv)
        self.crc = zlib.crc32(mv, self.crc)
        self.pos += n
        if self.pos == self.size and self.crc != self.crc_want:
            raise FormatError(f"{self.r.path}: chunk {self.name} is damaged (CRC mismatch)")
        return n


_PIECE = 32 << 20
_IO = ThreadPoolExecutor(8)


def _gf2_times(mat, vec):
    s, i = 0, 0
    while vec:
        if vec & 1:
            s ^= mat[i]
        vec >>= 1
        i += 1
    return s


def _gf2_compose(a, b):
    return [_gf2_times(a, b[n]) for n in range(32)]


@functools.lru_cache(maxsize=16)
def _zeros_op(nbytes):
    """the GF(2) operator that appending nbytes zero bytes applies to a CRC32 register"""
    op = [0xEDB88320] + [1 << n for n in range(31)]       # one zero bit
    for _ in range(3):
        op = _gf2_compose(op, op)                          # 2, 4, 8 bits: one byte
    out = None
    while nbytes:
        if nbytes & 1:
            out = op if out is None else _gf2_compose(op, out)
        nbytes >>= 1
        if nbytes:
            op = _gf2_compose(op, op)
    return out


def crc32_combine(crc1, crc2, len2):
    """CRC32 of A + B from crc32(A), crc32(B) and len(B) (zlib's crc32_combine)"""
    return crc1 if len2 <= 0 else _gf2_times(_zeros_op(len2), crc1) ^ crc2
