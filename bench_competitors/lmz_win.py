"""Runs the lmz CLI on Windows. lmz 1.3.0 is written for POSIX: it calls os.pread (missing on
Windows) and opens files with os.open without O_BINARY (text mode on Windows, which alters the
bytes read). The shim adds O_BINARY and a pread made of seek + read; nothing else is changed."""
import os
import sys
import threading

if hasattr(os, "O_BINARY"):
    _open = os.open

    def _bin_open(path, flags, *a, **kw):
        return _open(path, flags | os.O_BINARY, *a, **kw)

    os.open = _bin_open

if not hasattr(os, "pread"):
    _lock = threading.Lock()

    def pread(fd, n, offset):
        with _lock:
            cur = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, offset, os.SEEK_SET)
            try:
                parts, left = [], n
                while left > 0:
                    b = os.read(fd, left)
                    if not b:
                        break
                    parts.append(b); left -= len(b)
                return b"".join(parts)
            finally:
                os.lseek(fd, cur, os.SEEK_SET)

    os.pread = pread

from lmz.cli import main  # noqa: E402

sys.exit(main())
