"""ZipNN in memory (run inside WSL, where it builds): zipnn_bench.py FILE_OR_DIR...
Every file is read first (I/O not timed), compressed and decompressed with ZipNN's byte API
(BF16 byte grouping, 12 threads), and checked byte-exact. Prints one JSON object."""
import json
import os
import sys
import time

import zipnn


def files_of(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            for root, dirs, fs in os.walk(a):
                dirs[:] = [d for d in dirs if d not in (".git", ".cache")]
                out += [os.path.join(root, f) for f in sorted(fs)]
        else:
            out.append(a)
    return out


tc = td = 0.0
size = csize = 0
ok = True
z = zipnn.ZipNN(input_format="byte", bytearray_dtype="bfloat16", threads=12)
for p in files_of(sys.argv[1:]):
    with open(p, "rb") as f:
        data = f.read()
    ref = bytes(bytearray(data))       # ZipNN's compress() rearranges its input buffer in place
    t0 = time.perf_counter(); c = z.compress(data); t1 = time.perf_counter()
    d = z.decompress(c); t2 = time.perf_counter()
    ok &= bytes(d) == ref
    tc += t1 - t0; td += t2 - t1; size += len(data); csize += len(c)
    del data, ref, c, d
print(json.dumps({"tool": "ZipNN 0.5.4", "size": size, "archive": csize, "compress_s": tc, "decompress_s": td,
                  "lossless": ok, "note": "in memory (I/O excluded), WSL"}))
