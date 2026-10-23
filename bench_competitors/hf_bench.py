"""Benchmark for the Hugging Face release of Qwen3-0.6B: Seedpack against general-purpose and
model-specific lossless compressors, on the published file set and on three single files.

usage: hf_bench.py [dataset ...] [--keep]     -> bench_competitors/results_hf06.json
       (run with E:/bench_ext/.venv-bench, which has lmz, brotli, lz4, zstandard and blake3)

Datasets: hf-0.6b = the 30 files of the release (every GGUF of unsloth/Qwen3-0.6B-GGUF incl. BF16,
config.json, params, template, imatrix); bf16 / q8_0 / q4_k_m = single files.
Timing: compress = files on disk -> archive on disk; decompress = decode + BLAKE3 check of the
whole output, nothing written (varc test, lmz verify, 7z t; the others decode into a BLAKE3 hash).
Multi-file inputs go to the single-stream compressors as the concatenation of the files (tar-like).
Chunked tools (brotli, bzip2, gzip, lz4) work on 64 MB blocks with 12 workers, like pigz/pbzip2.
ZipNN runs inside WSL in memory (I/O excluded). HF Xet and DFloat11 are size estimates (estimates.py).
--keep keeps the Seedpack archive of hf-0.6b (the file that is published).
"""
import bz2
import json
import os
import subprocess
import sys
import time
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import blake3

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_bench as rb  # noqa: E402

OUT = rb.OUT
RES = os.path.join(HERE, "results_hf06.json")
U = os.path.join(rb.M, "unsloth")
DATASETS = {
    "hf-0.6b": ["E:/bench_ext/hf-0.6b"],
    "bf16-0.6b": [f"{U}/Qwen3-0.6B-BF16.gguf"],
    "q8_0-0.6b": [f"{U}/Qwen3-0.6B-Q8_0.gguf"],
    "q4_k_m-0.6b": [f"{U}/Qwen3-0.6B-Q4_K_M.gguf"],
}
SEVENZ = "C:/Program Files/7-Zip/7z.exe"
XZ = "C:/Program Files/Git/mingw64/bin/xz.exe"
BLOCK = 64 << 20
WORKERS = 12
KEEP = "--keep" in sys.argv


def stream(paths, n=BLOCK):
    for f in rb.files_of(paths):
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(n), b""):
                yield b


def digest(paths):
    h = blake3.blake3(max_threads=blake3.blake3.AUTO)
    for b in stream(paths):
        h.update(b)
    return h.digest()


def timed(cmd, **kw):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{cmd[:3]}: {r.stderr[-1500:]}")
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------------- tools
def t_seedpack(ds, paths, ref):
    arc = f"{OUT}/Qwen3-0.6B-SEEDPACK.varc" if ds == "hf-0.6b" else f"{OUT}/{ds}.varc"
    tc = timed([rb.PY, f"{rb.ROOT}/varc.py", "pack", *paths, "-o", arc, "-q", "-f"])
    td = timed([rb.PY, f"{rb.ROOT}/varc.py", "test", arc])
    size = os.path.getsize(arc)
    if not (KEEP and ds == "hf-0.6b"):
        os.remove(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=True)


def t_lmz(ds, paths, ref):
    r = rb.t_lmz(ds, paths)
    return r


def t_zipnn(ds, paths, ref):
    return rb.t_zipnn(ds, paths)


def t_zstd(level):
    def run(ds, paths, ref):
        import zstandard
        arc = f"{OUT}/{ds}.zst"
        p = zstandard.ZstdCompressionParameters.from_level(level, threads=WORKERS, enable_ldm=True, window_log=30,
                                                           job_size=256 << 20)
        t0 = time.perf_counter()
        with open(arc, "wb") as o:
            w = zstandard.ZstdCompressor(compression_params=p).stream_writer(o, closefd=False)
            for b in stream(paths):
                w.write(b)
            w.flush(zstandard.FLUSH_FRAME)
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        h = blake3.blake3()
        with open(arc, "rb") as f:
            r = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(f)
            for b in iter(lambda: r.read(BLOCK), b""):
                h.update(b)
        td = time.perf_counter() - t0
        size = os.path.getsize(arc)
        os.remove(arc)
        return dict(archive=size, compress_s=tc, decompress_s=td, lossless=h.digest() == ref)
    return run


def t_7z(ds, paths, ref):
    arc = f"{OUT}/{ds}.7z"
    rb.rm(arc)
    files = rb.files_of(paths)
    root = paths[0] if os.path.isdir(paths[0]) else os.path.dirname(paths[0])
    names = [os.path.relpath(f, root) for f in files]
    tc = timed([SEVENZ, "a", "-t7z", "-mx=9", f"-mmt={WORKERS}", "-bd", arc, *names], cwd=root)
    td = timed([SEVENZ, "t", "-bd", arc])          # CRC32 of every file checked
    size = os.path.getsize(arc)
    os.remove(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=True)


def t_xz(ds, paths, ref):
    arc = f"{OUT}/{ds}.xz"
    t0 = time.perf_counter()
    with open(arc, "wb") as o:
        p = subprocess.Popen([XZ, "-6", f"-T{WORKERS}", "-c"], stdin=subprocess.PIPE, stdout=o)
        for b in stream(paths):
            p.stdin.write(b)
        p.stdin.close()
        assert p.wait() == 0
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    h = blake3.blake3()
    p = subprocess.Popen([XZ, "-d", f"-T{WORKERS}", "-c", arc], stdout=subprocess.PIPE)
    for b in iter(lambda: p.stdout.read(BLOCK), b""):
        h.update(b)
    assert p.wait() == 0
    td = time.perf_counter() - t0
    size = os.path.getsize(arc)
    os.remove(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=h.digest() == ref)


def _brotli_c(b):
    import brotli
    return brotli.compress(b, quality=9, lgwin=24)


def _brotli_d(b):
    import brotli
    return brotli.decompress(b)


def _gzip_c(b):
    c = zlib.compressobj(9, zlib.DEFLATED, 31)
    return c.compress(b) + c.flush()


def _lz4_c(b):
    import lz4.frame
    return lz4.frame.compress(b, compression_level=9)


def _lz4_d(b):
    import lz4.frame
    return lz4.frame.decompress(b)


CHUNKED = {   # name: (compress, decompress, process pool?)
    "brotli": (_brotli_c, _brotli_d, True),
    "bzip2": (lambda b: bz2.compress(b, 9), bz2.decompress, False),
    "gzip": (_gzip_c, lambda b: zlib.decompress(b, 31), False),
    "lz4": (_lz4_c, _lz4_d, False),
}


def _ordered(ex, fn, items, sink):
    q = deque()
    for it in items:
        q.append(ex.submit(fn, it))
        if len(q) >= 2 * WORKERS:
            sink(q.popleft().result())
    while q:
        sink(q.popleft().result())


def t_chunked(kind):
    comp, decomp, procs = CHUNKED[kind]

    def run(ds, paths, ref):
        arc = f"{OUT}/{ds}.{kind}"
        sizes = []
        pool = ProcessPoolExecutor if procs else ThreadPoolExecutor
        with pool(WORKERS) as ex:
            t0 = time.perf_counter()
            with open(arc, "wb") as o:
                def put(c):
                    sizes.append(len(c))
                    o.write(c)
                _ordered(ex, comp, stream(paths), put)
            tc = time.perf_counter() - t0
            t0 = time.perf_counter()
            h = blake3.blake3()
            with open(arc, "rb") as f:
                _ordered(ex, decomp, (f.read(n) for n in sizes), h.update)
            td = time.perf_counter() - t0
        size = os.path.getsize(arc)
        os.remove(arc)
        return dict(archive=size, compress_s=tc, decompress_s=td, lossless=h.digest() == ref)
    return run


def t_est(kind):
    def run(ds, paths, ref):
        return rb.t_est(kind)(ds, paths)
    return run


TOOLS = {   # name: (function, datasets it applies to; None = all)
    "SeedPack": (t_seedpack, None),
    "lmz 1.3.0": (t_lmz, None),
    "ZipNN 0.5.4": (t_zipnn, None),
    "zstd 1.5.7 -19 --long=30": (t_zstd(19), None),
    "zstd 1.5.7 -3 --long=30": (t_zstd(3), None),
    "7-Zip 23.01 -mx9 (LZMA2)": (t_7z, None),
    "xz 5.4.1 -6": (t_xz, None),
    "brotli 1.2 -q9": (t_chunked("brotli"), None),
    "bzip2 -9": (t_chunked("bzip2"), None),
    "gzip -9": (t_chunked("gzip"), None),
    "lz4 1.9.4 -9 (HC)": (t_chunked("lz4"), None),
    "HF Xet (estimate)": (t_est("xet"), None),
    "DFloat11 (estimate)": (t_est("dfloat11"), {"bf16-0.6b"}),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = [a for a in args if a in TOOLS]
    for ds in [a for a in args if a in DATASETS] or list(DATASETS):
        paths = DATASETS[ds]
        size = sum(os.path.getsize(f) for f in rb.files_of(paths))
        ref = digest(paths)
        for tool, (fn, scope) in TOOLS.items():
            if (scope is not None and ds not in scope) or (only and tool not in only):
                continue
            key = f"{ds} | {tool}"
            if key in res and res[key].get("archive") and not only:
                continue
            print(f"{ds:12s} {tool:26s} ...", flush=True)
            try:
                r = fn(ds, paths, ref)
            except Exception as e:
                r = {"error": str(e)[-500:]}
            r["size"] = size
            res[key] = r
            json.dump(res, open(RES, "w"), indent=1)
            if "archive" in r:
                print(f"   {100 * r['archive'] / size:6.2f}%  compress {r.get('compress_s', float('nan')):.1f} s"
                      f"  decompress {r.get('decompress_s', float('nan')):.1f} s  lossless {r.get('lossless')}", flush=True)
            else:
                print("   ", r.get("error", "")[-300:], flush=True)


if __name__ == "__main__":
    main()
