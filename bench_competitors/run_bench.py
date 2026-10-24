"""Benchmark SeedPack against other lossless compressors on GGUF data.

usage: run_bench.py [dataset ...]      (default: all)    -> bench_competitors/results.json

Timing: "compress" = from the files on disk to the archive; "decompress" = decode + check
without writing the files (seedpack test / lmz verify / zstd and xz to a null sink), except ZipNN
which is measured in memory inside WSL (I/O excluded). DFloat11 and Xet are size estimates
(estimates.py): DFloat11's tool takes transformers models only, Xet is server-side.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = "E:/bench_ext/out"
LMZ = ["E:/bench_ext/.venv-bench/Scripts/python.exe", os.path.join(ROOT, "bench_competitors", "lmz_win.py")]
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
RES = os.path.join(ROOT, "bench_competitors", "results.json")

M = os.path.join(ROOT, "models")
DATASETS = {
    "bf16-0.6b": [f"{M}/unsloth/Qwen3-0.6B-BF16.gguf"],
    "bf16-2b": [f"{M}/qwen35-2b/Qwen3.5-2B-BF16.gguf"],
    "q8_0-0.6b": [f"{M}/unsloth/Qwen3-0.6B-Q8_0.gguf"],
    "q4_k_m-0.6b": [f"{M}/unsloth/Qwen3-0.6B-Q4_K_M.gguf"],
    "repo-0.6b": [f"{M}/unsloth"],
    "repo-0.8b": [f"{M}/qwen35-08b"],
    "repo-2b": [f"{M}/qwen35-2b"],
}


def files_of(paths):
    out = []
    for a in paths:
        if os.path.isdir(a):
            for root, dirs, fs in os.walk(a):
                dirs[:] = sorted(d for d in dirs if d not in (".git", ".cache"))
                out += [os.path.join(root, f) for f in sorted(fs)]
        else:
            out.append(a)
    return out


def sh(cmd, **kw):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    dt = time.perf_counter() - t0
    if r.returncode:
        raise RuntimeError(f"{cmd[:3]} failed: {r.stderr[-2000:]}")
    return dt, r.stdout


def rm(p):
    if os.path.exists(p):
        os.remove(p)


# ---------------------------------------------------------------------------------- tools
def t_seedpack(name, paths):
    arc = f"{OUT}/{name}.seedpack"
    rm(arc)
    tc, _ = sh([PY, "-m", "seedpack", "pack", *paths, "-o", arc, "-q", "-f"])
    td, _ = sh([PY, "-m", "seedpack", "test", arc])
    size = os.path.getsize(arc)
    rm(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=True)


def t_lmz(name, paths):
    arc = f"{OUT}/{name}.lmz"
    rm(arc)
    src = paths[0]
    tc, _ = sh(LMZ + ["compress", src, arc, "-q", "-f"])
    td, _ = sh(LMZ + ["verify", arc])
    size = os.path.getsize(arc)
    rm(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=True)


def t_zstd(level, long=False):
    def run(name, paths):
        import zstandard
        p = zstandard.ZstdCompressionParameters.from_level(level, threads=12, enable_ldm=long,
                                                           window_log=30 if long else 0,
                                                           job_size=(256 << 20) if long else 0)
        c = zstandard.ZstdCompressor(compression_params=p)
        h_in = hashlib.blake2b()
        t0 = time.perf_counter()
        chunks = []
        keep = sum(os.path.getsize(f) for f in files_of(paths)) < (6 << 30)
        w = c.stream_writer(_Counter(chunks if keep else None), closefd=False)
        for f in files_of(paths):
            with open(f, "rb") as fh:
                for b in iter(lambda: fh.read(64 << 20), b""):
                    h_in.update(b); w.write(b)
        w.flush(zstandard.FLUSH_FRAME)
        tc = time.perf_counter() - t0
        out = dict(compress_s=tc)
        cnt = _Counter.last
        out["archive"] = cnt.n
        if keep:
            t0 = time.perf_counter()
            d = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            h_out = hashlib.blake2b()
            r = d.stream_reader(_Chunks(chunks))
            for b in iter(lambda: r.read(64 << 20), b""):
                h_out.update(b)
            out["decompress_s"] = time.perf_counter() - t0
            out["lossless"] = h_in.digest() == h_out.digest()
        return out
    return run


class _Counter:
    last = None

    def __init__(self, keep):
        self.n, self.keep = 0, keep
        _Counter.last = self

    def write(self, b):
        self.n += len(b)
        if self.keep is not None:
            self.keep.append(bytes(b))
        return len(b)

    def flush(self):
        pass


class _Chunks:
    def __init__(self, chunks):
        self.it, self.buf = iter(chunks), b""

    def read(self, n=-1):
        while n < 0 or len(self.buf) < n:
            try:
                self.buf += next(self.it)
            except StopIteration:
                break
        if n < 0:
            n = len(self.buf)
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


def t_xz(name, paths):
    f = paths[0]
    arc = f"{OUT}/{name}.xz"
    t0 = time.perf_counter()
    with open(arc, "wb") as o:
        subprocess.run(["xz", "-6", "-T12", "-c", f], stdout=o, check=True)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = subprocess.run(["xz", "-d", "-T12", "-c", arc], stdout=subprocess.PIPE, check=True)
    td = time.perf_counter() - t0
    ok = hashlib.blake2b(r.stdout).digest() == hashlib.blake2b(open(f, "rb").read()).digest()
    size = os.path.getsize(arc)
    rm(arc)
    return dict(archive=size, compress_s=tc, decompress_s=td, lossless=ok)


def t_zipnn(name, paths):
    wp = [p.replace("\\", "/").replace("E:/", "/mnt/e/") for p in paths]
    _, out = sh(["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                 "cd ~ && ~/zvenv/bin/python -W ignore /mnt/e/Github/vibe/archivator/bench_competitors/zipnn_bench.py "
                 + " ".join(wp)])
    r = json.loads(out.strip().splitlines()[-1])
    return {k: r[k] for k in ("archive", "compress_s", "decompress_s", "lossless")} | {"note": r["note"]}


def t_est(kind):
    def run(name, paths):
        _, out = sh([PY, f"{ROOT}/bench_competitors/estimates.py", kind, *paths])
        r = json.loads(out)
        if kind == "dfloat11":
            return dict(archive=sum(x["dfloat11_est"] for x in r), note="size estimate (format model)")
        return dict(archive=r["xet_est"], note=f"size estimate (CDC dedup + {r['chunk_codec']})")
    return run


TOOLS = {
    "SeedPack": (t_seedpack, "all"),
    "lmz 1.3.0": (t_lmz, "all"),
    "ZipNN 0.5.4": (t_zipnn, "all"),
    "zstd -3 --long=30": (t_zstd(3, True), "all"),
    "zstd -19": (t_zstd(19), "single"),
    "xz -6": (t_xz, "single"),
    "DFloat11 (estimate)": (t_est("dfloat11"), "bf16"),
    "HF Xet (estimate)": (t_est("xet"), "all"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    names = sys.argv[1:] or list(DATASETS)
    for ds in names:
        paths = DATASETS[ds]
        size = sum(os.path.getsize(f) for f in files_of(paths))
        single = not os.path.isdir(paths[0])
        for tool, (fn, scope) in TOOLS.items():
            if (scope == "single" and not single) or (scope == "bf16" and not ds.startswith("bf16")):
                continue
            key = f"{ds} | {tool}"
            if key in res and res[key].get("archive"):
                continue
            print(f"{ds:12s} {tool:22s} ...", flush=True)
            try:
                r = fn(ds, paths)
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
