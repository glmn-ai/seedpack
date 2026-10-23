"""seedpack - lossless GPU archiver for GGUF model repositories.

    seedpack pack   INPUT... -o ARCHIVE.varc        archive files and/or directories
    seedpack unpack ARCHIVE [NAME|GLOB ...] [-o DIR] extract everything or only the given files
    seedpack list   ARCHIVE                         show the files in an archive
    seedpack test   ARCHIVE [NAME|GLOB ...]         decompress in memory and verify every checksum

ARCHIVE may be a local .varc file or an archive on the Hugging Face Hub:
hf://OWNER/REPO[/FILE.varc][@REVISION] or its https://huggingface.co/... URL (downloaded once
into the Hugging Face cache; `list` reads only the index).

All BF16 GGUFs among the inputs (the model, mmproj) are coded as "bases"; every other GGUF is
stored as the difference to what llama.cpp's quantiser produces from its base (GPU, bit-exact),
so a whole repository of quantisations costs about as much as its BF16 file. Any other file is
stored with LZMA (small files) or zstd. Extraction is always byte-exact and checked by BLAKE3 (or SHA-256) checksums.
Requires an NVIDIA GPU (CUDA, CuPy).
"""
import argparse
import fnmatch
import hashlib
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from . import __version__

TOOL = f"seedpack {__version__}"


def _hasher(algo):
    """file checksum: BLAKE3 (multithreaded, ~20 GB/s) when installed, else SHA-256"""
    if algo == "blake3":
        import blake3
        return lambda d: blake3.blake3(d, max_threads=blake3.blake3.AUTO).hexdigest()
    return lambda d: hashlib.sha256(d).hexdigest()


def physical_ram():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("virt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            ms = MS(); ms.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.total
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 16 << 30


def cache_bytes(mb):
    """--cache MB, or by default a fifth of the RAM (at most 6 GB): predictions of tensors that
    repeat across files are kept until their last use (Qwen3.5-2B needs about 6 GB)"""
    return mb << 20 if mb is not None else min(6 << 30, physical_ram() // 5)


def default_algo():
    try:
        import blake3  # noqa: F401
        return "blake3"
    except ImportError:
        return "sha256"
SKIP_DIRS = {".git", ".cache", "__pycache__"}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.2f} {u}"
        n /= 1000


def fail(msg):
    print(f"seedpack: error: {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------------- Hugging Face
def hf_spec(spec):
    """hf://OWNER/REPO[/PATH][@REV] or https://huggingface.co/OWNER/REPO[/resolve|blob/REV/PATH]
    -> (repo id, path in the repo or None, revision or None); None for a local path"""
    if spec.startswith(("https://huggingface.co/", "http://huggingface.co/")):
        parts = spec.split("huggingface.co/", 1)[1].split("?")[0].strip("/").split("/")
        if len(parts) >= 4 and parts[2] in ("resolve", "blob"):
            return "/".join(parts[:2]), "/".join(parts[4:]) or None, parts[3]
        return "/".join(parts[:2]), None, None
    if not spec.startswith("hf://"):
        return None
    rest, rev = spec[5:], None
    if "@" in rest:
        rest, rev = rest.rsplit("@", 1)
    parts = rest.strip("/").split("/")
    if len(parts) < 2:
        fail(f"{spec}: expected hf://OWNER/REPO[/FILE.varc][@REVISION]")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None, rev


def _hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        fail("hf:// archives need huggingface_hub (pip install huggingface_hub)")


def _hf_file(repo, path, rev):
    """the archive's path in the repo: as given, or the repo's only .varc file"""
    if path:
        return path
    try:
        cands = [f for f in _hub().list_repo_files(repo, revision=rev) if f.endswith(".varc")]
    except Exception as e:
        fail(f"hf://{repo}: {e.__class__.__name__}: {e}")
    if len(cands) != 1:
        fail(f"hf://{repo} has {len(cands)} .varc files{': ' + ', '.join(cands) if cands else ''}; "
             f"name one: hf://{repo}/FILE.varc")
    return cands[0]


def open_archive(spec, index_only=False):
    """container.Reader for a local path or a Hugging Face archive. With index_only, a Hub archive
    is read remotely (only its header and index are fetched); otherwise it is downloaded (once)."""
    from .container import Reader, FormatError
    hs = hf_spec(spec)
    try:
        if hs is None:
            return Reader(spec)
        repo, path, rev = hs
        path = _hf_file(repo, path, rev)
        hub = _hub()
        if index_only:
            fs = hub.HfFileSystem()
            url = f"{repo}@{rev}/{path}" if rev else f"{repo}/{path}"
            return Reader(f"hf://{url}", fileobj=fs.open(url, "rb", block_size=1 << 20), size=fs.size(url))
        print(f"hf://{repo}/{path}: downloading into the Hugging Face cache (once)", flush=True)
        return Reader(hub.hf_hub_download(repo, path, revision=rev))
    except (FormatError, OSError) as e:
        fail(str(e))
    except Exception as e:
        if e.__class__.__module__.startswith("huggingface_hub"):
            fail(f"{spec}: {e.__class__.__name__}: {e}")
        raise


def gpu_check():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception as e:                    # no driver / no device / no CuPy
        fail(f"an NVIDIA GPU with CUDA is required ({e.__class__.__name__}: {e})")


# ---------------------------------------------------------------------------------- inputs
def collect(inputs):
    """INPUT paths -> {archive name: path}; directories are walked, names are relative to them"""
    out = {}
    for inp in inputs:
        if os.path.isdir(inp):
            for root, dirs, fs in os.walk(inp):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
                for f in sorted(fs):
                    p = os.path.join(root, f)
                    out.setdefault(os.path.relpath(p, inp).replace(os.sep, "/"), []).append(p)
        elif os.path.isfile(inp):
            out.setdefault(os.path.basename(inp), []).append(inp)
        else:
            fail(f"{inp}: no such file or directory")
    dup = [n for n, ps in out.items() if len(ps) > 1]
    if dup:
        fail(f"several inputs map to the same name in the archive: {', '.join(dup[:5])}")
    return {n: ps[0] for n, ps in out.items()}


_IMATRIX_OF = {}                          # path -> the imatrix file named in its metadata


def gguf_role(path):
    """'base' for a GGUF whose weights are mostly BF16, 'derived' for another GGUF, else None.
    Notes the imatrix file a quant was made with (quantize.imatrix.file) in _IMATRIX_OF."""
    from .gguf_reader import GGUF, GGML_BF16
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
    size = 32 << 20
    while True:
        with open(path, "rb") as f:
            head = f.read(size)
        try:
            g = GGUF.from_bytes(head, light=True)
            break
        except Exception:
            if len(head) < size:
                return None                  # not a parsable GGUF: store as a plain file
            size *= 4
    total = sum(t.nbytes for t in g.tensors) or 1
    bf16 = sum(t.nbytes for t in g.tensors if t.type == GGML_BF16)
    if g.meta.get("quantize.imatrix.file"):
        _IMATRIX_OF[path] = str(g.meta["quantize.imatrix.file"])
    return "base" if bf16 / total >= 0.5 else "derived"


# ---------------------------------------------------------------------------------- pack
def cmd_pack(a):
    cp = gpu_check()
    from .container import Writer
    from .repo3 import M28RepoGPU, M29RepoGPU
    files = collect(a.inputs)
    if not files:
        fail("nothing to archive")
    out = a.output
    if os.path.exists(out) and not a.force:
        fail(f"{out} exists (use --force to overwrite)")
    roles = {n: (gguf_role(p) if n.lower().endswith(".gguf") else None) for n, p in files.items()}
    bases = {n: p for n, p in files.items() if roles[n] == "base"}
    derived = {n: p for n, p in files.items() if roles[n] == "derived"}
    extras = {n: p for n, p in files.items() if roles[n] is None}
    total = sum(os.path.getsize(p) for p in files.values())
    print(f"{len(files)} files, {human(total)}: {len(bases)} BF16 base(s), {len(derived)} derived GGUF(s), "
          f"{len(extras)} other file(s)")
    want_im = sorted({os.path.basename(_IMATRIX_OF[p].replace("\\", "/")) for p in derived.values() if p in _IMATRIX_OF})
    if want_im and not any("imatrix" in n.lower() for n in extras):
        print(f"  note: the quants were made with an importance matrix ({', '.join(want_im[:3])}); adding that file "
              "to the inputs makes the archive about 15-20% smaller")
    if derived and not bases:
        print("  note: no BF16 GGUF among the inputs - the most precise quant is the seed the others are derived "
              "from (with the BF16 model the archive is smaller still)")
    m = (M28RepoGPU if a.method == "m28" else M29RepoGPU)(level=a.level, cache_bytes=cache_bytes(a.cache))
    m.verbose = not a.quiet
    algo = default_algo()
    digest = _hasher(algo)
    hashes = {}
    hp = ThreadPoolExecutor(2)
    jobs = []

    def on_data(name, data):
        # called once per input, before it is modified. A derived GGUF arrives in the reader
        # thread (already parallel to the GPU work) and is XOR-ed right after: hashed here.
        # Bases and other files are not modified: hashed in the pool.
        if roles[name] == "derived":
            hashes[name] = digest(data)
        else:
            jobs.append((name, hp.submit(lambda d=data: digest(d))))

    w = Writer(out)
    t0 = time.perf_counter()
    try:
        m.compress(bases, derived, extras, put=w.put, on_data=on_data)
        for n, f in jobs:
            hashes[n] = f.result()
        catalog = []
        for n, p in files.items():
            catalog.append({"name": n, "size": os.path.getsize(p), "hash": hashes[n],
                            "role": "derived, M21" if n in m.q8_files else roles[n] or "file", "stored": m.sizes.get(n)})
        w.close({"tool": TOOL, "method": m.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "hash": algo,
                 "files": catalog, "total": total})
    except BaseException:
        w.abort()
        raise
    dt = time.perf_counter() - t0
    size = os.path.getsize(out)
    print(f"{out}: {human(total)} -> {human(size)} ({100 * size / total:.2f}%"
          + (f", {100 * size / sum(os.path.getsize(p) for p in bases.values()):.1f}% of the BF16 base(s)" if bases else "")
          + f") in {dt:.1f} s ({total / dt / 1e6:.0f} MB/s)")


# ---------------------------------------------------------------------------------- unpack / test
def select(meta, patterns):
    names = [f["name"] for f in meta["files"]]
    if not patterns:
        return None, names
    chosen = []
    for pat in patterns:
        hit = [n for n in names if fnmatch.fnmatchcase(n, pat) or n == pat or os.path.basename(n) == pat]
        if not hit:
            fail(f"{pat}: no such file in the archive (see 'seedpack list')")
        chosen += [n for n in hit if n not in chosen]
    return set(chosen), chosen


class Output:
    """receives decompressed files: verifies checksum and (unless testing) writes them, both in
    background threads, with at most `budget` bytes waiting"""

    def __init__(self, catalog, algo, outdir=None, budget=3 << 30):
        self.cat = {f["name"]: f for f in catalog}
        self.digest = _hasher(algo)
        self.outdir, self.budget = outdir, budget
        self.pool = ThreadPoolExecutor(4)
        self.pending = deque()
        self.done, self.bad, self.bytes = [], [], 0

    def __call__(self, name, data):
        self.pending.append((name, self.pool.submit(self._one, name, data), len(data)))
        while len(self.pending) > 1 and sum(s for _, _, s in self.pending) > self.budget:
            self._collect()

    def _one(self, name, data):
        ok = self.digest(data) == self.cat[name]["hash"]
        if self.outdir is not None and ok:
            path = os.path.join(self.outdir, *name.split("/"))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
        return ok

    def _collect(self):
        name, f, size = self.pending.popleft()
        ok = f.result()
        (self.done if ok else self.bad).append(name)
        self.bytes += size
        print(f"  {'OK ' if ok else 'BAD'} {name} ({human(size)})", flush=True)

    def finish(self):
        while self.pending:
            self._collect()


def _extract(a, outdir):
    gpu_check()
    from .container import FormatError
    from .repo3 import M28RepoGPU, M29RepoGPU
    r = open_archive(a.archive)
    meta = r.meta
    sel, names = select(meta, a.files)
    if outdir is not None and not a.force:
        clash = [n for n in names if os.path.exists(os.path.join(outdir, *n.split("/")))]
        if clash:
            fail(f"{len(clash)} file(s) already exist in {outdir}, e.g. {clash[0]} (use --force to overwrite)")
    cls = M28RepoGPU if meta["method"].startswith("M28") else M29RepoGPU
    m = cls(cache_bytes=cache_bytes(a.cache))
    out = Output(meta["files"], meta["hash"], outdir)
    t0 = time.perf_counter()
    try:
        m.decompress(r.get, out, select=sel)
        out.finish()
    except FormatError as e:
        fail(str(e))
    dt = time.perf_counter() - t0
    missing = [n for n in names if n not in out.done and n not in out.bad]
    verb = "extracted" if outdir is not None else "tested"
    print(f"{verb} {len(out.done)} file(s), {human(out.bytes)} in {dt:.1f} s ({out.bytes / max(dt, 1e-9) / 1e6:.0f} MB/s)")
    if out.bad or missing:
        fail(f"{len(out.bad)} file(s) failed the checksum check, {len(missing)} missing: {', '.join((out.bad + missing)[:5])}")


def cmd_unpack(a):
    _extract(a, a.output)


def cmd_test(a):
    _extract(a, None)


def cmd_list(a):
    r = open_archive(a.archive, index_only=True)
    meta = r.meta
    w = max(len(f["name"]) for f in meta["files"])
    print(f"{a.archive}: {meta['tool']}, {meta['method']}, created {meta['created']}")
    print(f"{'name':{w}s} {'size':>12s} {'stored':>12s}  role")
    for f in meta["files"]:
        st = human(f["stored"]) if f.get("stored") is not None else "(bundle)"
        print(f"{f['name']:{w}s} {human(f['size']):>12s} {st:>12s}  {f['role']}")
    print(f"{len(meta['files'])} files, {human(meta['total'])} -> archive {human(r.size)} "
          f"({100 * r.size / meta['total']:.2f}%)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="seedpack", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("--version", action="version", version=TOOL)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pack", aliases=["a"], help="archive files and/or directories")
    s.add_argument("inputs", nargs="+", help="files or directories (directory contents keep their relative paths)")
    s.add_argument("-o", "--output", required=True, help="archive to create (.varc)")
    s.add_argument("--method", choices=["m29", "m28"], default="m29", help="m29 (default): BF16 v2 base codec; m28: M24 codec")
    s.add_argument("--level", type=int, default=7, help="zstd level of the residuals (default 7; 9: about 1.5%% smaller, slower)")
    s.add_argument("--cache", type=int, default=None, help="prediction cache in MB (default: a fifth of the RAM, at most 6144)")
    s.add_argument("-f", "--force", action="store_true", help="overwrite the archive")
    s.add_argument("-q", "--quiet", action="store_true", help="no per-file progress")
    s.set_defaults(fn=cmd_pack)
    for name, alias, fn, hlp in (("unpack", "x", cmd_unpack, "extract all or selected files"),
                                 ("test", "t", cmd_test, "decompress in memory and verify checksum")):
        s = sub.add_parser(name, aliases=[alias], help=hlp)
        s.add_argument("archive", help="a .varc file or hf://OWNER/REPO[/FILE.varc][@REVISION]")
        s.add_argument("files", nargs="*", help="names or glob patterns (e.g. '*Q4_K_M*'); default: everything")
        if name == "unpack":
            s.add_argument("-o", "--output", default=".", help="output directory (default: current)")
            s.add_argument("-f", "--force", action="store_true", help="overwrite existing files")
        else:
            s.set_defaults(force=True)
        s.add_argument("--cache", type=int, default=None, help="prediction cache in MB (default: a fifth of the RAM, at most 6144)")
        s.set_defaults(fn=fn)
    s = sub.add_parser("list", aliases=["l"], help="list the files in an archive")
    s.add_argument("archive", help="a .varc file or hf://OWNER/REPO[/FILE.varc][@REVISION]")
    s.set_defaults(fn=cmd_list)
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except KeyboardInterrupt:
        fail("interrupted")


if __name__ == "__main__":
    main()
