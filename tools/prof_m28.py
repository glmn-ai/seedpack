"""Stage timings of M28 compression / decompression (cProfile, cumulative)."""
import cProfile, glob, os, pstats, sys, time
sys.path.insert(0, "src")
from repo3 import M28RepoGPU
from gpuquant import GPUQuant
d = "models/unsloth"
base_path = os.path.join(d, "Qwen3-0.6B-BF16.gguf")
files = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(d, "*.gguf"))) if p != base_path}
extras = {os.path.basename(p): p for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*"))
          if os.path.isfile(p) and not p.endswith(".gguf")}
base = open(base_path, "rb").read()
m = M28RepoGPU(); m.gq = GPUQuant()
base = {"Qwen3-0.6B-BF16.gguf": base}
t = time.perf_counter(); [open(p, "rb").read() for p in files.values()]; print(f"plain read of all quant files: {time.perf_counter() - t:.1f} s")
pr = cProfile.Profile(); pr.enable(); arc = m.compress(base, files, extras); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
pr = cProfile.Profile(); pr.enable(); m.decompress(arc, lambda n, b: None); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
