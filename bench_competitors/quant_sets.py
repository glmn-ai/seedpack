"""Quant sets without the BF16 model (hard-linked directories under E:/bench_ext):
quants-<model>    every quantised GGUF of the repository, no BF16, mmproj or imatrix
quantsim-<model>  the same plus the repository's imatrix file
quants-le4-0.6b   Qwen3-0.6B quants up to Q4 (the senior quant is Q4_1)
Seedpack, lmz and zstd; results go to results.json (run_bench's format), skipping existing ones.

usage: quant_sets.py [dataset ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import run_bench as rb

SETS = ["quants-0.6b", "quants-0.8b", "quants-2b", "quants-le4-0.6b", "quantsim-0.6b", "quantsim-0.8b", "quantsim-2b"]
TOOLS = ("Seedpack (varc)", "lmz 1.3.0", "zstd -3 --long=30")


def main():
    res = json.load(open(rb.RES))
    for ds in sys.argv[1:] or SETS:
        paths = [f"E:/bench_ext/{ds}"]
        rb.DATASETS[ds] = paths
        size = sum(os.path.getsize(f) for f in rb.files_of(paths))
        for tool in TOOLS:
            key = f"{ds} | {tool}"
            if key in res and res[key].get("archive"):
                continue
            r = rb.TOOLS[tool][0](ds, paths)
            r["size"] = size
            res[key] = r
            json.dump(res, open(rb.RES, "w"), indent=1)
            print(f"{ds:16s} {tool:20s} {100 * r['archive'] / size:6.2f}%  compress {r['compress_s']:.1f}s  "
                  f"decompress {r.get('decompress_s', 0):.1f}s  lossless {r.get('lossless')}", flush=True)


if __name__ == "__main__":
    main()
