"""results.json -> markdown tables (one per dataset), written to bench_competitors/RESULTS.md"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "results.json")))
TITLES = {
    "bf16-0.6b": "Qwen3-0.6B-BF16.gguf (1.20 GB)",
    "bf16-2b": "Qwen3.5-2B-BF16.gguf (3.78 GB)",
    "q8_0-0.6b": "Qwen3-0.6B-Q8_0.gguf (0.64 GB), alone",
    "q4_k_m-0.6b": "Qwen3-0.6B-Q4_K_M.gguf (0.40 GB), alone",
    "repo-0.6b": "unsloth/Qwen3-0.6B-GGUF, whole repository (10.95 GB, 32 files)",
    "repo-0.8b": "unsloth/Qwen3.5-0.8B-GGUF, whole repository (14.08 GB, 28 files)",
    "repo-2b": "unsloth/Qwen3.5-2B-GGUF, whole repository (34.57 GB, 28 files)",
}


def mbs(size, s):
    return f"{size / s / 1e6:,.0f} MB/s" if s else "—"


lines = []
for ds, title in TITLES.items():
    rows = [(k.split(" | ")[1], v) for k, v in res.items() if k.startswith(ds + " | ") and "archive" in v]
    if not rows:
        continue
    rows.sort(key=lambda r: r[1]["archive"])
    lines += [f"### {title}", "", "| tool | archive | % of input | compress | decompress | lossless |",
              "|---|---:|---:|---:|---:|---|"]
    for tool, v in rows:
        lines.append(f"| {'**' + tool + '**' if tool.startswith('Seedpack') else tool} | {v['archive'] / 1e6:,.1f} MB | "
                     f"{100 * v['archive'] / v['size']:.2f}% | {mbs(v['size'], v.get('compress_s'))} | "
                     f"{mbs(v['size'], v.get('decompress_s'))} | "
                     f"{v.get('lossless') if v.get('lossless') is not None else v.get('note', '')} |")
    lines.append("")
open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
