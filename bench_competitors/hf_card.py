"""Model card of the Hugging Face release neurosquad/Qwen3-0.6B-SEEDPACK.

usage: hf_card.py OUT_DIR [--extract-one SECONDS] [--extract-all SECONDS]
Writes OUT_DIR/README.md (SeedPack header + benchmark tables from results_hf06.json + the original
unsloth/Qwen3-0.6B-GGUF card) and OUT_DIR/assets/seedpack-banner.png.
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = "neurosquad/Qwen3-0.6B-SEEDPACK"
ARCHIVE = "Qwen3-0.6B-SEEDPACK.varc"
GITHUB = "https://github.com/glmn-ai/seedpack"
res = json.load(open(os.path.join(HERE, "results_hf06.json")))
files = json.load(open(os.path.join(HERE, "hf06_files.json")))
opt = {a: float(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a in ("--extract-one", "--extract-all")}
out_dir = sys.argv[1]

DATA = {"hf-0.6b": "This release, 30 files", "bf16-0.6b": "BF16", "q8_0-0.6b": "Q8_0", "q4_k_m-0.6b": "Q4_K_M"}
NOTES = {"ZipNN 0.5.4": "¹", "HF Xet (estimate)": "²", "DFloat11 (estimate)": "²"}


def size(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def rows(ds):
    out = [(k.split(" | ", 1)[1], v) for k, v in res.items() if k.startswith(ds + " | ") and "archive" in v]
    return sorted(out, key=lambda x: x[1]["archive"])


def bold(s, on):
    return f"**{s}**" if on else s


def full_table(ds):
    rs = rows(ds)
    n = rs[0][1]["size"]
    measured = [r for _, r in rs if "compress_s" in r]
    best_a = min(r["archive"] for _, r in rs)
    best_c = min(r["compress_s"] for r in measured)
    best_d = min(r["decompress_s"] for r in measured)
    t = ["| | Compressor | Archive | Ratio | Compression | Decompression + check | Lossless |",
         "|---:|---|---:|---:|---:|---:|:---:|"]
    for i, (tool, r) in enumerate(rs, 1):
        a = r["archive"]
        name = ("**SeedPack**" if tool == "SeedPack" else tool) + NOTES.get(tool, "")
        c = d = "—"
        if "compress_s" in r:
            c = bold(f"{r['compress_s']:.1f} s · {n / r['compress_s'] / 1e6:,.0f} MB/s", r["compress_s"] == best_c)
            d = bold(f"{r['decompress_s']:.1f} s · {n / r['decompress_s'] / 1e6:,.0f} MB/s", r["decompress_s"] == best_d)
        lossless = {True: "✓", False: "✗", None: "—"}[r.get("lossless")]
        t.append(f"| {i} | {name} | {bold(size(a), a == best_a)} | {bold(f'{100 * a / n:.2f}%', a == best_a)} | {c} | {d} | {lossless} |")
    return t


def single_table():
    sets = ["bf16-0.6b", "q8_0-0.6b", "q4_k_m-0.6b"]
    tools = sorted({k.split(" | ", 1)[1] for k in res if k.split(" | ")[0] in sets},
                   key=lambda t: sum(res.get(f"{s} | {t}", {}).get("archive", 1e12) / res[f"{s} | SeedPack"]["size"]
                                     for s in sets))
    best = {s: min(v["archive"] for k, v in res.items() if k.startswith(s + " | ") and "archive" in v) for s in sets}
    head = "| Compressor | " + " | ".join(f"{DATA[s]} ({size(res[f'{s} | SeedPack']['size'])})" for s in sets) + " |"
    t = [head, "|---|" + "---:|" * len(sets)]
    for tool in tools:
        cells = []
        for s in sets:
            r = res.get(f"{s} | {tool}")
            if not r or "archive" not in r:
                cells.append("n/a")
                continue
            p = f"{100 * r['archive'] / r['size']:.2f}%"
            if "compress_s" in r:
                p += f" · {r['size'] / r['compress_s'] / 1e6:,.0f} / {r['size'] / r['decompress_s'] / 1e6:,.0f} MB/s"
            cells.append(bold(p, r["archive"] == best[s]))
        name = ("**SeedPack**" if tool == "SeedPack" else tool) + NOTES.get(tool, "")
        t.append(f"| {name} | " + " | ".join(cells) + " |")
    return t


hf = rows("hf-0.6b")
sp = res["hf-0.6b | SeedPack"]
n_in = sp["size"]
others = [r for t, r in hf if t != "SeedPack" and "estimate" not in t]
runner = min(others, key=lambda r: r["archive"])
runner_name = next(t for t, r in hf if r is runner)
bf16_alone = res["bf16-0.6b | SeedPack"]["archive"]
quant_cost = sp["archive"] - bf16_alone
sys.path.insert(0, ROOT)
from seedpack import __version__  # noqa: E402

ext = ""
if opt:
    ext = (f"\nOn the benchmark machine below, extracting one quant takes about {opt['--extract-one']:.0f} s and the whole "
           f"repository about {opt['--extract-all']:.0f} s, file writes included (plus 2–4 s to start).\n") if len(opt) == 2 else ""

card_src = open(os.path.join(ROOT, "models", "unsloth", "README.md"), encoding="utf-8").read()
m = re.match(r"---\n(.*?)\n---\n", card_src, re.S)
orig_body = card_src[m.end():] if m else card_src

contents = ["| File | Size | SHA-256 |", "|---|---:|---|"]
for name in sorted(files, key=lambda x: (not x.endswith(".gguf"), x)):
    contents.append(f"| `{name}` | {size(files[name]['size'])} | `{files[name]['sha256']}` |")

header = f"""---
base_model: Qwen/Qwen3-0.6B
base_model_relation: quantized
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- seedpack
- gguf
- lossless-compression
- qwen3
- qwen
- unsloth
---

<p align="center">
  <a href="{GITHUB}"><img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>

# Qwen3-0.6B-SEEDPACK

**The complete [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) repository in one {size(sp['archive'])} archive instead of {size(n_in)}.** That covers the BF16 model, all 25 quants from UD-IQ1_S to UD-Q8_K_XL, `config.json`, the Ollama `template` and `params`, and the imatrix. The archive is {100 * sp['archive'] / n_in:.1f}% of the original size, {runner['archive'] / sp['archive']:.1f}× smaller than the best other compressor ({runner_name}).

Every file is restored **byte for byte**. The SHA-256 of each file matches unsloth/Qwen3-0.6B-GGUF; the full list is [below](#contents-of-the-archive).

## What is SeedPack?

[SeedPack]({GITHUB}) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad).

A GGUF repository ships the same weights 20–30 times: the BF16 model plus every quantization of it. General-purpose compressors see gigabytes of almost incompressible bytes. SeedPack uses the fact that every quant is a deterministic function of the same weights:

- **The seed is stored once.** The BF16 model is coded with an entropy coder built for BF16 weights ({100 * bf16_alone / res['bf16-0.6b | SeedPack']['size']:.1f}% of its size).
- **Quants are grown, not stored.** Each quant is computed again from the seed on the GPU. This uses a bit-exact CUDA port of llama.cpp's quantizers (`ggml-quants.c`) and the repository's importance matrix.
- **Only the difference is kept.** Where a re-derived quant differs from the published file, SeedPack stores a residual. The residual is coded with the quantizer's own probabilities. Here the 25 quants and the four small files together add only **{quant_cost / 1e6:.1f} MB** on top of the seed.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

When a repository has no BF16 model, the most precise quant (for example Q8_0) becomes the seed and the other quants grow from it.

## How to use

### 1. Install SeedPack

```bash
pip install seedpack
```

Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer).
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 under Windows 10 and Ubuntu (WSL2). No CUDA Toolkit is needed.

The first run compiles the GPU quantizers for your card (about 15 s) and caches them.

### 2. Extract quants or the whole repository

SeedPack reads the archive straight from this repository. `list` fetches only the index; `x` downloads the archive once into the Hugging Face cache.

```bash
seedpack list hf://{REPO}                                          # what is inside
seedpack x hf://{REPO} Qwen3-0.6B-Q4_K_M.gguf -o models            # one quant
seedpack x hf://{REPO} "*Q4_K*" "*Q8_0*" -o models                 # several, by name or mask
seedpack x hf://{REPO} -o models                                   # the whole repository
seedpack test hf://{REPO}                                          # decode and verify everything, write nothing
```

Only the files you ask for are written; the seed is decoded in memory. A downloaded copy works the same way:

```bash
hf download {REPO} {ARCHIVE} --local-dir .
seedpack x {ARCHIVE} Qwen3-0.6B-Q4_K_M.gguf -o models
```
{ext}
### 3. Run the model

The extracted files are the original Unsloth GGUFs, so any GGUF runtime works.

```bash
# llama.cpp
llama-cli -m models/Qwen3-0.6B-Q4_K_M.gguf --jinja --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 32768

# Ollama
echo FROM ./models/Qwen3-0.6B-Q4_K_M.gguf > Modelfile
ollama create qwen3-0.6b -f Modelfile
ollama run qwen3-0.6b
```

LM Studio, Jan, KoboldCpp and Open WebUI can open the extracted `.gguf` directly. The recommended sampling settings are in the original card below: thinking mode uses `temp 0.6, top_p 0.95, top_k 20, min_p 0`.

## Benchmark

The input is all 30 files of this release, {size(n_in)}. The best value in each column is in **bold**.

Test machine:
- NVIDIA RTX 5070 (12 GB).
- Intel Core i7-8700K (6 cores / 12 threads).
- 32 GB RAM, NVMe SSD, Windows 10.

Compression goes from files on disk to an archive on disk. Decompression decodes everything and checks the full output (BLAKE3 or CRC) without writing files.

""" + "\n".join(full_table("hf-0.6b")) + f"""

### Single files

Each cell shows the ratio, then compression / decompression speed.

""" + "\n".join(single_table()) + f"""

Notes:
- ¹ ZipNN runs in memory inside WSL; disk I/O is not counted.
- ² Size estimates from a model of the format. HF Xet is how the Hub stores files: content-defined chunks, deduplication and LZ4. DFloat11 applies to BF16 weights only.
- ZipLLM is not included: it accepts safetensors only.
- Single-stream compressors (zstd, xz) read the files concatenated, like a tar. brotli, bzip2, gzip and lz4 compress 64 MB blocks on 12 workers, like pigz and pbzip2. 7-Zip uses its own multi-threaded LZMA2.
- brotli runs at quality 7. Qualities 9–11 compress BF16 data at about 1 MB/s per core (hours for this set) and gain less than 0.5%.
- SeedPack times include process start and CUDA initialization (2–4 s). The archive was made with SeedPack {__version__}.
- To reproduce, run [`bench_competitors/hf_bench.py`]({GITHUB}/blob/main/bench_competitors/hf_bench.py).

## Contents of the archive

<details>
<summary>{len(files)} files, {size(n_in)}. Click to expand.</summary>

""" + "\n".join(contents) + """

</details>

## Original model card

> The original card of [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) follows, unchanged. The model is by the [Qwen team](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0), and the GGUF quantizations are by [Unsloth](https://unsloth.ai).

---

"""

os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(header + orig_body)
shutil.copyfile(os.path.join(ROOT, "assets", "banner.png"), os.path.join(out_dir, "assets", "seedpack-banner.png"))
print(f"{out_dir}/README.md: archive {size(sp['archive'])} ({100 * sp['archive'] / n_in:.2f}%), quants {quant_cost / 1e6:.2f} MB, "
      f"runner-up {runner_name} {runner['archive'] / sp['archive']:.1f}x, seedpack {__version__}")
