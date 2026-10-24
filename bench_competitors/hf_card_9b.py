"""Model card of the Hugging Face release neurosquad/Qwen3.5-9B-SEEDPACK.

usage: hf_card_9b.py ARCHIVE ORIGINAL_README OUT_DIR
Writes OUT_DIR/README.md (SeedPack header, timings from results_hf9b.json, per-file sizes read from
the archive, SHA-256 from hf9b_files.json, then the original unsloth/Qwen3.5-9B-GGUF card),
OUT_DIR/assets/seedpack-banner.png and seedpack-stats.png (both made by hf_banners.py).
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from seedpack.container import Reader  # noqa: E402
from seedpack.codecs_gguf import zstd_d  # noqa: E402

REPO = "neurosquad/Qwen3.5-9B-SEEDPACK"
SRC = "unsloth/Qwen3.5-9B-GGUF"
GITHUB = "https://github.com/glmn-ai/seedpack"
archive, orig_readme, out_dir = sys.argv[1:4]
ARCHIVE = os.path.basename(archive)
files = json.load(open(os.path.join(HERE, "hf9b_files.json")))
t = json.load(open(os.path.join(HERE, "results_hf9b.json")))
NB = " "


def size(b):
    return f"{b / 1e9:.2f}{NB}GB" if b >= 1e9 else f"{b / 1e6:.1f}{NB}MB" if b >= 1e6 else f"{b / 1e3:.1f}{NB}KB"


r = Reader(archive)
meta = r.meta
manifest = json.loads(zstd_d(r.get("manifest")))
n_in, n_out = meta["total"], r.size
stored = {}                                   # what each file costs in the archive
for i, name in enumerate(manifest["bases"]):
    stored[name] = r.chunks[f"b{i}"][1]
for i, m in enumerate(manifest["files"]):
    stored[m["name"]] = r.chunks[f"f{i}"][1] + (r.chunks[f"q{i}"][1] if f"q{i}" in r.chunks else 0)
for name, (chunk, _) in manifest["extras"]["single"].items():
    stored[name] = r.chunks[chunk][1]
bundle = r.chunks["extras"][1]
base = stored["Qwen3.5-9B-BF16.gguf"]
quants = [m["name"] for m in manifest["files"] if m["name"].startswith("Qwen3.5-9B-")]
quant_cost = sum(stored[n] for n in quants)
biggest = max(quants, key=lambda n: files[n]["size"])

contents = ["| File | Size | In the archive | SHA-256 |", "|---|---:|---:|---|"]
for name in sorted(files, key=lambda x: (not x.endswith(".gguf"), x)):
    st = size(stored[name]) if name in stored else "(bundle)"
    sha = files[name]["sha256"] or "git blob " + files[name]["blob"]
    contents.append(f"| `{name}` | {size(files[name]['size'])} | {st} | `{sha}` |")

card_src = open(orig_readme, encoding="utf-8").read()
m = re.match(r"---\n(.*?)\n---\n", card_src, re.S)
orig_body = card_src[m.end():] if m else card_src

header = f"""---
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- seedpack
- gguf
- lossless-compression
- qwen3.5
- qwen
- unsloth
---

<p align="center">
  <a href="{GITHUB}"><img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>
<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-stats.png" alt="Qwen3.5-9B: {len(files)} files, {size(n_in)}, in one {size(n_out)} archive, {n_in / n_out:.1f}× smaller" width="860">
</p>

# Qwen3.5-9B-SEEDPACK

**The complete [{SRC}](https://huggingface.co/{SRC}) repository in one {size(n_out)} archive instead of {size(n_in)}.** That covers the BF16 model, all {len(quants)} quants from UD-IQ2_XXS to UD-Q8_K_XL, the vision projector in BF16, F16 and F32, the imatrix and the README. The archive is {100 * n_out / n_in:.2f}% of the original size and smaller than the repository's largest single file, `{biggest}` ({size(files[biggest]['size'])}).

Every file is restored **byte for byte**. The SHA-256 of each file matches {SRC}; the full list is [below](#contents-of-the-archive).

## What is SeedPack?

[SeedPack]({GITHUB}) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad).

A GGUF repository ships the same weights 20–30 times: the BF16 model plus every quantization of it. SeedPack uses the fact that every quant is a deterministic function of the same weights:

- **The seed is stored once.** The BF16 model is coded with an entropy coder built for BF16 weights: {size(files['Qwen3.5-9B-BF16.gguf']['size'])} → {size(base)} ({100 * base / files['Qwen3.5-9B-BF16.gguf']['size']:.1f}%).
- **Quants are grown, not stored.** Each quant is computed again from the seed on the GPU with a bit-exact CUDA port of llama.cpp's quantizers (`ggml-quants.c`) and the repository's importance matrix.
- **Only the difference is kept.** Where a re-derived quant differs from the published file, SeedPack stores a residual coded with the quantizer's own probabilities. Here all {len(quants)} quants together cost **{size(quant_cost)}** on top of the seed.
- **Big models are streamed.** The BF16 model is larger than a 12 GB GPU, so packing and unpacking go through it in windows of about 1 GB of tensors. The BF16 file is never held whole, in RAM or on disk, unless you extract it.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

## How to use

### 1. Install SeedPack

```bash
pip install "seedpack>=0.2.2"
```

Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer). The model does not have to fit in GPU memory.
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 (12 GB) under Windows 10. No CUDA Toolkit is needed.
- Disk space for the archive ({size(n_out)}) plus the files you extract.

The first run compiles the GPU quantizers for your card (about 15 s) and caches them.

### 2. Extract quants or the whole repository

SeedPack reads the archive straight from this repository. `list` fetches only the index; `x` downloads the archive once into the Hugging Face cache.

```bash
seedpack list hf://{REPO}                                       # what is inside
seedpack x hf://{REPO} Qwen3.5-9B-Q4_K_M.gguf -o models          # one quant
seedpack x hf://{REPO} "*Q4_K_M*" mmproj-F16.gguf -o models      # several, by name or mask
seedpack x hf://{REPO} -o models                                # the whole repository
seedpack test hf://{REPO}                                       # decode and verify everything, write nothing
```

A downloaded copy works the same way:

```bash
hf download {REPO} {ARCHIVE} --local-dir .
seedpack x {ARCHIVE} Qwen3.5-9B-Q4_K_M.gguf -o models
```

### 3. Run the model

The extracted files are the original Unsloth GGUFs, so any GGUF runtime works:

```bash
llama-cli -m models/Qwen3.5-9B-Q4_K_M.gguf --mmproj models/mmproj-F16.gguf --jinja
```

Ollama, LM Studio, Jan and KoboldCpp open the extracted `.gguf` files directly. Recommended settings are in the original card below.

## Speed

Measured on an NVIDIA RTX 5070 (12 GB), an Intel Core i7-8700K (6 cores / 12 threads), 32 GB RAM and Windows 10, with the archive on two SATA SSDs in a stripe (about 1 GB/s) and a cold OS cache. Times include process start and file writes; a single extracted quant goes to another SATA SSD (about 380 MB/s, which is most of its time).

| Operation | Time | Speed |
|---|---:|---:|
| Pack the whole repository ({size(n_in)}) | {t['pack_s']:.0f} s | {n_in / t['pack_s'] / 1e6:,.0f} MB/s |
| Extract one quant (Q4_K_M, {size(files['Qwen3.5-9B-Q4_K_M.gguf']['size'])}) | {t['extract_one_s']:.0f} s | {files['Qwen3.5-9B-Q4_K_M.gguf']['size'] / t['extract_one_s'] / 1e6:,.0f} MB/s |
| Extract the whole repository | {t['extract_all_s']:.0f} s | {n_in / t['extract_all_s'] / 1e6:,.0f} MB/s |
| Test the whole repository (decode and verify, no writes) | {t['test_all_s']:.0f} s | {n_in / t['test_all_s'] / 1e6:,.0f} MB/s |

Peak GPU memory is {t['peak_vram_pack_gb']:.1f} GB when packing and {t['peak_vram_unpack_gb']:.1f} GB when unpacking. Extracting one quant reads the whole coded BF16 model ({size(base)}), so on a slower disk it takes about as long as reading that much. IQ quants take longer: their quantizers are slow even on the GPU (UD-IQ2_XXS: about {t['extract_iq2_xxs_s']:.0f} s).

The archive was made with SeedPack {meta['tool'].split()[-1]}.

## Contents of the archive

<details>
<summary>{len(files)} files, {size(n_in)}. Click to expand.</summary>

""" + "\n".join(contents) + f"""

`README.md` and `.gitattributes` are in the small-files bundle ({size(bundle)} with the imatrix).

</details>

## Original model card

> The original card of [{SRC}](https://huggingface.co/{SRC}) follows, unchanged. The model is by the [Qwen team](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache 2.0), and the GGUF quantizations are by [Unsloth](https://unsloth.ai).

---

"""

os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(header + orig_body)
shutil.copyfile(os.path.join(ROOT, "assets", "banner.png"), os.path.join(out_dir, "assets", "seedpack-banner.png"))
shutil.copyfile(os.path.join(ROOT, "assets", "stats-qwen3.5-9b.png"), os.path.join(out_dir, "assets", "seedpack-stats.png"))
print(f"{out_dir}/README.md: archive {size(n_out)} ({100 * n_out / n_in:.2f}%), quants {size(quant_cost)}, "
      f"base {size(base)}, {meta['tool']}")
