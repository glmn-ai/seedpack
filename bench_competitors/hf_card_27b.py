"""Model card of the Hugging Face release neurosquad/Qwen3.8-27B-SEEDPACK.

usage: hf_card_27b.py ARCHIVE ORIGINAL_README OUT_DIR
Writes OUT_DIR/README.md (SeedPack header, timings from results_hf27b.json, per-file sizes read from
the archive, SHA-256 from hf27b_files.json, then the original unsloth/Qwen3.8-27B-GGUF card),
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

MODEL = "Qwen3.8-27B"
REPO = f"neurosquad/{MODEL}-SEEDPACK"
SRC = f"unsloth/{MODEL}-GGUF"
GITHUB = "https://github.com/glmn-ai/seedpack"
ONE = f"{MODEL}-UD-Q4_K_XL.gguf"                # the quant the examples and the timings use
OLD_IMATRIX = [f"{MODEL}-Q4_0.gguf", f"{MODEL}-Q4_1.gguf"]   # made with an earlier imatrix (45 chunks)
archive, orig_readme, out_dir = sys.argv[1:4]
ARCHIVE = os.path.basename(archive)
files = json.load(open(os.path.join(HERE, "hf27b_files.json")))
t = json.load(open(os.path.join(HERE, "results_hf27b.json")))
NB = " "


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
shards = sorted(n for n in manifest["bases"] if n.startswith("BF16/"))
base, base_in = sum(stored[n] for n in shards), sum(files[n]["size"] for n in shards)
quants = [m["name"] for m in manifest["files"] if m["name"].startswith(f"{MODEL}-")]
mtp = next(m["name"] for m in manifest["files"] if m["name"].startswith("MTP/"))
ud = [n for n in quants if "-UD-" in n]
exact = [n for n in quants if n not in ud and n not in OLD_IMATRIX] + [mtp, "mmproj-F16.gguf"]
cost = {k: sum(stored[n] for n in v) for k, v in (("ud", ud), ("exact", exact), ("old", OLD_IMATRIX))}
ud_in = sum(files[n]["size"] for n in ud)
fitted = r.chunks["imx"][1] if "imx" in r.chunks else 0

contents = ["| File | Size | In the archive | SHA-256 |", "|---|---:|---:|---|"]
for name in sorted(files, key=lambda x: (not x.endswith(".gguf"), x)):
    st = size(stored[name]) if name in stored else "(bundle)"
    sha = files[name]["sha256"] or "git blob " + files[name]["blob"]
    contents.append(f"| `{name}` | {size(files[name]['size'])} | {st} | `{sha}` |")

card_src = open(orig_readme, encoding="utf-8").read()
m = re.match(r"---\n(.*?)\n---\n", card_src, re.S)
orig_body = card_src[m.end():] if m else card_src
old = " and ".join(f"{n.removeprefix(MODEL + '-').removesuffix('.gguf')} ({size(stored[n])} of {size(files[n]['size'])})"
                   for n in OLD_IMATRIX)
plain = ", ".join(n.removeprefix(MODEL + "-").removesuffix(".gguf") for n in exact if n.startswith(MODEL))
refs = sum(e[1] == "ref" for m in manifest["files"] for e in m["plan"])

rows = [(f"Pack the whole repository ({size(n_in)})", n_in, t.get("pack_s")),
        (f"Extract one quant (UD-Q4_K_XL, {size(files[ONE]['size'])})", files[ONE]["size"], t.get("extract_one_s")),
        ("Extract the whole repository", n_in, t.get("extract_all_s")),
        ("Test the whole repository (decode and verify, no writes)", n_in, t.get("test_all_s"))]
speed = "\n".join(f"| {lab} | {sec:.0f} s | {n / sec / 1e6:,.0f} MB/s |" for lab, n, sec in rows if sec)

header = f"""---
base_model: Qwen/{MODEL}
base_model_relation: quantized
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/{MODEL}/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- seedpack
- gguf
- lossless-compression
- qwen3.8
- qwen
- unsloth
---

<p align="center">
  <a href="{GITHUB}"><img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>
<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-stats.png" alt="{MODEL}: {len(files)} files, {size(n_in)}, in one {size(n_out)} archive, {n_in / n_out:.1f}× smaller" width="860">
</p>

# {MODEL}-SEEDPACK

**The complete [{SRC}](https://huggingface.co/{SRC}) repository in one {size(n_out)} archive instead of {size(n_in)}.** That covers the BF16 model in its {len(shards)} shards, all {len(quants)} quants from UD-IQ1_S to UD-Q8_K_XL, the multi-token-prediction layer (`{mtp}`), the vision projector in BF16 and F16, the imatrix and the README. The archive is {100 * n_out / n_in:.2f}% of the original size{f", smaller than the BF16 model alone ({size(base_in)})" if n_out < base_in else ""}.

Every file is restored **byte for byte**. The SHA-256 of each file matches {SRC}; the full list is [below](#contents-of-the-archive).

## What is SeedPack?

[SeedPack]({GITHUB}) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad).

A GGUF repository ships the same weights 20–30 times: the BF16 model plus every quantization of it. SeedPack uses the fact that every quant is a deterministic function of the same weights:

- **The seed is stored once.** The BF16 model is coded with an entropy coder built for BF16 weights: {size(base_in)} → {size(base)} ({100 * base / base_in:.1f}%). Its shards stay separate files, each restored as it is, but the quants are grown from them as from one model.
- **Quants are grown, not stored.** Each quant is computed again from the seed on the GPU with a bit-exact CUDA port of llama.cpp's quantizers (`ggml-quants.c`) and the repository's importance matrix.
- **Only the difference is kept.** Where a re-derived quant differs from the published file, SeedPack stores a residual coded with the quantizer's own probabilities:
  - {plain}, the MTP layer and the F16 vision projector come out exactly: **{size(cost['exact'])}** together.
  - Q4_0 and Q4_1 were made with an earlier importance matrix (45 chunks, as their metadata records) than the one in the repository (1251 chunks). SeedPack fits that matrix back from the quants themselves, by least squares on their block scales, and keeps it in the archive ({size(fitted)}): {old}.
  - The {len(ud)} Unsloth Dynamic (UD) quants come from Unsloth's own pipeline, whose codes drift from plain rounding along each row, so they cannot be re-derived exactly from BF16 and the imatrix. Their residuals cost **{size(cost['ud'])}**, {100 * cost['ud'] / ud_in:.1f}% of their {size(ud_in)}: the coder knows where in the row each weight is, and a tensor that several quants share (the same Q5_K `ffn_up` in UD-Q5_K_M and UD-Q5_K_XL, say) is stored once, the later files keeping a reference to it ({refs} of them here).
- **Big models are streamed.** The BF16 model is larger than a 12 GB GPU and than the test machine's 32 GB of RAM, so packing and unpacking go through it in windows of about 1 GB of tensors. It is never held whole, in RAM or on disk, unless you extract it.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

## How to use

### 1. Install SeedPack

```bash
pip install "seedpack>=0.3.0"
```

Version 0.3.0 or newer is needed: it is the first to handle a model split into shards, to code the IQ grid types and the drift of the Dynamic quants, and to fit an importance matrix back. Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer). The model does not have to fit in GPU memory.
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 (12 GB) under Windows 10. No CUDA Toolkit is needed.
- Disk space for the archive ({size(n_out)}) plus the files you extract.

The first run compiles the GPU quantizers for your card (about 15 s) and caches them.

### 2. Extract quants or the whole repository

SeedPack reads the archive straight from this repository. `list` fetches only the index; `x` downloads the archive once into the Hugging Face cache.

```bash
seedpack list hf://{REPO}                                           # what is inside
seedpack x hf://{REPO} {ONE} -o models               # one quant
seedpack x hf://{REPO} "*UD-Q4_K_XL*" mmproj-F16.gguf -o models      # several, by name or mask
seedpack x hf://{REPO} -o models                                    # the whole repository
seedpack test hf://{REPO}                                           # decode and verify everything, write nothing
```

A downloaded copy works the same way:

```bash
hf download {REPO} {ARCHIVE} --local-dir .
seedpack x {ARCHIVE} {ONE} -o models
```

### 3. Run the model

The extracted files are the original Unsloth GGUFs, so any GGUF runtime works:

```bash
llama-cli -m models/{ONE} --mmproj models/mmproj-F16.gguf --jinja
```

Ollama, LM Studio, Jan and KoboldCpp open the extracted `.gguf` files directly. Recommended settings are in the original card below.

## Speed

Measured on an NVIDIA RTX 5070 (12 GB), an Intel Core i7-8700K (6 cores / 12 threads), 32 GB RAM and Windows 10, with the source files on two SATA SSDs in a stripe (about 1 GB/s) and the archive written to another SATA SSD. Times include process start and file writes.

| Operation | Time | Speed |
|---|---:|---:|
{speed}

Peak GPU memory is {t['peak_vram_pack_gb']:.1f} GB when packing. Extracting one quant reads the whole coded BF16 model ({size(base)}), so it takes at least as long as the disk needs for that; IQ quants take longer, their quantizers are slow even on the GPU.

The archive was made with SeedPack {meta['tool'].split()[-1]}.

## Contents of the archive

<details>
<summary>{len(files)} files, {size(n_in)}. Click to expand.</summary>

""" + "\n".join(contents) + f"""

`README.md`, `config.json` and `.gitattributes` are in the small-files bundle ({size(bundle)} with the imatrix).

</details>

## Original model card

> The original card of [{SRC}](https://huggingface.co/{SRC}) follows, unchanged. The model is by the [Qwen team](https://huggingface.co/Qwen/{MODEL}) (Apache 2.0), and the GGUF quantizations are by [Unsloth](https://unsloth.ai).

---

"""

os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(header + orig_body)
shutil.copyfile(os.path.join(ROOT, "assets", "banner.png"), os.path.join(out_dir, "assets", "seedpack-banner.png"))
shutil.copyfile(os.path.join(ROOT, "assets", f"stats-{MODEL.lower()}.png"), os.path.join(out_dir, "assets", "seedpack-stats.png"))
print(f"{out_dir}/README.md: archive {size(n_out)} ({100 * n_out / n_in:.2f}%), exact {size(cost['exact'])}, "
      f"UD {size(cost['ud'])}, old imatrix {old} + {size(fitted)}, base {size(base)}, {meta['tool']}")
