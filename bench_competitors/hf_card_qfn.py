"""Model card of the Hugging Face release neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK.

usage: hf_card_qfn.py ARCHIVE ORIGINAL_README BASE_LICENSE OUT_DIR
Writes OUT_DIR/README.md (SeedPack header; what the archive spends on each kind of tensor, read from its
manifest; timings from results_hfqfn.json; SHA-256 from hfqfn_files.json; then the original
ISTA-DASLab card, its relative image links pointed at that repository), OUT_DIR/LICENSE (the base
model's, which asks for its notice in every copy) and OUT_DIR/assets/seedpack-banner.png and
seedpack-stats.png (made by hf_banners.py).
"""
import json
import os
import re
import shutil
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from seedpack.container import Reader  # noqa: E402
from seedpack.codecs_gguf import zstd_d  # noqa: E402
from seedpack.gguf_reader import TYPE_NAMES  # noqa: E402

MODEL = "Qwen3.8-Flash-Next"
NAME = f"{MODEL}-GSQ-RCO"
REPO = f"neurosquad/{NAME}-SEEDPACK"
SRC = f"ISTA-DASLab/{NAME}-GGUF"
SRC_REV = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
GITHUB = "https://github.com/glmn-ai/seedpack"
SIZES = ["IQ3_S", "IQ3_XXS", "IQ2_XS", "Q2_0"]
MMPROJ = f"mmproj-{MODEL}-BF16.gguf"
archive, orig_readme, base_license, out_dir = sys.argv[1:5]
ARCHIVE = os.path.basename(archive)
files = json.load(open(os.path.join(HERE, "hfqfn_files.json")))
tpath = os.path.join(HERE, "results_hfqfn.json")
t = json.load(open(tpath)) if os.path.exists(tpath) else {}
NB = " "


def size(b):
    return f"{b / 1e9:.2f}{NB}GB" if b >= 1e9 else f"{b / 1e6:.1f}{NB}MB" if b >= 1e6 else f"{b / 1e3:.1f}{NB}KB"


def pct(a, b):
    return f"{100 * a / b:.1f}%"


def shard(q, i):
    return f"{q}/{NAME}-{q}-0000{i}-of-00002.gguf"


r = Reader(archive)
meta = r.meta
man = json.loads(zstd_d(r.get("manifest")))
n_in, n_out = meta["total"], r.size
fman, groups = man["files"], man["groups"]
names = [m["name"] for m in fman]
chunk = {k: v[1] for k, v in r.chunks.items()}

# what each version of each tensor costs: its g{gi}.{k}.{j} chunks; small tensors share the "small" chunk
cost = defaultdict(int)
for k, n in chunk.items():
    m = re.fullmatch(r"g(\d+)\.(\d+)\.\d+", k)
    if m:
        cost[int(m[1]), int(m[2])] += n
small_in = sum(v["nbytes"] for g in groups for v in g["vers"] if "small" in v)
in_mmproj = {g for g, _, _, _ in fman[names.index(MMPROJ)]["tensors"]}
ngram = [gi for gi, g in enumerate(groups) if g["name"].startswith("per_layer_token_embd")]
assert len(ngram) == 1, ngram
NG = ngram[0]
first, later, mm = [0, 0], [0, 0], [0, 0]           # bytes in, bytes in the archive
pairs = defaultdict(lambda: [0, 0, 0])             # (type, type of the chain's first version): in, out, tensors
for gi, g in enumerate(groups):
    for k, v in enumerate(g["vers"]):
        if "small" in v or gi == NG:
            continue
        acc = mm if gi in in_mmproj else first if k == 0 else later
        acc[0] += v["nbytes"]
        acc[1] += cost[gi, k]
        if k and gi not in in_mmproj:
            p = pairs[TYPE_NAMES[v["type"]], TYPE_NAMES[g["vers"][0]["type"]]]
            p[0] += v["nbytes"]
            p[1] += cost[gi, k]
            p[2] += 1
ng_in, ng_out = groups[NG]["vers"][0]["nbytes"], cost[NG, 0]
distinct = sum(v["nbytes"] for g in groups for v in g["vers"])
gguf_in = sum(m["size"] for m in fman)
heads_in = sum(m["head"][1] + sum(x[2] for x in m["gaps"]) for m in fman if m.get("same") is None)
dup = gguf_in - distinct - heads_in                     # identical files and tensors stored once
same_files = [m["name"] for m in fman if m.get("same") is not None]
rest_out = n_out - ng_out - first[1] - later[1] - mm[1]
rest_in = n_in - gguf_in + heads_in + small_in

ng_copies = 1 + len(same_files)
breakdown = [
    (f"The n-gram table (`per_layer_token_embd`, IQ4_NL): shard 2, the same file in all {ng_copies} sizes",
     ng_copies * ng_in, ng_out, f"stored once; {pct(ng_out, ng_in)} of one copy"),
    ("Tensors that are the same bytes in several sizes (most dense tensors of the same type)",
     dup - (ng_copies - 1) * ng_in, 0, "stored once"),
    ("The first (largest) version of every other tensor, coded alone", first[0], first[1], pct(first[1], first[0])),
    ("Every later version, coded given the earlier ones", later[0], later[1], pct(later[1], later[0])),
    (f"The vision projector (`{MMPROJ}`)", mm[0], mm[1], pct(mm[1], mm[0])),
    ("GGUF headers, small tensors, the allocations, the README, the plots", rest_in, rest_out, pct(rest_out, rest_in)),
]
assert abs(sum(b[1] for b in breakdown) - n_in) < 1e6, (sum(b[1] for b in breakdown), n_in)
btable = ["| Part of the repository | Original | In the archive | |", "|---|---:|---:|---|"]
btable += [f"| {lab} | {size(a)} | {size(b) if b else '0'} | {note} |" for lab, a, b, note in breakdown]
btable.append(f"| **Total** | **{size(n_in)}** | **{size(n_out)}** | **{pct(n_out, n_in)}** |")
top = sorted(pairs.items(), key=lambda kv: -kv[1][0])[:6]
ptable = ["| Version | Chain starts at | Tensors | Original | In the archive | |", "|---|---|---:|---:|---:|---:|"]
ptable += [f"| {a} | {b} | {c[2]} | {size(c[0])} | {size(c[1])} | {pct(c[1], c[0])} |" for (a, b), c in top]

sizes_in = {q: files[shard(q, 1)]["size"] + files[shard(q, 2)]["size"] for q in SIZES}
smallest, largest = min(sizes_in.values()), max(sizes_in.values())

contents = ["| File | Size | SHA-256 |", "|---|---:|---|"]
for name in sorted(files, key=lambda x: (not x.endswith(".gguf"), x)):
    sha = files[name]["sha256"] or "git blob " + files[name]["blob"]
    contents.append(f"| `{name}` | {size(files[name]['size'])} | `{sha}` |")

card_src = open(orig_readme, encoding="utf-8").read()
m = re.match(r"---\n(.*?)\n---\n", card_src, re.S)
orig_body = card_src[m.end():] if m else card_src
orig_body = re.sub(r'(src="|\]\()(assets/[^")]+)', rf"\1https://huggingface.co/{SRC}/resolve/{SRC_REV}/\2", orig_body)

rows = [(f"Pack the whole repository ({size(n_in)})", n_in, t.get("pack_s")),
        (f"Extract one size (IQ3_S, {size(sizes_in['IQ3_S'])})", sizes_in["IQ3_S"], t.get("extract_one_s")),
        ("Extract the whole repository", n_in, t.get("extract_all_s")),
        ("Test the whole repository (decode and verify, no writes)", n_in, t.get("test_all_s"))]
speed = "\n".join(f"| {lab} | {sec:.0f} s | {n / sec / 1e6:,.0f} MB/s |" for lab, n, sec in rows if sec)
speed_section = f"""## Speed

Measured on an NVIDIA RTX 5070 (12 GB), an Intel Core i7-8700K (6 cores / 12 threads), 32 GB RAM and Windows 10, with the source files on two SATA SSDs in a stripe (about 1 GB/s) and the archive written to another SATA SSD. Times include process start and file writes.

| Operation | Time | Speed |
|---|---:|---:|
{speed}
{f"{chr(10)}Peak GPU memory is {t['peak_vram_pack_gb']:.1f} GB when packing and {t['peak_vram_test_gb']:.1f} GB when testing; process memory {t['peak_private_pack_gb']:.1f} GB and {t['peak_private_test_gb']:.1f} GB." if t.get('peak_vram_test_gb') else ""}

Unpacking several files writes them all at once: each tensor's chain is decoded once for every size that holds one of its versions. One size alone needs the versions before its own in each chain, so the smaller sizes take longer to extract than their bytes alone would.
""" if speed else ""

header = f"""---
base_model: Qwen/{MODEL}
base_model_relation: quantized
language:
- en
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- seedpack
- gguf
- lossless-compression
- qwen3.8
- qwen
- gsq
- rco
- moe
---

<p align="center">
  <a href="{GITHUB}"><img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>
<p align="center">
  <img src="https://huggingface.co/{REPO}/resolve/main/assets/seedpack-stats.png" alt="{MODEL} GSQ-RCO: {len(files)} files, {size(n_in)}, in one {size(n_out)} archive, {n_in / n_out:.1f}× smaller" width="860">
</p>

# {NAME}-SEEDPACK

**The complete [{SRC}](https://huggingface.co/{SRC}) repository in one {size(n_out)} archive instead of {size(n_in)}.** That is all four sizes, {", ".join(SIZES)}, two shards each, plus the BF16 vision projector, the RCO tensor allocations, the plots and the README. The archive is {pct(n_out, n_in)} of the original, {n_in / n_out:.2f}× smaller.

Every file is restored **byte for byte**. The SHA-256 of each file matches {SRC} at revision [`{SRC_REV[:8]}`](https://huggingface.co/{SRC}/tree/{SRC_REV}); the full list is [below](#contents-of-the-archive).

> **Do you need only one size?** Then download it from [the original repository]({f"https://huggingface.co/{SRC}"}): one size is {size(smallest)} to {size(largest)}, less than this archive. The archive is for keeping, mirroring or moving **all four** sizes: together they cost {n_out / largest:.2f}× the largest one alone.

## What is SeedPack?

[SeedPack]({GITHUB}) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad). For repositories that ship a BF16 model it grows the quants back from it. This one has no BF16 model, so SeedPack 0.4.0 codes the four sizes against each other (method M30):

- **The same bytes are stored once.** Shard 2 of every size is the same {size(ng_in)} file: the n-gram embedding table, IQ4_NL. Most dense tensors that two sizes keep at the same type are the same bytes too.
- **The rest are versions of one tensor.** ISTA-DASLab's GSQ quantizes every tensor anew for each size, so even two IQ3_S versions of an expert tensor in two sizes differ. SeedPack sorts the versions of each tensor by size and codes the largest alone ({pct(first[1], first[0])} of its bytes: GSQ's codes are nearly uniform, so there is little to take without a reference).
- **Each later version is coded given the earlier ones.** SeedPack dequantizes them, takes their precision-weighted mean, quantizes it to the next version's type with a bit-exact CUDA port of llama.cpp's quantizers, and codes the published file against that prediction: scales first, then each weight's code given the true scale and the estimate. All later versions together cost {pct(later[1], later[0])} of their bytes.
- **Big tensors are cut in slices.** The tensors of a 512-expert MoE run to several GB; slices of 64M weights {f"kept packing within {t['peak_vram_pack_gb']:.1f} GB of GPU memory and {t['peak_private_pack_gb']:.1f} GB of RAM" if t.get('peak_private_pack_gb') else "keep packing and unpacking within a 12 GB GPU and 32 GB of RAM"}.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

### Where the bytes go

{chr(10).join(btable)}

The largest groups of later versions:

{chr(10).join(ptable)}

## How to use

### 1. Install SeedPack

```bash
pip install "seedpack>=0.4.0"
```

Version 0.4.0 or newer is needed: it is the first with method M30 and the Q2_0 type. Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer). The model does not have to fit in GPU memory.
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 (12 GB) under Windows 10. No CUDA Toolkit is needed.
- Disk space for the archive ({size(n_out)}) plus the files you extract.

### 2. Extract sizes or the whole repository

```bash
seedpack list hf://{REPO}                                    # what is inside
seedpack x hf://{REPO} "IQ3_S/*" {MMPROJ} -o models   # one size and the vision projector
seedpack x hf://{REPO} -o models                             # the whole repository
seedpack test hf://{REPO}                                    # decode and verify everything, write nothing
```

`list` fetches only the index; `x` and `test` download the archive once into the Hugging Face cache. A downloaded copy works the same way:

```bash
hf download {REPO} {ARCHIVE} --local-dir .
seedpack x {ARCHIVE} "IQ3_XXS/*" -o models
```

### 3. Run the model

The extracted files are ISTA-DASLab's GGUFs, so everything in the original card below applies, for example:

```bash
llama-cli -m models/{shard("IQ3_S", 1)} -lm mmap --lazy-mode on -ngl 99
```

[Strata](https://github.com/Niko1221/Strata) takes a folder of shards you already have: `strata launch --model-path models/IQ3_S`, or `strata models import models/IQ3_S --plan` to see what it would make of it first.

{speed_section}
The archive was made with SeedPack {meta['tool'].split()[-1]}.

## Contents of the archive

<details>
<summary>{len(files)} files, {size(n_in)}. Click to expand.</summary>

""" + "\n".join(contents) + f"""

</details>

## License

The weights are [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/{MODEL}) by the Qwen team, under the Qwen Community License 1.0 ([LICENSE](LICENSE), copied from the base model: Copyright (c) 2026 Qwen). The GGUF quantizations are by the [Deep Algorithms and Systems Lab](https://github.com/IST-DASLab), ISTA, made with GSQ and RCO. SeedPack only packs their files, unchanged.

## Original model card

> The original card of [{SRC}](https://huggingface.co/{SRC}) follows, unchanged except that its images are linked from that repository. Its `<this-repo>` is ISTA-DASLab's repository.

---

"""

os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(header + orig_body)
shutil.copyfile(base_license, os.path.join(out_dir, "LICENSE"))
shutil.copyfile(os.path.join(ROOT, "assets", "banner.png"), os.path.join(out_dir, "assets", "seedpack-banner.png"))
shutil.copyfile(os.path.join(ROOT, "assets", f"stats-{NAME.lower()}.png"), os.path.join(out_dir, "assets", "seedpack-stats.png"))
print(f"{out_dir}/README.md: archive {size(n_out)} ({pct(n_out, n_in)}); n-gram {size(ng_out)}, first {size(first[1])} "
      f"of {size(first[0])}, later {size(later[1])} of {size(later[0])}, mmproj {size(mm[1])}, rest {size(rest_out)}; "
      f"{meta['tool']}")
