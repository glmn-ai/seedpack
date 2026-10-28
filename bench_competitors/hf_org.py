"""Organization card of NeuroSquad on Hugging Face (the README.md of the Space neurosquad/README).

usage: hf_org.py OUT_DIR
Writes OUT_DIR/README.md and OUT_DIR/assets/neurosquad-banner.png, seedpack-benchmarks.png: the
banner (NeuroSquad, SeedPack, the Qwen3.8-27B result) and the benchmarks (the three releases, the
archivers compared on all of unsloth/Qwen3-0.6B-GGUF, the 27B timings). Sizes from hf_banners.RELEASES,
the 27B archive's parts and timings from results_hf27b.json. Rendered and compressed as the release
banners (hf_banners.py: headless Chrome, imagequant, oxipng), on a narrower page: the organization
card's column shows them at about 430 px.
"""
import html
import json
import os
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_banners  # noqa: E402
from hf_banners import ASSETS, RELEASES, render, size, squeeze, uri  # noqa: E402

PAGE = 860                                     # CSS width: the card's column shows the banners about 430 px wide

ORG = "neurosquad"
SPACE = f"{ORG}/README"
GITHUB = "https://github.com/glmn-ai/seedpack"
PYPI = "https://pypi.org/project/seedpack/"
# all of unsloth/Qwen3-0.6B-GGUF with other archivers (results_hf06.json): label, key
ARCHIVERS = [("SeedPack", "SeedPack"), ("lmz 1.3", "lmz 1.3.0"), ("ggufpacker 0.4", "ggufpacker 0.4.2"),
             ("HF Xet (estimate)", "HF Xet (estimate)"),
             ("zstd -19 --long", "zstd 1.5.7 -19 --long=30"), ("7-Zip -mx9", "7-Zip 23.01 -mx9 (LZMA2)"),
             ("ZipNN 0.5", "ZipNN 0.5.4"), ("gzip -9", "gzip -9"), ("lz4 -9", "lz4 1.9.4 -9 (HC)")]
NB = " "

out_dir = sys.argv[1]
rels = []
for rel in RELEASES:
    files = json.load(open(os.path.join(HERE, rel["files"])))
    sizes = sorted({n.split("/")[0] for n in files if n.endswith(".gguf") and "/" in n and not n.startswith("BF16/")})
    bf16 = any("-BF16" in n and not n.startswith("mmproj") for n in files)
    rels.append(dict(rel, n_in=sum(v["size"] for v in files.values()), n_files=len(files), bf16=bf16,
                     n_quants=len(sizes) if not bf16 else
                     sum(1 for n in files if n.endswith(".gguf") and "-BF16" not in n and "mmproj" not in n
                         and not os.path.basename(n).startswith(("mtp-", "imatrix")))))
big = next(r for r in rels if r["model"] == "Qwen3.8-27B")
qfn = next((r for r in rels if r.get("name") == "Qwen3.8-Flash-Next-GSQ-RCO"), None)
b27 = json.load(open(os.path.join(HERE, "hf27b_files.json")))
bf16_27 = sum(v["size"] for n, v in b27.items() if n.startswith("BF16/"))
res06 = json.load(open(os.path.join(HERE, "results_hf06.json")))
t27 = json.load(open(os.path.join(HERE, "results_hf27b.json")))


def gb(b):
    return size(b).replace(NB, " ")


def dur(s):
    """pack time: seconds under a minute, else minutes; None (an estimate) -> a dash"""
    if s is None:
        return "—"
    return f"{s:.0f}{NB}s" if s < 60 else f"{s / 60:.1f}{NB}min"


STYLE = """
  html, body { margin: 0; background: transparent; }
  .b { box-sizing: border-box; width: %dpx; border-radius: 32px; background: #0E1311; border: 3px solid #1F2722;
       position: relative; overflow: hidden; font-family: 'Exo 2', sans-serif; color: #fff; font-variant-numeric: tabular-nums; }
  .b > * { position: relative; }
  .g { background: linear-gradient(90deg, #9BE34A, #29C074); -webkit-background-clip: text; color: transparent; padding-right: 6px; }
  .muted { color: #A9B5AE; }
""" % PAGE


def hero():
    ratio = big["n_in"] / big["archive"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,500;0,600;1,800&display=block" rel="stylesheet">
<style>{STYLE}
  .b {{ padding: 40px 46px 44px; }}
  .b::before {{ content: ""; position: absolute; left: -200px; top: -260px; width: 760px; height: 760px;
                background: radial-gradient(closest-side, rgba(110, 214, 80, .15), rgba(110, 214, 80, 0)); }}
  .b::after {{ content: ""; position: absolute; right: -240px; bottom: -320px; width: 760px; height: 760px;
               background: radial-gradient(closest-side, rgba(41, 192, 116, .10), rgba(41, 192, 116, 0)); }}
  .org {{ display: flex; align-items: center; gap: 16px; font-size: 28px; font-weight: 600; letter-spacing: 5px; color: #C9D3CD; }}
  .org img {{ width: 56px; height: 56px; border-radius: 13px; }}
  h1 {{ margin: 26px 0 0; display: flex; align-items: center; gap: 22px; font-size: 118px; line-height: 1; font-weight: 800;
        font-style: italic; letter-spacing: -1px; }}
  h1 img {{ width: 104px; height: 104px; }}
  .sub {{ margin: 16px 0 0; font-size: 32px; font-weight: 500; }}
  .card {{ margin-top: 30px; display: flex; align-items: center; justify-content: space-between; gap: 24px;
           border-radius: 24px; padding: 24px 30px 26px; background: linear-gradient(180deg, #17221A, #111814);
           border: 2px solid #33502C; box-shadow: 0 0 40px rgba(110, 214, 80, .10); }}
  .card .m {{ font-size: 40px; font-weight: 800; font-style: italic; }}
  .card .s {{ margin-top: 2px; font-size: 22px; font-weight: 500; color: #8C9891; }}
  .card .from {{ margin-top: 14px; font-size: 28px; font-weight: 500; color: #A9B5AE; white-space: nowrap; }}
  .card .from s {{ text-decoration-thickness: 2px; text-decoration-color: rgba(169, 181, 174, .6); }}
  .card .r {{ text-align: right; }}
  .card .to {{ font-size: 86px; line-height: 1; font-weight: 800; font-style: italic; white-space: nowrap; }}
  .card .x {{ margin-top: 8px; font-size: 26px; font-weight: 600; color: #E8EEEA; white-space: nowrap; }}
  .card .x b {{ font-weight: 800; }}
  .foot {{ margin-top: 20px; font-size: 24px; font-weight: 500; color: #8C9891; }}
  .foot b {{ color: #BFE7A6; font-weight: 600; }}
</style></head>
<body>
<div class="b">
  <div class="org"><img src="{uri(os.path.join(ASSETS, "neurosquad.png"))}" alt="">NEUROSQUAD</div>
  <h1><img src="{uri(os.path.join(ASSETS, "logo.svg"))}" alt=""><span>Seed<span class="g">Pack</span></span></h1>
  <p class="sub muted">Lossless GPU archiver for GGUF model repositories</p>
  <div class="card">
    <div>
      <div class="m">{html.escape(big["model"])}</div>
      <div class="s">{html.escape(big["src"])}</div>
      <div class="from"><s>{gb(big["n_in"])}</s> · {big["n_files"]} files</div>
    </div>
    <div class="r">
      <div class="to g">{gb(big["archive"])}</div>
      <div class="x"><b>{ratio:.1f}×</b> smaller</div>
    </div>
  </div>
  <div class="foot"><b>BF16 + every quant in one file</b> · every byte restored · open source</div>
</div>
</body></html>
"""


def bench():
    rows = []
    for r in rels:
        pct = 100 * r["archive"] / r["n_in"]
        rows.append(f"""
    <div class="rel">
      <div class="rn"><b>{html.escape(r.get("label", r["model"]))}</b><span>{f"BF16 + {r['n_quants']} quants" if r["bf16"]
                                                                          else f"{r['n_quants']} GSQ-RCO sizes, no BF16"}</span></div>
      <div class="track"><i style="width:{pct:.3f}%"></i><em{f' style="left:{pct + 2:.1f}%; font-size: 20px"' if pct > 10 else ""}>{gb(r["n_in"])} → <b>{gb(r["archive"])}</b></em></div>
      <div class="rx g">{r["n_in"] / r["archive"]:.1f}×</div>
    </div>""")
    bars = []
    vals = [(lab, 100 * res06[f"hf-0.6b | {key}"]["archive"] / res06[f"hf-0.6b | {key}"]["size"],
             res06[f"hf-0.6b | {key}"].get("compress_s")) for lab, key in ARCHIVERS]
    n06 = res06["hf-0.6b | SeedPack"]["size"]
    bars.append(f"""
      <div class="ar ah"><span class="al"></span><span class="at">archive, % of the {gb(n06)} repository</span>
        <span class="av">size</span><span class="ap">pack</span></div>""")
    for lab, pct, sec in vals:
        me = lab == "SeedPack"
        bars.append(f"""
      <div class="ar{" me" if me else ""}"><span class="al">{html.escape(lab)}</span>
        <div class="at"><i style="width:{pct:.3f}%"></i></div><span class="av">{pct:.1f}%</span>
        <span class="ap">{dur(sec)}</span></div>""")
    bars.append("""
      <div class="an">Pack times on one PC (i7-8700K, 32 GB): SeedPack on its RTX 5070, the others on the CPU
        (ZipNN in memory, without disk I/O).</div>""")
    tiles = [(f"Pack all {gb(big['n_in'])}", f"{t27['pack_s'] / 60:.1f}{NB}min", f"{big['n_in'] / t27['pack_s'] / 1e6:,.0f}{NB}MB/s"),
             ("Peak GPU memory", f"{t27['peak_vram_pack_gb']:.1f}{NB}GB", "12 GB card, 54.7 GB model"),
             ("Stored once", f"{t27['references']:,}", "tensors the quants share")]
    cells = "".join(f"""<div class="sp"><div class="sl">{html.escape(lab)}</div><div class="sv">{val}</div>
          <div class="ss">{html.escape(sub)}</div></div>""" for lab, val, sub in tiles)
    speed = f"""
  <h2>{html.escape(big["model"])} on one RTX 5070 (12 GB) and 32 GB of RAM</h2>
  <div class="speed">{cells}</div>"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,500;0,600;1,800&display=block" rel="stylesheet">
<style>{STYLE}
  .b {{ padding: 36px 44px 42px; }}
  .b::before {{ content: ""; position: absolute; right: -200px; top: -280px; width: 760px; height: 760px;
                background: radial-gradient(closest-side, rgba(110, 214, 80, .12), rgba(110, 214, 80, 0)); }}
  h1 {{ margin: 0; font-size: 56px; line-height: 1; font-weight: 800; font-style: italic; }}
  h2 {{ margin: 34px 0 16px; font-size: 25px; font-weight: 600; color: #C9D3CD; }}
  h2 small {{ font-size: 20px; font-weight: 500; color: #8C9891; }}
  .rel {{ display: flex; align-items: center; gap: 22px; margin-top: 14px; }}
  .rn {{ flex: 0 0 214px; }}
  .rn b {{ display: block; font-size: 28px; white-space: nowrap; font-weight: 800; font-style: italic; }}
  .rn span {{ font-size: 18px; font-weight: 500; color: #8C9891; }}
  .track {{ flex: 1; position: relative; height: 46px; border-radius: 10px; background: #1A2420; border: 2px solid #26312B; }}
  .track i {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 8px; background: linear-gradient(90deg, #9BE34A, #29C074);
              box-shadow: 0 0 18px rgba(110, 214, 80, .35); }}
  .track em {{ position: absolute; left: 13%; top: 0; bottom: 0; display: flex; align-items: center; font-style: normal;
               font-size: 22px; font-weight: 500; color: #A9B5AE; white-space: nowrap; }}
  .track em b {{ margin-left: 8px; color: #fff; font-weight: 600; }}
  .rx {{ flex: 0 0 104px; text-align: right; font-size: 40px; font-weight: 800; font-style: italic; }}
  .ar {{ display: flex; align-items: center; gap: 18px; margin-top: 9px; }}
  .al {{ flex: 0 0 196px; font-size: 22px; font-weight: 500; color: #C9D3CD; }}
  .at {{ flex: 1; height: 26px; border-radius: 6px; background: #161E1A; }}
  .at i {{ display: block; height: 100%; border-radius: 6px; background: #3A4741; }}
  .av {{ flex: 0 0 86px; text-align: right; font-size: 23px; font-weight: 600; color: #A9B5AE; }}
  .ap {{ flex: 0 0 92px; text-align: right; font-size: 21px; font-weight: 500; color: #8C9891; white-space: nowrap; }}
  .ah {{ margin-top: 0; }}
  .ah .at {{ height: auto; background: none; font-size: 17px; font-weight: 500; color: #6F7B74; }}
  .ah .av, .ah .ap {{ font-size: 17px; font-weight: 500; color: #6F7B74; }}
  .an {{ margin-top: 12px; font-size: 17px; font-weight: 500; color: #6F7B74; }}
  .me .al {{ font-weight: 800; font-style: italic; font-size: 25px; color: #fff; }}
  .me .at i {{ background: linear-gradient(90deg, #9BE34A, #29C074); box-shadow: 0 0 16px rgba(110, 214, 80, .4); }}
  .me .av {{ color: #9BE34A; font-size: 25px; font-weight: 800; }}
  .me .ap {{ color: #9BE34A; font-size: 23px; font-weight: 800; }}
  .speed {{ display: flex; gap: 14px; }}
  .sp {{ flex: 1; background: #141B17; border: 2px solid #212B25; border-radius: 14px; padding: 14px 18px 16px; }}
  .sl {{ font-size: 19px; font-weight: 500; color: #8C9891; }}
  .sv {{ margin-top: 2px; font-size: 40px; font-weight: 800; font-style: italic; }}
  .ss {{ font-size: 18px; font-weight: 500; color: #A9B5AE; }}
</style></head>
<body>
<div class="b">
  <h1>Seed<span class="g">Pack</span> benchmarks</h1>
  <h2>Whole repositories <small>· full bar = the original</small></h2>
  {"".join(rows)}
  <h2>Other archivers on unsloth/Qwen3-0.6B-GGUF <small>· lower is better</small></h2>
  <div class="ars">{"".join(bars)}
  </div>{speed}
</div>
</body></html>
"""


os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
hf_banners.PAGE, hf_banners.SHOWN = PAGE, 430
with tempfile.TemporaryDirectory() as tmp:
    for name, page in (("neurosquad-banner", hero()), ("seedpack-benchmarks", bench())):
        src = os.path.join(tmp, name + ".html")
        open(src, "w", encoding="utf-8").write(page)
        png = os.path.join(out_dir, "assets", name + ".png")
        render(src, png)
        w, h = Image.open(png).size
        before, after, psnr = squeeze(png)
        print(f"{png}: {w}x{h}, {before / 1e3:.0f} KB -> {after / 1e3:.0f} KB, PSNR {psnr:.1f} dB")

ud = [n for n in b27 if "-UD-" in n]
img = f"https://huggingface.co/spaces/{SPACE}/resolve/main/assets"
link = {r["model"]: f"https://huggingface.co/{ORG}/{r.get('name', r['model'])}-SEEDPACK" for r in rels}
table = "\n".join(f"| [{r.get('title', r['model'])}]({link[r['model']]}) | {gb(r['n_in'])} → **{gb(r['archive'])}** | "
                   f"{r['n_in'] / r['archive']:.1f}× |" for r in reversed(rels))
qfn_section = f"""### Qwen3.8-Flash-Next GSQ-RCO: {gb(qfn['n_in'])} → {gb(qfn['archive'])}

[{qfn['src']}](https://huggingface.co/{qfn['src']}), the GGUFs most Qwen3.8-Flash-Next users run (Strata's default), has no BF16 model to grow its sizes from. SeedPack 0.4.0 codes its {qfn['n_quants']} sizes against each other instead: identical files and tensors once, and every other tensor as a chain of versions, each coded given the precision-weighted mean of the earlier ones. All four sizes fit in one archive of **{gb(qfn['archive'])}**, {qfn['n_in'] / qfn['archive']:.1f}× smaller than the repository. [The archive]({link[qfn['model']]}) is for keeping all four sizes: one size alone is still smaller.

""" if qfn else ""
card = f"""---
title: NeuroSquad
emoji: 🌱
colorFrom: green
colorTo: gray
sdk: static
pinned: false
---

<p align="center">
  <a href="{GITHUB}"><img src="{img}/neurosquad-banner.png" alt="NeuroSquad · SeedPack: all of {big['src']}, {gb(big['n_in'])}, in {gb(big['archive'])}" width="860"></a>
</p>

**NeuroSquad** makes open tools that keep large language models small to store and quick to ship. The first one is **[SeedPack]({GITHUB})**.

## SeedPack

A GGUF repository ships the same weights 20–30 times over: the BF16 model plus every quantization of it. SeedPack stores the BF16 model once and **grows the quants back from it** on the GPU, with a bit-exact CUDA port of llama.cpp's quantizers and the repository's importance matrix. Only what differs from the published files is kept, coded with the quantizer's own probabilities. A whole repository becomes one archive smaller than its BF16 model, and every file comes back **byte for byte**.

### {big['model']}: {gb(big['n_in'])} → {gb(big['archive'])}

All of [{big['src']}](https://huggingface.co/{big['src']}) — the BF16 model in 2 shards, {big['n_quants']} quants from UD-IQ1_S to UD-Q8_K_XL, the multi-token-prediction layer, two vision projectors and the imatrix — is one archive of **{gb(big['archive'])}**: {big['n_in'] / big['archive']:.1f}× smaller than the {big['n_files']} files, and smaller than the BF16 model alone ({gb(bf16_27)}).

- **Bigger than the machine.** The {gb(bf16_27)} BF16 model never has to fit anywhere: one RTX 5070 with 12 GB and a PC with 32 GB of RAM pack the whole repository in {t27['pack_s'] / 60:.0f} minutes, window by window, with the GPU's memory at most {t27['peak_vram_pack_gb']:.1f} GB.
- **Unsloth Dynamic quants.** The {len(ud)} UD quants come from Unsloth's own pipeline, whose codes drift from plain rounding along each row, so they cannot be re-derived exactly. SeedPack still stores all of them ({gb(sum(b27[n]['size'] for n in ud))}) in **{gb(t27['ud_bytes'])}**: its coder knows where in the row every weight is, and the tensors they share are stored once.
- **A missing importance matrix, recovered.** Q4_0 and Q4_1 were made with an imatrix that is not in the repository. SeedPack fits it back from the quants themselves, by least squares on their block scales: {t27['blocks_equal'][1]:.0f}% of the blocks then come out exactly instead of {t27['blocks_equal'][0]:.0f}%, and the two files cost {gb(t27['old_imatrix_bytes'])} instead of about 1 GB.

<p align="center">
  <img src="{img}/seedpack-benchmarks.png" alt="SeedPack benchmarks: the releases and other archivers" width="860">
</p>

{qfn_section}### Releases

Each archive holds a whole repository: every file of it, restored byte for byte.

| Model | Repository → archive | Smaller |
|---|---:|---:|
{table}

### Try it

```bash
pip install seedpack
seedpack list hf://{ORG}/{big['model']}-SEEDPACK                                   # what is inside
seedpack x hf://{ORG}/{big['model']}-SEEDPACK {big['model']}-UD-Q4_K_XL.gguf -o models   # one quant
```

SeedPack runs on an NVIDIA GPU (Turing or newer) with a CUDA 13 driver; without one, `--cpu` (0.5.0, `pip install "seedpack[cpu]"`) makes the same archives byte for byte, 10–20 times slower. Python 3.11+; the model does not have to fit in GPU memory. Source code, documentation and the roadmap: [GitHub]({GITHUB}) · [PyPI]({PYPI}).
"""
open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8", newline="\n").write(card)
print(f"{out_dir}/README.md: {big['model']} {gb(big['n_in'])} -> {gb(big['archive'])}")
