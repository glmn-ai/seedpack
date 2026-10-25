"""Banners of the Hugging Face releases.

usage: hf_banners.py
Writes assets/banner.png (from assets/banner.html) and, per release, assets/stats-<model>.png: the
model, every file of the original repository with its size, the archive size and the ratio. The
pages are rendered by headless Chrome or Edge at twice the width they are shown at (860 px), then
reduced to 256 colours with dithering (imagequant) and optimized losslessly (oxipng): 50-100 KB
each, visually the same. Needs pillow, imagequant and pyoxipng (pip), which SeedPack does not use.
"""
import base64
import html
import json
import os
import re
import subprocess
import tempfile
import urllib.request

import imagequant
import numpy as np
import oxipng
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
SHOWN = 860                                    # width of the banners in the cards and READMEs
PAGE = 1280                                    # CSS width of the pages
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium"]
RELEASES = [  # archive: size of the published .seedpack file
    dict(model="Qwen3-0.6B", src="unsloth/Qwen3-0.6B-GGUF", files="hf06_files.json", archive=781678552),
    dict(model="Qwen3.5-9B", src="unsloth/Qwen3.5-9B-GGUF", files="hf9b_files.json", archive=12396550582),
    dict(model="Qwen3.8-27B", src="unsloth/Qwen3.8-27B-GGUF", files="hf27b_files.json", archive=45410617876),
]
FONT = "https://github.com/google/fonts/raw/main/ofl/exo2/"        # Exo 2, variable: normal and italic
FONT_FILES = {"normal": "Exo2%5Bwght%5D.ttf", "italic": "Exo2-Italic%5Bwght%5D.ttf"}
NB = "\u00a0"


def size(b):
    return f"{b / 1e9:.2f}{NB}GB" if b >= 1e9 else f"{b / 1e6:.1f}{NB}MB"


def stats_page(model, src, files, archive):
    """the stats banner of one release as an HTML page"""
    gguf = {n: v["size"] for n, v in files.items() if n.endswith(".gguf") and not n.startswith("imatrix")}
    shards = sorted(n for n in gguf if re.fullmatch(rf"(BF16/)?{re.escape(model)}-BF16(-\d{{5}}-of-\d{{5}})?\.gguf", n))
    base = "BF16"                                  # the BF16 model, whole or split into shards
    gguf = {base: sum(gguf.pop(n) for n in shards), **gguf}
    mmproj = sorted((n for n in gguf if n.startswith("mmproj")), key=lambda n: -gguf[n])
    mtp = sorted(n for n in gguf if os.path.basename(n).startswith("mtp-"))
    quants = sorted((n for n in gguf if n != base and n not in mmproj and n not in mtp), key=lambda n: -gguf[n])
    tiles = quants + mtp + mmproj
    cols = next((c for c in (6, 5, 7, 4) if len(tiles) % c == 0), 6)     # full rows when possible
    n_in = sum(v["size"] for v in files.values())
    what = ["BF16" + (f" in {len(shards)} shards" if len(shards) > 1 else ""), f"{len(quants)} quants"]         + (["the MTP layer"] if mtp else []) + ([f"{len(mmproj)} vision projectors"] if mmproj else [])
    what = ", ".join(what[:-1]) + " and " + what[-1]

    def short(n):
        return os.path.basename(n).replace(f"{model}-", "").removesuffix(".gguf")

    tile_html = "\n".join(f'    <div class="t"><div class="n">{short(n)}</div><div class="s">{size(gguf[n])}</div></div>'
                          for n in tiles)
    seed = ' class="seed"'
    segments = "".join(f'<i{seed if n == base else ""} style="flex-grow:{gguf[n]}"></i>' for n in [base] + tiles)
    pct = 100 * archive / n_in
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,500;0,600;1,800&display=block" rel="stylesheet">
<style>
  html, body {{ margin: 0; background: transparent; }}
  .b {{ box-sizing: border-box; width: {PAGE}px; border-radius: 32px; background: #0E1311; border: 3px solid #1F2722;
        padding: 40px 56px 44px; position: relative; overflow: hidden; font-family: 'Exo 2', sans-serif; color: #fff;
        font-variant-numeric: tabular-nums; }}
  .b::before {{ content: ""; position: absolute; right: -180px; top: -260px; width: 760px; height: 760px;
                background: radial-gradient(closest-side, rgba(110, 214, 80, .13), rgba(110, 214, 80, 0)); }}
  .b > * {{ position: relative; }}
  .head {{ display: flex; justify-content: space-between; align-items: flex-end; }}
  h1 {{ margin: 0; font-size: 66px; line-height: 1; font-weight: 800; font-style: italic; letter-spacing: -1px; }}
  .src {{ margin: 14px 0 0; font-size: 25px; font-weight: 500; color: #A9B5AE; }}
  .ratio {{ text-align: right; }}
  .ratio b {{ display: block; font-size: 100px; line-height: .86; font-weight: 800; font-style: italic; padding-right: 8px;
              background: linear-gradient(90deg, #9BE34A, #29C074); -webkit-background-clip: text; color: transparent; }}
  .ratio span {{ display: block; margin-top: 10px; font-size: 25px; font-weight: 500; color: #A9B5AE; }}
  .files {{ display: flex; gap: 12px; margin-top: 32px; }}
  .t {{ background: #141B17; border: 2px solid #212B25; border-radius: 12px; padding: 8px 12px 9px; }}
  .n {{ font-size: 20.5px; font-weight: 600; color: #E8EEEA; white-space: nowrap; }}
  .s {{ margin-top: 1px; font-size: 19px; font-weight: 500; color: #8C9891; white-space: nowrap; }}
  .base {{ flex: 0 0 150px; box-sizing: border-box; display: flex; flex-direction: column; border-color: #3C5A32;
           background: linear-gradient(180deg, #18241A, #121A15); }}
  .base .n {{ font-size: 32px; font-weight: 800; font-style: italic; }}
  .base .s {{ font-size: 21px; color: #A9B5AE; }}
  .base .k {{ margin-top: auto; font-size: 19px; font-weight: 500; color: #8FD35A; }}
  .grid {{ flex: 1; display: grid; grid-template-columns: repeat({cols}, minmax(0, 1fr)); gap: 10px; }}
  .bars {{ margin-top: 34px; }}
  .row {{ display: flex; justify-content: space-between; align-items: baseline; font-size: 22px; font-weight: 500; color: #A9B5AE; }}
  .row b {{ font-size: 26px; font-weight: 600; color: #fff; }}
  .bar {{ display: flex; gap: 3px; height: 26px; margin: 10px 0 0; }}
  .bar i {{ flex-basis: 0; min-width: 0; border-radius: 4px; background: #2F3B35; }}
  .bar i:nth-child(even) {{ background: #3A4741; }}
  .bar i.seed {{ background: #3C5A32; }}
  .arc {{ margin-top: 22px; }}
  .arcbar {{ display: flex; align-items: center; gap: 18px; margin-top: 10px; }}
  .arcbar i {{ flex: 0 0 {pct:.4f}%; height: 26px; border-radius: 4px; background: linear-gradient(90deg, #9BE34A, #29C074);
               box-shadow: 0 0 22px rgba(110, 214, 80, .35); }}
  .arcbar > b {{ font-size: 26px; font-weight: 600; }}
  .arcbar > span {{ font-size: 22px; font-weight: 500; color: #8C9891; }}
  .brand {{ margin-left: auto; display: flex; align-items: center; gap: 10px; font-size: 26px; font-weight: 800; font-style: italic; }}
  .brand img {{ width: 36px; height: 36px; }}
  .brand em {{ font-style: inherit; background: linear-gradient(90deg, #9BE34A, #29C074); -webkit-background-clip: text;
               color: transparent; padding-right: 4px; }}
</style></head>
<body>
<div class="b">
  <div class="head">
    <div>
      <h1>{html.escape(model)}</h1>
      <p class="src">{html.escape(src)}: {what}</p>
    </div>
    <div class="ratio"><b>{n_in / archive:.1f}×</b><span>smaller, byte for byte</span></div>
  </div>
  <div class="files">
    <div class="t base"><div class="n">BF16</div><div class="s">{size(gguf[base])}</div><div class="k">the seed</div></div>
    <div class="grid">
{tile_html}
    </div>
  </div>
  <div class="bars">
    <div class="row"><span>Original repository, {len(files)} files</span><b>{size(n_in)}</b></div>
    <div class="bar">{segments}</div>
    <div class="row arc"><span>SeedPack archive, one file: {html.escape(model)}.seedpack</span></div>
    <div class="arcbar"><i></i><b>{size(archive)}</b><span>{pct:.2f}% of the original</span>
      <div class="brand"><img src="{uri(os.path.join(ASSETS, "logo.svg"))}" alt=""><span>Seed<em>Pack</em></span></div></div>
  </div>
</div>
</body></html>
"""


def uri(path):
    return "file:///" + os.path.abspath(path).replace(os.sep, "/")


def font_faces():
    """@font-face rules of Exo 2 with the fonts inline: a headless screenshot does not wait for a web
    font, and one that comes late is drawn in a fallback font"""
    cache = os.path.join(tempfile.gettempdir(), "seedpack-banner-fonts")
    os.makedirs(cache, exist_ok=True)
    rules = []
    for style, name in FONT_FILES.items():
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(FONT + name, path)
        data = base64.b64encode(open(path, "rb").read()).decode()
        rules.append(f"@font-face {{ font-family: 'Exo 2'; font-style: {style}; font-weight: 100 900; font-display: block;"
                     f" src: url(data:font/ttf;base64,{data}); }}")
    return "\n".join(rules)


def render(page, png):
    """screenshot of an HTML page whose content is one box on a transparent background, cut to the box"""
    exe = next(p for p in BROWSERS if os.path.exists(p))
    src, n = re.subn(r'<link href="https://fonts\.googleapis\.com/[^"]*" rel="stylesheet">',
                     lambda m: f'<base href="{uri(os.path.dirname(page))}/">\n<style>{font_faces()}</style>',
                     open(page, encoding="utf-8").read())
    assert n == 1, page
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "page.html")
        open(local, "w", encoding="utf-8").write(src)
        subprocess.run([exe, "--headless=new", "--hide-scrollbars", f"--force-device-scale-factor={2 * SHOWN / PAGE}",
                        f"--window-size={PAGE},1200", "--virtual-time-budget=15000", "--default-background-color=00000000",
                        f"--user-data-dir={os.path.join(tmp, 'profile')}", f"--screenshot={os.path.abspath(png)}", uri(local)],
                       check=True, capture_output=True, timeout=120)
    img = Image.open(png)
    img.load()
    img.crop(img.getchannel("A").getbbox()).save(png)


def squeeze(png):
    """256 colours with dithering, then a lossless PNG optimization; returns (bytes before, after, PSNR of
    what is seen: colours times alpha)"""
    before = os.path.getsize(png)
    img = Image.open(png).convert("RGBA")
    q = imagequant.quantize_pil_image(img, dithering_level=1.0, max_colors=256, min_quality=90, max_quality=100)
    q.save(png)
    oxipng.optimize(png, level=6, strip=oxipng.StripChunks.safe())
    a = np.asarray(img, np.float64)
    b = np.asarray(Image.open(png).convert("RGBA"), np.float64)
    mse = ((a[..., :3] * a[..., 3:] - b[..., :3] * b[..., 3:]) ** 2).mean() / 255 ** 2
    return before, os.path.getsize(png), 10 * np.log10(255 ** 2 / mse) if mse else float("inf")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        jobs = [(os.path.join(ASSETS, "banner.html"), os.path.join(ASSETS, "banner.png"))]
        for rel in RELEASES:
            page = os.path.join(tmp, f"stats-{rel['model'].lower()}.html")
            files = json.load(open(os.path.join(HERE, rel["files"])))
            open(page, "w", encoding="utf-8").write(stats_page(rel["model"], rel["src"], files, rel["archive"]))
            jobs.append((page, os.path.join(ASSETS, f"stats-{rel['model'].lower()}.png")))
        for page, png in jobs:
            render(page, png)
            w, h = Image.open(png).size
            before, after, psnr = squeeze(png)
            print(f"{png}: {w}x{h}, {before / 1e3:.0f} KB -> {after / 1e3:.0f} KB, PSNR {psnr:.1f} dB")
