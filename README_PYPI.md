<p align="center">
  <img src="https://raw.githubusercontent.com/glmn-ai/seedpack/main/assets/banner.png" alt="SeedPack" width="860">
</p>

# SeedPack

**Lossless GPU archiver for GGUF model repositories.** A whole repository, the BF16 model and all its quants from IQ1 to Q8, packs into one archive about the size of a single compressed BF16 file. Every file comes back byte for byte.

| unsloth/Qwen3-0.6B-GGUF, BF16 + 25 quants (10.95 GB) | Archive | Ratio |
|---|---:|---:|
| **SeedPack** | **782 MB** | **7.14%** |
| lmz 1.3.0 | 4.54 GB | 41.47% |
| zstd 1.5.7 -19 --long=30 | 6.57 GB | 59.99% |
| 7-Zip 23.01 -mx9 | 10.21 GB | 93.26% |
| ZipNN 0.5.4 | 10.33 GB | 94.34% |

The archive is smaller than the repository's largest single quant (UD-Q8_K_XL, 844 MB). The full benchmark against 10 other compressors, with speeds and single files, is on the [example archive's page](https://huggingface.co/neurosquad/Qwen3-0.6B-SEEDPACK).

Models bigger than the GPU's memory are streamed: all of unsloth/Qwen3.5-9B-GGUF (BF16, 21 quants and the vision projector, 147.8 GB) packs into [12.4 GB](https://huggingface.co/neurosquad/Qwen3.5-9B-SEEDPACK) in under 4 minutes on a 12 GB card, and its Q4_K_M comes back in about 20 s. All of unsloth/Qwen3.8-27B-GGUF (BF16 in two shards, 24 quants, the MTP layer and two vision projectors, 472.1 GB) packs into [45.4 GB](https://huggingface.co/neurosquad/Qwen3.8-27B-SEEDPACK) in 23 minutes on the same card with 32 GB of RAM. Unsloth's Dynamic quants cannot be re-derived exactly; their residuals are coded with position-aware contexts, the tensors they share are stored once, and the imatrix that the plain Q4_0 / Q4_1 were made with, which is not in the repository, is fitted back from the quants.

Without a BF16 model, `--method m30` (0.4.0) codes the sizes of one model against each other: identical files and tensors once, every other tensor as a chain of versions, each coded given the precision-weighted mean of the earlier ones. All of ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (four sizes, IQ3_S to Q2_0, in two shards each and the vision projector, 294.8 GB) packs into [112.7 GB](https://huggingface.co/neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK) in 29 minutes, with 2.7 GB of GPU memory.

## Install

```bash
pip install seedpack
```

Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer).
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 under Windows 10 (Python 3.11) and Ubuntu on WSL2 (Python 3.14).
- No CUDA Toolkit is needed: NVRTC, nvJitLink and the CUDA headers come from pip.
- No NVIDIA GPU? `--cpu` (0.5.0) runs the same kernels on the CPU: the same archives byte for byte, 10–20 times slower. It compiles the kernels once and needs a C++ compiler: `pip install "seedpack[cpu]"` (ziglang), or clang++ / g++ on the PATH.

Archives are portable between operating systems: an archive packed on Windows unpacks bit-exactly on Linux, and the other way round. The first run compiles the GPU quantizers for your card (about 15 s) and caches them in `%LOCALAPPDATA%\seedpack\cache` or `~/.cache/seedpack`; set `SEEDPACK_CACHE` to move the cache.

## Use

```bash
# straight from the Hugging Face Hub: download once, extract what you need
seedpack list hf://neurosquad/Qwen3-0.6B-SEEDPACK
seedpack x    hf://neurosquad/Qwen3-0.6B-SEEDPACK Qwen3-0.6B-Q4_K_M.gguf -o models
seedpack x    hf://neurosquad/Qwen3-0.6B-SEEDPACK "*Q4_K*" "*Q8_0*" -o models
seedpack x    hf://neurosquad/Qwen3-0.6B-SEEDPACK -o models      # everything

# local archives
seedpack pack  models/Qwen3-0.6B-GGUF -o qwen3.seedpack           # files and/or folders
seedpack x     qwen3.seedpack Qwen3-0.6B-Q4_K_M.gguf -o out
seedpack test  qwen3.seedpack                                         # decode and verify, write nothing
```

Notes on the commands:
- `hf://OWNER/REPO[/FILE.seedpack][@REVISION]` and `https://huggingface.co/...` links work wherever an archive is expected.
- `list` reads only the archive's index over the network; `x` and `test` download the archive once into the Hugging Face cache.
- Extracted files are checked against BLAKE3 hashes. Every chunk of the archive carries a CRC32.

## How it works

A GGUF repository ships the same weights 20–30 times: the BF16 model plus each quantization of it. To a general-purpose compressor that looks like gigabytes of almost incompressible bytes. SeedPack uses the fact that every quant is a deterministic function of the same weights:

- **The seed is stored once.** The most precise model (the BF16 file, or the senior quant when there is none) is coded with an entropy coder built for its weights.
- **Quants are re-derived, not stored.** Each quant is computed again on the GPU by a bit-exact CUDA port of llama.cpp's quantizers (`ggml-quants.c`, compiled without FMA contraction), with the repository's importance matrix.
- **Only the residual is kept.** Where a re-derived quant differs from the published file, the residual is coded with the quantizer's own probabilities: scales first, then each code relative to its rounding boundary. For Qwen3-0.6B, all 25 quants together cost about 1 MB on top of the seed.
- **Duplicates are stored once.** Identical tensors in different files are kept once. Files that are not GGUF are compressed with LZMA or zstd.

## Links

- Source and research notes: <https://github.com/glmn-ai/seedpack>
- Example archives: <https://huggingface.co/neurosquad/Qwen3-0.6B-SEEDPACK>, <https://huggingface.co/neurosquad/Qwen3.5-9B-SEEDPACK>, <https://huggingface.co/neurosquad/Qwen3.8-27B-SEEDPACK>
- A [NeuroSquad](https://huggingface.co/neurosquad) project.

## License

Apache-2.0. The bundled llama.cpp / ggml quantization sources (`seedpack/ggml/`) are MIT-licensed; see `NOTICE`.
