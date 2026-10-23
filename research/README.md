# research/ — new methods vs M24 / M28

All numbers on Qwen3-0.6B (unsloth repo), RTX 5070, measured while other GPU/CPU benchmarks ran
concurrently, so timings are noisy (±15 %). Every size below comes from a byte-exact round trip.
Reference: M24 = 782,650,261 B (65.32 %) for Qwen3-0.6B-BF16.gguf; M28 repo archive = 783,707,377 B.

| id | file | idea | result |
|----|------|------|--------|
| R1 | analyze_bf16.py, analyze_bf16_v2.py, analyze_bf16_v3.py, pb_study.py, side_entropy.py | Entropy study of BF16: what is left after M24's model | M24 model ideal 781.2 MB. Neighbour contexts (left/up weight, sign of neighbours, per-row/column sign tables) gain <=0.01 bpw; oracle tile variance (1x128, 32x32, 16x256) gains less than its side info would cost; mantissa low 4 bits = 3.997 bit (random). Gains that are real: 1/8-octave contexts (+0.5 MB), role-shared tables (cheap tables), 15-bit probabilities (+0.25 MB), duplicate rows (+0.6 MB). The rank-1 variance model is at the floor: ideal for the R2 model is 779.37 MB. |
| R2 | bf16v2.py, bench_bf16v2.py | BF16 v2 codec on GPU: rfi/cfi at 1/8 octave; exponent ctx = predicted log2 sigma (512), mantissa ctx = exp - prediction (128), sign ctx = column bias (16); tables shared per tensor role (blk.*.ffn_up ...), PB=15; exact duplicate rows coded once; tensors processed in groups (GPU pool <= 2.4 GB, the file does not have to fit) | **780,618,332 B = 65.15 % (-2,031,929 B vs M24)**, 378 MB/s compress, 1591 MB/s decompress. Qwen3.5-0.8B-BF16: 985,299,803 B = 64.96 % vs M22 (CPU twin of M24) 986,875,210 B (-1.58 MB). |
| R3 | repo_sparse.py, bench_repo_sparse.py, profile_sparse.py | M28 with sparse residual: per quant file only the bytes outside predicted tensors (zstd-9 + BF16 header dictionary) and the XOR of tensors that are not reproduced exactly; tensor table in the manifest, so the decoder does not parse GGUFs: it allocates the file, fills the gaps and copies the predictions in. | Per quant file 2.3–2.7 KB instead of 9–28 KB. Repo (R2 base + R3 + R5): **781,363,658 B = 99.70 % of M28** (781,569,739 with the PB=12 R2) (M28: 783,707,377; same R2 base with the M28 residual: 781,881,545). Decompress without SHA-256: 10.2 s (one run 17.8 s under heavy concurrent load) vs 13.3 s (M28 residual path, same base and cache). Compress: 22–26 s vs 21–25 s (no gain, it is noise-level). |
| R4 | profile_m28.py, profile_quant_types.py | Profiling M28 | Before the coordinator's fixes: GGUF parse 5.5 s/side, GPU quant 12–17 s. Now: in bench_repo3 the **SHA-256 check of 11 GB takes ~20 s of the 30 s "decompress" time** (single-thread hashlib); the codec itself decompresses in ~10 s. |
| R5 | extras_codec.py | legacy imatrix .dat: float payload split into 4 byte planes, raw LZMA2 (lc=lp=pb=0); other small files lzma -9e | 638,889 B vs 695,028 B (-56 KB), lossless |

## Not worth it (measured)
* left/up neighbour contexts for exponent/mantissa/sign: < 0.8 MB ideal before table cost, needs 1-4 M table cells.
* per-tensor tables with fine contexts: table cost (0.8–1.6 MB) > gain.
* tile variance side info, finer sign classes (scl64 x magnitude): < 0.05 MB.
* side info (rfi/cfi/scl) entropy coding instead of zstd: ~20 KB.

## To integrate
1. R2 instead of M24 (same interface via `repo_sparse.R2Base`): -2.03 MB, bounded GPU memory, same speed class.
2. R3 sparse residual + manifest tensor table: -0.3 MB and ~3 s less decode; no zstd over zeros.
3. R5 imatrix byte planes: -56 KB.
4. bench: hash outputs in a thread pool (or blake2b / xxh3) — the benchmark, not the codec, dominates decompress time.
