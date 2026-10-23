### Qwen3-0.6B-BF16.gguf (1.20 GB)

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 780.6 MB | 65.15% | 165 MB/s | 337 MB/s | True |
| lmz 1.3.0 | 790.4 MB | 65.97% | 762 MB/s | 1,233 MB/s | True |
| ZipNN 0.5.4 | 797.4 MB | 66.55% | 1,165 MB/s | 6,613 MB/s | True |
| DFloat11 (estimate) | 800.4 MB | 66.81% | — | — | size estimate (format model) |
| xz -6 | 846.0 MB | 70.61% | 13 MB/s | 69 MB/s | True |
| zstd -19 | 902.8 MB | 75.35% | 22 MB/s | 177 MB/s | True |
| zstd -3 --long=30 | 934.6 MB | 78.01% | 269 MB/s | 233 MB/s | True |
| HF Xet (estimate) | 1,039.5 MB | 86.76% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |

### Qwen3.5-2B-BF16.gguf (3.78 GB)

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 2,463.4 MB | 65.24% | 213 MB/s | 493 MB/s | True |
| lmz 1.3.0 | 2,487.1 MB | 65.87% | 914 MB/s | 1,416 MB/s | True |
| ZipNN 0.5.4 | 2,511.2 MB | 66.51% | 1,168 MB/s | 6,668 MB/s | True |
| DFloat11 (estimate) | 2,516.1 MB | 66.64% | — | — | size estimate (format model) |
| xz -6 | 2,671.2 MB | 70.75% | 16 MB/s | 33 MB/s | True |
| zstd -19 | 2,852.6 MB | 75.55% | 24 MB/s | 198 MB/s | True |
| zstd -3 --long=30 | 2,945.1 MB | 78.00% | 160 MB/s | 249 MB/s | True |
| HF Xet (estimate) | 3,251.5 MB | 86.12% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |

### Qwen3-0.6B-Q8_0.gguf (0.64 GB), alone

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 576.4 MB | 90.13% | 67 MB/s | 172 MB/s | True |
| lmz 1.3.0 | 594.7 MB | 93.01% | 368 MB/s | 855 MB/s | True |
| zstd -19 | 608.5 MB | 95.17% | 35 MB/s | 299 MB/s | True |
| zstd -3 --long=30 | 609.1 MB | 95.25% | 208 MB/s | 284 MB/s | True |
| xz -6 | 613.6 MB | 95.97% | 22 MB/s | 94 MB/s | True |
| HF Xet (estimate) | 634.6 MB | 99.24% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |
| ZipNN 0.5.4 | 636.5 MB | 99.54% | 929 MB/s | 10,033 MB/s | True |

### Qwen3-0.6B-Q4_K_M.gguf (0.40 GB), alone

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 372.1 MB | 93.79% | 76 MB/s | 165 MB/s | True |
| lmz 1.3.0 | 376.9 MB | 95.01% | 102 MB/s | 430 MB/s | True |
| zstd -19 | 384.7 MB | 96.97% | 29 MB/s | 285 MB/s | True |
| zstd -3 --long=30 | 385.9 MB | 97.28% | 191 MB/s | 309 MB/s | True |
| xz -6 | 386.3 MB | 97.37% | 18 MB/s | 97 MB/s | True |
| HF Xet (estimate) | 392.0 MB | 98.82% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |
| ZipNN 0.5.4 | 393.7 MB | 99.25% | 1,049 MB/s | 7,965 MB/s | True |

### unsloth/Qwen3-0.6B-GGUF, whole repository (10.95 GB, 32 files)

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 781.7 MB | 7.14% | 554 MB/s | 1,101 MB/s | True |
| lmz 1.3.0 | 4,540.2 MB | 41.47% | 303 MB/s | 2,123 MB/s | True |
| HF Xet (estimate) | 5,110.7 MB | 46.68% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |
| zstd -3 --long=30 | 6,608.9 MB | 60.36% | 375 MB/s | — |  |
| ZipNN 0.5.4 | 10,329.2 MB | 94.34% | 1,129 MB/s | 8,596 MB/s | True |

### unsloth/Qwen3.5-0.8B-GGUF, whole repository (14.08 GB, 28 files)

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 1,120.5 MB | 7.96% | 582 MB/s | 1,155 MB/s | True |
| lmz 1.3.0 | 5,799.7 MB | 41.20% | 400 MB/s | 2,264 MB/s | True |
| HF Xet (estimate) | 6,680.9 MB | 47.46% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |
| zstd -3 --long=30 | 8,701.0 MB | 61.81% | 350 MB/s | — |  |
| ZipNN 0.5.4 | 12,801.8 MB | 90.94% | 1,108 MB/s | 7,755 MB/s | True |

### unsloth/Qwen3.5-2B-GGUF, whole repository (34.57 GB, 28 files)

| tool | archive | % of input | compress | decompress | lossless |
|---|---:|---:|---:|---:|---|
| **Seedpack (varc)** | 2,908.5 MB | 8.41% | 598 MB/s | 1,134 MB/s | True |
| lmz 1.3.0 | 15,120.6 MB | 43.74% | 464 MB/s | 2,035 MB/s | True |
| HF Xet (estimate) | 17,417.1 MB | 50.38% | — | — | size estimate (CDC dedup + lz4 / bg4+lz4) |
| zstd -3 --long=30 | 29,820.9 MB | 86.26% | 323 MB/s | — |  |
| ZipNN 0.5.4 | 31,392.3 MB | 90.80% | 1,048 MB/s | 7,949 MB/s | True |
