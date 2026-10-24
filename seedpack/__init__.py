"""SeedPack: lossless GPU archiver for GGUF model repositories.

The quants of a repository are re-derived on the GPU from its most precise model (the seed) with
a bit-exact port of llama.cpp's quantisers; only what differs is stored. Command line: `seedpack`.
"""
__version__ = "0.2.0"
