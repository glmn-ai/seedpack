#!/bin/sh
# Builds ggmlq.dll (Windows, MinGW gcc) / libggmlq.so: llama.cpp's quantisers (ggml-quants.c, via
# export.c which adds accessors for the IQ lookup tables) plus minimal stubs of ggml.c.
# -ffp-contract=off / -fno-fast-math: plain IEEE single precision, as the GPU port reproduces.
set -e
cd "$(dirname "$0")"
FLAGS="-O2 -std=c11 -ffp-contract=off -fno-fast-math -I."
gcc $FLAGS -c export.c -o export.o
gcc $FLAGS -c stubs.c -o stubs.o
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) gcc -shared -o ggmlq.dll export.o stubs.o -Wl,--export-all-symbols -lm ;;
    *) gcc -shared -fPIC -o libggmlq.so export.o stubs.o -lm ;;
esac
echo "built"
