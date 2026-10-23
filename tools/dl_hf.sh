#!/bin/bash
# usage: dl_hf.sh <hf repo> <dest dir> — whole repository through the Xet high-performance client
export HF_XET_HIGH_PERFORMANCE=1
.venv/Scripts/hf.exe download "$1" --local-dir "$2" && echo "done $1"
