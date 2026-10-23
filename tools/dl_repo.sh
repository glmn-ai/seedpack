#!/bin/bash
# usage: dl_repo.sh <hf repo> <dest dir> — every file of the repository, 4 at a time, resumable
repo=$1; dest=$2; mkdir -p "$dest"
curl -s "https://huggingface.co/api/models/$repo/tree/main?recursive=1" | .venv/Scripts/python -c "
import json,sys
for f in json.load(sys.stdin):
    if f['type']=='file': print(f['path'])" | tr -d '\r' > "$dest/.files"
xargs -P 8 -I{} sh -c "curl -sL -C - --retry 5 -o \"$dest/{}\" \"https://huggingface.co/$repo/resolve/main/{}\" || echo FAIL {}" < "$dest/.files"
echo "done $repo"
