"""Compare local files with the sizes listed by the Hugging Face API: check_dl.py <repo> <dir>"""
import json, os, sys, urllib.request
repo, d = sys.argv[1], sys.argv[2]
t = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"))
ok = bad = 0
for f in t:
    if f["type"] != "file":
        continue
    p = os.path.join(d, f["path"]); have = os.path.getsize(p) if os.path.exists(p) else -1
    if have == f["size"]:
        ok += 1
    else:
        bad += 1; print(f"  incomplete {f['path']}: {have:,} / {f['size']:,}")
print(f"{repo}: {ok} complete, {bad} incomplete")
