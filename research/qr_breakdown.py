"""Pack a directory with QR statistics on: entropy of the flag / scale / code streams per type."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import qr
from gguf_reader import TYPE_NAMES
qr.STATS = {}
import runpy
out = os.path.join(tempfile.gettempdir(), "qr_breakdown.varc")
sys.argv = ["varc.py", "pack", sys.argv[1], "-o", out, "-f", "-q"]
try:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "varc.py"), run_name="__main__")
except SystemExit:
    pass
os.remove(out)
types = sorted({t for t, _ in qr.STATS})
print(f"{'type':7s} {'flags MB':>9s} {'scales MB':>10s} {'codes MB':>9s}")
for t in types:
    print(f"{TYPE_NAMES[t]:7s} {qr.STATS.get((t,'flags'),0)/1e6:9.2f} {qr.STATS.get((t,'scales'),0)/1e6:10.2f} {qr.STATS.get((t,'codes'),0)/1e6:9.2f}")
print(f"total   {sum(v for (t,k),v in qr.STATS.items() if k=='flags')/1e6:9.2f} {sum(v for (t,k),v in qr.STATS.items() if k=='scales')/1e6:10.2f} {sum(v for (t,k),v in qr.STATS.items() if k=='codes')/1e6:9.2f}")
