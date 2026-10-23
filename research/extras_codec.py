"""R5: better coding of the repository's small files (M28 'other files' = lzma of everything).
imatrix (legacy .dat): names/counts kept as a small header, the float32 payload split into
byte planes (sign+exponent byte, then mantissa bytes), each plane lzma'd; everything else lzma.
Lossless round trip checked."""
import lzma, struct, sys, os, glob
import numpy as np

def split_imatrix(buf):
    (n,), pos = struct.unpack_from("<i", buf), 4
    hdr, vals = [buf[:4]], []
    for _ in range(n):
        (ln,) = struct.unpack_from("<i", buf, pos)
        ncall, nval = struct.unpack_from("<ii", buf, pos + 4 + ln)
        hdr.append(buf[pos:pos + 12 + ln]); pos += 12 + ln
        vals.append(buf[pos:pos + 4 * nval]); pos += 4 * nval
    hdr.append(buf[pos:])
    return b"".join(hdr), b"".join(vals), [len(h) for h in hdr], [len(v) for v in vals]

def enc_imatrix(buf):
    hdr, vals, hl, vl = split_imatrix(buf)
    a = np.frombuffer(vals, np.uint8).reshape(-1, 4)
    planes = b"".join(a[:, k].tobytes() for k in (3, 2, 1, 0))
    meta = struct.pack("<I", len(hl)) + struct.pack(f"<{len(hl)}I", *hl) + struct.pack(f"<{len(vl)}I", *vl)
    f = lzma.FORMAT_RAW; filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    p = lzma.compress(planes, format=f, filters=filt)
    h = lzma.compress(meta + hdr, preset=9 | lzma.PRESET_EXTREME)
    return struct.pack("<II", len(h), len(vals)) + h + p

def dec_imatrix(z):
    lh, nv = struct.unpack_from("<II", z)
    mh = lzma.decompress(z[8:8 + lh])
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    planes = np.frombuffer(lzma.decompress(z[8 + lh:], format=lzma.FORMAT_RAW, filters=filt), np.uint8)
    (nh,) = struct.unpack_from("<I", mh); hl = struct.unpack_from(f"<{nh}I", mh, 4); vl = struct.unpack_from(f"<{nh - 2}I", mh, 4 + 4 * nh)
    hdr = mh[4 + 4 * nh + 4 * (nh - 2):]
    q = nv // 4; a = np.empty((q, 4), np.uint8)
    for i, k in enumerate((3, 2, 1, 0)): a[:, k] = planes[i * q:(i + 1) * q]
    vals = a.tobytes(); out, hp, vp = [], 0, 0
    for i in range(nh):
        out.append(hdr[hp:hp + hl[i]]); hp += hl[i]
        if 1 <= i <= nh - 2: out.append(vals[vp:vp + vl[i - 1]]); vp += vl[i - 1]
    return b"".join(out)

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "models/unsloth"
    files = {os.path.basename(p): open(p, "rb").read() for p in glob.glob(os.path.join(d, "*")) + glob.glob(os.path.join(d, ".*"))
             if os.path.isfile(p) and not p.endswith(".gguf") and os.path.basename(p) != ".files"}
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from seedpack.codecs_gguf import pack
    base = len(lzma.compress(pack(files), preset=9))
    im = {k: v for k, v in files.items() if k.startswith("imatrix")}
    rest = {k: v for k, v in files.items() if k not in im}
    tot = len(lzma.compress(pack(rest), preset=9 | lzma.PRESET_EXTREME))
    for k, v in im.items():
        z = enc_imatrix(v); assert dec_imatrix(z) == v, "round trip"
        print(k, len(v), "lzma alone", len(lzma.compress(v, preset=9 | lzma.PRESET_EXTREME)), "planes", len(z))
        tot += len(z)
    print(f"M28 extras (lzma of pack): {base:,} B; R5: {tot:,} B; saving {base - tot:,} B")
