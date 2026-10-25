"""Check a remote GGUF's tensor list (header only) via HTTP range request. Supports v2+v3."""
import struct, sys, urllib.request

def fetch(url, start, end):
    if url.startswith("C:\\") or url.startswith("/"):
        with open(url, "rb") as f: f.seek(start); return f.read(end-start+1)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def read_u64(buf, off): return struct.unpack_from("<Q", buf, off)[0]
def read_str(buf, off):
    n = read_u64(buf, off); s = buf[off+8:off+8+n].decode(); return s, off+8+n

# gguf-py v3 enum: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool,8 string,9 array,10 u64,11 i64,12 f64
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_val(buf, off, t):
    if t == 8:   # string
        _, off = read_str(buf, off)
    elif t in SZ: off += SZ[t]
    elif t == 9:  # array
        et, off = struct.unpack_from("<I", buf, off)[0], off+4
        cnt = read_u64(buf, off); off += 8
        if et == 8:
            for _ in range(cnt): _, off = read_str(buf, off)
        elif et in SZ: off += cnt*SZ[et]
    return off

def check(url):
    buf = fetch(url, 0, 32_000_000)
    magic, ver = struct.unpack_from("<II", buf, 0)
    assert magic == 0x46554747, hex(magic)
    tc, kc = read_u64(buf, 8), read_u64(buf, 16)
    off = 24
    arch = None
    for _ in range(kc):
        if ver >= 3:
            k, off = read_str(buf, off)
            t, off = struct.unpack_from("<I", buf, off)[0], off+4
        else:
            t, off = struct.unpack_from("<I", buf, off)[0], off+4
            k, off = read_str(buf, off)
        if k == "general.architecture" and t == 8:
            s, off = read_str(buf, off); arch = s; continue
        off = skip_val(buf, off, t)
    tops = {}
    for _ in range(tc):
        n, off = read_str(buf, off)
        ndim, off = struct.unpack_from("<I", buf, off)[0], off+4
        dims = struct.unpack_from(f"<{ndim}Q", buf, off) if ndim else ()
        off += 8*ndim + 4 + 8
        if not n.startswith("blk."):
            tops[n] = dims
    print(f"v{ver}", arch, "| tensors:", tc, "| top-level:", tops)

for u in sys.argv[1:]:
    try: check(u)
    except Exception as e: print("ERR", u, repr(e)[:200])
