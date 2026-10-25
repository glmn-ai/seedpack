"""How much more host memory SeedPack may take, read from the system at every decision.

The room is the memory the system has available (free pages plus the file cache it can drop) minus
a reserve left to the system and the other programs, at most the RAM left by them and the reserve
minus all of this process's private memory, and on Windows also the commit charge's room (RAM +
page file) minus a reserve. The packer's and the unpacker's budgets (prediction cache, files
read ahead, coded data waiting for zstd, files decoded ahead) grow while there is room and shrink
when another program takes it, so an archive is made with what the machine has, all of it, but no
more. The reserve is 20% of the RAM, at least 6 GB and at most 12 GB (SEEDPACK_RESERVE_MB sets it):
long runs at the edge of the memory are where an unstable machine fails.
"""
import os

COMMIT_RESERVE = 4 << 30                  # commit charge (Windows) left to the system


def _own():
    """(this process's resident bytes, its private bytes, resident or not), (0, 0) if unknown"""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD), ("peak_ws", ctypes.c_size_t),
                            ("ws", ctypes.c_size_t), ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t),
                            ("qnpp", ctypes.c_size_t), ("qnp", ctypes.c_size_t), ("pf", ctypes.c_size_t),
                            ("peak_pf", ctypes.c_size_t), ("private", ctypes.c_size_t)]
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.ws, pmc.private
            return 0, 0
        info = {}
        with open("/proc/self/status") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("VmRSS", "RssAnon", "VmSwap"):
                    info[k] = int(v.split()[0]) * 1024
        return info.get("VmRSS", 0), info.get("RssAnon", 0) + info.get("VmSwap", 0)
    except Exception:                         # (the room then rests on the system's figures alone)
        return 0, 0


def _status():
    """(total RAM, available RAM, commit charge left or None) in bytes; available None if unknown"""
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("pf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                        ("virt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
        ms = MS(); ms.len = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return ms.total, ms.avail, ms.apf
        return None, None, None
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemFree"]), None
    except (OSError, KeyError, ValueError):
        return None, None, None


def reserve(total=None):
    """bytes of RAM left to the system and the other programs"""
    env = os.environ.get("SEEDPACK_RESERVE_MB")
    if env:
        return int(env) << 20
    total = _status()[0] if total is None else total
    return 6 << 30 if total is None else min(12 << 30, max(6 << 30, total // 5))


def room():
    """bytes this process may still take (negative: it should give some back); very large if
    the system does not say. Its own private bytes count whole, resident or paged out: when the
    system pages a process's idle memory out, the available memory grows again, and taking that
    would only make the system page more"""
    total, avail, commit = _status()
    if avail is None:
        return 1 << 62
    res = reserve(total)
    ws, private = _own()
    others = max(0, total - avail - ws)       # what the system and the other programs hold
    r = min(avail - res, total - res - others - private)
    if commit is not None:
        r = min(r, commit - COMMIT_RESERVE)
    return r


def fits(n):
    """whether n more bytes keep the system's reserve"""
    return room() >= n


def budget(cap, floor=256 << 20, share=3):
    """a buffer budget: at most `cap`, at most a `share`-th of the room (several buffers grow at
    once), at least `floor` (the work goes on, slower, when memory is short)"""
    return int(max(floor, min(cap, room() // share)))
