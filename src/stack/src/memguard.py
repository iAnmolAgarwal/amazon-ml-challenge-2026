"""Memory watchdog: aborts the job (and its worker processes) before it exhausts RAM (limits from MEM_LIMIT_GB / MEM_MIN_AVAIL_GB).

Usage: `import memguard; memguard.start(limit_gb=3.0)` at the top of any heavy entry point.
The limit is on total RSS of this process + children; it also aborts if system free memory + inactive drops too low.
"""
import os
import sys

# Arrow's jemalloc/mimalloc pools keep freed buffers; the system allocator returns memory to the OS.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import threading
import time

import psutil

LIMIT_GB = float(os.environ.get("MEM_LIMIT_GB", "3.0"))
MIN_AVAIL_GB = float(os.environ.get("MEM_MIN_AVAIL_GB", "0.8"))
_peak = [0.0]


MIN_FREE_PCT = int(os.environ.get("MEM_MIN_FREE_PCT", "20"))


def free_pct():
    """macOS kernel memory level (% free incl. reclaimable cache, same as `memory_pressure`); None elsewhere."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        v = ctypes.c_int(0)
        sz = ctypes.c_size_t(ctypes.sizeof(v))
        if libc.sysctlbyname(b"kern.memorystatus_level", ctypes.byref(v), ctypes.byref(sz), None, 0) == 0:
            return v.value
    except Exception:
        pass
    return None


def rss_gb():
    p = psutil.Process()
    tot = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            tot += c.memory_info().rss
        except psutil.Error:
            pass
    return tot / 2**30


def peak_gb():
    return _peak[0]


def _watch(limit, min_avail, interval):
    while True:
        r = rss_gb()
        _peak[0] = max(_peak[0], r)
        fp = free_pct()
        if fp is not None:
            low = fp < MIN_FREE_PCT
            sysmsg = f"system free={fp}% (floor {MIN_FREE_PCT}%)"
        else:
            avail = psutil.virtual_memory().available / 2**30
            low = avail < min_avail
            sysmsg = f"system available={avail:.2f} GB"
        if r > limit or low:
            sys.stderr.write(f"\n[memguard] ABORT: rss={r:.2f} GB (limit {limit}), {sysmsg}\n")
            sys.stderr.flush()
            for c in psutil.Process().children(recursive=True):
                try:
                    c.kill()
                except psutil.Error:
                    pass
            os._exit(137)
        time.sleep(interval)


def start(limit_gb=None, min_avail_gb=None, interval=0.5):
    t = threading.Thread(target=_watch, args=(limit_gb or LIMIT_GB, min_avail_gb or MIN_AVAIL_GB, interval),
                         daemon=True)
    t.start()
    return t
