"""Lightweight Raspberry Pi health metrics (stdlib only, read from /proc and /sys).

SoC temperature, throttling, CPU usage, RAM, swap (`SwapTotal` / `SwapFree`)
and disk read/write throughput. CPU and disk are rates, so `SystemStats`
keeps the previous sample and computes the delta on every `snapshot()`.
"""

import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

# CPU and GPU share the same SoC die and a single thermal sensor.
THERMAL_ZONE = Path("sys/class/thermal/thermal_zone0/temp")
MEMINFO = Path("proc/meminfo")
PROC_STAT = Path("proc/stat")
DISKSTATS = Path("proc/diskstats")
# bits 0-3: under-voltage / frequency capped / throttled / soft temp limit
THROTTLED_NOW_MASK = 0xF
THROTTLED_TTL_S = 5.0  # vcgencmd spawns a process: don't call it every poll
SECTOR_BYTES = 512  # /proc/diskstats always counts 512-byte sectors
# Whole disks only (partitions would be counted twice); no loop/ram/zram
DISK_RE = re.compile(r"^(mmcblk\d+|sd[a-z]+|nvme\d+n\d+)$")


def _vcgencmd_throttled() -> str:
    return subprocess.run(
        ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
    ).stdout  # e.g. "throttled=0x0"


class SystemStats:
    def __init__(
        self,
        root: Path = Path("/"),
        clock: Callable[[], float] = time.monotonic,
        throttled_cmd: Callable[[], str] = _vcgencmd_throttled,
    ) -> None:
        self._root = root
        self._clock = clock
        self._throttled_cmd = throttled_cmd
        self._lock = threading.Lock()
        self._cpu_prev: tuple[int, int] | None = None  # (busy, total) jiffies
        self._disk_prev: tuple[float, int, int] | None = None  # (t, read, written)
        self._throttled: tuple[float, bool | None] | None = None

    def snapshot(self) -> dict:
        """All metrics; a value is None when it can't be read on this machine."""
        with self._lock:
            ram, swap = self.memory_mb()
            return {
                "soc": {"temp_c": self.soc_temp(), "throttled": self.is_throttled()},
                "cpu": {"pct": self.cpu_percent()},
                "ram": ram,
                "swap": swap,
                "disk": self.disk_mb_s(),
            }

    def _read(self, path: Path) -> str:
        return (self._root / path).read_text()

    def soc_temp(self) -> float | None:
        """SoC (CPU+GPU) temperature in °C."""
        try:
            return int(self._read(THERMAL_ZONE)) / 1000
        except (OSError, ValueError):
            return None

    def memory_mb(self) -> tuple[dict, dict]:
        """RAM (MemTotal - MemAvailable) and swap (SwapTotal - SwapFree) in MB."""
        empty = {"used_mb": None, "total_mb": None}
        try:
            info = {}
            for line in self._read(MEMINFO).splitlines():
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0])  # kB
        except (OSError, ValueError):
            return empty, dict(empty)

        def used_total(total_key: str, free_key: str) -> dict:
            if total_key not in info or free_key not in info:
                return dict(empty)
            total = info[total_key]
            return {
                "used_mb": (total - info[free_key]) // 1024,
                "total_mb": total // 1024,
            }

        return used_total("MemTotal", "MemAvailable"), used_total(
            "SwapTotal", "SwapFree"
        )

    def cpu_percent(self) -> float | None:
        """CPU busy % since the previous call (None on the first one)."""
        try:
            fields = [int(v) for v in self._read(PROC_STAT).splitlines()[0].split()[1:]]
        except (OSError, ValueError, IndexError):
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
        total = sum(fields[:8])  # user..steal (guest is already in user)
        busy = total - idle
        prev, self._cpu_prev = self._cpu_prev, (busy, total)
        if prev is None or total <= prev[1]:
            return None
        return round(100 * (busy - prev[0]) / (total - prev[1]), 1)

    def disk_mb_s(self) -> dict:
        """Read / write throughput of the whole disks (MB/s) since the last call."""
        empty = {"read_mb_s": None, "write_mb_s": None}
        try:
            read = written = 0
            for line in self._read(DISKSTATS).splitlines():
                parts = line.split()
                if len(parts) >= 10 and DISK_RE.match(parts[2]):
                    read += int(parts[5])
                    written += int(parts[9])
        except (OSError, ValueError):
            return empty
        now = self._clock()
        prev, self._disk_prev = self._disk_prev, (now, read, written)
        if prev is None or now <= prev[0]:
            return empty
        dt = now - prev[0]

        def rate(sectors: int) -> float:
            return round(max(sectors, 0) * SECTOR_BYTES / dt / 1e6, 2)

        return {
            "read_mb_s": rate(read - prev[1]),
            "write_mb_s": rate(written - prev[2]),
        }

    def is_throttled(self) -> bool | None:
        """True if the Pi is throttling now (heat or under-voltage); cached."""
        now = self._clock()
        if self._throttled is not None and now - self._throttled[0] < THROTTLED_TTL_S:
            return self._throttled[1]
        try:
            out = self._throttled_cmd()
            value = bool(int(out.strip().split("=")[1], 16) & THROTTLED_NOW_MASK)
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            value = None
        self._throttled = (now, value)
        return value
