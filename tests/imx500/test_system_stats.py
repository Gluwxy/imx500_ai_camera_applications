"""SystemStats on a fake /proc and /sys tree."""

from pathlib import Path

import pytest

from imx500_core.monitors.system_stats import SystemStats

MEMINFO = """MemTotal:         426000 kB
MemFree:           50000 kB
MemAvailable:     206000 kB
SwapTotal:        204800 kB
SwapFree:         153600 kB
"""


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _stat(busy: int, idle: int) -> str:
    # user nice system idle iowait irq softirq steal
    return f"cpu  {busy} 0 0 {idle} 0 0 0 0\ncpu0 1 2 3 4\n"


def _disk(read_sectors: int, write_sectors: int) -> str:
    row = "179 0 {name} 10 0 {r} 0 20 0 {w} 0 0 0 0"
    return "\n".join(
        [
            row.format(name="mmcblk0", r=read_sectors, w=write_sectors),
            row.format(name="mmcblk0p2", r=read_sectors, w=write_sectors),  # partition
            row.format(name="zram0", r=10**9, w=10**9),  # swap in RAM: ignored
        ]
    )


@pytest.fixture
def fake(tmp_path):
    _write(tmp_path, "sys/class/thermal/thermal_zone0/temp", "52100\n")
    _write(tmp_path, "proc/meminfo", MEMINFO)
    _write(tmp_path, "proc/stat", _stat(busy=100, idle=900))
    _write(tmp_path, "proc/diskstats", _disk(0, 0))
    clock = _Clock()
    stats = SystemStats(
        root=tmp_path, clock=clock, throttled_cmd=lambda: "throttled=0x50005\n"
    )
    return stats, tmp_path, clock


def test_first_snapshot(fake):
    stats, _, _ = fake
    s = stats.snapshot()
    assert s["soc"] == {"temp_c": 52.1, "throttled": True}  # bit 0 and 2 set
    assert s["ram"] == {"used_mb": 214, "total_mb": 416}  # MemTotal - MemAvailable
    assert s["swap"] == {"used_mb": 50, "total_mb": 200}  # SwapTotal - SwapFree
    # rates need a previous sample
    assert s["cpu"] == {"pct": None}
    assert s["disk"] == {"read_mb_s": None, "write_mb_s": None}


def test_cpu_and_disk_are_rates_between_snapshots(fake):
    stats, root, clock = fake
    stats.snapshot()
    _write(root, "proc/stat", _stat(busy=100 + 30, idle=900 + 70))
    # 2 s later: 4 MB read (8192 sectors), 1 MB written (2048 sectors)
    _write(root, "proc/diskstats", _disk(8192, 2048))
    clock.now += 2
    s = stats.snapshot()
    assert s["cpu"] == {"pct": 30.0}
    assert s["disk"] == {"read_mb_s": 2.1, "write_mb_s": 0.52}


def test_missing_sources_give_none(tmp_path):
    stats = SystemStats(root=tmp_path, throttled_cmd=lambda: "")
    s = stats.snapshot()
    assert s["soc"] == {"temp_c": None, "throttled": None}
    assert s["ram"] == {"used_mb": None, "total_mb": None}
    assert s["swap"] == {"used_mb": None, "total_mb": None}
    assert s["cpu"] == {"pct": None}


def test_no_swap_reports_zero_total(fake):
    stats, root, _ = fake
    _write(
        root,
        "proc/meminfo",
        "MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
    )
    assert stats.snapshot()["swap"] == {"used_mb": 0, "total_mb": 0}


def test_throttled_is_cached(fake):
    stats, _, clock = fake
    calls = []
    stats._throttled_cmd = lambda: calls.append(1) or "throttled=0x0"
    stats._throttled = None
    stats.is_throttled()
    stats.is_throttled()
    clock.now += 10
    stats.is_throttled()
    assert len(calls) == 2
