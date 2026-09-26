"""Run long CLI jobs detached from the notebook and watch them with a few
status lines.

Streaming a long `lerobot-eval` into a Colab cell crashed the browser tab, so
the job writes only to a log file and the cell prints one compact status line
per interval: elapsed time, free RAM, the job's RAM and its latest output.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

_SPLIT = re.compile(r"[\r\n]+")


def start_background(cmd: list[str], log_path: str | Path) -> subprocess.Popen:
    """Start `cmd` with stdout+stderr going to `log_path` only."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def tail_lines(log_path: str | Path, n: int = 20, max_bytes: int = 256 * 1024) -> list[str]:
    """Last `n` non-empty lines, treating progress-bar redraws (`\\r`) as lines."""
    path = Path(log_path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        text = f.read().decode("utf-8", errors="replace")
    lines = [s.strip() for s in _SPLIT.split(text) if s.strip()]
    return lines[-n:]


def _meminfo_gb() -> tuple[float, float] | None:
    """(total, available) system RAM in GB, Linux only."""
    try:
        info = dict(
            line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
        )
        kb = lambda key: int(info[key].split()[0])  # noqa: E731
        return kb("MemTotal") / 1e6, kb("MemAvailable") / 1e6
    except (OSError, KeyError, ValueError):
        return None


def _rss_gb(pid: int) -> float | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    except (OSError, ValueError):
        pass
    return None


def status_line(proc: subprocess.Popen, log_path: str | Path, t0: float) -> str:
    minutes = (time.monotonic() - t0) / 60
    parts = [f"[{minutes:5.1f} min]"]
    mem = _meminfo_gb()
    if mem:
        parts.append(f"RAM free {mem[1]:.1f}/{mem[0]:.1f} GB")
    rss = _rss_gb(proc.pid)
    if rss is not None:
        parts.append(f"job {rss:.1f} GB")
    last = tail_lines(log_path, 1)
    if last:
        parts.append("| " + last[0][-120:])
    return "  ".join(parts)


def watch(proc: subprocess.Popen, log_path: str | Path, interval: float = 60.0, tail_on_error: int = 30) -> None:
    """Block until `proc` exits, printing one status line per `interval`.
    Raises with the log tail if the job fails (exit code -9 = killed, usually out of RAM)."""
    t0 = time.monotonic()
    while True:
        try:
            proc.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            print(status_line(proc, log_path, t0), flush=True)
    print(status_line(proc, log_path, t0), flush=True)
    if proc.returncode != 0:
        hint = " (killed - most likely out of RAM)" if proc.returncode == -9 else ""
        raise RuntimeError(
            f"job failed with exit code {proc.returncode}{hint}; log: {log_path}\n"
            + "\n".join(tail_lines(log_path, tail_on_error))
        )
