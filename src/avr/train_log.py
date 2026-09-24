"""Parse `lerobot-train` log lines, e.g.

INFO 2026-09-24 21:03:12 ot_train.py:412 step:1K smpl:8K ep:20 epch:0.40 loss:0.512 grdn:23.1 lr:1.0e-05 updt_s:0.312 data_s:0.004
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_KV = re.compile(r"\b(step|loss|grdn|lr|updt_s|data_s):([0-9.eE+-]+[KMB]?)")
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def _number(text: str) -> float:
    if text[-1] in _SUFFIX:
        return float(text[:-1]) * _SUFFIX[text[-1]]
    return float(text)


def parse_train_log(lines) -> list[dict]:
    """One dict per metrics line: time, step, loss, grdn, lr, updt_s, data_s.
    Note: LeRobot abbreviates steps >= 1000 (`1K`), so they are approximate."""
    records = []
    for line in lines:
        if "step:" not in line or "loss:" not in line:
            continue
        rec = {k: _number(v) for k, v in _KV.findall(line)}
        ts = _TS.search(line)
        rec["time"] = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S") if ts else None
        records.append(rec)
    return records


def read_train_log(path: str | Path) -> list[dict]:
    return parse_train_log(Path(path).read_text(errors="replace").splitlines())


def seconds_per_step(records: list[dict], skip_first: int = 1) -> float:
    """Wall-clock seconds per training step from log timestamps. The first
    interval(s) include startup and are skipped."""
    recs = [r for r in records if r.get("time") is not None][skip_first:]
    if len(recs) < 2:
        raise ValueError("need at least two timed log lines after skipping warm-up")
    dt = (recs[-1]["time"] - recs[0]["time"]).total_seconds()
    dsteps = recs[-1]["step"] - recs[0]["step"]
    return dt / dsteps
