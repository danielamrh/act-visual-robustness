"""Helpers for running CLI tools from Colab notebooks."""

from __future__ import annotations

import subprocess
from collections import deque
from pathlib import Path


def run_streaming(cmd: list[str], log_path: str | Path | None = None, tail_on_error: int = 40) -> None:
    """Run a command and stream its output into the notebook cell.

    Plain `subprocess.run` in Colab sends the child's output to the kernel log
    instead of the cell, so errors are invisible. Output is optionally appended
    to `log_path`; on failure the last lines are repeated in the exception.
    """
    print(" ".join(cmd), flush=True)
    tail: deque[str] = deque(maxlen=tail_on_error)
    log = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                tail.append(line)
                if log is not None:
                    log.write(line)
                    log.flush()
            returncode = proc.wait()
    finally:
        if log is not None:
            log.close()
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; last output:\n" + "".join(tail)
        )
