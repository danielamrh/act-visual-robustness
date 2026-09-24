"""Helpers for running CLI tools from Colab notebooks."""

from __future__ import annotations

import subprocess
from collections import deque


def run_streaming(cmd: list[str], tail_on_error: int = 40) -> None:
    """Run a command and stream its output into the notebook cell.

    Plain `subprocess.run` in Colab sends the child's output to the kernel log
    instead of the cell, so errors are invisible. On failure the last lines are
    repeated in the exception message.
    """
    print(" ".join(cmd), flush=True)
    tail: deque[str] = deque(maxlen=tail_on_error)
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
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; last output:\n" + "".join(tail)
        )
