"""Helpers for Colab notebooks: GPU rendering setup and running CLI tools."""

from __future__ import annotations

import codecs
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


_SEGMENT = re.compile(r"([^\r\n]*)(\r\n|\n|\r)")


def run_streaming(
    cmd: list[str],
    log_path: str | Path | None = None,
    tail_on_error: int = 40,
    progress_every: float = 30.0,
) -> None:
    """Run a command and stream its output into the notebook cell.

    Plain `subprocess.run` in Colab sends the child's output to the kernel log
    instead of the cell, so errors are invisible. Progress bars redraw with
    `\\r` hundreds of thousands of times during a long eval; printing each
    redraw as a line crashes the browser tab, so `\\r`-terminated updates are
    shown at most every `progress_every` seconds. Output is optionally appended
    to `log_path`; on failure the last lines are repeated in the exception.
    """
    print(" ".join(cmd), flush=True)
    tail: deque[str] = deque(maxlen=tail_on_error)
    log = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8")

    def emit(text: str) -> None:
        print(text, flush=True)
        tail.append(text + "\n")
        if log is not None:
            log.write(text + "\n")
            log.flush()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    last_progress = 0.0
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while True:
                chunk = os.read(fd, 65536)
                buf += decoder.decode(chunk, final=not chunk)
                pos = 0
                for m in _SEGMENT.finditer(buf):
                    pos = m.end()
                    text, end = m.group(1), m.group(2)
                    if not text.strip():
                        continue
                    if end == "\r":  # progress-bar redraw
                        now = time.monotonic()
                        if now - last_progress >= progress_every:
                            last_progress = now
                            emit(text)
                    else:
                        emit(text)
                buf = buf[pos:]
                if not chunk:
                    break
            if buf.strip():
                emit(buf)
            returncode = proc.wait()
    finally:
        if log is not None:
            log.close()
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; last output:\n" + "".join(tail)
        )


NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"


def ensure_nvidia_egl() -> bool:
    """Register NVIDIA's EGL driver with glvnd. Without this file Colab's EGL
    falls back to Mesa's CPU renderer (llvmpipe), which makes MuJoCo rendering
    ~100x slower. Same fix as the official MuJoCo/dm_control Colab tutorials.
    Returns True if the file was created."""
    if os.path.exists(NVIDIA_EGL_ICD):
        return False
    os.makedirs(os.path.dirname(NVIDIA_EGL_ICD), exist_ok=True)
    with open(NVIDIA_EGL_ICD, "w") as f:
        json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}, f)
    return True


def gl_renderer() -> str:
    """OpenGL renderer that a fresh MuJoCo process gets (e.g. 'Tesla T4/PCIe/SSE2'
    on GPU, 'llvmpipe (...)' on CPU). Runs in a subprocess, like lerobot-eval."""
    code = (
        "import mujoco; ctx = mujoco.GLContext(64, 64); ctx.make_current(); "
        "from OpenGL import GL; print(GL.glGetString(GL.GL_RENDERER).decode()); ctx.free()"
    )
    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return out.stdout.strip() or f"failed: {out.stderr.strip()[-500:]}"
