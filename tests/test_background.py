import sys

import pytest

from avr.background import start_background, tail_lines, watch
from avr.eval.chunked import run_chunked_eval
from avr.lerobot_cli import eval_cmd


def test_background_job_and_tail(tmp_path, capsys):
    log = tmp_path / "job.log"
    code = "import sys\nfor i in range(1000): sys.stdout.write(f'\\rstep {i}')\nprint()\nprint('done')"
    watch(start_background([sys.executable, "-c", code], log), log, interval=0.05)
    assert tail_lines(log, 2) == ["step 999", "done"]
    assert len(capsys.readouterr().out.splitlines()) < 50  # compact status only


def test_background_failure(tmp_path):
    log = tmp_path / "job.log"
    proc = start_background([sys.executable, "-c", "import sys; print('boom'); sys.exit(2)"], log)
    with pytest.raises(RuntimeError, match="exit code 2") as exc:
        watch(proc, log, interval=0.05)
    assert "boom" in str(exc.value)


def test_eval_cmd_seed():
    assert "--seed=1100" in eval_cmd("p", "o", seed=1100)
    assert not any(a.startswith("--seed") for a in eval_cmd("p", "o"))


def test_chunked_eval_skips_finished_and_seeds(tmp_path, monkeypatch):
    import avr.eval.chunked as chunked

    calls = []

    def fake_start(cmd, log_path):
        calls.append(cmd)
        out = next(a.split("=", 1)[1] for a in cmd if a.startswith("--output_dir="))
        code = f"import os; os.makedirs(r'{out}', exist_ok=True); open(r'{out}/eval_info.json','w').write('{{}}')"
        return start_background([sys.executable, "-c", code], log_path)

    monkeypatch.setattr(chunked, "start_background", fake_start)
    root = tmp_path / "evalroot"
    (root / "chunk00" / "sub").mkdir(parents=True)
    (root / "chunk00" / "sub" / "eval_info.json").write_text("{}")  # already finished
    (root / "chunk01").mkdir()  # crashed leftover, must be redone

    paths = run_chunked_eval("pol", str(root), n_episodes=250, chunk_size=100, seed=1000,
                             log_dir=str(tmp_path / "logs"), interval=0.05)
    assert len(paths) == 3
    seeds = [next(a for a in c if a.startswith("--seed=")) for c in calls]
    assert seeds == ["--seed=1100", "--seed=1200"]
    assert "--eval.n_episodes=50" in calls[-1]
