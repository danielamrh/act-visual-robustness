import sys

import pytest

from avr.colab import run_streaming

PROGRESS = (
    "import sys\n"
    "print('start', flush=True)\n"
    "for i in range(5000):\n"
    "    sys.stdout.write(f'\\rstep {i}/5000')\n"
    "sys.stdout.write('\\n')\n"
    "print('done')\n"
)


def test_progress_redraws_are_throttled(capsys, tmp_path):
    log = tmp_path / "out.log"
    run_streaming([sys.executable, "-c", PROGRESS], log_path=log, progress_every=3600)
    expected = ["start", "step 0/5000", "step 4999/5000", "done"]
    # first redraw (throttle starts open), final state (terminated by \n), plain lines
    assert log.read_text().splitlines() == expected
    assert capsys.readouterr().out.splitlines()[-4:] == expected


def test_error_tail(capsys):
    code = "import sys; print('boom'); sys.exit(3)"
    with pytest.raises(RuntimeError, match="exit code 3") as exc:
        run_streaming([sys.executable, "-c", code])
    assert "boom" in str(exc.value)


def test_unterminated_last_line(capsys):
    run_streaming([sys.executable, "-c", "import sys; sys.stdout.write('no newline')"])
    assert capsys.readouterr().out.splitlines()[-1] == "no newline"
