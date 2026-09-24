import pytest

from avr.lerobot_cli import eval_cmd, resume_cmd, train_cmd
from avr.train_log import parse_train_log, seconds_per_step

LOG = """\
INFO 2026-09-24 21:00:00 ot_train.py:300 Start offline training on a fixed dataset
INFO 2026-09-24 21:00:30 ot_train.py:412 step:50 smpl:400 ep:1 epch:0.02 loss:12.345 grdn:150.2 lr:1.0e-05 updt_s:0.410 data_s:0.020
INFO 2026-09-24 21:00:50 ot_train.py:412 step:100 smpl:800 ep:2 epch:0.04 loss:5.100 grdn:80.0 lr:1.0e-05 updt_s:0.400 data_s:0.010
INFO 2026-09-24 21:01:10 ot_train.py:412 step:150 smpl:1K ep:3 epch:0.06 loss:3.200 grdn:60.0 lr:1.0e-05 updt_s:0.400 data_s:0.010
INFO 2026-09-24 22:00:00 ot_train.py:412 step:12K smpl:96K ep:240 epch:4.80 loss:0.150 grdn:9.0 lr:1.0e-05 updt_s:0.300 data_s:0.005
"""


def test_parse_metrics_and_suffixes():
    recs = parse_train_log(LOG.splitlines())
    assert len(recs) == 4
    assert recs[0]["step"] == 50 and recs[0]["loss"] == pytest.approx(12.345)
    assert recs[0]["lr"] == pytest.approx(1e-5)
    assert recs[-1]["step"] == 12_000


def test_seconds_per_step_skips_warmup():
    recs = parse_train_log(LOG.splitlines()[:4])
    # from step 100 to 150: 20 s / 50 steps
    assert seconds_per_step(recs) == pytest.approx(0.4)


def test_commands():
    t = train_cmd("/content/outputs/run", steps=500, save_checkpoint=False, env_eval_freq=0)
    assert t[0] == "lerobot-train"
    assert "--policy.push_to_hub=false" in t
    assert "--save_checkpoint=false" in t and "--env_eval_freq=0" in t
    assert "--eval.use_async_envs=false" in eval_cmd("p", "o")
    assert resume_cmd("c.json") == ["lerobot-train", "--config_path=c.json", "--resume=true"]
