import pytest

from avr.eval.stats import summarize_eval, wilson_ci


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(83, 100)
    assert lo < 0.83 < hi
    assert hi - lo == pytest.approx(0.147, abs=0.01)


def test_wilson_ci_edges():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1


def test_summarize_nested_layout():
    # newer lerobot versions nest results per task
    info = {
        "per_task": [
            {"metrics": {"per_episode": [
                {"success": True, "max_reward": 4.0},
                {"success": False, "max_reward": 2.0},
            ]}},
        ],
        "overall": {"pc_success": 50.0},
    }
    s = summarize_eval(info)
    assert s["n_episodes"] == 2
    assert s["success_rate"] == 0.5
    assert s["stage_counts"] == {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}


def test_summarize_flat_layout():
    info = {"per_episode": [{"success": False, "max_reward": 0.0}] * 3}
    s = summarize_eval(info)
    assert s["successes"] == 0
    assert s["stage_counts"][0] == 3


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize_eval({"aggregated": {}})
