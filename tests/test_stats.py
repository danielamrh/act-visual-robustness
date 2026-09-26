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


def test_summarize_lerobot_06_layout():
    # eval_info.json as written by lerobot-eval 0.6.1 (eval_policy_all)
    info = {
        "per_task": [
            {"task_group": "aloha", "task_id": 0, "metrics": {
                "sum_rewards": [120.0, 30.0, 0.0],
                "max_rewards": [4.0, 2.0, 0.0],
                "successes": [True, False, False],
                "video_paths": ["a.mp4"],
                "predicted_video_paths": [],
            }},
        ],
        "per_group": {"aloha": {"pc_success": 33.3, "n_episodes": 3}},
        "overall": {"pc_success": 33.3, "n_episodes": 3, "video_paths": ["a.mp4"]},
    }
    s = summarize_eval(info)
    assert s["n_episodes"] == 3 and s["successes"] == 1
    assert s["stage_counts"] == {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}


def test_summarize_flat_layout():
    info = {"per_episode": [{"success": False, "max_reward": 0.0}] * 3}
    s = summarize_eval(info)
    assert s["successes"] == 0
    assert s["stage_counts"][0] == 3


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize_eval({"aggregated": {}})


def test_merge_eval_infos(tmp_path):
    import json

    from avr.eval.stats import merge_eval_infos

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"per_episode": [{"success": True, "max_reward": 4}] * 3}))
    b.write_text(json.dumps({"per_task": [{"metrics": {"per_episode": [{"success": False, "max_reward": 1}]}}]}))
    s = summarize_eval(merge_eval_infos([a, b]))
    assert s["n_episodes"] == 4 and s["successes"] == 3
