import pytest

from avr.analysis.robustness import summary_rows


def _cell(factor, level, successes, max_rewards=None):
    max_rewards = max_rewards or [4 if s else 1 for s in successes]
    eps = [{"seed": 1000 + i, "success": s, "max_reward": r} for i, (s, r) in enumerate(zip(successes, max_rewards))]
    return {"factor": factor, "level": level, "episodes": eps}


def test_paired_rows():
    clean = _cell(None, 0, [True] * 8 + [False] * 2 + [True] * 10)  # 20 seeds, 18 ok
    cells = {
        ("clean", 0): clean,
        ("noise", 4): _cell("noise", 4, [False] * 8 + [False] * 2),  # first 10 seeds: 8 lost
        ("blur", 1): _cell("blur", 1, [True] * 8 + [False] * 2),  # identical to clean
    }
    rows = {(r["factor"], r["level"]): r for r in summary_rows(cells, clean=clean)}
    assert "p" not in rows[("clean", 0)]
    noise = rows[("noise", 4)]
    assert noise["clean_same_seeds"] == pytest.approx(0.8)
    assert noise["delta"] == pytest.approx(-0.8)
    assert (noise["only_clean"], noise["only_perturbed"]) == (8, 0)
    assert noise["p"] == pytest.approx(2 / 2**8)
    assert rows[("blur", 1)]["p"] == 1.0
    assert noise["p_holm"] == pytest.approx(2 * 2 / 2**8)
    assert list(rows)[0] == ("clean", 0)  # clean first, then factor order


def test_missing_clean_seed_is_an_error():
    clean = _cell(None, 0, [True] * 5)
    cells = {("noise", 1): _cell("noise", 1, [True] * 6)}
    with pytest.raises(ValueError):
        summary_rows(cells, clean=clean)
