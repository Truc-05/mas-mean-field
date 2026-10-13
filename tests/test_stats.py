import numpy as np

from mfg_llm.metrics.stats import summarize, run_replicates, summarize_trajectories, bootstrap_ci


def test_summarize_known_values():
    s = summarize([1.0, 2.0, 3.0])
    assert s.mean == 2.0
    assert abs(s.std - 1.0) < 1e-12
    assert s.n == 3
    assert s.ci_low < 2.0 < s.ci_high


def test_summarize_single_sample_no_nan():
    s = summarize([5.0])
    assert s.mean == 5.0
    assert s.std == 0.0
    assert s.ci_low == s.ci_high == 5.0


def test_run_replicates_collects_all_keys():
    def run(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        return {"acc": float(rng.random()), "trace": float(rng.random() * 10)}

    out = run_replicates(run, n_seeds=8)
    assert set(out) == {"acc", "trace"}
    assert out["acc"].n == 8
    assert 0.0 <= out["acc"].mean <= 1.0
    assert out["acc"].std > 0.0


def test_summarize_trajectories_pointwise():
    trajs = [np.arange(5, dtype=float) + i for i in range(4)]
    out = summarize_trajectories(trajs)
    assert out["mean"].shape == (5,)
    assert np.allclose(out["mean"], np.arange(5) + 1.5)
    assert np.allclose(out["std"], out["std"][0])


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=1.0, size=200)
    lo, hi = bootstrap_ci(x, n_boot=2000, rng=rng)
    assert lo < 3.0 < hi
    assert hi - lo < 1.0
