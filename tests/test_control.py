import numpy as np
import pytest

from mfg_llm.control.hjb_solver import (
    solve_lqr_hjb, optimal_feedback, steady_state_control, time_to_truth,
)
from mfg_llm.control.mf_rl.env import MeanFieldDebateEnv
from mfg_llm.opinion_state import one_hot_truth


def test_lqr_riccati_solution_is_positive_definite():
    K = 3
    A = np.eye(K) * 0.5
    B = np.eye(K) * 0.2
    s = np.zeros(K)
    P, K_gain = solve_lqr_hjb(A, B, s)
    eig = np.linalg.eigvalsh(0.5 * (P + P.T))
    assert np.all(eig > -1e-8)
    assert K_gain.shape == (K, K)


def test_time_to_truth_finite_when_controllable():
    K = 3
    A = np.eye(K) * 0.3
    B = np.eye(K) * 0.2
    a_star = one_hot_truth("A", ["A", "B", "C"])
    s = 0.1 * a_star
    P, K_gain = solve_lqr_hjb(A, B, s)
    x0 = np.zeros(K)
    t = time_to_truth(A, B, s, x0, a_star, K_gain, eta=0.05, tol=1.0, max_steps=500)
    assert t < 500


def test_feedforward_makes_target_exact_equilibrium():
    K = 3
    A = np.eye(K) * 0.3
    B = np.eye(K) * 0.2
    a_star = one_hot_truth("A", ["A", "B", "C"])
    s = 0.1 * a_star + np.array([0.05, -0.02, -0.03])
    P, K_gain = solve_lqr_hjb(A, B, s)
    u_ss = steady_state_control(A, B, s, a_star)

    x = np.zeros(K)
    for _ in range(4000):
        u = optimal_feedback(x, a_star, K_gain, u_ss)
        x = x + 0.05 * (-A @ x + B @ x + s + u)
    assert np.linalg.norm(x - a_star) < 1e-3


def test_env_step_shapes_and_reward_bounds():
    K = 3
    N = 20
    A = np.eye(K) * 0.5
    B = np.eye(K) * 0.2
    a_star = one_hot_truth("A", ["A", "B", "C"])
    s = 0.3 * a_star
    env = MeanFieldDebateEnv(N, K, A, B, s, sigma=0.2, a_star_idx=0, eta=0.1, max_steps=5)
    x, obs = env.reset()
    assert x.shape == (N, K)
    assert obs.shape == (2 * K,)

    for _ in range(5):
        x, obs, reward, done, info = env.step(np.array([1.0, 1.0]))
        assert x.shape == (N, K)
        assert obs.shape == (2 * K,)
        assert np.all(obs[K:] >= 0.0)
        assert 0.0 <= info["accuracy"] <= 1.0
    assert done


def test_policy_gradient_smoke():
    torch = pytest.importorskip("torch")
    from mfg_llm.control.mf_rl.train import train_policy_gradient, rollout_policy

    K = 3
    a_star = one_hot_truth("A", ["A", "B", "C"])
    env = MeanFieldDebateEnv(N=10, K=K, A=np.eye(K) * 0.5, B=np.eye(K) * 0.2,
                             s=0.3 * a_star, sigma=0.2, a_star_idx=0, eta=0.1, max_steps=5,
                             rng=np.random.default_rng(0))
    policy, returns = train_policy_gradient(env, n_episodes=3, seed=0)
    assert len(returns) == 3
    roll = rollout_policy(env, policy)
    assert roll["accuracy_traj"].shape == (5,)
