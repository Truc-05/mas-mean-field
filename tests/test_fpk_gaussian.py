import numpy as np

from mfg_llm.dynamics.fpk_gaussian import (
    simulate_gaussian_fpk, simulate_gaussian_fpk_exact, fixed_point, consensus_collapses,
)


def test_mean_converges_to_fixed_point():
    K = 3
    A = np.eye(K) * 1.0
    B = np.eye(K) * 0.2
    s = np.array([1.0, 0.5, -0.3])
    sigma = 0.1
    m0 = np.zeros(K)
    cov0 = np.eye(K)

    m_traj, cov_traj = simulate_gaussian_fpk(m0, cov0, A, B, s, sigma, dt=0.01, n_steps=5000)
    m_inf, Sigma_inf, stable = fixed_point(A, B, s, sigma)

    assert stable
    assert np.allclose(m_traj[-1], m_inf, atol=1e-2)


def test_covariance_stays_symmetric_and_psd():
    K = 4
    A = np.eye(K) * 0.5
    B = np.eye(K) * 0.1
    s = np.zeros(K)
    sigma = 0.2
    m0 = np.zeros(K)
    cov0 = np.eye(K) * 2.0

    _, cov_traj = simulate_gaussian_fpk(m0, cov0, A, B, s, sigma, dt=0.01, n_steps=1000)
    for Sigma in cov_traj:
        assert np.allclose(Sigma, Sigma.T, atol=1e-8)
        eig = np.linalg.eigvalsh(Sigma)
        assert np.all(eig > -1e-6)


def test_consensus_collapse_flag():
    K = 2
    A_strong = np.eye(K) * 5.0
    assert consensus_collapses(A_strong, sigma=0.1)
    A_weak = np.eye(K) * 0.01
    assert not consensus_collapses(A_weak, sigma=1.0)
    A_unstable = -np.eye(K) * 0.1
    assert not consensus_collapses(A_unstable, sigma=0.01)


def test_consensus_collapse_matches_stationary_trace():
    # criterion should agree with tr(Sigma_inf) = tr(sigma * A^{-1}) for symmetric A
    A = np.diag([2.0, 4.0])
    sigma = 0.05
    trace_inf = sigma * (1 / 2.0 + 1 / 4.0)
    assert consensus_collapses(A, sigma, tol=trace_inf * 1.1)
    assert not consensus_collapses(A, sigma, tol=trace_inf * 0.9)


def test_exact_fpk_matches_euler_and_fixed_point():
    K = 3
    A = np.array([[1.0, 0.2, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.9]])
    B = np.eye(K) * 0.3
    s = np.array([0.5, -0.2, 0.1])
    sigma = 0.1
    m0 = np.ones(K)
    cov0 = np.eye(K) * 2.0

    m_ex, cov_ex = simulate_gaussian_fpk_exact(m0, cov0, A, B, s, sigma, dt=0.05, n_steps=400)
    m_eu, cov_eu = simulate_gaussian_fpk(m0, cov0, A, B, s, sigma, dt=0.001, n_steps=20000)
    assert np.allclose(m_ex[-1], m_eu[-1], atol=1e-2)
    assert np.allclose(cov_ex[-1], cov_eu[-1], atol=1e-2)

    m_inf, Sigma_inf, stable = fixed_point(A, B, s, sigma)
    assert stable
    assert np.allclose(m_ex[-1], m_inf, atol=1e-3)
    assert np.allclose(cov_ex[-1], Sigma_inf, atol=1e-3)


def test_unstable_matrix_returns_nan_fixed_point():
    K = 2
    A = np.eye(K) * 0.1
    B = np.eye(K) * 1.0
    s = np.ones(K)
    _, _, stable = fixed_point(A, B, s, sigma=0.1)
    assert not stable
