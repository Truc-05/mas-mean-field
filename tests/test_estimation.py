import numpy as np

from mfg_llm.dynamics.particle_system import ParticleSystem
from mfg_llm.estimation.regression import (
    DebateLog, estimate_matrices, estimate_matrices_two_stage, spectrum_report,
)
from mfg_llm.estimation.validate_fit import validate_prediction


def _synthetic_log(seed: int = 0) -> tuple[DebateLog, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    K, N, T = 3, 200, 40
    A = np.array([[1.0, 0.1, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.9]])
    B = np.eye(K) * 0.3
    s = np.array([0.5, -0.2, 0.1])
    sigma = 0.05
    eta = 0.1

    x0 = rng.normal(size=(N, K))
    ps = ParticleSystem(x0, A, B, s, sigma, eta, rng=rng)
    traj = ps.run(T)

    log = DebateLog(states_by_round=[traj[t] for t in range(T + 1)], eta=eta)
    return log, A, B, s, sigma


def test_estimate_matrices_recovers_approx_parameters():
    log, A_true, B_true, s_true, sigma_true = _synthetic_log()
    params = estimate_matrices(log)

    assert params.A.shape == A_true.shape
    assert np.linalg.norm(params.A - A_true) / np.linalg.norm(A_true) < 0.5
    assert np.linalg.norm(params.s - s_true) / (np.linalg.norm(s_true) + 1e-9) < 1.0


def test_spectrum_report_keys():
    log, A_true, B_true, s_true, sigma_true = _synthetic_log()
    params = estimate_matrices(log)
    report = spectrum_report(params)
    for key in ["lambda_min_A", "lambda_max_B", "min_re_lambda_M", "stable_fixed_point",
                "lambda_min_A_H", "lambda_max_B_H", "min_re_lambda_M_H", "stable_fixed_point_H"]:
        assert key in report
    assert len(report["eig_A_H"]) == params.A.shape[0] - 1


def test_sigma_recovery():
    log, A_true, B_true, s_true, sigma_true = _synthetic_log()
    params = estimate_matrices(log)
    assert abs(params.sigma - sigma_true) / sigma_true < 0.25


def _synthetic_logs_diverse(n_logs: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    K, N, T = 3, 200, 40
    A = np.array([[1.0, 0.1, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.9]])
    B = np.eye(K) * 0.3
    s = np.array([0.5, -0.2, 0.1])
    sigma, eta = 0.05, 0.1

    logs = []
    for _ in range(n_logs):
        x0 = rng.normal(size=(N, K)) + rng.normal(scale=1.5, size=K)
        ps = ParticleSystem(x0, A, B, s, sigma, eta, rng=rng)
        traj = ps.run(T)
        logs.append(DebateLog(states_by_round=[traj[t] for t in range(T + 1)], eta=eta))
    return logs, A, B, s, sigma


def test_two_stage_recovers_B_from_diverse_logs():
    logs, A_true, B_true, s_true, sigma_true = _synthetic_logs_diverse()
    joint = estimate_matrices(logs[0])
    two_stage = estimate_matrices_two_stage(logs)
    err_joint = np.linalg.norm(joint.B - B_true) / np.linalg.norm(B_true)
    err_two = np.linalg.norm(two_stage.B - B_true) / np.linalg.norm(B_true)
    assert err_two < err_joint
    assert err_two < 0.3
    assert np.linalg.norm(two_stage.A - A_true) / np.linalg.norm(A_true) < 0.1
    assert abs(two_stage.sigma - sigma_true) / sigma_true < 0.25


def test_validate_prediction_error_bounded():
    log, A_true, B_true, s_true, sigma_true = _synthetic_log()
    params = estimate_matrices(log)
    report = validate_prediction(log, params)
    assert report["mean_relative_error"] < 1.0
    assert report["trace_relative_error"] < 2.0
