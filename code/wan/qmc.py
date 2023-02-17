import numpy as np
from oracle import Oracle
from qiskit.algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,
)
from qiskit.primitives import Sampler


def fake_qmc(oracle: Oracle, eps: float, delta: float) -> tuple[float, int]:

    estimate = oracle.p + np.random.uniform(-eps, eps)
    oracle_calls = int(1 / eps * np.log(2 / delta))

    return estimate, oracle_calls


def canonical_ae(
    problem: EstimationProblem,
    n_iter: int,
    delta: float,
) -> tuple[float, int]:

    sampler = Sampler()
    sampler.set_options(shots=int(np.log(1 / delta)))
    ae = AmplitudeEstimation(
        num_eval_qubits=max(int(np.log2(n_iter)), 1),
        sampler=sampler,
    )
    ae_res = ae.estimate(problem)
    est = ae_res.mle
    oracle_calls = ae_res.num_oracle_queries + 1
    return est, oracle_calls


def faster_ae(
    problem: EstimationProblem,
    n_iter: int,
    delta: float,
) -> tuple[float, int]:

    sampler = Sampler()
    ae = FasterAmplitudeEstimation(
        maxiter=n_iter,
        delta=delta,
        rescale=False,
        sampler=sampler,
    )
    ae_res = ae.estimate(problem)
    est = ae_res.estimation
    oracle_calls = ae_res.num_oracle_queries
    return est, oracle_calls


def qmc(
    oracle: Oracle, n_iter: int, delta: float, method: str = "canonical"
) -> tuple[float, int]:

    problem = EstimationProblem(
        state_preparation=oracle,
        objective_qubits=[0],
    )

    if method == "canonical":
        estimate, oracle_calls = canonical_ae(problem, n_iter, delta)
    elif method == "faster":
        estimate, oracle_calls = faster_ae(problem, n_iter, delta)
    else:
        raise ValueError(f"Unknown method {method}")

    return estimate, oracle_calls
