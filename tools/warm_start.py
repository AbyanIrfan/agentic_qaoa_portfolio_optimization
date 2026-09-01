"""
Warm-start tools — your thesis contribution lives here.
QP relaxation via cvxpy + covariance-ranked XY-mixer truncation.
"""

import numpy as np
import cvxpy as cp


def qp_relaxation(qubo: np.ndarray) -> np.ndarray:
    """
    Solve the continuous QP relaxation of the QUBO problem.
    Returns a real-valued solution vector in [0, 1]^n.

    This vector is used to:
      1. Initialize quantum state amplitudes (warm-start)
      2. Rank XY-mixer edges via covariance magnitude (truncation)
    """
    n = qubo.shape[0]
    x = cp.Variable(n)

    objective   = cp.Minimize(cp.quad_form(x, qubo))
    constraints = [x >= 0, x <= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP relaxation failed: {prob.status}")

    return np.array(x.value)


def get_truncated_pairs(
    warm_start_vec: np.ndarray,
    qubo: np.ndarray,
    top_k_fraction: float = 0.5
) -> list[tuple[int, int]]:
    """
    Rank qubit pairs by covariance magnitude and retain the top fraction.
    These pairs define the sparse XY-mixer topology.

    Args:
        warm_start_vec:   QP relaxation solution vector
        qubo:             QUBO matrix (used for covariance weighting)
        top_k_fraction:   Fraction of pairs to retain (default: top 50%)

    Returns:
        List of (i, j) qubit pairs for the XY-mixer
    """
    n = len(warm_start_vec)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            # Covariance magnitude as edge weight
            weight = abs(qubo[i, j]) * warm_start_vec[i] * warm_start_vec[j]
            pairs.append((i, j, weight))

    # Sort by weight descending, retain top fraction
    pairs.sort(key=lambda x: x[2], reverse=True)
    k = max(1, int(len(pairs) * top_k_fraction))

    return [(i, j) for i, j, _ in pairs[:k]]
