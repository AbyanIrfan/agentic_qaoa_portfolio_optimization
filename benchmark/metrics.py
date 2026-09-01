"""
Metrics — computes approximation ratio, CVaR, and energy from QAOA results.
"""

import numpy as np


def bitstring_energy(bitstring: str, qubo: np.ndarray) -> float:
    x = np.array([int(b) for b in bitstring])
    return float(x @ qubo @ x)


def compute_energy(quasi_dist: dict, qubo: np.ndarray) -> float:
    """Expected energy E[x^T Q x] over the quasi-distribution."""
    return sum(
        prob * bitstring_energy(bs, qubo)
        for bs, prob in quasi_dist.items()
    )


def compute_approx_ratio(quasi_dist: dict, qubo: np.ndarray) -> float:
    """
    Approximation ratio = best_found_energy / classical_optimal_energy.
    Classical optimal approximated by brute-force over all bitstrings.
    """
    n = qubo.shape[0]
    energies  = [bitstring_energy(f"{i:0{n}b}", qubo) for i in range(2**n)]
    optimal   = min(energies)
    best_found = min(bitstring_energy(bs, qubo) for bs in quasi_dist)

    if optimal == 0:
        return 1.0
    return float(best_found / optimal)


def compute_cvar(quasi_dist: dict, qubo: np.ndarray, alpha: float = 0.25) -> float:
    """
    CVaR at level alpha — expected energy of the worst-alpha tail of samples.
    Lower CVaR = better tail-risk performance.
    """
    sorted_samples = sorted(
        quasi_dist.items(),
        key=lambda x: bitstring_energy(x[0], qubo)
    )

    cumulative = 0.0
    cvar_sum   = 0.0

    for bs, prob in sorted_samples:
        energy = bitstring_energy(bs, qubo)
        if cumulative + prob <= alpha:
            cvar_sum   += prob * energy
            cumulative += prob
        else:
            remaining   = alpha - cumulative
            cvar_sum   += remaining * energy
            break

    return cvar_sum / alpha if alpha > 0 else 0.0
