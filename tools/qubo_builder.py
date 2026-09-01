"""
QUBO builder — constructs the QUBO matrix and SparsePauliOp Hamiltonian
for portfolio optimization problems.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def build_qubo(problem_input: str, n: int = 6, budget: int = 3,
               q: float = 0.5) -> tuple[np.ndarray, SparsePauliOp]:
    """
    Placeholder QUBO builder — constructs the Woerner n=6, B=3, q=0.5
    benchmark instance by default.

    In full implementation: parse problem_input to extract
    covariance matrix, expected returns, and constraints.

    Returns:
        qubo:        (n x n) numpy QUBO matrix
        hamiltonian: SparsePauliOp for use in Qiskit circuits
    """
    # Placeholder: random symmetric QUBO for scaffolding
    np.random.seed(42)
    Q = np.random.randn(n, n)
    Q = (Q + Q.T) / 2                         # Symmetrize
    np.fill_diagonal(Q, np.abs(np.diag(Q)))   # Positive diagonal

    # Build a minimal Hamiltonian (ZZ + Z terms)
    pauli_list = []
    for i in range(n):
        z_str = "I" * i + "Z" + "I" * (n - i - 1)
        pauli_list.append((z_str, Q[i, i]))
        for j in range(i + 1, n):
            zz_str = (
                "I" * i + "Z" +
                "I" * (j - i - 1) + "Z" +
                "I" * (n - j - 1)
            )
            pauli_list.append((zz_str, Q[i, j]))

    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return Q, hamiltonian
