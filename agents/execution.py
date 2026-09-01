"""
Node 3: Execution — CVaR-COBYLA optimization loop over the circuit
produced by Node 2, run against either the Aer simulator or IBM Quantum
hardware (controlled by config.settings.USE_HARDWARE).
"""

from __future__ import annotations

import os
from itertools import combinations as icombs
from typing import Callable, Iterable

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from scipy.optimize import minimize as scipy_minimize

from config.settings import SEED, USE_HARDWARE

# ── Fast Pauli evaluation (IBM tutorial — bitwise XOR, orders of magnitude ────
# ── faster than string-parsing over Pauli terms)  ────────────────────────────
_PARITY = np.array(
    [-1 if bin(i).count("1") % 2 else 1 for i in range(256)],
    dtype=np.complex128,
)

def evaluate_sparse_pauli(state_int: int, observable) -> complex:
    """Evaluate <state|H|state> for diagonal Ising H via bitwise XOR parity."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes  = np.frombuffer(
        state_int.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8
    )
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def _get_cvar_aggregation(alpha: float) -> Callable:
    """
    Returns CVaR aggregation function for given alpha.
    alpha=1.0  → standard expectation value
    alpha=0.25 → average of best 25% of samples (Barkoutsos et al. 2020)
    alpha=0.10 → aggressive tail focus
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    def cvar_aggregation(objective_dict: Iterable) -> float:
        sorted_measurements = sorted(objective_dict, key=lambda x: x[1])
        accumulated, cvar = 0.0, 0.0
        for probability, value in sorted_measurements:
            contribution = min(probability, alpha - accumulated)
            cvar        += value * contribution
            accumulated += probability
            if accumulated >= alpha:
                break
        return cvar / alpha

    return cvar_aggregation


class CVaRObjective:
    """
    Tracks optimization history alongside CVaR, expectation value,
    and optimal solution probability per iteration.
    Adapted from IBM advanced QAOA tutorial + Woerner thesis implementation.
    """

    def __init__(self, ansatz, hamiltonian, offset,
                 alpha, sampler, n_qubits, optimal_value=None):
        self.ansatz        = ansatz
        self.hamiltonian   = hamiltonian
        self.offset        = offset
        self.alpha         = alpha
        self.sampler       = sampler
        self.n_qubits      = n_qubits
        self.optimal_value = optimal_value
        self.aggregator    = _get_cvar_aggregation(alpha)

        self.best_bitstring     = None
        self.best_value         = np.inf
        self.final_params       = None
        self.final_distribution = {}

        self.cvar_history     = []
        self.expectation_hist = []
        self.opt_prob_history = []
        self.iteration        = 0

    def evaluate(self, params: np.ndarray) -> float:
        self.iteration += 1

        job        = self.sampler.run([(self.ansatz, params)])
        result     = job.result()
        int_counts = result[0].data.meas.get_int_counts()
        shots      = sum(int_counts.values())
        dist       = {k: v / shots for k, v in int_counts.items()}

        evaluated = {}
        for state_int, prob in dist.items():
            energy = float(np.real(
                evaluate_sparse_pauli(state_int, self.hamiltonian)
            )) + self.offset
            evaluated[state_int] = (prob, energy)

            if energy < self.best_value:
                self.best_value     = energy
                self.best_bitstring = format(state_int, f'0{self.n_qubits}b')

        cvar = self.aggregator(evaluated.values())
        self.cvar_history.append(cvar)

        exp_val = sum(p * v for p, v in evaluated.values())
        self.expectation_hist.append(exp_val)

        if self.optimal_value is not None:
            opt_prob = sum(
                p for p, v in evaluated.values()
                if v <= self.optimal_value + 1e-6
            )
            self.opt_prob_history.append(opt_prob)

        if self.iteration % 10 == 0:
            print(f"  Iter {self.iteration:3d} | "
                  f"CVaR(α={self.alpha}): {cvar:.4f} | "
                  f"E[H]: {exp_val:.4f}")

        self.final_distribution = evaluated
        self.final_params       = params.copy()
        return cvar


def compute_classical_optimal(state):
    """
    Enumerate all C(n,k) feasible portfolios, return min Ising energy.
    Used for P(optimal) tracking in CVaRObjective and AR computation.
    """
    n      = state["n_assets"]
    budget = state["budget"]
    mu_v   = state["mu"]
    sigma_m= state["sigma"]
    q      = state["risk_factor"]
    offset = state.get("offset", 0.0)

    best = np.inf
    for combo in icombs(range(n), budget):
        x    = np.zeros(n)
        x[list(combo)] = 1
        val  = q * x @ sigma_m @ x - mu_v @ x
        best = min(best, val)

    return best


def execution_node(state):
    ALPHA        = 0.25
    MAX_ITER     = 100
    N_SHOTS      = state.get("n_shots", 1024)
    n_qubits     = state["n_assets"]

    circuit     = state["circuit"]
    hamiltonian = state["hamiltonian"]
    offset      = state.get("offset", 0.0)

    print("=" * 60)
    print(f"[Execution] CVaR-COBYLA optimization")
    print(f"  α={ALPHA} | max_iter={MAX_ITER} | shots={N_SHOTS}")

    if USE_HARDWARE:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
        service  = QiskitRuntimeService(
            token   = os.environ.get("IBM_QUANTUM_TOKEN"),
            channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum"),
        )
        backend      = service.backend(os.environ.get("DEFAULT_HARDWARE", "ibm_brisbane"))
        backend_used = backend.name
    else:
        backend      = AerSimulator()
        backend_used = "aer_simulator"

    pm         = generate_preset_pass_manager(
                     backend=backend, optimization_level=1, seed_transpiler=SEED)
    transpiled = pm.run(circuit)

    print(f"  Backend:  {backend_used}")
    print(f"  Circuit:  depth={transpiled.depth()} | "
          f"gates={transpiled.size()} | "
          f"params={transpiled.num_parameters}")

    if USE_HARDWARE:
        sampler = RuntimeSampler(mode=backend)
    else:
        sampler = AerSamplerV2()
    sampler.options.default_shots = N_SHOTS

    optimal_value = compute_classical_optimal(state)
    print(f"  Classical optimal (Ising): {optimal_value:.4f}")

    rng         = np.random.default_rng(SEED)
    init_params = rng.uniform(0, 2 * np.pi, transpiled.num_parameters)

    obj = CVaRObjective(
        ansatz        = transpiled,
        hamiltonian   = hamiltonian,
        offset        = offset,
        alpha         = ALPHA,
        sampler       = sampler,
        n_qubits      = n_qubits,
        optimal_value = optimal_value,
    )

    print(f"\n[Execution] Running COBYLA...")
    opt_result = scipy_minimize(
        obj.evaluate,
        init_params,
        method  = "COBYLA",
        options = {"maxiter": MAX_ITER, "rhobeg": 0.5},
    )

    job_results = {
        format(s_int, f'0{n_qubits}b'): prob
        for s_int, (prob, _) in obj.final_distribution.items()
    }

    top = max(job_results, key=job_results.get)

    print(f"\n[Execution] Complete:")
    print(f"  Final CVaR:         {opt_result.fun:.4f}")
    print(f"  Classical optimal:  {optimal_value:.4f}")
    print(f"  Converged:          {opt_result.success}")
    print(f"  COBYLA iterations:  {opt_result.nfev}")
    print(f"  Top bitstring:      {top} (prob={job_results[top]:.4f})")
    print(f"  Best found:         {obj.best_bitstring} (energy={obj.best_value:.4f})")
    print(f"  Unique outcomes:    {len(job_results)}")

    best_bits   = [int(b) for b in reversed(obj.best_bitstring)]
    best_assets = [i for i, b in enumerate(best_bits) if b == 1]
    print(f"  Best assets selected: {best_assets}")

    print(f"\n✓ Node 3 complete.")

    return {
        "job_results":       job_results,
        "backend_chosen":    backend_used,
        "optimizer":         "COBYLA",
        "alpha":             ALPHA,
        "n_iterations":      opt_result.nfev,
        "cobyla_converged":  opt_result.success,
        "final_cvar_opt":    opt_result.fun,
        "classical_optimal": optimal_value,
        "cvar_history":      obj.cvar_history,
        "exp_history":       obj.expectation_hist,
        "opt_prob_history":  obj.opt_prob_history,
        "optimal_params":    obj.final_params,
        "best_bitstring":    obj.best_bitstring,
        "best_energy":       obj.best_value,
        "best_assets": best_assets,
        "optimizer_init": "random uniform",
    }