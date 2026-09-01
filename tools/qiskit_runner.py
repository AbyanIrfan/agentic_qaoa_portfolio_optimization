"""
Qiskit runner — builds QAOA circuits and executes them on
simulator (Aer) or IBM Quantum hardware (Runtime).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from config.settings import IBM_QUANTUM_TOKEN, IBM_QUANTUM_CHANNEL, BENCHMARK_SHOTS


# ── Circuit builders ──────────────────────────────────────────────────────────

def build_qaoa_circuit(
    hamiltonian,
    variant:         str,
    p:               int,
    warm_start_vec=  None,
    truncated_pairs= None,
) -> QuantumCircuit:
    """
    Dispatch to the correct QAOA circuit builder based on variant.
    """
    if variant == "warm_start_xy":
        return _build_warm_start_xy(hamiltonian, p, warm_start_vec, truncated_pairs)
    elif variant == "cvar":
        return _build_cvar_qaoa(hamiltonian, p)
    else:
        return _build_standard_qaoa(hamiltonian, p)


def _build_standard_qaoa(hamiltonian, p: int) -> QuantumCircuit:
    n = hamiltonian.num_qubits
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    qc = QuantumCircuit(n)
    qc.h(range(n))                                 # Initial superposition

    for layer in range(p):
        # Cost layer
        for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            qc.append(term.to_instruction([gamma[layer]]), range(n))
        # Mixer layer (X-mixer)
        for q in range(n):
            qc.rx(2 * beta[layer], q)

    qc.measure_all()
    return qc


def _build_warm_start_xy(hamiltonian, p, warm_start_vec, truncated_pairs) -> QuantumCircuit:
    """Your thesis method: warm-start amplitudes + truncated XY-mixer."""
    n = hamiltonian.num_qubits
    gamma = ParameterVector("γ", p)
    beta  = ParameterVector("β", p)

    qc = QuantumCircuit(n)

    # Warm-start initialization (Ry rotations from QP relaxation)
    for i, val in enumerate(warm_start_vec):
        angle = 2 * np.arcsin(np.sqrt(np.clip(val, 0, 1)))
        qc.ry(angle, i)

    for layer in range(p):
        # Cost layer
        for term, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            qc.append(term.to_instruction([gamma[layer]]), range(n))
        # Sparse XY-mixer (only truncated pairs)
        for (i, j) in (truncated_pairs or []):
            qc.rxx(beta[layer], i, j)
            qc.ryy(beta[layer], i, j)

    qc.measure_all()
    return qc


def _build_cvar_qaoa(hamiltonian, p: int) -> QuantumCircuit:
    # Placeholder — same structure as standard; CVaR applied at post-processing
    return _build_standard_qaoa(hamiltonian, p)


# ── Execution ─────────────────────────────────────────────────────────────────

def run_circuit_simulator(circuit: QuantumCircuit, backend: str = "aer_simulator") -> dict:
    sim = AerSimulator()
    bound = circuit.assign_parameters(
        {p: np.random.uniform(0, np.pi) for p in circuit.parameters}
    )
    job    = sim.run(bound, shots=BENCHMARK_SHOTS)
    counts = job.result().get_counts()
    total  = sum(counts.values())
    return {k: v / total for k, v in counts.items()}   # Normalized quasi-dist


def run_circuit_hardware(circuit: QuantumCircuit, backend: str) -> dict:
    service = QiskitRuntimeService(
        channel=IBM_QUANTUM_CHANNEL,
        token=IBM_QUANTUM_TOKEN,
    )
    backend_obj = service.backend(backend)
    sampler     = Sampler(mode=backend_obj)
    bound = circuit.assign_parameters(
        {p: np.random.uniform(0, np.pi) for p in circuit.parameters}
    )
    job    = sampler.run([bound], shots=BENCHMARK_SHOTS)
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total  = sum(counts.values())
    return {k: v / total for k, v in counts.items()}
