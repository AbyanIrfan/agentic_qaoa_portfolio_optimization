"""
AgentState — the typed dict that flows through every node in the graph.
Every field is Optional so nodes can be added incrementally.
"""

from typing import TypedDict, Optional, Any
from qiskit import QuantumCircuit


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    problem_input:    str                    # Raw problem description

    # ── Problem init node output ──────────────────────────────────────────────
    n_assets:         Optional[int]
    tickers_used:      Optional[list]
    mu:                Optional[Any]          # numpy ndarray
    sigma:             Optional[Any]          # numpy ndarray
    risk_factor:       Optional[float]
    budget:            Optional[int]
    penalty:           Optional[int]
    n_shots:           Optional[int]
    reps:              Optional[int]

    # ── Literature search node output (Node 0) ────────────────────────────────
    literature_recommendation: Optional[str]
    cited_papers:               Optional[list]
    raw_papers:                  Optional[list]
    p_layers_recommended:        Optional[int]
    truncation_fraction:         Optional[float]

    # ── Literature validation node output (Node 0.5) ──────────────────────────
    literature_validation_skipped:   Optional[bool]
    arxiv_sources_used:               Optional[list]
    validation_supported_claims:      Optional[str]
    validation_contradicted_claims:   Optional[str]
    validation_unverified_claims:     Optional[str]
    validated_recommendation:         Optional[str]
    validation_found_contradiction:   Optional[bool]
    validated_p_layers:               Optional[int]
    validated_truncation_fraction:    Optional[float]
    validated_penalty:                Optional[int]

    # ── Formulation node output ───────────────────────────────────────────────
    qubo_matrix:      Optional[Any]          # numpy ndarray
    hamiltonian:      Optional[Any]          # SparsePauliOp
    warm_start_vec:   Optional[Any]          # QP relaxation solution vector (cvxpy)
    truncated_pairs:  Optional[list]         # Covariance-ranked XY-mixer edges
    offset:                     Optional[float]
    generated_formulation_code: Optional[str]

    # ── Circuit design node output ────────────────────────────────────────────
    circuit:          Optional[QuantumCircuit]
    variant_chosen:   Optional[str]          # "standard" | "warm_start_xy" | "cvar"
    p_layers:         Optional[int]
    # ── Circuit design node output (additional) ────────────────────────────────
    variant_reasoning:         Optional[str]
    generated_circuit_code:    Optional[str]
    hw_preservation_fraction:  Optional[float]
    static_validation_passed:  Optional[bool]

    # ── Execution node output ─────────────────────────────────────────────────
    job_results:      Optional[dict]         # Raw counts / quasi-distribution
    backend_chosen:   Optional[str]          # Simulator or hardware name
        # ── Execution node output (additional) ──────────────────────────────────────
    optimizer:            Optional[str]
    alpha:                 Optional[float]
    n_iterations:           Optional[int]
    cobyla_converged:        Optional[bool]
    final_cvar_opt:           Optional[float]
    classical_optimal:         Optional[float]
    cvar_history:                Optional[list]
    exp_history:                  Optional[list]
    opt_prob_history:              Optional[list]
    optimal_params:                 Optional[Any]
    best_bitstring:                  Optional[str]
    best_energy:                      Optional[float]
    best_assets:                       Optional[list]
    optimizer_init:                     Optional[str]

    # ── Analysis node output ──────────────────────────────────────────────────
    approx_ratio:     Optional[float]
    cvar_value:       Optional[float]
    energy:           Optional[float]
    metrics:          Optional[dict]         # Full metrics dict

    # ── Report node output ────────────────────────────────────────────────────
    report:           Optional[str]          # Full natural language report

    # ── Control ───────────────────────────────────────────────────────────────
    llm_name:         str                    # "claude" | "gpt" | "gemini" | "hetero"
    retry_count:      int
    errors:           Optional[list]         # Accumulated error messages
