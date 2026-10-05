"""
Node 4: Analysis — computes approximation ratio, CVaR, and energy from
Node 3's job results against the Ising Hamiltonian, gets an LLM
interpretation, and writes a timestamped report to results/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from itertools import combinations as icombs
import numpy as np
from llm.router import get_node_llm, get_response_text


RESULTS_DIR = Path("results")


def bitstring_energy_ising(bitstring, hamiltonian, offset=0.0):
    bits  = [int(b) for b in reversed(bitstring)]
    spins = [1 - 2*b for b in bits]

    energy = 0.0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        pauli_str = pauli.to_label()
        z_qubits  = [i for i, c in enumerate(reversed(pauli_str)) if c == 'Z']
        term      = float(coeff.real)
        for q in z_qubits:
            term *= spins[q]
        energy += term
    return energy + offset


def compute_energy(quasi_dist, hamiltonian, offset=0.0):
    return sum(
        prob * bitstring_energy_ising(bs, hamiltonian, offset)
        for bs, prob in quasi_dist.items()
    )


def compute_feasible_optimal(hamiltonian, offset, n_assets, budget):
    """
    True classical optimum restricted to the budget-feasible subspace
    (Hamming weight == budget). Brute-forcing over all 2^n bitstrings
    instead would only give the right answer by coincidence — it relies
    on the Hamiltonian's penalty term being strong enough that no
    infeasible bitstring ever scores lower than the true feasible
    optimum, which isn't verified anywhere and isn't guaranteed to hold
    for every instance size/scale.
    """
    best = None
    for combo in icombs(range(n_assets), budget):
        state_int = sum(1 << i for i in combo)
        bitstring = f"{state_int:0{n_assets}b}"
        energy = bitstring_energy_ising(bitstring, hamiltonian, offset)
        if best is None or energy < best:
            best = energy
    return best


def compute_approx_ratio(quasi_dist, hamiltonian, offset, n_assets, budget):
    optimal    = compute_feasible_optimal(hamiltonian, offset, n_assets, budget)
    best_found = min(bitstring_energy_ising(bs, hamiltonian, offset) for bs in quasi_dist)
    return 1.0 if optimal == 0 else float(best_found / optimal)


def compute_cvar(quasi_dist, hamiltonian, alpha=0.25, offset=0.0):
    sorted_samples = sorted(
        quasi_dist.items(),
        key=lambda x: bitstring_energy_ising(x[0], hamiltonian, offset)
    )
    cumulative, cvar_sum = 0.0, 0.0
    for bs, prob in sorted_samples:
        energy = bitstring_energy_ising(bs, hamiltonian, offset)
        if cumulative + prob <= alpha:
            cvar_sum   += prob * energy
            cumulative += prob
        else:
            cvar_sum += (alpha - cumulative) * energy
            break
    return cvar_sum / alpha if alpha > 0 else 0.0

def compute_sharpe_ratio(mu, sigma, selected_assets, budget, risk_free_rate=0.0):
    """
    Sharpe ratio of the EQUAL-WEIGHTED portfolio over the selected assets
    (1/budget each). Equal-weighting is used because this pipeline only
    performs discrete selection, not continuous allocation — there is no
    optimized weight vector to draw on (unlike e.g. Mancilla et al.'s
    separate SLSQP allocation stage).

    mu/sigma are already annualized (see finance_data.py), so
    risk_free_rate must also be annual for the units to match.
    """
    w = np.zeros(len(mu))
    for i in selected_assets:
        w[i] = 1.0 / budget
    portfolio_return = float(w @ mu)
    portfolio_variance = float(w @ sigma @ w)
    portfolio_std = float(np.sqrt(max(portfolio_variance, 0.0)))
    if portfolio_std < 1e-12:
        return None  # degenerate — avoid dividing by ~zero
    return (portfolio_return - risk_free_rate) / portfolio_std


def _write_report(state, metrics, interpretation) -> Path:
    """Write a timestamped analysis report to results/. One file per run,
    not overwritten, so past runs stay comparable."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = state.get("llm_name", "unknown")
    report_path = RESULTS_DIR / f"analysis_{backend}_{timestamp}.txt"

    lines = [
        f"Analysis Report — {timestamp}",
        "=" * 60,
        f"Problem:          {state.get('problem_input', 'N/A')}",
        f"Variant:          {metrics.get('variant')}",
        f"p_layers:         {metrics.get('p_layers')}",
        f"Backend:          {metrics.get('backend')}",
        f"Retry count:      {metrics.get('retry')}",
        "",
        f"Approx. ratio:    {metrics['approx_ratio']:.4f}",
        f"CVaR:             {metrics['cvar']:.4f}",
        f"Energy:           {metrics['energy']:.4f}",
        "",
        "Interpretation:",
        interpretation,
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def analysis_node(state):
    from config.settings import RISK_FREE_RATE
    llm = get_node_llm("analysis", state.get("llm_name"))                # analysis.py

    print("[Analysis] Computing metrics...")

    job_results  = state["job_results"]
    hamiltonian  = state["hamiltonian"]
    offset = state["offset"]

    approx_ratio = compute_approx_ratio(job_results, hamiltonian, offset, state["n_assets"], state["budget"])
    sharpe_ratio = compute_sharpe_ratio(
        state["mu"], state["sigma"], state["best_assets"], state["budget"],
        risk_free_rate=RISK_FREE_RATE,
    )

    # CVaR is an optimization-time training signal, computed fresh every
    # COBYLA iteration in execution.py — not something to independently
    # re-derive from a stored bitstring snapshot after the fact. Report the
    # value the optimizer itself actually used and converged to, rather
    # than recomputing a second, potentially-divergent CVaR here.
    cvar_value = state["final_cvar_opt"]

    # Kept as a cheap sanity check, not the reported value: now that
    # job_results is guaranteed fresh (see execution.py), this SHOULD match
    # cvar_value almost exactly. If it ever doesn't, that's a real signal
    # the two CVaR implementations disagree on something beyond staleness.
    _cvar_cross_check = compute_cvar(job_results, hamiltonian, alpha=0.25, offset=offset)
    if abs(_cvar_cross_check - cvar_value) > 1e-6:
        print(f"  [Warning] CVaR cross-check mismatch: optimizer reported "
              f"{cvar_value:.6f}, recomputed from final distribution "
              f"{_cvar_cross_check:.6f} — investigate before trusting this run's CVaR.")

    energy = compute_energy(job_results, hamiltonian, offset)

    metrics = {
        "approx_ratio": approx_ratio,
        "cvar":         cvar_value,
        "energy":       energy,
        "sharpe_ratio": sharpe_ratio,
        "variant":      state.get("variant_chosen"),
        "p_layers":     state.get("p_layers"),
        "backend":      state.get("backend_chosen"),
        "retry":        state.get("retry_count"),
    }

    prompt = f"""
    You are a quantum optimization researcher analyzing QAOA results.
    Interpret the following metrics for a portfolio optimization run:

    {metrics}

    In 2-3 sentences, explain:
    1. Whether the approximation ratio is acceptable
    2. What the CVaR value suggests about tail-risk performance
    3. Whether the Sharpe ratio (equal-weighted, annualized) is reasonable for this asset universe
    4. A specific recommendation for improvement if retry is needed

    Be concise and technical.
    """
    interpretation = get_response_text(llm.invoke(prompt))

    sharpe_str = f"{sharpe_ratio:.4f}" if sharpe_ratio is not None else "N/A"
    print(f"[Analysis] AR={approx_ratio:.4f} | CVaR={cvar_value:.4f} | Energy={energy:.4f} | Sharpe={sharpe_str}")
    print(f"[Analysis] Interpretation:\n{interpretation}")

    report_path = _write_report(state, metrics, interpretation)
    print(f"[Analysis] Report written to {report_path}")

    print("\nNode 4 complete.")

    return {
        "approx_ratio": approx_ratio,
        "cvar_value":   cvar_value,
        "energy":       energy,
        "sharpe_ratio": sharpe_ratio,
        "metrics":      {**metrics, "interpretation": interpretation},
        "retry_count":  state.get("retry_count", 0) + 1,
    }