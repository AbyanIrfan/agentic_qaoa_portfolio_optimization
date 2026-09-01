"""
Node 4: Analysis — computes approximation ratio, CVaR, and energy from
Node 3's job results against the Ising Hamiltonian, gets an LLM
interpretation, and writes a timestamped report to results/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_anthropic import ChatAnthropic

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


def compute_approx_ratio(quasi_dist, hamiltonian, offset=0.0):
    n          = hamiltonian.num_qubits
    energies   = [bitstring_energy_ising(f"{i:0{n}b}", hamiltonian, offset) for i in range(2**n)]
    optimal    = min(energies)
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


def _write_report(state, metrics, interpretation) -> Path:
    """Write a timestamped analysis report to results/. One file per run,
    not overwritten, so past runs stay comparable."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"analysis_{timestamp}.txt"

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
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    print("[Analysis] Computing metrics...")

    job_results  = state["job_results"]
    hamiltonian  = state["hamiltonian"]
    offset = state["offset"]

    approx_ratio = compute_approx_ratio(job_results, hamiltonian, offset)
    cvar_value   = compute_cvar(job_results, hamiltonian, alpha=0.25, offset=offset)
    energy       = compute_energy(job_results, hamiltonian, offset)

    metrics = {
        "approx_ratio": approx_ratio,
        "cvar":         cvar_value,
        "energy":       energy,
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
    3. A specific recommendation for improvement if retry is needed

    Be concise and technical.
    """
    interpretation = llm.invoke(prompt).content

    print(f"[Analysis] AR={approx_ratio:.4f} | CVaR={cvar_value:.4f} | Energy={energy:.4f}")
    print(f"[Analysis] Interpretation:\n{interpretation}")

    report_path = _write_report(state, metrics, interpretation)
    print(f"[Analysis] Report written to {report_path}")

    print("\n✓ Node 4 complete.")

    return {
        "approx_ratio": approx_ratio,
        "cvar_value":   cvar_value,
        "energy":       energy,
        "metrics":      {**metrics, "interpretation": interpretation},
        "retry_count":  state.get("retry_count", 0) + 1,
    }