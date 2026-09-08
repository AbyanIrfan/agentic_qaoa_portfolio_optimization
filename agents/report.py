"""
Node 5: Report (V2 — grounded). Synthesizes the full pipeline into a
structured technical report, with warm-start and truncation claims
grounded in facts extracted from the actual circuit/literature-validation
output rather than the canonical description of the named technique
(see module docstring history below for why that grounding matters).

Original module docstring:
Root cause of the fabrication in the previous version: the prompt gave the
LLM scalar summary facts (variant_chosen='warm_start_xy', a pair COUNT of
15) but never the actual generated circuit or the actual literature
rationale behind the truncation decision. With only a technique NAME and
no artifact to describe, the LLM's best completion is the textbook
description of what that technique should look like — which is exactly
what happened in Section 3.1 (canonical Ry-based continuous warm-start,
when the real circuit uses hard X gates) and Section 3.2 (an invented
tractability rationale, when the real reason was Node 0.5's full-text
finding that full connectivity outperforms ring topology).

Fix: extract FACTS from the real artifacts programmatically (cheap, and
immune to the LLM re-hallucinating regardless of prompt wording), and
feed those facts — not just a label — into the prompt, with an explicit
instruction to describe what's actually there rather than the canonical
version of the named technique.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from llm.router import get_node_llm, get_response_text

RESULTS_DIR = Path("results")


def _describe_warm_start_from_circuit(circuit) -> str:
    """
    Determine what warm-start technique the circuit ACTUALLY implements,
    from its real gate composition — rather than trusting the LLM to
    correctly infer this from a variant label like 'warm_start_xy' alone.
    Cheap: circuit.count_ops() is already computed elsewhere in Node 2.
    """
    ops = circuit.count_ops()
    has_ry = ops.get("ry", 0) > 0
    has_x = ops.get("x", 0) > 0

    if has_ry and not has_x:
        return (
            "Continuous-angle warm-start: initial state prepared via "
            "per-qubit Ry rotations, biasing each qubit's measurement "
            "probability proportional to the QP relaxation value "
            "(theta_i = 2*arcsin(sqrt(x_i)))."
        )
    if has_x and not has_ry:
        return (
            "DETERMINISTIC ROUNDED warm-start (not continuous-angle): "
            "initial state prepared via hard X gates on the qubits "
            "corresponding to the top-`budget` highest-value relaxation "
            "entries. Fractional relaxation values are discarded rather "
            "than encoded as rotation angles — this is a different, "
            "simpler technique than the Ry-based continuous warm-start "
            "described in Egger et al., not an implementation of it."
        )
    if has_ry and has_x:
        return "Mixed warm-start: circuit uses both Ry rotations and hard X gates."
    return (
        "No warm-start rotation gates (Ry or X) detected in initial state "
        "preparation — likely a uniform superposition start with no "
        "warm-start bias applied."
    )


def _write_report(report_text: str, backend: str = "unknown") -> Path:
    """Write a timestamped final report to results/. One file per run,
    not overwritten, matching Node 4's analysis-report convention."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"report_{backend}_{timestamp}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def report_node(state):
    llm = get_node_llm("report", state.get("llm_name"))                  # report.py

    print("[Report] Generating final report...")

    ar    = f"{state['approx_ratio']:.4f}" if state.get('approx_ratio') is not None else "N/A"
    cvar  = f"{state['cvar_value']:.4f}"   if state.get('cvar_value')   is not None else "N/A"
    enrgy = f"{state['energy']:.4f}"       if state.get('energy')       is not None else "N/A"
    sharpe = f"{state['sharpe_ratio']:.4f}" if state.get('sharpe_ratio') is not None else "N/A"

    circuit = state.get("circuit")
    warm_start_fact = (
        _describe_warm_start_from_circuit(circuit)
        if circuit is not None
        else "Circuit not available — cannot verify warm-start implementation."
    )

    truncation_rationale = state.get(
        "validated_recommendation",
        state.get("literature_recommendation", "No literature-grounded rationale recorded."),
    )
    truncation_contradicted = state.get("validation_contradicted_claims", "none recorded")

    n_pairs = len(state.get('truncated_pairs') or [])
    n_assets = state.get('n_assets')
    max_pairs = n_assets * (n_assets - 1) // 2 if n_assets else None

    prompt = f"""
    You are a quantum computing researcher writing a technical report
    on a QAOA optimization run. Synthesize the full pipeline into a
    structured report with the following sections:

    1. Problem Summary
    2. Approach & Circuit Design Choices (justify variant and p-layers)
    3. Warm-Start Analysis (QP relaxation vector, XY-mixer truncation)
    4. Execution Details (backend, shots)
    5. Results (approximation ratio, CVaR, energy, Sharpe ratio)
    6. Reasoning & Conclusion

    CRITICAL GROUNDING INSTRUCTION: Base every claim about the warm-start
    technique and mixer truncation ONLY on the verified facts provided
    below — do NOT describe the canonical/textbook version of the named
    variant. If the verified fact indicates a simpler or different
    implementation than the technique's usual description, report that
    difference explicitly rather than the idealized version. Do not
    invent a rationale for the truncation pair count that isn't given
    below — quote or paraphrase the actual literature rationale provided.

    Pipeline state:
    - Problem:             {state['problem_input']}
    - Variant chosen:      {state.get('variant_chosen')}
    - p-layers:            {state.get('p_layers')}

    - VERIFIED warm-start implementation (extracted from the actual
      circuit's gate composition, not the variant label):
      {warm_start_fact}

    - Truncated XY pairs:  {n_pairs} of {max_pairs}
    - ACTUAL literature rationale for this pair count (from full-text
      cross-check against the arXiv corpus):
      {truncation_rationale}
    - Claims the cross-check specifically contradicted (report these as
      corrections applied, if any, not as open questions):
      {truncation_contradicted}

    - Backend:             {state.get('backend_chosen')}
    - Optimizer:           {state.get('optimizer', 'N/A')}
    - Optimizer init:      {state.get('optimizer_init', 'N/A')}
    - Shots:               {state.get('n_shots', 'N/A')}
    - COBYLA iterations:   {state.get('n_iterations', 'N/A')}
    - CVaR alpha:          {state.get('alpha', 'N/A')}
    - Classical optimal:   {state.get('classical_optimal', 'N/A')}
    - Best bitstring:      {state.get('best_bitstring', 'N/A')}
    - Best assets:         {state.get('best_assets', 'N/A')}
    - Approximation ratio: {ar}
    - CVaR:                {cvar}
    - Energy:              {enrgy}
    - Sharpe ratio (equal-weighted, annualized): {sharpe}
    - Static doc-store validation: {state.get('static_validation_passed', 'N/A')}
    - Hamming-weight preservation: {state.get('hw_preservation_fraction', 'N/A')}
    - Analysis node passes:  {state.get('retry_count', 0)}
    - Analysis note:       {state.get('metrics', {}).get('interpretation', 'N/A')}

    Write in a technical but clear style suitable for a research paper appendix.
    """

    report = get_response_text(llm.invoke(prompt))

    print("[Report] Done.\n")
    print("=" * 60)
    print(report)
    print("=" * 60)

    report_path = _write_report(report, backend=state.get("llm_name", "unknown"))
    print(f"[Report] Report written to {report_path}")

    print("\n✓ Full pipeline complete — all 5 nodes executed successfully.")

    return {"report": report}