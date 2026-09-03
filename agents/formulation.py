"""
Node 1: Problem Formulation (v3) — LLM-generated Ising Hamiltonian, QP-
relaxation warm-start vector, and truncated XY-mixer pairs, executed in a
controlled namespace with retry-on-failure and output validation.
"""

from __future__ import annotations

import traceback

import cvxpy as cp
import numpy as np
from llm.router import get_node_llm, get_response_text
from qiskit.quantum_info import SparsePauliOp
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.converters import LinearEqualityToPenalty


def formulation_node(state):
    llm = get_node_llm("formulation", state.get("llm_name"))             # formulation.py

    MAX_RETRIES    = 3
    last_error     = None
    generated_code = None

    print("=" * 60)
    print("[Formulation v3] LLM-generated problem formulation\n")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Formulation v3] Attempt {attempt}/{MAX_RETRIES}")

        error_context = "" if attempt == 1 else f"""
Your previous attempt failed with this error:
{last_error}

Your previous code was:
{generated_code}

Fix the issue and regenerate the complete corrected code.
"""

        code_prompt = f"""
You are a quantum computing expert specializing in QAOA for
combinatorial optimization.

Problem description:
{state['problem_input']}

Literature recommendation (use this to inform your approach):
{state.get('validated_recommendation', 'Use warm_start_xy variant.')}

The following variables are already defined in your execution scope.
Use them exactly as named — do NOT redefine them:
  mu                  : numpy array, shape ({state['n_assets']},)
  sigma               : numpy array, shape ({state['n_assets']},{state['n_assets']})
  n_assets            : int = {state['n_assets']}
  budget              : int = {state['budget']}
  risk_factor         : float = {state['risk_factor']}
  penalty             : int = {state['penalty']}
  truncation_fraction : float = {state.get('validated_truncation_fraction', state.get('truncation_fraction', 0.5))}

The following are already imported — do NOT add import statements:
  numpy as np
  cvxpy as cp
  SparsePauliOp
  PortfolioOptimization
  LinearEqualityToPenalty

Your code must produce exactly these four variables:

  hamiltonian      : SparsePauliOp  — Ising Hamiltonian for this problem
  offset           : float          — constant energy offset
  warm_start_vec   : numpy array, shape (n_assets,), values in [0, 1]
  truncated_pairs  : list of (i, j) tuples — XY-mixer pair subset

Requirements:
  - warm_start_vec must sum to approximately {state['budget']}
  - truncated_pairs must contain between 1 and {state['n_assets']*(state['n_assets']-1)//2} pairs
  - All pair indices must be in range [0, {state['n_assets']-1}]

{error_context}

Return ONLY executable Python code.
No import statements. No print statements. No markdown. No explanation.
Assign results to exactly: hamiltonian, offset, warm_start_vec, truncated_pairs
"""

        raw = get_response_text(llm.invoke(code_prompt)).strip()

        if "```" in raw:
            blocks = raw.split("```")
            for block in blocks[1::2]:
                if block.startswith("python"):
                    block = block[6:]
                generated_code = block.strip()
                break
        else:
            generated_code = raw

        print(f"  Generated {len(generated_code.splitlines())} lines of code")
        print(f"  Executing...")

        namespace = {
            "np":                    np,
            "cp":                    cp,
            "SparsePauliOp":         SparsePauliOp,
            "PortfolioOptimization": PortfolioOptimization,
            "LinearEqualityToPenalty": LinearEqualityToPenalty,
            "mu":                    state["mu"],
            "sigma":                 state["sigma"],
            "n_assets":              state["n_assets"],
            "budget":                state["budget"],
            "risk_factor":           state["risk_factor"],
            "penalty":               state["penalty"],
            "truncation_fraction":   state.get("validated_truncation_fraction", state.get("truncation_fraction", 0.5)),
        }

        try:
            exec(generated_code, namespace)

            hamiltonian     = namespace.get("hamiltonian")
            offset          = namespace.get("offset")
            warm_start_vec  = namespace.get("warm_start_vec")
            truncated_pairs = namespace.get("truncated_pairs")

            assert hamiltonian is not None, \
                "hamiltonian not produced"
            assert hasattr(hamiltonian, "num_qubits"), \
                "hamiltonian is not a SparsePauliOp"
            assert hamiltonian.num_qubits == state["n_assets"], \
                f"hamiltonian has {hamiltonian.num_qubits} qubits, expected {state['n_assets']}"

            assert offset is not None, \
                "offset not produced"
            offset = float(np.real(offset))

            assert warm_start_vec is not None, \
                "warm_start_vec not produced"
            warm_start_vec = np.array(warm_start_vec)
            assert warm_start_vec.shape == (state["n_assets"],), \
                f"warm_start_vec wrong shape: {warm_start_vec.shape}"
            assert np.all(warm_start_vec >= -1e-4) and np.all(warm_start_vec <= 1 + 1e-4), \
                f"warm_start_vec values out of [0,1]: {warm_start_vec}"
            warm_start_vec = np.clip(warm_start_vec, 0, 1)

            ws_sum = float(np.sum(warm_start_vec))
            assert abs(ws_sum - state["budget"]) < 0.1, \
                f"warm_start_vec sums to {ws_sum:.4f}, expected ≈ {state['budget']}"

            # Reject a near-uniform vector (e.g. every entry == budget/n_assets)
            # as a proxy for "the generated code never actually solved the QP
            # relaxation against mu/sigma" — a real per-asset optimization
            # against distinct return/risk inputs essentially never lands on
            # exact uniformity. Skipped when budget == n_assets, where the only
            # feasible relaxation solution IS the uniform all-selected vector.
            if state["n_assets"] != state["budget"]:
                ws_std = float(np.std(warm_start_vec))
                assert ws_std > 1e-3, \
                    (f"warm_start_vec looks degenerate/uniform (std={ws_std:.6f}, "
                     f"values={np.round(warm_start_vec, 4)}) — this usually means "
                     f"the generated code didn't actually solve the QP relaxation "
                     f"against mu/sigma, just filled in a placeholder satisfying "
                     f"the sum constraint.")

            assert truncated_pairs is not None, \
                "truncated_pairs not produced"
            assert isinstance(truncated_pairs, list), \
                "truncated_pairs must be a list"
            assert len(truncated_pairs) > 0, \
                "truncated_pairs is empty"
            for pair in truncated_pairs:
                assert len(pair) == 2, f"each pair must be (i,j), got: {pair}"
                i, j = pair
                assert 0 <= i < state["n_assets"] and 0 <= j < state["n_assets"], \
                    f"pair indices out of range: ({i},{j})"

            print(f"\n  ✓ All outputs validated:")
            print(f"    Hamiltonian Pauli terms: {len(hamiltonian)}"
                  f"  (reference: 21)")
            print(f"    Offset:                  {offset:.6f}"
                  f"  (reference: 15.610925)")
            print(f"    Warm-start vector:       {np.round(warm_start_vec, 4)}")
            print(f"    Warm-start sum:          {ws_sum:.4f}"
                  f"  (should be ≈ {state['budget']})")
            print(f"    XY pairs retained:       {len(truncated_pairs)}"
                  f" of {state['n_assets']*(state['n_assets']-1)//2}")
            print(f"    Truncated pairs:         {truncated_pairs}")

            print(f"\n  QP relaxation solution (x*):")
            for i, val in enumerate(warm_start_vec):
                print(f"    asset_{i}: {val:.4f}")

            print(f"\n✓ Node 1 (v3) complete.")

            return {
                "hamiltonian":               hamiltonian,
                "offset":                    offset,
                "warm_start_vec":            warm_start_vec,
                "truncated_pairs":           truncated_pairs,
                "generated_formulation_code": generated_code,
            }

        except Exception as e:
            last_error = traceback.format_exc()
            print(f"  ✗ Attempt {attempt} failed: {str(e)}")
            if attempt < MAX_RETRIES:
                print(f"  Feeding error back to LLM...\n")

    raise RuntimeError(
        f"[Formulation v3] Failed after {MAX_RETRIES} attempts.\n"
        f"Last error:\n{last_error}"
    )