"""
Node 2: Circuit Design (V4 — LLM-generated circuit synthesis with static
Qiskit doc-store validation before execution, and arXiv-corpus-grounded
retry guidance on failure).
"""

from __future__ import annotations

import traceback

import numpy as np
from llm.router import get_node_llm, get_response_text
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit_aer import AerSimulator

from rag.call_validator import validate_generated_code
from rag.corpus_query import query_corpus_for_implementation_detail
from rag.store_utils import get_arxiv_runtime_cached, get_qiskit_doc_store

from graph.execution_context import describe_execution_context


def circuit_design_node(state):
    llm = get_node_llm("circuit_design", state.get("llm_name"))          # circuit_design.py

    qiskit_doc_store = get_qiskit_doc_store()
    arxiv_store, embed_client, embedding_model = get_arxiv_runtime_cached()

    MAX_RETRIES = 3
    last_error  = None
    generated_code = None

    p_layers = state.get("validated_p_layers", state.get("p_layers_recommended", 1))
    n_assets = state["n_assets"]
    budget   = state["budget"]

    print("=" * 60)
    print(f"[Circuit Design v3] LLM-generated circuit synthesis")
    print(f"  p={p_layers} | n_qubits={n_assets} | budget={budget}")
    print(f"  XY pairs from Node 1: {state['truncated_pairs']}\n")

    # ── Step 1: Variant selection ───────────────────────────────────────────
    variant_prompt = f"""
    You are a quantum circuit design expert specializing in QAOA.
    Given the following context, select the most appropriate QAOA variant.

    {describe_execution_context()}

    Problem: {state['problem_input']}
    Warm-start vector available: True
    Truncated XY pairs from Node 1: {state['truncated_pairs']}
    p-layers recommended by literature: {p_layers}
    Previous approx ratio: {state.get('approx_ratio', 'N/A')}

    Literature recommendation:
    {state.get('validated_recommendation', 'Use warm_start_xy variant.')}

    Options:
    - "standard":      Standard QAOA with X-mixer
    - "warm_start_xy": Warm-start XY-mixer with truncated pairs (thesis method)
    - "cvar":          CVaR-QAOA for tail-risk optimization

    Respond in exactly this format:
    VARIANT: <one of the three option strings>
    REASONING: <2-3 sentences citing specific papers from the literature>
    """

    raw_variant = get_response_text(llm.invoke(variant_prompt)).strip()
    variant     = "warm_start_xy"
    reasoning   = ""

    for line in raw_variant.splitlines():
        if line.startswith("VARIANT:"):
            variant   = line.replace("VARIANT:", "").strip().strip('"').lower()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    if variant not in {"standard", "warm_start_xy", "cvar"}:
        variant = "warm_start_xy"

    print(f"[Circuit Design v3] Variant selected: {variant}")
    print(f"[Circuit Design v3] Reasoning: {reasoning}\n")

    # ── Step 2: LLM-generated circuit code ──────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Circuit Design v3] Code generation attempt {attempt}/{MAX_RETRIES}")

        if attempt == 1:
            error_context = ""
        else:
            retrieved_guidance = ""
            if arxiv_store is not None:
                print(f"  Attempt {attempt} follows a failure — searching arXiv "
                      f"corpus for relevant guidance...")
                retrieval_question = (
                    f"How should a QAOA '{variant}' circuit be constructed to "
                    f"avoid this specific implementation error: {last_error}"
                )[:2000]
                corpus_result = query_corpus_for_implementation_detail(
                    question=retrieval_question, store=arxiv_store,
                    embed_client=embed_client, llm=llm, top_k=4,
                )
                if corpus_result["found"]:
                    retrieved_guidance = f"""
Relevant technical guidance retrieved from the literature corpus, based on
this specific failure:
{corpus_result['answer']}
"""
                    print(f"  Found relevant guidance from: "
                          f"{', '.join(s['title'][:50] for s in corpus_result['sources'][:2])}")
                else:
                    print(f"  No directly relevant guidance found in the corpus for this failure.")

            error_context = f"""
Your previous attempt failed with this error:
{last_error}

Your previous code was:
{generated_code}
{retrieved_guidance}
Fix the issue and regenerate the complete corrected code.
"""

        code_prompt = f"""
You are a quantum computing expert specializing in QAOA circuit design.

Problem: {state['problem_input']}
Variant to implement: {variant}
QAOA depth: p = {p_layers}

Literature recommendation:
{state.get('validated_recommendation', 'Use warm_start_xy variant.')}

The following variables are already in your execution scope.
Use them exactly as named — do NOT redefine them:
  hamiltonian     : SparsePauliOp — the Ising cost Hamiltonian ({len(state['hamiltonian'])} Pauli terms)
  warm_start_vec  : numpy array shape ({n_assets},) = {state['warm_start_vec'].tolist()}
  truncated_pairs : list of (i,j) tuples = {state['truncated_pairs']}
  n_assets        : int = {n_assets}
  budget          : int = {budget}
  p_layers        : int = {p_layers}

The following are already imported — do NOT add import statements:
  QuantumCircuit, ParameterVector, XXPlusYYGate (from qiskit)
  numpy as np

Your code must produce exactly one variable:
  circuit : QuantumCircuit

Hard requirements:
  - Exactly {n_assets} qubits
  - Exactly {2 * p_layers} free parameters
    (use ParameterVector for gamma and beta, {p_layers} each)
  - Must end with circuit.measure_all()
  - Must implement the {variant} variant
  - Must use ONLY the pairs in truncated_pairs for the XY mixer
  - The XY mixer MUST preserve the budget constraint


{error_context}

Return ONLY executable Python code.
No import statements. No print statements. No markdown. No explanation.
Assign result to exactly: circuit
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

        # ── Static validation against the Qiskit doc store, BEFORE exec ────
        if qiskit_doc_store is not None:
            print(f"  Checking generated code against Qiskit doc store...")
            static_problems = validate_generated_code(
                generated_code, qiskit_doc_store,
                known_var_types={"hamiltonian": type(state["hamiltonian"])},
            )
            if static_problems:
                last_error = (
                    "Static validation against the Qiskit API reference found "
                    "the following problems BEFORE execution:\n"
                    + "\n".join(f"  - {p}" for p in static_problems)
                )
                print(f"  ✗ Attempt {attempt} failed static validation:")
                for p in static_problems:
                    print(f"      - {p}")
                if attempt < MAX_RETRIES:
                    print(f"  Feeding error back to LLM...\n")
                continue
            print(f"  Static validation passed ({len(static_problems)} issues found)")
        else:
            print(f"  (Skipping static doc-store validation — store not loaded)")

        print(f"  Executing...")

        namespace = {
            "np":              np,
            "QuantumCircuit":  QuantumCircuit,
            "ParameterVector": ParameterVector,
            "XXPlusYYGate":    XXPlusYYGate,
            "hamiltonian":     state["hamiltonian"],
            "warm_start_vec":  state["warm_start_vec"],
            "truncated_pairs": state["truncated_pairs"],
            "n_assets":        n_assets,
            "budget":          budget,
            "p_layers":        p_layers,
        }

        try:
            exec(generated_code, namespace)
            circuit = namespace.get("circuit")

            assert circuit is not None, "circuit variable not produced"
            assert isinstance(circuit, QuantumCircuit), \
                f"circuit is not a QuantumCircuit, got {type(circuit)}"

            assert circuit.num_qubits == n_assets, \
                (f"circuit has {circuit.num_qubits} qubits, "
                 f"expected {n_assets}")

            assert circuit.num_parameters == 2 * p_layers, \
                (f"circuit has {circuit.num_parameters} parameters, "
                 f"expected {2 * p_layers} "
                 f"({p_layers} gamma + {p_layers} beta)")

            ops = circuit.count_ops()
            assert "measure" in ops, \
                "circuit has no measurements — missing measure_all()"

            test_params = {p: np.random.uniform(0, np.pi)
                          for p in circuit.parameters}
            bound  = circuit.assign_parameters(test_params)
            bound  = bound.decompose(reps=3)
            sim    = AerSimulator()
            job    = sim.run(bound, shots=200)
            counts = job.result().get_counts()

            hw_values    = [bs.count("1") for bs in counts.keys()]
            hw_correct   = [hw == budget for hw in hw_values]
            hw_fraction  = sum(hw_correct) / len(hw_correct)

            assert hw_fraction > 0.8, \
                (f"Only {hw_fraction:.0%} of output bitstrings satisfy "
                 f"the budget constraint of {budget} selected assets. "
                 f"The XY mixer is not correctly preserving feasibility. "
                 f"Hamming weights seen: {sorted(set(hw_values))}.")

            print(f"\n  ✓ All validations passed:")
            print(f"    Static doc-store check: "
                  f"{'passed' if qiskit_doc_store is not None else 'skipped (store not loaded)'}")
            print(f"    Qubits:          {circuit.num_qubits}"
                  f"  (expected {n_assets})")
            print(f"    Parameters:      {circuit.num_parameters}"
                  f"  (expected {2 * p_layers})")
            print(f"    Circuit depth:   {circuit.depth()}")
            print(f"    Total gates:     {circuit.size()}")
            print(f"    HW={budget} preserved: {hw_fraction:.0%} of 200 shots")
            print(f"\n  Circuit diagram:")
            print(circuit.draw("text"))

            print(f"\nNode 2 (v3 + static validation) complete.")

            return {
                "circuit":                  circuit,
                "variant_chosen":           variant,
                "variant_reasoning":        reasoning,
                "p_layers":                 p_layers,
                "generated_circuit_code":   generated_code,
                "hw_preservation_fraction": hw_fraction,
                "static_validation_passed": qiskit_doc_store is not None,
            }

        except Exception as e:
            last_error = traceback.format_exc()
            print(f"  ✗ Attempt {attempt} failed: {str(e)[:200]}")
            if attempt < MAX_RETRIES:
                print(f"  Feeding error back to LLM...\n")

    raise RuntimeError(
        f"[Circuit Design v3] Failed after {MAX_RETRIES} attempts.\n"
        f"Last error:\n{last_error}"
    )