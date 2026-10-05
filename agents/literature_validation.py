"""
Node 0.5: Literature Validation — cross-checks Node 0's Semantic-Scholar-
derived (abstract-only) recommendation against full-text excerpts from the
static arXiv corpus, revising numeric parameters if the full text disagrees.
"""

from __future__ import annotations

import json

from llm.router import get_node_llm, get_response_text

from rag.corpus_query import _embed_query
from rag.store_utils import get_arxiv_runtime_cached
from graph.execution_context import describe_execution_context


def literature_validation_node(state):
    arxiv_store, embed_client, embedding_model = get_arxiv_runtime_cached()

    if arxiv_store is None or len(arxiv_store) == 0:
        print("[Literature Validation] Skipped — arXiv corpus not available.")
        return {
            "literature_validation_skipped": True,
            "validated_recommendation": state.get("literature_recommendation"),
        }

    llm = get_node_llm("literature_validation", state.get("llm_name"))   # literature_validation.py

    print("=" * 60)
    print("[Literature Validation] Cross-checking against arXiv full text...\n")

    query_text = (
        f"{state['problem_input']} "
        f"{state.get('literature_recommendation', '')}"
    )[:2000]

    print("[Literature Validation] Embedding query and searching arXiv corpus...")
    query_vec = _embed_query(query_text, embed_client, embedding_model)
    hits = arxiv_store.search(query_vec, top_k=5, journal_boost=1.15)

    if not hits:
        print("[Literature Validation] No relevant excerpts found — passing "
              "Node 0's recommendation through unchanged.")
        return {
            "literature_validation_skipped": True,
            "validated_recommendation": state.get("literature_recommendation"),
            "validated_p_layers": state.get("p_layers_recommended"),
            "validated_truncation_fraction": state.get("truncation_fraction"),
            "validated_penalty": state.get("penalty"),
        }

    print(f"[Literature Validation] Retrieved {len(hits)} excerpts:")
    for record, score in hits:
        print(f"  - {record.title[:70]} (chunk {record.chunk_index}/{record.n_chunks}, "
              f"score={score:.3f})")

    excerpts_formatted = "\n\n---\n\n".join(
        f"Source: {r.title} ({r.published}, arXiv:{r.arxiv_id})\n"
        f"Excerpt (chunk {r.chunk_index+1}/{r.n_chunks}):\n{r.text[:1500]}"
        for r, _ in hits
    )

    validation_prompt = f"""
    You are a quantum computing researcher fact-checking a methodology
    recommendation against primary-source full text.

    {describe_execution_context()}

    Problem: {state['problem_input']}

    The following recommendation was synthesized from Semantic Scholar
    ABSTRACTS only (not full text):
    ---
    {state.get('literature_recommendation', 'N/A')}
    ---

    Here are full-text excerpts from the arXiv corpus, retrieved by semantic
    similarity to the problem and recommendation above:
    ---
    {excerpts_formatted}
    ---

    Cross-check the recommendation against these excerpts:
    1. Does the full text SUPPORT, CONTRADICT, or provide NO INFORMATION on
       each specific claim in the recommendation (QAOA variant choice,
       p-layers, warm-start benefit, XY-mixer truncation criterion)?
    2. If the full text suggests different values than the recommendation
       (e.g. a different p-layers range, a different truncation criterion),
       state the discrepancy explicitly and recommend which to trust and why.
    3. Flag any claim in the original recommendation that these excerpts do
       NOT support — do not assume it's still correct just because it
       wasn't contradicted.

    Return your response in exactly this format:
    SUPPORTED: <comma-separated list of claims the excerpts support>
    CONTRADICTED: <comma-separated list of claims the excerpts contradict, or "none">
    UNVERIFIED: <comma-separated list of claims the excerpts say nothing about, or "none">
    REVISED_RECOMMENDATION: <the recommendation, corrected if needed, under 150 words>
    """

    print("\n[Literature Validation] Cross-checking with LLM...")
    raw = get_response_text(llm.invoke(validation_prompt)).strip()

    sections = {"SUPPORTED": "", "CONTRADICTED": "", "UNVERIFIED": "", "REVISED_RECOMMENDATION": ""}
    current_key = None
    for line in raw.splitlines():
        matched = False
        for key in sections:
            if line.startswith(f"{key}:"):
                sections[key] = line.replace(f"{key}:", "").strip()
                current_key = key
                matched = True
                break
        if not matched and current_key:
            sections[current_key] += " " + line.strip()

    print("\n" + "=" * 60)
    print("CROSS-CHECK RESULT:")
    print("=" * 60)
    print(f"Supported:    {sections['SUPPORTED']}")
    print(f"Contradicted: {sections['CONTRADICTED']}")
    print(f"Unverified:   {sections['UNVERIFIED']}")
    print(f"\nRevised recommendation:\n{sections['REVISED_RECOMMENDATION']}")

    contradicted_flag = sections["CONTRADICTED"].strip().lower() not in ("none", "", "n/a")
    orig_p_layers = state.get("p_layers_recommended", 1)
    orig_truncation = state.get("truncation_fraction", 0.5)
    orig_penalty = state.get("penalty", 12)
    validated_p_layers, validated_truncation, validated_penalty = (
        orig_p_layers, orig_truncation, orig_penalty
    )

    if contradicted_flag:
        print(f"\nAt least one claim was contradicted by full-text evidence — "
              f"review before proceeding to Node 1.")

        param_prompt = f"""
        Based on this full-text cross-check of a QAOA methodology recommendation:
        
        {describe_execution_context()}

        SUPPORTED: {sections['SUPPORTED']}
        CONTRADICTED: {sections['CONTRADICTED']}
        UNVERIFIED: {sections['UNVERIFIED']}
        REVISED_RECOMMENDATION: {sections['REVISED_RECOMMENDATION']}

        Node 0's original numeric parameters (derived from abstracts only) were:
          p_layers:            {orig_p_layers}
          truncation_fraction: {orig_truncation}
          penalty:             {orig_penalty}

        Only change a parameter if the CONTRADICTED or UNVERIFIED sections above
        specifically justify a different value. If a parameter is not addressed
        by the cross-check, or is listed under SUPPORTED, keep Node 0's original
        value unchanged. Do not invent a new value from general knowledge — only
        from what the cross-check above actually established.

        Note specifically: if the cross-check found the full XY-mixer / full
        connectivity is supported over a truncated/ring topology, truncation_fraction
        should reflect that (closer to 1.0 = fewer pairs discarded).

        Return ONLY valid JSON with these three parameters, no explanation:
        {{
            "p_layers": <integer 1 to 5>,
            "truncation_fraction": <float 0.3 to 1.0>,
            "penalty": <integer 8 to 20>
        }}
        """

        print("\n[Literature Validation] Re-checking numeric parameters against the cross-check...")
        raw_params = get_response_text(llm.invoke(param_prompt)).strip()
        if raw_params.startswith("```"):
            raw_params = raw_params.split("```")[1]
            if raw_params.startswith("json"):
                raw_params = raw_params[4:]
        raw_params = raw_params.strip()

        try:
            params = json.loads(raw_params)
            validated_p_layers = max(1, min(5, int(params.get("p_layers", orig_p_layers))))
            validated_truncation = max(0.3, min(1.0, float(params.get("truncation_fraction", orig_truncation))))
            validated_penalty = max(8, min(20, int(params.get("penalty", orig_penalty))))
        except Exception as e:
            print(f"  [Warning] Numeric re-extraction failed to parse ({e}) — "
                  f"keeping Node 0's original values unchanged.")

        def _flag(orig, new):
            return "unchanged" if orig == new else f"REVISED from {orig}"

        print(f"  p_layers:            {validated_p_layers}  ({_flag(orig_p_layers, validated_p_layers)})")
        print(f"  truncation_fraction: {validated_truncation}  ({_flag(orig_truncation, validated_truncation)})")
        print(f"  penalty:             {validated_penalty}  ({_flag(orig_penalty, validated_penalty)})")

    print(f"\nNode 0.5 complete.")

    return {
        "literature_validation_skipped": False,
        "arxiv_sources_used": [
            {"arxiv_id": r.arxiv_id, "title": r.title, "chunk_index": r.chunk_index, "score": score}
            for r, score in hits
        ],
        "validation_supported_claims": sections["SUPPORTED"],
        "validation_contradicted_claims": sections["CONTRADICTED"],
        "validation_unverified_claims": sections["UNVERIFIED"],
        "validated_recommendation": sections["REVISED_RECOMMENDATION"] or state.get("literature_recommendation"),
        "validation_found_contradiction": contradicted_flag,
        "validated_p_layers": validated_p_layers,
        "validated_truncation_fraction": validated_truncation,
        "validated_penalty": validated_penalty,
    }