"""
Node 0: Literature Search — Semantic Scholar abstract-level discovery,
synthesized into a methodology recommendation via LLM.
"""

from __future__ import annotations

import json
import os
import time

import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


def s2_search_papers(query, limit=5, sleep_time=1.2, max_retries=4):
    """
    Search Semantic Scholar for papers matching query.
    Handles 429 rate limits with exponential backoff.
    No API key needed for prototyping (100 req/5min free tier).
    """
    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  limit,
        "fields": "title,year,authors,abstract,venue,citationCount"
    }
    headers = {"x-api-key": S2_API_KEY}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                data   = response.json()
                papers = data.get("data", [])
                time.sleep(sleep_time)
                return papers

            elif response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                print(f"  [S2] Rate limited — waiting {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)

            else:
                print(f"  [S2] HTTP {response.status_code} for query: '{query}'")
                return []

        except Exception as e:
            print(f"  [S2] Error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    print(f"  [S2] All retries exhausted for: '{query}'")
    return []


def literature_search_node(state):
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    print("=" * 60)
    print("[Literature Search] Starting...\n")

    if "n_assets" not in state:
        state["n_assets"] = len(state["mu"])

    query_prompt = f"""
    You are a quantum computing researcher.

    Given this optimization problem:
    {state['problem_input']}

    Generate exactly 3 short, specific Semantic Scholar search queries
    to find relevant QAOA and quantum optimization papers.

    Focus on:
    - The specific QAOA variant most likely to work (warm-start, XY-mixer, CVaR)
    - The constraint type (cardinality, portfolio, budget)
    - The warm-start or relaxation method

    Return ONLY valid JSON in this exact format, no explanation:
    {{
        "search_queries": [
            "query one here",
            "query two here",
            "query three here"
        ]
    }}
    """

    print("[Literature Search] Generating search queries...")
    raw = llm.invoke(query_prompt).content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        query_data = json.loads(raw)
        queries    = query_data["search_queries"]
    except Exception as e:
        print(f"  [Warning] JSON parse failed ({e}), using fallback queries")
        queries = [
            "QAOA warm start portfolio optimization",
            "XY mixer QAOA cardinality constraint",
            "CVaR quantum approximate optimization"
        ]

    print(f"\n[Literature Search] Queries generated:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print("\n[Literature Search] Searching Semantic Scholar...")
    all_papers = []

    for i, query in enumerate(queries):
        print(f"  Searching: '{query}'")
        papers = s2_search_papers(query, limit=4)
        all_papers.extend(papers)
        print(f"  → Found {len(papers)} papers")

    seen   = set()
    unique = []
    for p in all_papers:
        title = p.get("title", "")
        if title and title not in seen:
            seen.add(title)
            unique.append(p)

    print(f"\n[Literature Search] Total unique papers: {len(unique)}")
    unique.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
    print(f"[Literature Search] Sorted by citation count "
          f"(top: {unique[0].get('citationCount', 0)} citations)" if unique else "")

    formatted = []
    for p in unique:
        if p.get("abstract"):
            entry = (
                f"Title: {p.get('title', 'N/A')} "
                f"({p.get('year', 'N/A')}, {p.get('citationCount', 0)} citations)\n"
                f"Venue: {p.get('venue', 'N/A')}\n"
                f"Abstract: {p['abstract'][:500]}..."
            )
            formatted.append(entry)

    if not formatted:
        print("  [Warning] No abstracts found — using minimal context")
        literature_context = "No relevant papers found. Use standard QAOA best practices."
    else:
        literature_context = "\n\n---\n\n".join(formatted)

    synthesis_prompt = f"""
    You are a quantum computing researcher reviewing literature before
    designing a QAOA circuit for the following problem:

    Problem: {state['problem_input']}
    Warm-start vector available: True
    Budget constraint: {state['budget']} assets from {state['n_assets']}

    Relevant papers found:
    {literature_context}

    Based on this literature, provide a concise methodology recommendation:

    1. Which QAOA variant is best supported? (standard / warm_start_xy / cvar)
    2. What p-layers are recommended for this problem size?
    3. Is a warm-start beneficial according to the literature?
    4. Should XY-mixer pairs be truncated? If so, by what criterion?
    5. Which 2-3 papers are most directly relevant? (cite by title)

    Be specific and cite papers where possible. Keep it under 200 words.
    """

    print("\n[Literature Search] Synthesizing recommendations...")
    recommendation = llm.invoke(synthesis_prompt).content

    print("\n" + "=" * 60)
    print("LITERATURE RECOMMENDATION:")
    print("=" * 60)
    print(recommendation)

    print("\n" + "=" * 60)
    print(f"PAPERS FOUND ({len(unique)} total):")
    print("=" * 60)
    for i, p in enumerate(unique[:8], 1):
        print(f"{i}. {p.get('title', 'N/A')} "
              f"({p.get('year', 'N/A')}) — "
              f"{p.get('citationCount', 0)} citations")

    print("\n[Literature Search] Extracting concrete parameters...")

    param_prompt = f"""
    Based on this literature recommendation:
    {recommendation}

    And this problem:
    - Problem: {state['problem_input']}
    - n_assets: {state['n_assets']}
    - budget: {state['budget']}

    Return ONLY valid JSON with these three parameters.
    No explanation, no markdown, just the JSON object:
    {{
        "p_layers": <integer 1 to 5, recommended QAOA circuit depth>,
        "truncation_fraction": <float 0.3 to 1.0, fraction of XY mixer pairs to retain>,
        "penalty": <integer 8 to 20, penalty weight for the cardinality constraint>
    }}

    Base each value on specific evidence from the literature.
    """

    raw_params = llm.invoke(param_prompt).content.strip()

    if raw_params.startswith("```"):
        raw_params = raw_params.split("```")[1]
        if raw_params.startswith("json"):
            raw_params = raw_params[4:]
    raw_params = raw_params.strip()

    try:
        params = json.loads(raw_params)

        p_rec    = max(1, min(5,   int(params.get("p_layers",          1))))
        t_frac   = max(0.3, min(1.0, float(params.get("truncation_fraction", 0.5))))
        penalty  = max(8,   min(20,  int(params.get("penalty",           12))))

        print(f"  p_layers recommended:   {p_rec}")
        print(f"  truncation_fraction:    {t_frac}")
        print(f"  penalty weight:         {penalty}")

    except Exception as e:
        print(f"  [Warning] Parameter parsing failed ({e}) — using defaults")
        p_rec   = 1
        t_frac  = 0.5
        penalty = 12

    print(f"\n✓ Node 0 complete — {len(unique)} papers retrieved.")

    return {
        "literature_recommendation": recommendation,
        "cited_papers":              [p.get("title") for p in unique],
        "raw_papers":                unique,
        "p_layers_recommended":      p_rec,
        "truncation_fraction":       t_frac,
        "penalty":                   penalty,
        "n_assets":                  state["n_assets"],
    }