"""
Benchmark sweep — runs the full pipeline across multiple LLM backends
and problem instances. Core experiment for the multi-LLM comparison paper.
"""

import json
import time
from pathlib import Path
from tqdm import tqdm

from graph.builder   import build_graph
from config.settings import RESULTS_DIR

PROBLEM_INSTANCES = [
    "Portfolio optimization: 6 assets, budget=3, risk_aversion=0.5 (Woerner benchmark)",
    "Portfolio optimization: 10 assets, budget=4, risk_aversion=0.3",
    "Portfolio optimization: 12 assets, budget=5, risk_aversion=0.7",
]

LLM_MODES = [
    "claude",    # Homogeneous: Claude for all nodes
    "gpt",       # Homogeneous: GPT for all nodes
    "gemini",    # Homogeneous: Gemini for all nodes
    "hetero",    # Heterogeneous: per-node routing (our proposed method)
]


def run_sweep(
    problems: list[str] = PROBLEM_INSTANCES,
    llm_modes: list[str] = LLM_MODES,
    output_file: str = "sweep_results.json",
):
    app     = build_graph()
    results = []

    total = len(problems) * len(llm_modes)
    pbar  = tqdm(total=total, desc="Benchmark sweep")

    for problem in problems:
        for llm_name in llm_modes:
            pbar.set_postfix({"llm": llm_name, "problem": problem[:30]})
            start = time.time()

            try:
                final_state = app.invoke({
                    "problem_input": problem,
                    "llm_name":      llm_name,
                    "retry_count":   0,
                })

                results.append({
                    "problem":      problem,
                    "llm":          llm_name,
                    "approx_ratio": final_state.get("approx_ratio"),
                    "cvar":         final_state.get("cvar_value"),
                    "energy":       final_state.get("energy"),
                    "variant":      final_state.get("variant_chosen"),
                    "p_layers":     final_state.get("p_layers"),
                    "retries":      final_state.get("retry_count", 0) - 1,
                    "backend":      final_state.get("backend_chosen"),
                    "wall_time_s":  round(time.time() - start, 2),
                    "status":       "success",
                })

            except Exception as e:
                results.append({
                    "problem": problem,
                    "llm":     llm_name,
                    "status":  "error",
                    "error":   str(e),
                })

            pbar.update(1)

    pbar.close()

    out_path = Path(RESULTS_DIR) / output_file
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSweep complete. Results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_sweep()
