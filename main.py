"""
main.py — entry point for a single agentic QAOA pipeline run.
For the full benchmark sweep, run: python -m benchmark.sweep
"""

from dotenv import load_dotenv
load_dotenv()

from graph.builder import build_graph
from rich.console  import Console
from rich.markdown import Markdown

console = Console()


def run_single(llm_name: str = "hetero"):
    """
    Run one full pipeline pass and print the report.

    The problem instance itself (tickers, budget, risk factor, etc.) is
    sourced entirely from config/settings.py by problem_init_node — not
    passed in here — since that node always derives problem_input from
    config regardless of what's in the initial state.

    Args:
        llm_name: "claude" | "gpt" | "gemini" | "hetero"
                  (NOTE: not yet wired to per-node LLM routing — every
                  node currently hardcodes ChatAnthropic directly. This
                  is a known follow-up, not yet implemented.)
    """
    app = build_graph()

    console.rule(f"[bold]Agentic QAOA — {llm_name.upper()} mode")

    final_state = app.invoke({
        "llm_name":    llm_name,
        "retry_count": 0,
    })

    console.print(f"Problem: {final_state.get('problem_input', 'N/A')}\n")

    def _fmt(key):
        val = final_state.get(key)
        return f"{val:.4f}" if val is not None else "N/A"

    console.rule("[bold]Results")
    console.print(f"Variant:         {final_state.get('variant_chosen', 'N/A')}")
    console.print(f"p-layers:        {final_state.get('p_layers', 'N/A')}")
    console.print(f"Backend:         {final_state.get('backend_chosen', 'N/A')}")
    console.print(f"Approx ratio:    {_fmt('approx_ratio')}")
    console.print(f"CVaR (α=0.25):   {_fmt('cvar_value')}")
    console.print(f"Energy:          {_fmt('energy')}")
    console.print(f"Retries:         {final_state.get('retry_count', 0) - 1}")

    console.rule("[bold]Report")
    console.print(Markdown(final_state.get("report", "No report generated.")))

    return final_state


if __name__ == "__main__":
    run_single(llm_name="hetero")