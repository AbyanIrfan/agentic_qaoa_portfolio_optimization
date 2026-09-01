"""
Conditional edge functions for the LangGraph StateGraph.
These determine routing between nodes based on current state.
"""

from graph.state import AgentState
from config.settings import RETRY_THRESHOLD, MAX_RETRIES


def should_retry(state: AgentState) -> str:
    """
    After analysis: retry circuit design if approx ratio is below threshold
    and we haven't exceeded max retries. Otherwise proceed to report.
    """
    ratio   = state.get("approx_ratio") or 0.0
    retries = state.get("retry_count") or 0

    if ratio < RETRY_THRESHOLD and retries < MAX_RETRIES:
        print(f"[Edge] AR={ratio:.3f} < {RETRY_THRESHOLD} — retrying "
              f"(attempt {retries + 1}/{MAX_RETRIES})")
        return "retry"

    print(f"[Edge] AR={ratio:.3f} — proceeding to report")
    return "proceed"


def choose_backend(state: AgentState) -> str:
    """
    Before execution: route to simulator or hardware based on config.
    Extensible — could use LLM reasoning here later.
    """
    from config.settings import USE_HARDWARE
    return "hardware" if USE_HARDWARE else "simulator"
