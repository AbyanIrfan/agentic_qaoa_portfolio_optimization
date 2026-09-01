"""
Assembles the LangGraph StateGraph from individual agent nodes.
Call build_graph() to get a compiled, runnable app.
"""

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.edges import should_retry
from agents.problem_init          import problem_init_node
from agents.literature_search     import literature_search_node
from agents.literature_validation import literature_validation_node
from agents.formulation    import formulation_node
from agents.circuit_design import circuit_design_node
from agents.execution      import execution_node
from agents.analysis       import analysis_node
from agents.report         import report_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("problem_init",           problem_init_node)
    graph.add_node("literature_search",      literature_search_node)
    graph.add_node("literature_validation",  literature_validation_node)
    graph.add_node("formulation",    formulation_node)
    graph.add_node("circuit_design", circuit_design_node)
    graph.add_node("execution",      execution_node)
    graph.add_node("analysis",       analysis_node)
    graph.add_node("report",         report_node)

    # ── Linear edges ──────────────────────────────────────────────────────────
    graph.set_entry_point("problem_init")
    graph.add_edge("problem_init",          "literature_search")
    graph.add_edge("literature_search",     "literature_validation")
    graph.add_edge("literature_validation", "formulation")
    graph.add_edge("formulation",    "circuit_design")
    graph.add_edge("circuit_design", "execution")
    graph.add_edge("execution",      "analysis")

    # ── Conditional edge: retry loop or proceed ───────────────────────────────
    graph.add_conditional_edges(
        "analysis",
        should_retry,
        {
            "retry":   "circuit_design",   # Loop back with incremented p
            "proceed": "report",
        }
    )

    graph.add_edge("report", END)

    return graph.compile()
