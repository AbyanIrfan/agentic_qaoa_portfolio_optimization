"""
Global configuration for the Agentic QAOA pipeline.
Edit values here — no need to touch agent code directly.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM backends ──────────────────────────────────────────────────────────────
LLM_BACKENDS = {
    "claude":  "claude-sonnet-4-6",   # matches what's proven working from night one
    "gpt":     "gpt-5.6-sol",         # current OpenAI flagship (gpt-4o is stale)
    "gemini":  "gemini-3.1-pro-preview",  # current Gemini flagship (2.0-flash is shut down)
}

NODE_LLM_MAP = {
    "literature_search":     "gpt",     # tentative — see comparison report Section 8.1
    "literature_validation": "claude",
    "formulation":           "gpt",     # tentative — see comparison report Section 8.1
    "circuit_design":        "gemini",  # tentative — retest after thinking_level fix
    "analysis":              "claude",
    "report":                "claude",
    # "execution" removed — agents/execution.py makes no LLM call
}

# ── QAOA hyperparameters ──────────────────────────────────────────────────────
DEFAULT_P_LAYERS   = 1
MAX_P_LAYERS       = 5
RETRY_THRESHOLD    = 0.85          # Approx ratio below which we retry
MAX_RETRIES        = 3
CVAR_ALPHA         = 0.25          # CVaR confidence level

# ── Hardware ──────────────────────────────────────────────────────────────────
IBM_QUANTUM_TOKEN    = os.getenv("IBM_QUANTUM_TOKEN")
IBM_QUANTUM_CHANNEL  = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")
IBM_QUANTUM_INSTANCE = os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
DEFAULT_SIMULATOR    = os.getenv("DEFAULT_SIMULATOR", "aer_simulator")
DEFAULT_HARDWARE     = os.getenv("DEFAULT_HARDWARE", "ibm_brisbane")

# ── Benchmark ─────────────────────────────────────────────────────────────────
RESULTS_DIR        = "results"
BENCHMARK_SHOTS    = 1024
USE_HARDWARE       = False         # Flip to True for real device runs

# ═══════════════════════════════════════════════════════════════════════════
# Problem instance settings — edit these for your specific run
# ═══════════════════════════════════════════════════════════════════════════
USE_LIVE_DATA      = True                                  # False reproduces the exact Woerner benchmark
TICKERS            = ["AAPL", "XOM", "JPM", "JNJ", "GLD"]  # only used if USE_LIVE_DATA=True
BUDGET              = 3                                     # number of assets to select
RISK_FACTOR         = 0.5                                    # risk-aversion weight, conventionally in [0, 1]
LOOKBACK_DAYS       = 252                                     # ~1 trading year; see finance_data.py docstring
RETURN_CONVENTION   = "log"                                    # "log" or "simple"
END_DATE            = None                                     # e.g. "2026-02-16" to reproduce a historical window
PENALTY             = 12                                  # QUBO constraint-penalty starting weight (Node 0 may revise)
N_SHOTS             = 1024
REPS                = 1
SEED                = 123