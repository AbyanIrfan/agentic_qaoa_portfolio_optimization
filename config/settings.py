"""
Global configuration for the Agentic QAOA pipeline.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM backends ──────────────────────────────────────────────────────────────
LLM_BACKENDS = {
    "claude":  "claude-sonnet-4-6",   
    "gpt":     "gpt-5.6-sol",         
    "gemini":  "gemini-3.1-pro-preview", 
}

NODE_LLM_MAP = {
    "literature_search":     "gpt",      
    "literature_validation": "claude",  
    "formulation":           "gpt",     
    "circuit_design":        "gpt",     
    "analysis":              "claude",
    "report":                "claude",
    
}

# ── QAOA hyperparameters ──────────────────────────────────────────────────────
DEFAULT_P_LAYERS   = 1
MAX_P_LAYERS       = 5
RETRY_THRESHOLD    = 0.85          
MAX_RETRIES        = 3
CVAR_ALPHA         = 0.25          

# ── Hardware ──────────────────────────────────────────────────────────────────
IBM_QUANTUM_TOKEN    = os.getenv("IBM_QUANTUM_TOKEN")
IBM_QUANTUM_CHANNEL  = os.getenv("IBM_QUANTUM_CHANNEL", "ibm_quantum")
IBM_QUANTUM_INSTANCE = os.getenv("IBM_QUANTUM_INSTANCE", "ibm-q/open/main")
DEFAULT_SIMULATOR    = os.getenv("DEFAULT_SIMULATOR", "aer_simulator")
DEFAULT_HARDWARE     = os.getenv("DEFAULT_HARDWARE", "ibm_kingston")
USE_NOISY_SIM     = True
NOISY_SIM_SOURCE  = "real"            
NOISY_SIM_BACKEND = "FakeSherbrooke"   # used when NOISY_SIM_SOURCE == "fake"

# ── Benchmark ─────────────────────────────────────────────────────────────────
RESULTS_DIR        = "results"
BENCHMARK_SHOTS    = 1024
USE_HARDWARE       = False         # Flip to True for real device runs


# Problem instance settings 

USE_LIVE_DATA      = False                                 # False reproduces the exact Woerner benchmark
TICKERS     = ["LIN.DE", "BAYN.DE", "VNA.DE", "MTX.DE", "MUV2.DE",
               "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE"]  # only used if USE_LIVE_DATA=True
BUDGET              = 3                                     # number of assets to select
RISK_FACTOR         = 0.5                                    # risk-aversion weight, conventionally in [0, 1]
LOOKBACK_DAYS       = 252                                     # ~1 trading year; see finance_data.py docstring
RETURN_CONVENTION   = "log"                                    # "log" or "simple"
END_DATE            = "2026-09-02"                               
PENALTY             = 12                                  # QUBO constraint-penalty starting weight (Node 0 may revise)
RISK_FREE_RATE = 0.0   # ANNUAL rate, mu/sigma from finance_data.py are already
                        # annualized (252-day convention), so this must be annual too,
                        # not daily, or the Sharpe ratio will be silently wrong.
N_SHOTS             = 1024
REPS                = 1
SEED                = 123