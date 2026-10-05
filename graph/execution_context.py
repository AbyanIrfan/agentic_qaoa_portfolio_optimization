"""
execution_context.py

Describes the ACTUAL backend a run will execute on, in plain language, so
upstream nodes (literature_validation, circuit_design) can weigh
noise-relevant literature claims against what will really happen this run
— instead of reasoning about noise vs. noiseless in the abstract, detached
from this run's real configuration.

USE_HARDWARE / USE_NOISY_SIM / etc. are static config values known before
the graph even starts, so any node can safely import this directly.
"""

from config.settings import (
    USE_HARDWARE, USE_NOISY_SIM, NOISY_SIM_SOURCE, NOISY_SIM_BACKEND, DEFAULT_HARDWARE,
)


def describe_execution_context() -> str:
    if USE_HARDWARE:
        return (
            f"EXECUTION CONTEXT: this run will execute on REAL IBM Quantum hardware "
            f"({DEFAULT_HARDWARE}). Real gate errors, decoherence, and readout noise "
            f"will be present. Weight noise-robust mixer topology and noise-aware "
            f"depth/truncation findings from the literature ABOVE noiseless-regime findings."
        )
    if USE_NOISY_SIM:
        source_desc = (
            f"{DEFAULT_HARDWARE}'s live, current calibration data"
            if NOISY_SIM_SOURCE == "real"
            else f"{NOISY_SIM_BACKEND}'s historical calibration snapshot"
        )
        return (
            f"EXECUTION CONTEXT: this run will execute on a NOISY SIMULATION using "
            f"{source_desc} (via AerSimulator.from_backend) — this closely approximates "
            f"real hardware noise. Weight noise-robust mixer topology and noise-aware "
            f"depth/truncation findings from the literature ABOVE noiseless-regime findings."
        )
    return (
        "EXECUTION CONTEXT: this run will execute on an IDEAL, NOISELESS simulator "
        "(AerSimulator, no noise model). Noiseless-regime findings (e.g. full "
        "XY-mixer/QAMPA superiority) are directly applicable; noise-robustness "
        "findings are not relevant to this specific run's execution conditions."
    )