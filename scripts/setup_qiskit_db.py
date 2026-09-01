#!/usr/bin/env python3
"""
setup_qiskit_db.py

Builds the static Qiskit class-documentation store used to ground code
generation and validation for the circuit-design node. Mirrors Astronaut's
setup_local_file.py flag convention (--save_raw_json_path /
--save_formatted_json_path), adapted for Qiskit instead of PennyLane.

No LLM call anywhere in this pipeline — see format_docs.py's docstring
for why that's a deliberate choice, not an oversight.

Example:
    python3 setup_qiskit_db.py \\
        --save_raw_json_path="../../data/references/qiskit_raw.json" \\
        --save_formatted_json_path="../../data/references/qiskit_formatted.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag.extract_source import extract_module_tree, records_to_json_ready  # noqa: E402
from rag.format_docs import format_all  # noqa: E402

logger = logging.getLogger("setup_qiskit_db")

# Modules the QAOA-portfolio pipeline actually imports (Nodes 1 and 2).
# Passed via --modules to override/extend. qiskit.circuit.library and
# qiskit._accelerate.circuit are included here (unlike the original default)
# because the circuit-design node imports QuantumCircuit, ParameterVector,
# and XXPlusYYGate directly — a store missing those can't validate anything
# that node actually generates. See README for why ParameterVector still
# shows introspectable=False even though it's now included: it's a
# Rust-accelerated binding with no real signature `inspect` can see.
DEFAULT_MODULES = [
    "qiskit_optimization",
    "qiskit_optimization.applications",
    "qiskit_optimization.converters",
    "qiskit_algorithms.minimum_eigensolvers",
    "qiskit_algorithms.optimizers",
    "qiskit.quantum_info",
    "qiskit.circuit",
    "qiskit.circuit.library",
    "qiskit._accelerate.circuit",
    "qiskit_finance.applications.optimization",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Qiskit class-doc reference store.")
    p.add_argument("--save_raw_json_path", type=Path, required=True)
    p.add_argument("--save_formatted_json_path", type=Path, required=True)
    p.add_argument("--modules", type=str, default=",".join(DEFAULT_MODULES),
                    help="Comma-separated top-level module/package paths to walk.")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    logger.info(f"Target modules: {modules}")

    raw_records = extract_module_tree(modules)
    if not raw_records:
        logger.error("No classes extracted — check --modules and that all packages are installed.")
        sys.exit(1)

    raw_json = records_to_json_ready(raw_records)
    args.save_raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, ensure_ascii=False, indent=None)
    logger.info(f"Wrote raw doc dump: {args.save_raw_json_path} ({len(raw_records)} classes)")

    formatted_json = format_all(raw_records)
    n_no_args = sum(1 for v in formatted_json.values() if not v["args"])
    n_no_desc = sum(1 for v in formatted_json.values() if not v["description"])
    args.save_formatted_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_formatted_json_path, "w", encoding="utf-8") as f:
        json.dump(formatted_json, f, ensure_ascii=False, indent=None)
    logger.info(
        f"Wrote formatted doc store: {args.save_formatted_json_path} "
        f"({len(formatted_json)} classes, {n_no_args} with no args parsed, "
        f"{n_no_desc} with no description parsed)"
    )
    if n_no_desc > 0:
        logger.warning(
            f"{n_no_desc} classes had no parseable description — likely missing/"
            f"non-standard docstrings. Spot-check these before relying on them for validation."
        )


if __name__ == "__main__":
    main()
