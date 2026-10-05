"""
doc_lookup.py

Runtime lookup helper for the formatted Qiskit doc store — this is what
Node 2 (circuit design) and any static-validation step would actually call
during a pipeline run, analogous to embed_store.py's search() closing the
loop on the arXiv side.

Deliberately a plain keyed dict lookup, not semantic search: unlike paper
retrieval, you always know the exact class name you're validating usage
against, so exact lookup is both simpler and more correct here.
"""

from __future__ import annotations

import json
from pathlib import Path


class QiskitDocStore:
    def __init__(self, formatted_json_path: Path):
        with open(formatted_json_path, "r", encoding="utf-8") as f:
            self._docs: dict[str, dict] = json.load(f)
        # Index by bare class name too, since generated code usually refers
        # to e.g. "QAOA" rather than the fully-qualified
        # "qiskit_algorithms.minimum_eigensolvers.qaoa.QAOA".
        self._by_short_name: dict[str, list[str]] = {}
        for qualified_name in self._docs:
            short_name = qualified_name.rsplit(".", 1)[-1]
            self._by_short_name.setdefault(short_name, []).append(qualified_name)

    def get_by_qualified_name(self, qualified_name: str) -> dict | None:
        return self._docs.get(qualified_name)

    def get_by_short_name(self, short_name: str) -> list[dict]:
        """Returns all matches for a bare class name (e.g. 'QAOA') — a list
        because the same short name can exist in multiple modules; the
        caller (or an LLM prompt) should disambiguate by import path."""
        qualified_names = self._by_short_name.get(short_name, [])
        return [self._docs[q] for q in qualified_names]

    def validate_kwargs(self, short_name: str, kwargs: dict) -> list[str]:
        """Back-compat convenience wrapper — pure-keyword call, no positional args."""
        return self.validate_call(short_name, n_positional=0, keyword_names=list(kwargs.keys()))

    def validate_call(self, short_name: str, n_positional: int, keyword_names: list[str],
                       has_kwargs_splat: bool = False) -> list[str]:
        """
        Validate a constructor call site against the doc store, binding BOTH
        positional and keyword arguments — a kwargs-only check would false-
        positive on every positional call (e.g. ParameterVector("gamma", p)),
        which is how Qiskit code is commonly written.

        Returns a list of problems (empty = looks fine, or class known but
        not statically validatable — see `introspectable` below).
        """
        matches = self.get_by_short_name(short_name)
        if not matches:
            return [f"'{short_name}' not found in doc store — cannot validate."]

        best_problems = None
        for doc in matches:
            if not doc.get("introspectable", True):
                return []

            args = doc["args"]
            positional_names: list[str] = []
            keyword_only_names: list[str] = []
            required_names: set[str] = set()
            known_names: set[str] = set()
            var_positional_present = False
            has_kwargs_catch = False
            seen_star = False

            for a in args:
                nm = a["name"]
                if nm.startswith("**"):
                    has_kwargs_catch = True
                    continue
                if nm.startswith("*"):
                    var_positional_present = True
                    seen_star = True
                    continue
                known_names.add(nm)
                if a["required"]:
                    required_names.add(nm)
                (keyword_only_names if seen_star else positional_names).append(nm)

            problems: list[str] = []
            satisfied = set(keyword_names)

            if var_positional_present:
                pass
            else:
                bound = positional_names[:n_positional]
                satisfied.update(bound)
                if n_positional > len(positional_names):
                    problems.append(
                        f"Too many positional arguments for {short_name} "
                        f"({n_positional} given, {len(positional_names)} accepted)."
                    )

            if not has_kwargs_catch and not has_kwargs_splat:
                for k in keyword_names:
                    if k not in known_names:
                        problems.append(f"Unknown argument '{k}' for {short_name}.")

            for r in required_names:
                if r not in satisfied:
                    problems.append(f"Missing required argument '{r}' for {short_name}.")

            if best_problems is None or len(problems) < len(best_problems):
                best_problems = problems

        return best_problems or []

    def __len__(self) -> int:
        return len(self._docs)
