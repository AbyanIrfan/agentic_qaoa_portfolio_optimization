"""
call_validator.py

Statically scans a block of generated Python code for constructor-style
call sites (e.g. `QuantumCircuit(n_assets)`, `XXPlusYYGate(theta, beta=0)`)
and checks each one against the QiskitDocStore — BEFORE the code is ever
executed. This is what actually gets wired into a generation node's
retry loop: a cheap, static, pre-execution check that catches an LLM's
plausible-but-wrong argument name (or a forgotten required argument)
without spending a simulator run to find out.

Only bare-name and simple-attribute calls are inspected (`ClassName(...)`
or `module.ClassName(...)`) — arbitrary dynamic call targets are skipped,
since there's nothing meaningful to validate statically in that case.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from rag.doc_lookup import QiskitDocStore


@dataclass
class CallSite:
    class_name: str          # short name, e.g. "QuantumCircuit"
    lineno: int
    n_positional: int
    keyword_names: list[str]
    has_starargs: bool       # e.g. f(*args) — unpacked, can't count statically
    has_kwargs_splat: bool   # e.g. f(**kwargs) — unpacked, can't check names statically


def _call_target_short_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def extract_call_sites(code: str) -> list[CallSite]:
    """
    Parse `code` and return a CallSite for every function/constructor call
    found. Raises SyntaxError if the code doesn't parse — callers should
    treat that as its own validation failure (it's about to fail exec()
    anyway, so surfacing it here is strictly earlier/cheaper).
    """
    tree = ast.parse(code)
    sites: list[CallSite] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_target_short_name(node)
        if name is None or not name[0].isupper():
            # Heuristic: only look at calls that look like class
            # constructors (PascalCase). Skips ordinary function calls
            # (np.array(...), len(...), etc.) that aren't in the doc store
            # anyway and would just add noise.
            continue

        n_positional = 0
        has_starargs = False
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                has_starargs = True
            else:
                n_positional += 1

        keyword_names = []
        has_kwargs_splat = False
        for kw in node.keywords:
            if kw.arg is None:
                has_kwargs_splat = True  # **something
            else:
                keyword_names.append(kw.arg)

        sites.append(CallSite(
            class_name=name,
            lineno=node.lineno,
            n_positional=n_positional,
            keyword_names=keyword_names,
            has_starargs=has_starargs,
            has_kwargs_splat=has_kwargs_splat,
        ))

    return sites


def validate_generated_code(code: str, store: QiskitDocStore,
                             known_classes_only: bool = True) -> list[str]:
    """
    Run every call site in `code` through the doc store. Returns a list of
    human-readable problems (empty list = nothing flagged).

    known_classes_only=True (default): silently skip call sites whose
    class name isn't in the doc store at all — most PascalCase calls in
    generated code will be Qiskit classes, but some may be user-defined
    helpers, dataclasses, etc. that were never meant to be in this store.
    Set False if you want "class not found" surfaced as a problem too
    (useful for catching a genuinely hallucinated class name).
    """
    try:
        sites = extract_call_sites(code)
    except SyntaxError as e:
        return [f"Code does not parse (line {e.lineno}): {e.msg}"]

    problems: list[str] = []
    for site in sites:
        if known_classes_only and not store.get_by_short_name(site.class_name):
            continue

        site_problems = store.validate_call(
            site.class_name,
            n_positional=site.n_positional,
            keyword_names=site.keyword_names,
            has_kwargs_splat=site.has_kwargs_splat,
        )
        for p in site_problems:
            problems.append(f"Line {site.lineno}: {p}")

    return problems
