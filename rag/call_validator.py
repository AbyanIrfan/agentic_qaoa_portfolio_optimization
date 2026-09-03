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
    class_name: str
    lineno: int
    n_positional: int
    keyword_names: list[str]
    has_starargs: bool
    has_kwargs_splat: bool
    target_var: str | None = None   # bare variable a method was called on,
                                     # e.g. "hamiltonian" in hamiltonian.exp_i(...);
                                     # None for constructor-style ClassName(...) calls


def _call_target_info(call: ast.Call) -> tuple[str | None, str | None]:
    """Returns (short_name, target_var). target_var is the bare variable a
    method was called on, only when it's a simple Name (None for anything
    more complex, e.g. chained calls)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        target_var = func.value.id if isinstance(func.value, ast.Name) else None
        return func.attr, target_var
    return None, None


def extract_call_sites(code: str, known_var_types: dict[str, type] | None = None) -> list[CallSite]:
    known_var_types = known_var_types or {}
    tree = ast.parse(code)
    sites: list[CallSite] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name, target_var = _call_target_info(node)
        if name is None:
            continue

        is_constructor_style = name[0].isupper()
        is_known_var_method = target_var is not None and target_var in known_var_types
        if not is_constructor_style and not is_known_var_method:
            continue  # ordinary call we have no basis to validate

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
                has_kwargs_splat = True
            else:
                keyword_names.append(kw.arg)

        sites.append(CallSite(
            class_name=name, lineno=node.lineno, n_positional=n_positional,
            keyword_names=keyword_names, has_starargs=has_starargs,
            has_kwargs_splat=has_kwargs_splat, target_var=target_var,
        ))

    return sites


def validate_generated_code(code: str, store: QiskitDocStore,
                             known_classes_only: bool = True,
                             known_var_types: dict[str, type] | None = None) -> list[str]:
    known_var_types = known_var_types or {}
    try:
        sites = extract_call_sites(code, known_var_types=known_var_types)
    except SyntaxError as e:
        return [f"Code does not parse (line {e.lineno}): {e.msg}"]

    problems: list[str] = []
    for site in sites:
        if site.target_var and site.target_var in known_var_types:
            cls = known_var_types[site.target_var]
            if not hasattr(cls, site.class_name):
                problems.append(
                    f"Line {site.lineno}: '{site.target_var}.{site.class_name}(...)' \u2014 "
                    f"{cls.__name__} has no method '{site.class_name}'."
                )
            continue  # doc-store arg-checking below doesn't apply to methods

        if known_classes_only and not store.get_by_short_name(site.class_name):
            continue
        site_problems = store.validate_call(
            site.class_name, n_positional=site.n_positional,
            keyword_names=site.keyword_names, has_kwargs_splat=site.has_kwargs_splat,
        )
        for p in site_problems:
            problems.append(f"Line {site.lineno}: {p}")

    return problems