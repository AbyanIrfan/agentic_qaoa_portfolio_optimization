"""
extract_source.py

Static, LLM-free extraction of Qiskit class documentation: for every class
found in the target modules, pulls its __init__ signature and docstrings
via `inspect`. No LLM is used anywhere in this file — the whole point of
this store is to ground circuit-generation/validation against the *actual*
signatures, not an LLM's paraphrase of them. An LLM-summarized doc store
would defeat the purpose of using it to catch LLM hallucinations.

Produces "raw" records: one per class, containing the unprocessed
docstrings and signature string. format_docs.py turns these into the
structured (name/type/required/description) form actually consumed
at runtime.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from dataclasses import asdict, dataclass

logger = logging.getLogger("extract_source")


@dataclass
class RawClassRecord:
    qualified_name: str        # e.g. "qiskit_algorithms.minimum_eigensolvers.qaoa.QAOA"
    module: str
    class_name: str
    class_docstring: str | None
    init_docstring: str | None
    init_signature: str | None
    source_available: bool


def _iter_submodules(package_name: str):
    """Yield every importable submodule name under package_name, including itself."""
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        logger.warning(f"Could not import {package_name}: {e}")
        return

    yield package_name

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return  # not a package (no submodules), just a single module

    for _, name, _ in pkgutil.walk_packages(pkg_path, prefix=package_name + "."):
        yield name


def _iter_classes_in_module(module_name: str):
    """Yield (qualified_name, class_obj) for every class DEFINED in this module
    (not merely imported into it, to avoid duplicate entries across re-exports)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        logger.debug(f"Skipping unimportable module {module_name}: {e}")
        return

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_name:
            continue  # imported from elsewhere; will be visited at its defining module
        if name.startswith("_"):
            continue
        yield f"{obj.__module__}.{obj.__qualname__}", obj


def extract_module_tree(package_names: list[str]) -> list[RawClassRecord]:
    """
    Walk each package in package_names (recursively) and extract a
    RawClassRecord for every public class found, deduplicated by
    qualified name.
    """
    records: dict[str, RawClassRecord] = {}

    for package_name in package_names:
        logger.info(f"Walking package: {package_name}")
        n_modules = 0
        for module_name in _iter_submodules(package_name):
            n_modules += 1
            for qualified_name, cls in _iter_classes_in_module(module_name):
                if qualified_name in records:
                    continue
                records[qualified_name] = _extract_one_class(qualified_name, cls)
        logger.info(f"  scanned {n_modules} modules under {package_name}")

    logger.info(f"Extracted {len(records)} unique classes total.")
    return list(records.values())


def _extract_one_class(qualified_name: str, cls: type) -> RawClassRecord:
    class_doc = inspect.getdoc(cls)

    init_doc = None
    init_sig_str = None
    try:
        init_doc = inspect.getdoc(cls.__init__)
    except Exception:
        pass
    try:
        init_sig_str = str(inspect.signature(cls.__init__))
    except (ValueError, TypeError):
        # Some classes (C extensions, dynamically generated __init__) have
        # no introspectable signature. Recorded as unavailable, not fatal.
        pass

    source_available = True
    try:
        inspect.getsource(cls)
    except (OSError, TypeError):
        source_available = False

    return RawClassRecord(
        qualified_name=qualified_name,
        module=cls.__module__,
        class_name=cls.__qualname__,
        class_docstring=class_doc,
        init_docstring=init_doc,
        init_signature=init_sig_str,
        source_available=source_available,
    )


def records_to_json_ready(records: list[RawClassRecord]) -> dict:
    return {r.qualified_name: asdict(r) for r in records}
