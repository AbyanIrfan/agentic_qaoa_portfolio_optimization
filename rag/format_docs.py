"""
format_docs.py

Turns RawClassRecord objects (raw docstrings + signature strings) into the
structured form actually consumed at runtime: one entry per class, with a
short description and a list of {name, type, required, description} args.

Purely static parsing (docstring_parser + inspect) — no LLM call anywhere.
Shape mirrors the PennyLane JSON documented in Sakka, Mizukami & Mitarai
(arXiv:2606.13380), Listing 1, for familiarity: keyed by fully-qualified
class name, each value has "description" and "args".
"""

from __future__ import annotations

import inspect
import re
from dataclasses import asdict, dataclass

import docstring_parser
from docstring_parser import DocstringStyle
from docstring_parser.common import ParseError

from rag.extract_source import RawClassRecord


@dataclass
class ArgDoc:
    name: str
    type: str
    required: bool
    description: str


@dataclass
class FormattedClassDoc:
    qualified_name: str
    description: str
    args: list[ArgDoc]
    introspectable: bool = True
    # False when the real __init__ signature is opaque to `inspect` — most
    # commonly Rust-accelerated bindings (e.g. qiskit._accelerate.circuit.
    # ParameterVector) that report a generic "(self, /, *args, **kwargs)"
    # regardless of their actual accepted arguments. In that case `args` is
    # empty, but that emptiness means "unknown," not "takes no arguments" —
    # callers (call_validator.py) must skip validation entirely rather than
    # flag every real argument as unknown.


_GENERIC_SIGNATURE_RE = re.compile(
    r"^\(self\s*(,\s*/\s*)?(,\s*\*args)?(,\s*\*\*kwargs)?\s*\)"
    r"(\s*->\s*.+)?$"
)


def _is_generic_signature(init_signature: str | None) -> bool:
    """
    True if the signature carries no real per-argument information beyond
    an optional bare *args/**kwargs — the pattern `inspect` reports for
    Rust/C-extension __init__ methods it can't actually introspect.
    A signature with named parameters (even alongside *args/**kwargs)
    returns False, since those named parameters ARE real information.
    """
    if not init_signature:
        return True
    normalized = " ".join(init_signature.split())
    return bool(_GENERIC_SIGNATURE_RE.match(normalized))


def _safe_parse_google(doc_text: str | None):
    """
    Parse a Google-style docstring, tolerating the common Qiskit pattern
    where __init__'s docstring starts directly with 'Args:' and has no
    leading summary line (which the strict Google parser requires).
    """
    if not doc_text:
        return None
    text = doc_text.strip()
    if text.startswith(("Args:", "Arguments:", "Parameters:")):
        text = "Initializer.\n\n" + text
    try:
        return docstring_parser.parse(text, style=DocstringStyle.GOOGLE)
    except ParseError:
        return None


def _description_from_class_doc(class_doc: str | None) -> str:
    parsed = _safe_parse_google(class_doc)
    if parsed and parsed.short_description:
        parts = [parsed.short_description]
        if parsed.long_description:
            parts.append(parsed.long_description)
        return " ".join(p.strip() for p in parts if p.strip())
    # Fall back to the first non-empty line of the raw docstring.
    if class_doc:
        for line in class_doc.splitlines():
            if line.strip():
                return line.strip()
    return ""


def _clean_type(raw_annotation: str | None, docstring_type: str | None) -> str:
    """Prefer the real type annotation from the signature; fall back to the
    docstring's stated type if the annotation is missing (common for
    *args/**kwargs or loosely-typed legacy code)."""
    if raw_annotation and raw_annotation not in ("inspect._empty", ""):
        return raw_annotation
    return docstring_type or "Any"


def _args_from_signature_and_docstring(init_signature: str | None,
                                        init_doc: str | None) -> list[ArgDoc]:
    docstring_params = {}
    parsed = _safe_parse_google(init_doc)
    if parsed:
        for p in parsed.params:
            docstring_params[p.arg_name] = p

    if not init_signature:
        # No usable signature — fall back to whatever the docstring alone gives us.
        return [
            ArgDoc(
                name=p.arg_name,
                type=p.type_name or "Any",
                required=not p.is_optional,
                description=(p.description or "").strip(),
            )
            for p in docstring_params.values()
        ]

    # Re-parse the signature string back into a live Signature object isn't
    # possible without the callable, so we parse it textually instead —
    # good enough for name/annotation/default extraction, which is all we need.
    args: list[ArgDoc] = []
    inner = init_signature.strip()
    # Strip a trailing return-type annotation (e.g. "-> 'None'") BEFORE
    # touching the outer parens — otherwise the leading "(" is never
    # stripped (since the string then ends with the return type, not ")"),
    # which makes bracket-depth tracking below never return to zero and
    # silently swallows every comma split.
    if " -> " in inner:
        inner = inner.rsplit(" -> ", 1)[0].strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    # Split on top-level commas only (avoid splitting inside Union[...] etc.)
    parts, depth, current = [], 0, ""
    for ch in inner:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)

    for raw_part in parts:
        part = raw_part.strip()
        if not part or part == "self":
            continue
        if part in ("*", "/"):
            continue
        if part.startswith("**"):
            name = part[2:].split(":")[0].strip()
            args.append(ArgDoc(name=f"**{name}", type="dict", required=False,
                                description=docstring_params.get(name, None) and
                                            (docstring_params[name].description or "").strip() or ""))
            continue
        if part.startswith("*"):
            name = part[1:].split(":")[0].strip()
            args.append(ArgDoc(name=f"*{name}", type="tuple", required=False,
                                description=""))
            continue

        has_default = "=" in part
        name_and_type = part.split("=")[0].strip() if has_default else part
        if ":" in name_and_type:
            name, annotation = name_and_type.split(":", 1)
            name = name.strip()
            annotation = annotation.strip().strip("'\"")
        else:
            name = name_and_type.strip()
            annotation = None

        doc_p = docstring_params.get(name)
        args.append(ArgDoc(
            name=name,
            type=_clean_type(annotation, doc_p.type_name if doc_p else None),
            required=not has_default,
            description=(doc_p.description or "").strip() if doc_p else "",
        ))

    return args


def format_record(raw: RawClassRecord) -> FormattedClassDoc:
    description = _description_from_class_doc(raw.class_docstring)
    introspectable = not _is_generic_signature(raw.init_signature)
    args = (
        _args_from_signature_and_docstring(raw.init_signature, raw.init_docstring)
        if introspectable else []
    )
    return FormattedClassDoc(
        qualified_name=raw.qualified_name,
        description=description,
        args=args,
        introspectable=introspectable,
    )


def format_all(raw_records: list[RawClassRecord]) -> dict[str, dict]:
    formatted = {}
    for raw in raw_records:
        f = format_record(raw)
        formatted[f.qualified_name] = {
            "description": f.description,
            "args": [asdict(a) for a in f.args],
            "introspectable": f.introspectable,
        }
    return formatted
