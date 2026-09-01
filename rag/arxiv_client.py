"""
arxiv_client.py

Thin client for the public arXiv API (http://export.arxiv.org/api/query).
No API key required. Per arXiv's API Terms of Use, requests are throttled
to no more than one every 3 seconds.

Reference: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MIN_REQUEST_INTERVAL_SECONDS = 3.1  # arXiv ToS: >= 3s between requests


@dataclass
class ArxivPaper:
    arxiv_id: str          # e.g. "2606.13380" (version stripped)
    version: str           # e.g. "2606.13380v1" (as returned)
    title: str
    authors: list[str]
    abstract: str
    published: str         # ISO date string, e.g. "2026-06-11"
    updated: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    journal_ref: str = ""  # e.g. "Phys. Rev. A 105, 062431 (2022)" if published, else ""
    doi: str = ""          # resolved DOI URL if present, else ""
    # Both default to "" (not None) so an entry with neither present still
    # constructs cleanly, and downstream boost logic can just check
    # `if paper.journal_ref:` without a None-check everywhere.


def build_query(categories: list[str], keywords: list[str] | list[list[str]],
                 date_range: tuple[str, str]) -> str:
    """
    Build an arXiv API search_query string.

    categories: OR-combined via cat: field, e.g. ["quant-ph", "q-fin.PM", "q-fin.CP"]
    keywords:   either a flat list (all OR'd together — the original
                behavior, e.g. ["QAOA", "XY mixer"]), or a list of lists
                (AND across groups, OR within each group — e.g.
                [["QAOA", "warm-start QAOA"], ["portfolio optimization",
                "cardinality-constrained"]] requires at least one QAOA-ish
                term AND at least one portfolio-ish term in the same
                paper, which a single flat OR list can't express).
    date_range: (start, end) as "YYYYMMDD" strings, inclusive

    A flat OR list is precision-poor for a narrow topic like QAOA-for-
    portfolio-optimization: "QAOA" OR "portfolio optimization" matches any
    paper containing EITHER term, so it happily pulls in QAOA papers about
    MIMO detection or magic-state protocols that have nothing to do with
    portfolios. Grouped AND-of-OR requires both a QAOA-ish term and a
    portfolio-ish term in the same paper.
    """
    if not categories:
        raise ValueError("At least one category is required.")
    if not keywords:
        raise ValueError("At least one keyword (or keyword group) is required.")

    # Normalize to list[list[str]] — a flat list becomes a single group.
    groups: list[list[str]] = (
        keywords if keywords and isinstance(keywords[0], list) else [keywords]  # type: ignore[list-item]
    )
    for group in groups:
        if not group:
            raise ValueError("Keyword groups cannot be empty.")

    cat_clause = " OR ".join(f'cat:{c.strip()}' for c in categories)
    group_clauses = [
        "(" + " OR ".join(f'all:"{k.strip()}"' for k in group) + ")"
        for group in groups
    ]
    keyword_clause = " AND ".join(group_clauses)

    start, end = date_range
    # arXiv expects full-precision timestamps for range queries.
    start_ts = f"{start}000000"
    end_ts = f"{end}235959"
    date_clause = f"submittedDate:[{start_ts} TO {end_ts}]"

    return f"({cat_clause}) AND {keyword_clause} AND {date_clause}"


def _parse_entry(entry: ET.Element) -> ArxivPaper:
    def text(tag: str, ns: str = "atom") -> str:
        el = entry.find(f"{ns}:{tag}", ATOM_NS)
        return (el.text or "").strip() if el is not None else ""

    raw_id_url = text("id")  # e.g. http://arxiv.org/abs/2606.13380v1
    version = raw_id_url.rsplit("/", 1)[-1]
    arxiv_id = version.split("v")[0]

    authors = [
        (a.find("atom:name", ATOM_NS).text or "").strip()
        for a in entry.findall("atom:author", ATOM_NS)
        if a.find("atom:name", ATOM_NS) is not None
    ]

    categories = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", ATOM_NS)
        if c.attrib.get("term")
    ]
    primary_el = entry.find("arxiv:primary_category", ATOM_NS)
    primary_category = primary_el.attrib.get("term", "") if primary_el is not None else (
        categories[0] if categories else ""
    )

    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "")
            break
    if not pdf_url:
        # Fallback: derive from the abs URL.
        pdf_url = raw_id_url.replace("/abs/", "/pdf/")

    published = text("published")[:10]
    updated = text("updated")[:10]

    # journal_ref / doi live in the arxiv: namespace, same as primary_category —
    # both are optional elements, only present once a paper has been formally
    # published. Absence is NOT a reliable "unpublished" signal (many published
    # papers never get this backfilled) — treat presence as a soft positive
    # signal only, never absence as a negative one.
    journal_ref_el = entry.find("arxiv:journal_ref", ATOM_NS)
    journal_ref = (journal_ref_el.text or "").strip() if journal_ref_el is not None else ""
    doi_el = entry.find("arxiv:doi", ATOM_NS)
    doi = (doi_el.text or "").strip() if doi_el is not None else ""

    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=" ".join(text("title").split()),
        authors=authors,
        abstract=" ".join(text("summary").split()),
        published=published,
        updated=updated,
        primary_category=primary_category,
        categories=categories,
        pdf_url=pdf_url,
        journal_ref=journal_ref,
        doi=doi,
    )


def parse_atom_feed(xml_bytes: bytes) -> list[ArxivPaper]:
    root = ET.fromstring(xml_bytes)
    entries = root.findall("atom:entry", ATOM_NS)
    return [_parse_entry(e) for e in entries]


def fetch_page(query: str, start: int, max_results: int, sort_by: str = "submittedDate",
                sort_order: str = "descending") -> list[ArxivPaper]:
    """Fetch a single page of results. Caller is responsible for rate limiting."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "qaoa-portfolio-db/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()
    return parse_atom_feed(xml_bytes)


def iterate_all(query: str, max_results_per_request: int = 100, max_total: int = 2000,
                 sleep_seconds: float = MIN_REQUEST_INTERVAL_SECONDS):
    """
    Generator yielding ArxivPaper objects across paginated requests,
    stopping at max_total results or when the API returns an empty page.
    Sleeps between requests to respect arXiv's rate-limit terms of use.
    """
    start = 0
    yielded = 0
    last_request_time = 0.0

    while yielded < max_total:
        elapsed = time.monotonic() - last_request_time
        if elapsed < sleep_seconds:
            time.sleep(sleep_seconds - elapsed)

        page_size = min(max_results_per_request, max_total - yielded)
        last_request_time = time.monotonic()
        papers = fetch_page(query, start=start, max_results=page_size)

        if not papers:
            break

        for p in papers:
            if yielded >= max_total:
                break
            yield p
            yielded += 1

        start += len(papers)
        if len(papers) < page_size:
            # Fewer results than requested — we've exhausted the query.
            break


def fetch_by_ids(arxiv_ids: list[str]) -> list[ArxivPaper]:
    """
    Fetch specific papers directly by arXiv ID, bypassing the search_query
    entirely. Used to force-ingest a known shortlist regardless of whether
    keyword/category search happens to surface them — arXiv's own search
    is keyword-literal, not semantic, so it can miss papers whose title/
    abstract phrasing doesn't match the query terms even when the paper
    is clearly in-scope.
    """
    if not arxiv_ids:
        return []
    params = {"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)}
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "qaoa-portfolio-db/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_bytes = resp.read()
    return parse_atom_feed(xml_bytes)


def parse_date_range(date_range: str) -> tuple[str, str]:
    """Parse 'YYYYMMDD-YYYYMMDD' into ('YYYYMMDD', 'YYYYMMDD'), validating format."""
    try:
        start, end = date_range.split("-")
        datetime.strptime(start, "%Y%m%d")
        datetime.strptime(end, "%Y%m%d")
    except ValueError as e:
        raise ValueError(
            f"date_range must be 'YYYYMMDD-YYYYMMDD', got {date_range!r}"
        ) from e
    return start, end