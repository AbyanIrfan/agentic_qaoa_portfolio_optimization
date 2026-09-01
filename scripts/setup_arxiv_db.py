#!/usr/bin/env python3
"""
setup_arxiv_db.py

Builds (or incrementally extends) a FAISS-backed corpus of arXiv papers for
the QAOA-portfolio-optimization domain. Mirrors the Astronaut project's
setup_arxiv_db.py CLI, adapted for this project:

  - --init_db true   : start a fresh index (previous files for this
                        --index_name are overwritten)
  - --init_db false  : load the existing index and only ingest papers not
                        already in its manifest — download, text extraction,
                        and embedding calls are skipped entirely for papers
                        already ingested, since those are the expensive steps.

Requires OPENAI_API_KEY in the environment (or a .env file) for embeddings.
The arXiv API itself needs no key.

Example (flat OR — original behavior, still supported):
    python3 setup_arxiv_db.py \\
        --index_name "qaoa-portfolio-v1" \\
        --chunk_size 1024 \\
        --max_results_per_request 100 \\
        --init_db true \\
        --category "quant-ph,q-fin.PM,q-fin.CP" \\
        --keywords "QAOA,quantum approximate optimization algorithm,warm-start QAOA,XY mixer,portfolio optimization quantum,Ising formulation combinatorial optimization,CVaR quantum optimization,digitized counterdiabatic quantum optimization,QUBO" \\
        --date_range "20140101-20260819"

Example (AND-of-OR groups — recommended for a narrow topic: requires at
least one term from EACH group in the same paper, not just any term from
the combined list):
    python3 setup_arxiv_db.py \\
        --index_name "qaoa-portfolio-v1" \\
        --chunk_size 1024 \\
        --max_results_per_request 100 \\
        --init_db true \\
        --category "quant-ph,q-fin.PM,q-fin.CP" \\
        --keyword_groups "QAOA,quantum approximate optimization algorithm,warm-start QAOA,XY mixer,digitized counterdiabatic quantum optimization;portfolio optimization,cardinality-constrained,asset allocation,Markowitz,CVaR" \\
        --date_range "20140101-20260819"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from rag.arxiv_client import ArxivPaper, build_query, fetch_by_ids, iterate_all, parse_date_range  # noqa: E402
from rag.embed_store import ArxivVectorStore, ChunkRecord  # noqa: E402
from rag.pdf_extractor import chunk_text, download_pdf, extract_text  # noqa: E402

logger = logging.getLogger("setup_arxiv_db")

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/extend the QAOA-portfolio arXiv corpus.")
    p.add_argument("--index_name", type=str, required=True)
    p.add_argument("--chunk_size", type=int, required=True, default=1024)
    p.add_argument("--max_results_per_request", type=int, required=True, default=100)
    p.add_argument("--init_db", type=str2bool, required=True)
    p.add_argument("--category", type=str, required=True,
                    help="Comma-separated arXiv categories, e.g. 'quant-ph,q-fin.PM,q-fin.CP'")
    p.add_argument("--keywords", type=str, default="",
                    help="Comma-separated keywords, ALL OR'd together against title+abstract. "
                         "Mutually exclusive with --keyword_groups. For a narrow topic (e.g. "
                         "QAOA specifically applied to portfolio optimization), prefer "
                         "--keyword_groups instead — a flat OR list matches EITHER concept "
                         "alone and pulls in unrelated QAOA papers.")
    p.add_argument("--keyword_groups", type=str, default="",
                    help="Semicolon-separated groups of comma-separated keywords: terms "
                         "within a group are OR'd, groups are AND'd together — a paper must "
                         "match at least one term from EVERY group. Example: "
                         "'QAOA,warm-start QAOA,XY mixer;portfolio optimization,"
                         "cardinality-constrained,Markowitz' requires a QAOA-ish term AND a "
                         "portfolio-ish term in the same paper. Mutually exclusive with "
                         "--keywords.")
    p.add_argument("--date_range", type=str, required=True, help="'YYYYMMDD-YYYYMMDD'")

    # Non-required, sensible defaults:
    p.add_argument("--overlap_size", type=int, default=100,
                    help="Token overlap between consecutive chunks. Default: 100.")
    p.add_argument("--max_total_results", type=int, default=2000,
                    help="Safety cap on total papers fetched per run. Default: 2000.")
    p.add_argument("--pdf_dir", type=str, default="data/papers/arxiv",
                    help="Where downloaded PDFs are cached, relative to this script.")
    p.add_argument("--db_dir", type=str, default="data/references/arxiv_db",
                    help="Where the FAISS index/metadata/manifest are stored.")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                    choices=list(EMBEDDING_DIMS.keys()))
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--sleep_seconds", type=float, default=3.1,
                    help="Delay between arXiv API requests (ToS minimum: 3s).")
    p.add_argument("--seed_ids", type=str, default="",
                    help="Comma-separated arXiv IDs to force-ingest regardless of whether "
                         "the keyword/category query surfaces them (e.g. a novelty-check "
                         "shortlist). Fetched once via a direct id_list lookup before the "
                         "main search-query pagination begins.")
    p.add_argument("--keep_pdfs", type=str2bool, default=True,
                    help="If false, delete each PDF after it's been successfully chunked "
                         "and embedded (index+metadata are retained either way — this only "
                         "affects the raw PDF cache, which is ~20x the disk footprint of "
                         "everything else). Default: true (keep, matching Astronaut's "
                         "convention of caching PDFs for reuse across runs).")
    p.add_argument("--log_level", type=str, default="INFO")
    args = p.parse_args()

    has_flat = bool(args.keywords.strip())
    has_grouped = bool(args.keyword_groups.strip())
    if has_flat and has_grouped:
        p.error("--keywords and --keyword_groups are mutually exclusive — pick one.")
    if not has_flat and not has_grouped:
        p.error("Provide either --keywords (flat OR list) or --keyword_groups (AND-of-OR).")

    return args


def parse_keyword_groups(raw: str) -> list[list[str]]:
    """Parse '--keyword_groups' CLI syntax: ';' separates AND-groups, ',' separates
    OR-terms within a group. Blank groups/terms (e.g. a trailing ';') are dropped."""
    groups = []
    for group_str in raw.split(";"):
        terms = [t.strip() for t in group_str.split(",") if t.strip()]
        if terms:
            groups.append(terms)
    if not groups:
        raise ValueError(f"--keyword_groups parsed to zero usable groups from: {raw!r}")
    return groups


def embed_texts(client, texts: list[str], model: str, batch_size: int) -> np.ndarray:
    """Embed a list of texts in batches, L2-normalizing for cosine-via-inner-product search."""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        all_vecs.append(vecs)
    out = np.vstack(all_vecs)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def process_paper(paper: ArxivPaper, args: argparse.Namespace, pdf_dir: Path) -> list[str]:
    """Download + extract + chunk one paper. Returns the list of text chunks."""
    pdf_path = pdf_dir / f"{paper.arxiv_id}.pdf"
    try:
        download_pdf(paper.pdf_url, pdf_path)
    except Exception as e:
        logger.warning(f"Failed to download {paper.arxiv_id}: {e}")
        return []

    try:
        text = extract_text(pdf_path)
    except Exception as e:
        logger.warning(f"Failed to extract text from {paper.arxiv_id}: {e}")
        return []

    if not text.strip():
        logger.warning(f"No extractable text for {paper.arxiv_id} (scanned/image PDF?) — skipping.")
        return []

    chunks = chunk_text(text, chunk_size_tokens=args.chunk_size, overlap_tokens=args.overlap_size)

    if not args.keep_pdfs and chunks:
        # Only delete after successful extraction+chunking — never delete a
        # PDF we might still need to retry from.
        try:
            pdf_path.unlink()
        except OSError as e:
            logger.warning(f"Could not delete cached PDF for {paper.arxiv_id}: {e}")

    return chunks


def ingest_one(paper: ArxivPaper, args: argparse.Namespace, pdf_dir: Path,
                store: ArxivVectorStore, client) -> str:
    """
    Ingest a single paper end to end: skip if already in the manifest,
    otherwise download/extract/chunk/embed/add. Returns one of
    'skipped_duplicate', 'skipped_empty', 'skipped_embed_failed', 'ingested'.
    Shared by both the seed_ids pre-pass and the main search-query loop so
    the two paths can't drift out of sync.
    """
    if store.already_ingested(paper.arxiv_id):
        return "skipped_duplicate"

    chunks = process_paper(paper, args, pdf_dir)
    if not chunks:
        return "skipped_empty"

    try:
        embeddings = embed_texts(client, chunks, args.embedding_model, args.embedding_batch_size)
    except Exception as e:
        logger.warning(f"Embedding failed for {paper.arxiv_id}: {e}")
        return "skipped_embed_failed"

    records = [
        ChunkRecord(
            arxiv_id=paper.arxiv_id,
            version=paper.version,
            title=paper.title,
            authors=paper.authors,
            published=paper.published,
            primary_category=paper.primary_category,
            chunk_index=i,
            n_chunks=len(chunks),
            text=chunk,
            journal_ref=paper.journal_ref,
            doi=paper.doi,
        )
        for i, chunk in enumerate(chunks)
    ]
    store.add_paper_chunks(records, embeddings)
    return "ingested"


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set (checked environment and .env). Aborting.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI()

    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.db_dir)
    embedding_dim = EMBEDDING_DIMS[args.embedding_model]

    store = ArxivVectorStore(args.index_name, db_dir, embedding_dim)
    if args.init_db:
        store.init_new()
        logger.info(f"Initialized fresh index '{args.index_name}'.")
    else:
        loaded = store.load_or_init()
        if loaded:
            logger.info(
                f"Loaded existing index '{args.index_name}' "
                f"({len(store)} vectors, {len(store.manifest)} papers already ingested)."
            )
        else:
            logger.info(
                f"No existing index found for '{args.index_name}' — starting fresh "
                f"even though --init_db was false."
            )

    counts = {"skipped_duplicate": 0, "skipped_empty": 0, "skipped_embed_failed": 0, "ingested": 0}
    n_chunks_added = 0
    n_seen = 0
    t0 = time.time()

    # --- Seed IDs pass: force-ingest a known shortlist first, bypassing
    # keyword/category search entirely (arXiv's search is keyword-literal,
    # not semantic, so a shortlisted paper can otherwise be missed). ---
    seed_ids = [s.strip() for s in args.seed_ids.split(",") if s.strip()]
    if seed_ids:
        logger.info(f"Fetching {len(seed_ids)} seed ID(s) directly: {seed_ids}")
        try:
            seed_papers = fetch_by_ids(seed_ids)
        except Exception as e:
            logger.warning(f"Failed to fetch seed IDs: {e}")
            seed_papers = []

        found_ids = {p.arxiv_id for p in seed_papers}
        for missing in set(seed_ids) - found_ids:
            logger.warning(f"Seed ID '{missing}' not found on arXiv — check the ID is correct.")

        for paper in seed_papers:
            n_seen += 1
            logger.info(f"[seed] Processing {paper.arxiv_id}: {paper.title[:80]}")
            result = ingest_one(paper, args, pdf_dir, store, client)
            counts[result] += 1
            if result == "ingested":
                n_chunks_added += paper_chunk_count(store, paper.arxiv_id)

        if counts["ingested"] > 0:
            store.save()
        time.sleep(args.sleep_seconds)  # respect ToS before the main query pass begins

    # --- Main search-query pass ---
    categories = [c.strip() for c in args.category.split(",") if c.strip()]
    if args.keyword_groups.strip():
        keywords = parse_keyword_groups(args.keyword_groups)
        logger.info(f"Using {len(keywords)} AND'd keyword group(s): {keywords}")
    else:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    date_range = parse_date_range(args.date_range)
    query = build_query(categories, keywords, date_range)
    logger.info(f"arXiv query: {query}")

    for paper in iterate_all(
        query,
        max_results_per_request=args.max_results_per_request,
        max_total=args.max_total_results,
        sleep_seconds=args.sleep_seconds,
    ):
        n_seen += 1
        result = ingest_one(paper, args, pdf_dir, store, client)
        counts[result] += 1

        if result == "ingested":
            logger.info(f"[{n_seen}] Ingested {paper.arxiv_id}: {paper.title[:80]}")
            # Checkpoint periodically so a crash mid-run doesn't lose progress.
            if counts["ingested"] % 10 == 0:
                store.save()
                logger.info(f"Checkpoint saved at {counts['ingested']} papers ingested.")

    store.save()
    elapsed = time.time() - t0

    logger.info(
        "Done. seen=%d ingested=%d skipped_duplicate=%d skipped_no_text=%d "
        "skipped_embed_failed=%d total_vectors=%d elapsed=%.1fs",
        n_seen, counts["ingested"], counts["skipped_duplicate"], counts["skipped_empty"],
        counts["skipped_embed_failed"], len(store), elapsed,
    )


def paper_chunk_count(store: ArxivVectorStore, arxiv_id: str) -> int:
    return sum(1 for r in store.metadata if r.arxiv_id == arxiv_id)


if __name__ == "__main__":
    main()