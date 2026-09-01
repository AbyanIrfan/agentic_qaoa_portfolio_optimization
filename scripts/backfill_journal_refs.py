#!/usr/bin/env python3
"""
backfill_journal_refs.py

Updates journal_ref/doi on chunks ALREADY in the corpus — without
re-downloading PDFs, re-extracting text, re-chunking, or re-embedding
anything. None of that changed; only two metadata fields need
populating for papers ingested before those fields existed in the code.

Uses fetch_by_ids() (already built for --seed_ids) to pull each paper's
current lightweight Atom entry — no PDF download — and updates every
ChunkRecord belonging to that arxiv_id in place. Existing embeddings,
text, and the manifest are completely untouched: this can add two
previously-missing string fields, but it cannot corrupt or lose any
already-ingested content, even if interrupted partway through (each
paper's update is independent; a crash after paper 30 of 51 just leaves
the other 21 not-yet-backfilled, not lost).

Usage:
    python3 backfill_journal_refs.py \\
        --index_name "qaoa-portfolio-v1" \\
        --db_dir "data/references/arxiv_db"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag.arxiv_client import fetch_by_ids  # noqa: E402
from rag.embed_store import ArxivVectorStore  # noqa: E402

logger = logging.getLogger("backfill_journal_refs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill journal_ref/doi onto an existing arXiv corpus.")
    p.add_argument("--index_name", type=str, required=True)
    p.add_argument("--db_dir", type=str, required=True)
    p.add_argument("--embedding_dim", type=int, default=1536)
    p.add_argument("--sleep_seconds", type=float, default=3.1,
                    help="Delay between arXiv API requests (ToS minimum: 3s).")
    p.add_argument("--save_every", type=int, default=10,
                    help="Checkpoint save frequency, same rationale as setup_arxiv_db.py.")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    store = ArxivVectorStore(args.index_name, Path(args.db_dir), embedding_dim=args.embedding_dim)
    loaded = store.load_or_init()
    if not loaded:
        logger.error(f"No existing corpus found at '{args.db_dir}/{args.index_name}.*' — nothing to backfill.")
        sys.exit(1)

    arxiv_ids = sorted(store.manifest)
    logger.info(f"Loaded corpus: {len(store)} chunks, {len(arxiv_ids)} papers. Backfilling metadata...")

    n_processed = n_now_have_journal_ref = n_not_found = n_fetch_failed = 0

    for i, arxiv_id in enumerate(arxiv_ids, 1):
        try:
            papers = fetch_by_ids([arxiv_id])
        except Exception as e:
            logger.warning(f"[{i}/{len(arxiv_ids)}] Failed to fetch {arxiv_id}: {e}")
            n_fetch_failed += 1
            time.sleep(args.sleep_seconds)
            continue

        if not papers:
            logger.warning(f"[{i}/{len(arxiv_ids)}] {arxiv_id} not found on arXiv (withdrawn/moved?).")
            n_not_found += 1
            time.sleep(args.sleep_seconds)
            continue

        paper = papers[0]
        n_chunks_touched = sum(1 for r in store.metadata if r.arxiv_id == arxiv_id)
        for record in store.metadata:
            if record.arxiv_id == arxiv_id:
                record.journal_ref = paper.journal_ref
                record.doi = paper.doi

        n_processed += 1
        if paper.journal_ref:
            n_now_have_journal_ref += 1
            logger.info(f"[{i}/{len(arxiv_ids)}] {arxiv_id} ({n_chunks_touched} chunks): "
                        f"journal_ref = '{paper.journal_ref}'")
        else:
            logger.info(f"[{i}/{len(arxiv_ids)}] {arxiv_id} ({n_chunks_touched} chunks): "
                        f"no journal_ref (preprint, or not yet backfilled by arXiv)")

        if n_processed % args.save_every == 0:
            store.save()
            logger.info(f"Checkpoint saved at {n_processed} papers processed.")

        time.sleep(args.sleep_seconds)  # same ToS rate limit as the main ingestion script

    store.save()
    logger.info(
        "Done. processed=%d now_have_journal_ref=%d not_found=%d fetch_failed=%d total_papers=%d",
        n_processed, n_now_have_journal_ref, n_not_found, n_fetch_failed, len(arxiv_ids),
    )


if __name__ == "__main__":
    main()
