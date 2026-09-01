"""
embed_store.py

FAISS-backed vector store for arXiv paper chunks, with a JSON manifest
tracking which arxiv_ids have already been ingested. This is what makes
--init_db false a genuine incremental append rather than a full rebuild:
papers already in the manifest are skipped before any download, text
extraction, or embedding call happens (all three are the expensive steps).

Files written per index_name, under db_dir:
  <index_name>.faiss     — the FAISS index itself
  <index_name>.meta.json — parallel metadata for every vector (paper id,
                            title, authors, chunk text, chunk index)
  <index_name>.manifest.json — set of arxiv_ids already ingested
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class ChunkRecord:
    arxiv_id: str
    version: str
    title: str
    authors: list[str]
    published: str
    primary_category: str
    chunk_index: int
    n_chunks: int
    text: str
    journal_ref: str = ""  # non-empty only if the paper has a formal publication venue
    doi: str = ""
    # Both new fields default to "" so records already stored on disk from
    # before this change (missing these keys entirely in their JSON) still
    # construct cleanly via ChunkRecord(**r) — they just get no boost until
    # their paper is re-ingested, never a crash on load.


class ArxivVectorStore:
    def __init__(self, index_name: str, db_dir: Path, embedding_dim: int):
        self.index_name = index_name
        self.db_dir = Path(db_dir)
        self.embedding_dim = embedding_dim

        self.index_path = self.db_dir / f"{index_name}.faiss"
        self.meta_path = self.db_dir / f"{index_name}.meta.json"
        self.manifest_path = self.db_dir / f"{index_name}.manifest.json"

        self.index: faiss.Index | None = None
        self.metadata: list[ChunkRecord] = []
        self.manifest: set[str] = set()

    # ------------------------------------------------------------------ #
    # Construction / loading
    # ------------------------------------------------------------------ #

    def init_new(self) -> None:
        """Start a fresh index, discarding any existing files on disk for this name."""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        # Inner-product index over L2-normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.manifest = set()

    def load_or_init(self) -> bool:
        """
        Load an existing index if all three files are present; otherwise
        initialize a fresh one. Returns True if an existing index was loaded.
        """
        if self.index_path.exists() and self.meta_path.exists() and self.manifest_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = [ChunkRecord(**r) for r in raw]
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest = set(json.load(f))
            if self.index.d != self.embedding_dim:
                raise ValueError(
                    f"Existing index dim ({self.index.d}) != requested embedding_dim "
                    f"({self.embedding_dim}). Use a different --index_name or matching model."
                )
            return True

        self.init_new()
        return False

    # ------------------------------------------------------------------ #
    # Manifest checks — used to skip already-ingested papers BEFORE any
    # download/extract/embed work happens.
    # ------------------------------------------------------------------ #

    def already_ingested(self, arxiv_id: str) -> bool:
        return arxiv_id in self.manifest

    # ------------------------------------------------------------------ #
    # Adding data
    # ------------------------------------------------------------------ #

    def add_paper_chunks(self, records: list[ChunkRecord], embeddings: np.ndarray) -> None:
        """
        Add one paper's worth of (already-embedded) chunks to the index.
        embeddings shape: (len(records), embedding_dim), assumed L2-normalized.
        """
        if self.index is None:
            raise RuntimeError("Call init_new() or load_or_init() before adding data.")
        if len(records) == 0:
            return
        if embeddings.shape[0] != len(records):
            raise ValueError("embeddings row count must match number of records.")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings dim {embeddings.shape[1]} != store dim {self.embedding_dim}"
            )

        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(records)
        # All records in one call share an arxiv_id by construction.
        self.manifest.add(records[0].arxiv_id)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save — index was never initialized.")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.metadata], f, ensure_ascii=False, indent=None)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.manifest), f)

    # ------------------------------------------------------------------ #
    # Query (used later by the Discussion/critique node — included here
    # so the store is immediately usable, not just buildable)
    # ------------------------------------------------------------------ #

    def search(self, query_embedding: np.ndarray, top_k: int = 3,
               journal_boost: float = 1.0, over_fetch_multiplier: int = 3
               ) -> list[tuple[ChunkRecord, float]]:
        """
        journal_boost: multiplier applied to a chunk's similarity score if
        its paper has a non-empty journal_ref (i.e., a known publication
        venue). Default 1.0 is a true no-op — identical behavior to before
        this parameter existed, so every existing caller is unaffected
        unless it explicitly opts in with journal_boost > 1.0.

        This is a SOFT boost, not a filter: a paper with no journal_ref is
        never excluded or penalized, only not boosted — journal_ref is
        known to be under-populated even for genuinely published papers,
        so treating its absence as a negative signal would unfairly
        downrank good papers that just never got it backfilled.

        over_fetch_multiplier: with boost enabled, we fetch more raw
        candidates than top_k before re-ranking, since a boosted-but-
        outside-the-raw-top-k chunk otherwise has no chance to rise into
        the final results. Ignored when journal_boost == 1.0 (no re-rank
        needed, so no reason to over-fetch).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)

        raw_top_k = top_k * over_fetch_multiplier if journal_boost != 1.0 else top_k
        raw_top_k = min(raw_top_k, self.index.ntotal)
        scores, indices = self.index.search(q, raw_top_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = self.metadata[idx]
            boosted_score = score * journal_boost if record.journal_ref else score
            candidates.append((record, float(boosted_score)))

        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:top_k]

    def __len__(self) -> int:
        return 0 if self.index is None else self.index.ntotal