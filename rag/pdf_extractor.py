"""
pdf_extractor.py

Downloads a paper's PDF, extracts text, and splits it into token-bounded
chunks suitable for embedding. Chunk size is measured in tokens (via
tiktoken, matching the OpenAI embedding model's tokenizer) rather than
characters, so chunk boundaries are consistent with what the embedding
model actually sees.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import tiktoken
from pypdf import PdfReader


_encoding_cache: dict = {"enc": None, "checked": False}

WORDS_PER_TOKEN_APPROX = 0.75  # ~1.33 tokens/word, standard English approximation


def _get_encoding():
    if not _encoding_cache["checked"]:
        try:
            _encoding_cache["enc"] = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_cache["enc"] = None
        _encoding_cache["checked"] = True
    return _encoding_cache["enc"]


def download_pdf(pdf_url: str, dest_path: Path, timeout: int = 60) -> None:
    """Download a PDF to dest_path. Skips download if the file already exists."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "qaoa-portfolio-db/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    tmp_path = dest_path.with_suffix(".part")
    tmp_path.write_bytes(data)
    tmp_path.rename(dest_path)


def extract_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF, page by page, joined with double newlines."""
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    return "\n\n".join(pages_text)


def chunk_text(text: str, chunk_size_tokens: int = 1024, overlap_tokens: int = 100) -> list[str]:
    """
    Split text into overlapping chunks of chunk_size_tokens tokens each.

    Overlap defaults to ~10% of chunk size so that content spanning a chunk
    boundary is still retrievable in at least one whole chunk.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens.")

    encoding = _get_encoding()
    if encoding is not None:
        return _chunk_by_tokens(text, encoding, chunk_size_tokens, overlap_tokens)
    return _chunk_by_words_approx(text, chunk_size_tokens, overlap_tokens)


def _chunk_by_tokens(text: str, encoding, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = encoding.encode(text)
    if not tokens:
        return []

    stride = chunk_size_tokens - overlap_tokens
    chunks = []
    pos = 0
    while pos < len(tokens):
        window = tokens[pos: pos + chunk_size_tokens]
        stripped = encoding.decode(window).strip()
        if stripped:
            chunks.append(stripped)
        if pos + chunk_size_tokens >= len(tokens):
            break
        pos += stride

    return chunks


def _chunk_by_words_approx(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback used only when tiktoken's BPE file can't be fetched (no network)."""
    words = text.split()
    if not words:
        return []

    words_per_chunk = max(1, round(chunk_size_tokens * WORDS_PER_TOKEN_APPROX))
    overlap_words = round(overlap_tokens * WORDS_PER_TOKEN_APPROX)
    stride = max(1, words_per_chunk - overlap_words)

    chunks = []
    pos = 0
    while pos < len(words):
        window = words[pos: pos + words_per_chunk]
        stripped = " ".join(window).strip()
        if stripped:
            chunks.append(stripped)
        if pos + words_per_chunk >= len(words):
            break
        pos += stride

    return chunks


def extract_and_chunk(pdf_path: Path, chunk_size_tokens: int = 1024,
                       overlap_tokens: int = 100) -> list[str]:
    """Convenience wrapper: extract text from a PDF and chunk it in one call."""
    text = extract_text(pdf_path)
    return chunk_text(text, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens)
