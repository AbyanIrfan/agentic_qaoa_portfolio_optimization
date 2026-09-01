# ── One-time setup (reuses the exact same corpus + embedding client as
# literature_validation_node.py — copy that setup cell if not already run
# in this session) ─────────────────────────────────────────────────────
#
# from embed_store import ArxivVectorStore
# from openai import OpenAI as _OpenAIClient
# _arxiv_store = ArxivVectorStore(ARXIV_DB_NAME, ARXIV_DB_DIR, embedding_dim=1536)
# _arxiv_store.load_or_init()
# _embed_client = _OpenAIClient()

import numpy as np


def _embed_query(text: str, client, model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def search_within_paper(store, arxiv_id: str, query_embedding: np.ndarray, top_k: int = 10):
    """
    Rank a query against ALL of one paper's chunks directly, rather than
    filtering a global top-k after the fact. This matters because
    restrict_to_arxiv_id filtering a global search can miss a paper's
    most relevant chunks entirely if they don't happen to rank in the
    global top pool — confirmed empirically: a top_k=6 global-then-filter
    query against a specific paper returned only 1 of that paper's
    chunks, even with a 6x over-fetch. This function guarantees every
    chunk belonging to the target paper is actually considered.

    Uses FAISS's reconstruct() to pull back the exact stored vector for
    each of the paper's chunks (only possible with a Flat index, which
    ArxivVectorStore uses) rather than re-embedding anything.
    """
    indices = [i for i, r in enumerate(store.metadata) if r.arxiv_id == arxiv_id]
    if not indices:
        return []

    vecs = np.vstack([store.index.reconstruct(i) for i in indices]).astype(np.float32)
    scores = vecs @ query_embedding.astype(np.float32)  # cosine, since vectors are normalized
    order = np.argsort(-scores)[:top_k]
    return [(store.metadata[indices[i]], float(scores[i])) for i in order]


def query_corpus_for_implementation_detail(
    question: str,
    store,
    embed_client,
    llm,
    top_k: int = 6,
    restrict_to_arxiv_id: str | None = None,
) -> dict:
    """
    Reusable "ask the corpus a targeted implementation question" helper.

    Reuses the SAME retrieval infrastructure as literature_validation_node
    (embed query -> ArxivVectorStore.search against the same FAISS corpus)
    but with a prompt fit for extracting implementable technical detail,
    not cross-checking a recommendation. Use this whenever a later node
    (or you, manually) needs a specific technique's construction from a
    paper already in the corpus — the Dicke-state warm-start question is
    the first use, not the only one this is built for.

    restrict_to_arxiv_id: if set, only chunks from that specific paper are
    used (still ranked by similarity among that paper's own chunks) — use
    this when you already know which paper has the answer, as we do here
    (2305.03857), rather than trusting open retrieval to find it again.
    """
    query_vec = _embed_query(question, embed_client)

    if restrict_to_arxiv_id:
        hits = search_within_paper(store, restrict_to_arxiv_id, query_vec, top_k=top_k)
    else:
        hits = store.search(query_vec, top_k=top_k)

    if not hits:
        return {
            "found": False,
            "answer": None,
            "sources": [],
            "note": "No matching excerpts found" + (
                f" for arxiv_id={restrict_to_arxiv_id}." if restrict_to_arxiv_id else "."
            ),
        }

    excerpts_formatted = "\n\n---\n\n".join(
        f"Source: {r.title} ({r.published}, arXiv:{r.arxiv_id})\n"
        f"Excerpt (chunk {r.chunk_index + 1}/{r.n_chunks}):\n{r.text[:1800]}"
        for r, _ in hits
    )

    prompt = f"""
    You are a quantum computing researcher extracting an implementable
    technical construction from primary-source excerpts.

    Question: {question}

    Excerpts retrieved from the paper(s) most likely to answer this:
    ---
    {excerpts_formatted}
    ---

    Extract the construction as precisely and completely as the excerpts
    allow. Prioritize:
    1. The mathematical/algorithmic construction itself (state
       preparation circuit, gate sequence, amplitude weighting scheme —
       whatever the question asks for), in enough detail to implement.
    2. Any explicit formulas, algorithms, or pseudocode given.
    3. What's NOT covered by these excerpts — be explicit about gaps
       rather than filling them in from general knowledge. If the
       excerpts don't fully specify the construction, say so and name
       what's missing, rather than completing it from training knowledge.

    Do not pad with general background on why the technique is useful —
    focus entirely on the construction itself.
    """

    answer = llm.invoke(prompt).content

    return {
        "found": True,
        "answer": answer,
        "sources": [
            {"arxiv_id": r.arxiv_id, "title": r.title, "chunk_index": r.chunk_index, "score": score}
            for r, score in hits
        ],
        "note": None,
    }
