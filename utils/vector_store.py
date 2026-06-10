import numpy as np
import faiss
from typing import Any, List, Dict, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# =========================
# 1. EMBEDDING MODEL
# =========================
def create_embedding_model(api_key: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """Create Google Generative AI embedding model."""

    kwargs = {
        "model": "models/text-embedding-004",
        "api_version": "v1",
    }

    if api_key:
        kwargs["google_api_key"] = api_key

    return GoogleGenerativeAIEmbeddings(**kwargs)


# =========================
# 2. EMBEDD TEXTS
# =========================
def embed_texts(
    texts: List[str],
    embedding_model: GoogleGenerativeAIEmbeddings
) -> List[List[float]]:
    """Generate embeddings for text chunks."""

    if not texts:
        return []

    embeddings = embedding_model.embed_documents(texts)

    if not embeddings:
        raise ValueError("Embedding generation failed (empty output)")

    return embeddings


# =========================
# 3. BUILD FAISS INDEX
# =========================
def build_faiss_index(
    chunks: List[str],
    embedding_model: GoogleGenerativeAIEmbeddings
) -> Tuple[faiss.IndexFlatIP, int]:

    """Create FAISS index from embeddings."""

    if not chunks:
        raise ValueError("No text chunks provided.")

    embeddings = embed_texts(chunks, embedding_model)
    np_embeddings = np.array(embeddings, dtype=np.float32)

    if np_embeddings.ndim != 2 or np_embeddings.shape[0] == 0:
        raise ValueError("Invalid embeddings shape.")

    # Normalize for cosine similarity
    faiss.normalize_L2(np_embeddings)

    index = faiss.IndexFlatIP(np_embeddings.shape[1])
    index.add(np_embeddings)

    return index, len(chunks)


# =========================
# 4. RETRIEVE SIMILAR CHUNKS
# =========================
def retrieve_similar_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    embedding_model: GoogleGenerativeAIEmbeddings,
    top_k: int = 4
) -> List[Dict[str, Any]]:

    """Retrieve top matching chunks from FAISS."""

    if index.ntotal == 0:
        return []

    # embed query
    query_embedding = embedding_model.embed_query(query)
    query_vector = np.array(query_embedding, dtype=np.float32)[None, :]

    # normalize
    faiss.normalize_L2(query_vector)

    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            "index": int(idx),
            "score": float(score),
            "content": chunks[idx]
        })

    return results