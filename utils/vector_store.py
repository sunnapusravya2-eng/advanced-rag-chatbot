import os
import numpy as np
import faiss
import google.generativeai as genai


def create_embedding_model(api_key=None):
    """Initialize and return the Google Generative AI embedding model."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file."
        )

    genai.configure(api_key=key)

    return {
        "client": genai,
        "model": os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
    }


def build_faiss_index(chunks: list, embedding_model: dict) -> tuple:
    """
    Generate embeddings for all chunks and build a FAISS index.
    """
    if embedding_model is None:
        raise ValueError(
            "embedding_model is None. Make sure create_embedding_model() returned successfully."
        )

    client = embedding_model["client"]
    model = embedding_model["model"]

    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            result = client.embed_content(
                model=model,
                content=chunk,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        except Exception as e:
            raise RuntimeError(f"Failed to embed chunk {i}: {e}")

    if not embeddings:
        raise ValueError("No embeddings were generated. Check your chunks and API key.")

    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, len(chunks)


def retrieve_similar_chunks(
    query: str,
    index,
    chunks: list,
    embedding_model: dict,
    top_k: int = 4,
) -> list:
    """
    Embed the query and retrieve the most similar chunks from the FAISS index.
    """
    if embedding_model is None:
        raise ValueError(
            "embedding_model is None. Make sure create_embedding_model() returned successfully."
        )

    if index is None:
        raise ValueError("FAISS index is None. Please upload and process a PDF first.")

    if not query.strip():
        raise ValueError("Query is empty. Please enter a valid question.")

    client = embedding_model["client"]
    model = embedding_model["model"]

    try:
        result = client.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to embed query: {e}")

    query_vec = np.array([result["embedding"]]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "index": int(idx),
                "score": float(1 / (1 + dist)),
                "content": chunks[idx],
            })

    return results
