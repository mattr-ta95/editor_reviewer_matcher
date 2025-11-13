"""Similarity calculation utilities"""

import numpy as np
from typing import List, Tuple


def calculate_pairwise_similarity(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Calculate pairwise similarity between two sets of embeddings

    Args:
        embeddings1: First set of embeddings (N x D)
        embeddings2: Second set of embeddings (M x D)
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Similarity matrix (N x M)
    """
    if metric == "cosine":
        # Normalize vectors
        emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Compute cosine similarity (dot product of normalized vectors)
        return np.dot(emb1_norm, emb2_norm.T)

    elif metric == "euclidean":
        # Compute Euclidean distance
        # Convert to similarity: 1 / (1 + distance)
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings1, embeddings2, metric='euclidean')
        return 1.0 / (1.0 + distances)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def top_k_similar(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int = 10,
    metric: str = "cosine"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k most similar embeddings

    Args:
        query_embedding: Query embedding (D,)
        candidate_embeddings: Candidate embeddings (N x D)
        k: Number of top results
        metric: Similarity metric

    Returns:
        Tuple of (indices, scores)
    """
    # Calculate similarities
    similarities = calculate_pairwise_similarity(
        query_embedding.reshape(1, -1),
        candidate_embeddings,
        metric=metric
    )[0]

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    top_scores = similarities[top_indices]

    return top_indices, top_scores
