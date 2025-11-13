"""FAISS vector index operations"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FAISSVectorIndex:
    """Manages FAISS index for fast similarity search"""

    def __init__(self, dimension: int = 768):
        """
        Initialize FAISS index

        Args:
            dimension: Embedding dimension (768 for SPECTER2)
        """
        self.dimension = dimension
        self.index = None
        self.reviewer_ids = []
        self.metadata = {}

    def build_index(
        self,
        embeddings: np.ndarray,
        reviewer_ids: List[str],
        index_type: str = "Flat"
    ):
        """
        Build FAISS index from embeddings

        Args:
            embeddings: Array of embeddings (N x dimension)
            reviewer_ids: List of reviewer IDs corresponding to embeddings
            index_type: Type of index ('Flat' for exact search, 'IVF' for approximate)
        """
        if len(embeddings) != len(reviewer_ids):
            raise ValueError("Number of embeddings must match number of reviewer IDs")

        # Normalize embeddings
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # Create index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        elif index_type == "IVF":
            # Approximate search for larger datasets
            nlist = min(100, len(embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add embeddings to index
        self.index.add(embeddings)
        self.reviewer_ids = reviewer_ids

        logger.info(f"Built {index_type} index with {len(embeddings)} embeddings")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar reviewers

        Args:
            query_embedding: Query embedding vector (768-dim)
            k: Number of results to return

        Returns:
            Tuple of (reviewer_ids, similarity_scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Reshape and normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, k)

        # Convert to reviewer IDs and scores
        reviewer_ids = [self.reviewer_ids[idx] for idx in indices[0] if idx < len(self.reviewer_ids)]
        scores = distances[0].tolist()

        return reviewer_ids, scores

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 100
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Batch search for multiple queries

        Args:
            query_embeddings: Array of query embeddings (N x 768)
            k: Number of results per query

        Returns:
            Tuple of (list of reviewer_id lists, list of score lists)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Normalize queries
        queries = query_embeddings.astype('float32')
        faiss.normalize_L2(queries)

        # Search
        distances, indices = self.index.search(queries, k)

        # Convert to reviewer IDs and scores
        all_reviewer_ids = []
        all_scores = []

        for i in range(len(queries)):
            reviewer_ids = [self.reviewer_ids[idx] for idx in indices[i] if idx < len(self.reviewer_ids)]
            scores = distances[i].tolist()
            all_reviewer_ids.append(reviewer_ids)
            all_scores.append(scores)

        return all_reviewer_ids, all_scores

    def save(self, path: str):
        """
        Save index to disk

        Args:
            path: Path to save index
        """
        if self.index is None:
            raise ValueError("No index to save")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(output_path))

        # Save metadata
        metadata = {
            "reviewer_ids": self.reviewer_ids,
            "dimension": self.dimension,
            "index_size": self.index.ntotal
        }

        metadata_path = output_path.with_suffix(".meta")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Index saved to {path}")

    def load(self, path: str):
        """
        Load index from disk

        Args:
            path: Path to load index from
        """
        index_path = Path(path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = index_path.with_suffix(".meta")
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.reviewer_ids = metadata["reviewer_ids"]
                self.dimension = metadata["dimension"]

        logger.info(f"Loaded index from {path} with {self.index.ntotal} vectors")

    def get_stats(self) -> dict:
        """Get index statistics"""
        if self.index is None:
            return {"status": "not_built"}

        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "is_trained": self.index.is_trained
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query vectors and target vectors

    Args:
        queries: Query vectors (N x D)
        targets: Target vectors (M x D)

    Returns:
        Similarity matrix (N x M)
    """
    # Normalize
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)

    # Compute dot product
    return np.dot(queries, targets.T)
