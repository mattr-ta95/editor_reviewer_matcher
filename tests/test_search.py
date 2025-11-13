"""Tests for search functionality"""

import pytest
import numpy as np
from src.search.vector_index import FAISSVectorIndex, cosine_similarity


def test_cosine_similarity():
    """Test cosine similarity calculation"""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])

    # Identical vectors
    assert np.isclose(cosine_similarity(a, b), 1.0)

    # Orthogonal vectors
    assert np.isclose(cosine_similarity(a, c), 0.0)


def test_faiss_index_creation():
    """Test FAISS index creation"""
    embeddings = np.random.randn(100, 768).astype('float32')
    reviewer_ids = [f"reviewer_{i}" for i in range(100)]

    index = FAISSVectorIndex(dimension=768)
    index.build_index(embeddings, reviewer_ids, index_type="Flat")

    assert index.index is not None
    assert index.index.ntotal == 100


def test_faiss_search():
    """Test FAISS search"""
    # Create random embeddings
    embeddings = np.random.randn(100, 768).astype('float32')
    reviewer_ids = [f"reviewer_{i}" for i in range(100)]

    # Build index
    index = FAISSVectorIndex(dimension=768)
    index.build_index(embeddings, reviewer_ids)

    # Search with first embedding
    query = embeddings[0]
    result_ids, scores = index.search(query, k=5)

    # First result should be the query itself
    assert len(result_ids) == 5
    assert result_ids[0] == "reviewer_0"
    assert scores[0] > 0.99  # Should be very close to 1.0


def test_index_save_load(tmp_path):
    """Test saving and loading index"""
    embeddings = np.random.randn(50, 768).astype('float32')
    reviewer_ids = [f"reviewer_{i}" for i in range(50)]

    # Build and save index
    index1 = FAISSVectorIndex(dimension=768)
    index1.build_index(embeddings, reviewer_ids)

    index_path = tmp_path / "test.index"
    index1.save(str(index_path))

    # Load index
    index2 = FAISSVectorIndex(dimension=768)
    index2.load(str(index_path))

    # Test search on loaded index
    query = embeddings[0]
    result_ids, scores = index2.search(query, k=3)

    assert len(result_ids) == 3
    assert result_ids[0] == "reviewer_0"
