"""Tests for embedding generation"""

import pytest
import numpy as np
from src.data.embedding_generator import SPECTER2EmbeddingGenerator


@pytest.fixture
def generator():
    """Create embedding generator for tests"""
    return SPECTER2EmbeddingGenerator(model_name="allenai/specter2_base")


def test_embedding_generation(generator):
    """Test basic embedding generation"""
    text = "This is a test abstract about machine learning and neural networks."
    embedding = generator.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)
    assert not np.isnan(embedding).any()


def test_embedding_normalization(generator):
    """Test embeddings are normalized"""
    text = "Test abstract for normalization check."
    embedding = generator.encode(text, normalize=True)

    # Check unit length
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_batch_embedding(generator):
    """Test batch embedding generation"""
    texts = [
        "First abstract about deep learning",
        "Second abstract about computer vision",
        "Third abstract about natural language processing"
    ]

    embeddings = generator.encode(texts)

    assert embeddings.shape == (3, 768)
    assert not np.isnan(embeddings).any()


def test_paper_encoding(generator):
    """Test paper encoding with title and abstract"""
    title = "Neural Networks for Image Recognition"
    abstract = "We propose a deep learning approach for image classification using convolutional neural networks."

    embedding = generator.encode_paper(title, abstract)

    assert embedding.shape == (768,)
    assert not np.isnan(embedding).any()


def test_embedding_aggregation(generator):
    """Test embedding aggregation"""
    embeddings = np.random.randn(5, 768)

    # Test mean aggregation
    mean_emb = generator.aggregate_embeddings(embeddings, method="mean")
    assert mean_emb.shape == (768,)

    # Test max aggregation
    max_emb = generator.aggregate_embeddings(embeddings, method="max")
    assert max_emb.shape == (768,)

    # Test weighted aggregation
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    weighted_emb = generator.aggregate_embeddings(embeddings, method="weighted", weights=weights)
    assert weighted_emb.shape == (768,)
