"""Generate embeddings using SPECTER2 model"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SPECTER2EmbeddingGenerator:
    """Generate scientific paper embeddings using SPECTER2"""

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize SPECTER2 embedding generator

        Args:
            model_name: Hugging Face model name
            device: Device to use ('cpu' or 'cuda')
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Set cache directory
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Load tokenizer and model
        logger.info(f"Loading SPECTER2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts (title + abstract)
            batch_size: Batch size for processing
            max_length: Maximum token length
            normalize: Whether to normalize embeddings to unit length

        Returns:
            numpy array of embeddings (768-dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(batch_embeddings)

            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {i + batch_size}/{len(texts)} texts")

        # Concatenate all batches
        embeddings = np.vstack(embeddings)

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Return single embedding if input was single text
        if single_text:
            return embeddings[0]

        return embeddings

    def encode_paper(
        self,
        title: str,
        abstract: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single paper

        Args:
            title: Paper title
            abstract: Paper abstract
            normalize: Whether to normalize embedding

        Returns:
            768-dimensional embedding vector
        """
        # Combine title and abstract as recommended by SPECTER2
        text = f"{title} [SEP] {abstract}" if abstract else title
        return self.encode(text, normalize=normalize)

    def encode_papers_batch(
        self,
        papers: List[dict],
        batch_size: int = 16,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple papers

        Args:
            papers: List of paper dictionaries with 'title' and 'abstract' keys
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings

        Returns:
            Array of embeddings (N x 768)
        """
        texts = []
        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title} [SEP] {abstract}" if abstract else title
            texts.append(text)

        return self.encode(texts, batch_size=batch_size, normalize=normalize)

    def aggregate_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = "mean",
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Aggregate multiple embeddings into a single embedding

        Args:
            embeddings: Array of embeddings (N x 768)
            method: Aggregation method ('mean', 'max', 'weighted')
            weights: Optional weights for weighted aggregation

        Returns:
            Single aggregated embedding (768-dim)
        """
        if len(embeddings) == 0:
            return np.zeros(768)

        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "max":
            return np.max(embeddings, axis=0)
        elif method == "weighted" and weights is not None:
            weights = weights / np.sum(weights)  # Normalize weights
            return np.sum(embeddings * weights[:, np.newaxis], axis=0)
        else:
            return np.mean(embeddings, axis=0)

    def generate_reviewer_embedding(
        self,
        papers: List[dict],
        method: str = "mean",
        recent_years: int = 3,
        citation_weighted: bool = False
    ) -> np.ndarray:
        """
        Generate aggregate embedding for a reviewer based on their papers

        Args:
            papers: List of reviewer's papers
            method: Aggregation method
            recent_years: Only use papers from last N years
            citation_weighted: Weight by citation count

        Returns:
            Single embedding representing reviewer's expertise
        """
        if not papers:
            return np.zeros(768)

        # Filter recent papers
        from datetime import datetime
        current_year = datetime.now().year
        recent_papers = [
            p for p in papers
            if p.get("year", 0) >= current_year - recent_years
        ]

        if not recent_papers:
            # Fall back to all papers if no recent ones
            recent_papers = papers

        # Generate embeddings
        embeddings = self.encode_papers_batch(recent_papers, normalize=True)

        # Prepare weights if citation-weighted
        weights = None
        if citation_weighted:
            citations = np.array([p.get("citations", 1) for p in recent_papers])
            # Add 1 to avoid zero weights, use log scale
            weights = np.log1p(citations)

        # Aggregate
        aggregated = self.aggregate_embeddings(embeddings, method=method, weights=weights)

        # Normalize final embedding
        return aggregated / np.linalg.norm(aggregated)


def create_embedding_cache(
    papers: List[dict],
    generator: SPECTER2EmbeddingGenerator,
    output_path: str,
    batch_size: int = 32
) -> None:
    """
    Create and save embedding cache for papers

    Args:
        papers: List of paper dictionaries
        generator: SPECTER2 embedding generator
        output_path: Path to save embeddings
        batch_size: Batch size for processing
    """
    logger.info(f"Generating embeddings for {len(papers)} papers...")

    embeddings = generator.encode_papers_batch(papers, batch_size=batch_size)

    # Save embeddings
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, embeddings)
    logger.info(f"Embeddings saved to {output_path}")

    # Save paper IDs for reference
    paper_ids = [p.get("paper_id", f"paper_{i}") for i, p in enumerate(papers)]
    np.save(output_file.with_suffix(".ids.npy"), paper_ids)
    logger.info(f"Paper IDs saved to {output_file.with_suffix('.ids.npy')}")
