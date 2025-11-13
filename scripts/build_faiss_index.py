#!/usr/bin/env python3
"""Script to build FAISS index from reviewer embeddings"""

import sys
import os
from pathlib import Path
import argparse
import logging
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.operations import ReviewerDatabase
from src.data.embedding_generator import SPECTER2EmbeddingGenerator
from src.search.vector_index import FAISSVectorIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from reviewer database")
    parser.add_argument(
        "--db-path",
        default="data/reviewers.db",
        help="Path to reviewer database"
    )
    parser.add_argument(
        "--index-path",
        default="data/indices/specter2.index",
        help="Path to save FAISS index"
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings/reviewer_embeddings.npy",
        help="Path to save embeddings"
    )
    parser.add_argument(
        "--model-name",
        default="allenai/specter2_base",
        help="SPECTER2 model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--index-type",
        choices=["Flat", "IVF"],
        default="Flat",
        help="Type of FAISS index"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if embeddings exist"
    )

    args = parser.parse_args()

    # Create output directories
    Path(args.embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)

    # Load database
    logger.info("Loading reviewer database...")
    db = ReviewerDatabase(args.db_path)
    reviewers = db.get_all_reviewers(active_only=True)
    logger.info(f"Loaded {len(reviewers)} active reviewers")

    if len(reviewers) == 0:
        logger.error("No reviewers found in database. Run build_reviewer_database.py first.")
        return

    # Check if embeddings already exist
    embeddings_file = Path(args.embeddings_path)
    if embeddings_file.exists() and not args.force:
        logger.info("Loading existing embeddings...")
        embeddings = np.load(embeddings_file)
        reviewer_ids = [r.reviewer_id for r in reviewers]
    else:
        # Initialize embedding generator
        logger.info("Initializing SPECTER2 model...")
        generator = SPECTER2EmbeddingGenerator(model_name=args.model_name)

        # Generate embeddings for all reviewers
        logger.info("Generating embeddings for reviewers...")
        embeddings = []
        reviewer_ids = []

        for reviewer in tqdm(reviewers, desc="Generating embeddings"):
            if reviewer.recent_papers:
                try:
                    # Generate aggregate embedding
                    embedding = generator.generate_reviewer_embedding(
                        papers=[p.to_dict() for p in reviewer.recent_papers],
                        method="mean",
                        recent_years=3,
                        citation_weighted=True
                    )
                    embeddings.append(embedding)
                    reviewer_ids.append(reviewer.reviewer_id)
                except Exception as e:
                    logger.error(f"Error generating embedding for {reviewer.name}: {e}")
                    continue

        embeddings = np.array(embeddings)
        logger.info(f"Generated embeddings for {len(embeddings)} reviewers")

        # Save embeddings
        np.save(embeddings_file, embeddings)
        np.save(embeddings_file.with_suffix(".ids.npy"), reviewer_ids)
        logger.info(f"Saved embeddings to {embeddings_file}")

    # Build FAISS index
    logger.info(f"Building {args.index_type} FAISS index...")
    index = FAISSVectorIndex(dimension=768)
    index.build_index(
        embeddings=embeddings,
        reviewer_ids=reviewer_ids,
        index_type=args.index_type
    )

    # Save index
    index.save(args.index_path)
    logger.info(f"Saved index to {args.index_path}")

    # Test search
    logger.info("\nTesting index with sample query...")
    test_query = embeddings[0]  # Use first reviewer as test
    result_ids, scores = index.search(test_query, k=5)
    logger.info("Top 5 similar reviewers:")
    for i, (rid, score) in enumerate(zip(result_ids, scores), 1):
        reviewer = db.get_reviewer(rid)
        if reviewer:
            logger.info(f"  {i}. {reviewer.name} (score: {score:.3f})")

    logger.info("\n" + "="*50)
    logger.info("Index build complete!")
    logger.info(f"Total vectors: {len(embeddings)}")
    logger.info(f"Dimension: {embeddings.shape[1]}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
