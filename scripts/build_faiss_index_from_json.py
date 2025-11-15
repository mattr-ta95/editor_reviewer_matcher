#!/usr/bin/env python3
"""Build FAISS index directly from JSON (no SQLite needed!)"""

import sys
import os
from pathlib import Path
import argparse
import logging
import numpy as np
import json
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.embedding_generator import SPECTER2EmbeddingGenerator
from src.search.vector_index import FAISSVectorIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from JSON")
    parser.add_argument(
        "--json-path",
        default="data/processed/metadata/reviewers.json",
        help="Path to reviewers JSON file"
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

    args = parser.parse_args()

    # Check if JSON exists
    json_file = Path(args.json_path)
    if not json_file.exists():
        logger.error(f"❌ JSON file not found: {args.json_path}")
        logger.error("Run: python scripts/build_json_database.py first")
        return 1

    # Load JSON
    logger.info("Loading reviewer data from JSON...")
    with open(json_file) as f:
        reviewers = json.load(f)

    logger.info(f"Loaded {len(reviewers)} reviewers")

    if len(reviewers) == 0:
        logger.error("❌ No reviewers in JSON file")
        return 1

    # Create output directories
    Path(args.embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize embedding generator
    logger.info("Initializing SPECTER2 model...")
    generator = SPECTER2EmbeddingGenerator(model_name=args.model_name)

    # Generate embeddings for all reviewers
    logger.info("Generating embeddings for reviewers...")
    embeddings = []
    reviewer_ids = []

    for reviewer in tqdm(reviewers, desc="Generating embeddings"):
        if reviewer["recent_papers"]:
            try:
                # Generate aggregate embedding from papers
                embedding = generator.generate_reviewer_embedding(
                    papers=reviewer["recent_papers"],
                    method="mean",
                    recent_years=3,
                    citation_weighted=True
                )
                embeddings.append(embedding)
                reviewer_ids.append(reviewer["reviewer_id"])
            except Exception as e:
                logger.error(f"Error generating embedding for {reviewer['name']}: {e}")
                continue

    embeddings = np.array(embeddings)
    logger.info(f"Generated embeddings for {len(embeddings)} reviewers")

    # Save embeddings
    np.save(args.embeddings_path, embeddings)
    np.save(Path(args.embeddings_path).with_suffix(".ids.npy"), reviewer_ids)
    logger.info(f"Saved embeddings to {args.embeddings_path}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    index = FAISSVectorIndex(dimension=768)
    index.build_index(
        embeddings=embeddings,
        reviewer_ids=reviewer_ids,
        index_type="Flat"
    )

    # Save index
    index.save(args.index_path)
    logger.info(f"Saved index to {args.index_path}")

    # Test search
    logger.info("\nTesting index with sample query...")
    test_query = embeddings[0]
    result_ids, scores = index.search(test_query, k=5)
    logger.info("Top 5 similar reviewers:")
    for i, (rid, score) in enumerate(zip(result_ids, scores), 1):
        # Find reviewer name
        reviewer = next((r for r in reviewers if r["reviewer_id"] == rid), None)
        if reviewer:
            logger.info(f"  {i}. {reviewer['name']} (score: {score:.3f})")

    logger.info("\n" + "="*70)
    logger.info("✅ Index build complete!")
    logger.info("="*70)
    logger.info(f"Total vectors: {len(embeddings)}")
    logger.info(f"Dimension: {embeddings.shape[1]}")
    logger.info(f"Index saved to: {args.index_path}")
    logger.info("="*70)
    logger.info("\n✅ System is ready!")
    logger.info("Next steps:")
    logger.info("  1. Start API: ./scripts/run_api.sh &")
    logger.info("  2. Start UI: streamlit run app/streamlit_app.py")
    logger.info("  3. Visit: http://localhost:8501")


if __name__ == "__main__":
    main()
