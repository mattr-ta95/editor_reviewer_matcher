#!/usr/bin/env python3
"""Script to build reviewer database from Semantic Scholar"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.reviewer_builder import ReviewerDatabaseBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build reviewer database from Semantic Scholar")
    parser.add_argument(
        "--db-path",
        default="data/reviewers.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key (optional)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10000,
        help="Target number of reviewers"
    )
    parser.add_argument(
        "--min-h-index",
        type=int,
        default=10,
        help="Minimum h-index for inclusion"
    )
    parser.add_argument(
        "--min-publications",
        type=int,
        default=5,
        help="Minimum publication count"
    )
    parser.add_argument(
        "--method",
        choices=["fields", "search"],
        default="search",
        help="Data collection method"
    )

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    # Initialize builder
    logger.info("Initializing reviewer database builder...")
    builder = ReviewerDatabaseBuilder(
        db_path=args.db_path,
        api_key=api_key,
        config_path=args.config
    )

    # Build database
    if args.method == "fields":
        # Build from predefined fields
        fields = builder.config.get("database", {}).get("fields", [
            "machine learning",
            "deep learning",
            "natural language processing",
            "computer vision",
            "bioinformatics",
            "computational biology"
        ])

        logger.info(f"Building database from fields: {fields}")
        count = builder.build_from_fields(
            fields=fields,
            min_h_index=args.min_h_index,
            min_publications=args.min_publications,
            target_count=args.target_count
        )

    else:  # search method
        # Build from paper searches
        queries = [
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "neural networks",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "bioinformatics",
            "computational biology",
            "protein structure",
            "genomics",
            "systems biology"
        ]

        logger.info(f"Building database from {len(queries)} search queries")
        count = builder.build_from_paper_search(
            queries=queries,
            target_count=args.target_count
        )

    # Print statistics
    stats = builder.get_database_stats()
    logger.info("\n" + "="*50)
    logger.info("Database Build Complete!")
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Active reviewers: {stats['active_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
