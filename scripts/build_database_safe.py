#!/usr/bin/env python3
"""Alternative script to build reviewer database using author search"""

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
    parser = argparse.ArgumentParser(description="Build reviewer database using author search")
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
        help="Semantic Scholar API key (recommended for higher rate limits)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of reviewers (start with 1000 for testing)"
    )

    args = parser.parse_args()

    # Get API key from env if not provided
    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    if not api_key:
        logger.warning("⚠️  No API key provided. You may hit rate limits quickly.")
        logger.warning("Get a free API key at: https://www.semanticscholar.org/product/api")
        logger.warning("Then set: export SEMANTIC_SCHOLAR_API_KEY=your_key")
        logger.warning("")

    # Initialize builder
    logger.info("Initializing reviewer database builder...")
    builder = ReviewerDatabaseBuilder(
        db_path=args.db_path,
        api_key=api_key,
        config_path=args.config
    )

    # Use field-based search (more reliable than paper search)
    fields = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "bioinformatics"
    ]

    logger.info(f"Building database using field-based author search")
    logger.info(f"Target: {args.target_count} reviewers")
    logger.info(f"Fields: {', '.join(fields)}")
    logger.info("")
    logger.info("⏱️  This may take 15-30 minutes...")
    logger.info("")

    try:
        count = builder.build_from_fields(
            fields=fields,
            min_h_index=5,  # Lower threshold for testing
            min_publications=3,
            target_count=args.target_count
        )

        # Print statistics
        stats = builder.get_database_stats()
        logger.info("\n" + "="*50)
        logger.info("✅ Database Build Complete!")
        logger.info(f"Total reviewers: {stats['total_reviewers']}")
        logger.info(f"Active reviewers: {stats['active_reviewers']}")
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"Average h-index: {stats['avg_h_index']}")
        logger.info("="*50)

        if stats['total_reviewers'] < 100:
            logger.warning("\n⚠️  Low reviewer count. Consider:")
            logger.warning("  1. Getting an API key for higher rate limits")
            logger.warning("  2. Running again to add more reviewers")
            logger.warning("  3. Using the test database script instead")

    except KeyboardInterrupt:
        logger.info("\n\n⏸️  Build interrupted by user")
        stats = builder.get_database_stats()
        logger.info(f"Partial build: {stats['total_reviewers']} reviewers added")
        logger.info("You can run the script again to continue adding reviewers")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error during build: {e}")
        logger.error("See error above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
