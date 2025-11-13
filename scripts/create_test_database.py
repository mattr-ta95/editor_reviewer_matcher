#!/usr/bin/env python3
"""Create a small test database with known author IDs (bypasses search API)"""

import sys
import os
from pathlib import Path
import argparse
import logging
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.semantic_scholar_client import SemanticScholarClient, extract_author_info, extract_paper_info
from src.database.models import Reviewer, Paper, Affiliation
from src.database.operations import ReviewerDatabase
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Known prominent researchers in ML/AI (publicly available IDs)
KNOWN_AUTHOR_IDS = [
    "1741101",      # Yoshua Bengio
    "1688882",      # Yann LeCun
    "1741101",      # Geoffrey Hinton
    "48417367",     # Andrew Ng
    "2262347",      # Fei-Fei Li
    "144794037",    # Jitendra Malik
    "145253968",    # Daphne Koller
    "1741101",      # Michael I. Jordan
    "1769686",      # Christopher Manning
    "1775664",      # Peter Norvig
    "1710790",      # Sebastian Thrun
    "1790679",      # Judea Pearl
    # Add more as needed...
]


def fetch_author_by_id(client, author_id: str, db: ReviewerDatabase) -> bool:
    """Fetch a single author by ID and add to database"""
    try:
        logger.info(f"Fetching author {author_id}...")

        # Get author details
        author_data = client.get_author_details(author_id)
        if not author_data:
            logger.warning(f"Could not fetch author {author_id}")
            return False

        # Extract info
        info = extract_author_info(author_data)
        logger.info(f"  Found: {info['name']} (h-index: {info['h_index']})")

        # Get recent papers
        current_year = datetime.now().year
        papers_data = client.get_author_papers(
            author_id=author_id,
            limit=20,
            year_min=current_year - 5
        )

        # Convert to Paper objects
        papers = []
        for paper_data in papers_data:
            if paper_data.get("abstract"):
                paper_info = extract_paper_info(paper_data)
                papers.append(Paper(**paper_info))

        logger.info(f"  Papers: {len(papers)} with abstracts")

        # Check if author is active
        active = any(p.year >= current_year - 3 for p in papers)

        # Convert affiliations
        affiliations = [Affiliation(**aff) for aff in info["affiliations"]]

        # Determine fields from papers
        fields = set()
        for paper in papers:
            fields.update(paper.fields)

        # Create Reviewer object
        reviewer = Reviewer(
            reviewer_id=info["reviewer_id"],
            name=info["name"],
            affiliations=affiliations,
            h_index=info["h_index"],
            publication_count=info["publication_count"],
            citation_count=info["citation_count"],
            fields=list(fields),
            active=active,
            recent_papers=papers,
            last_updated=datetime.now()
        )

        # Add to database
        db.add_reviewer(reviewer)
        logger.info(f"  ✅ Added to database")

        return True

    except Exception as e:
        logger.error(f"Error processing author {author_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create test database with known authors")
    parser.add_argument(
        "--db-path",
        default="data/reviewers.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Target number of reviewers"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    logger.info("="*60)
    logger.info("Creating test database with known author IDs")
    logger.info("="*60)

    if not api_key:
        logger.warning("⚠️  No API key - using public rate limits (slow)")
        logger.warning("Get free key at: https://www.semanticscholar.org/product/api")

    # Initialize
    client = SemanticScholarClient(api_key=api_key, rate_limit=1)  # 1 req/sec to be safe
    db = ReviewerDatabase(args.db_path)

    added = 0
    failed = 0

    # Fetch each known author
    for i, author_id in enumerate(KNOWN_AUTHOR_IDS[:args.count], 1):
        logger.info(f"\n[{i}/{min(args.count, len(KNOWN_AUTHOR_IDS))}]")

        if fetch_author_by_id(client, author_id, db):
            added += 1
        else:
            failed += 1

        # Small delay to avoid rate limits
        if i < len(KNOWN_AUTHOR_IDS):
            time.sleep(1.5)

    # Print results
    stats = db.get_stats()
    logger.info("\n" + "="*60)
    logger.info("✅ Test Database Created!")
    logger.info("="*60)
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Active reviewers: {stats['active_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']}")
    logger.info(f"Success: {added}, Failed: {failed}")
    logger.info("="*60)

    if stats['total_reviewers'] >= 10:
        logger.info("\n✅ Database is ready for testing!")
        logger.info("Next step: Run 'python scripts/build_faiss_index.py'")
    else:
        logger.warning("\n⚠️  Database has few reviewers. Consider:")
        logger.warning("  1. Getting an API key for better access")
        logger.warning("  2. Waiting a bit and running again")


if __name__ == "__main__":
    main()
