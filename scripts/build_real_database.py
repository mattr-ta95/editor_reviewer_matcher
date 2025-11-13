#!/usr/bin/env python3
"""Build reviewer database using direct author search (more reliable)"""

import sys
import os
from pathlib import Path
import argparse
import logging

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


def build_from_author_search(
    client: SemanticScholarClient,
    db: ReviewerDatabase,
    queries: list,
    target_count: int = 1000,
    min_h_index: int = 5
):
    """
    Build database using direct author search (bypasses broken paper search)

    This uses the /author/search endpoint which is more reliable
    """
    logger.info("Building reviewer database using author search API")
    logger.info(f"Target: {target_count} reviewers")

    reviewers_added = 0
    author_ids_seen = set()

    for query in queries:
        if reviewers_added >= target_count:
            break

        logger.info(f"\nSearching authors for: '{query}'")

        try:
            # Use author search directly (more reliable than paper search)
            authors = client.search_authors(
                query=query,
                fields=[],  # Empty to search broadly
                limit=min(100, target_count - reviewers_added)
            )

            logger.info(f"Found {len(authors)} authors")

            for author_data in authors:
                if reviewers_added >= target_count:
                    break

                author_id = author_data.get("authorId")
                if not author_id or author_id in author_ids_seen:
                    continue

                author_ids_seen.add(author_id)

                try:
                    # Get full details
                    full_author = client.get_author_details(author_id)
                    if not full_author:
                        continue

                    h_index = full_author.get("hIndex", 0)
                    paper_count = full_author.get("paperCount", 0)

                    # Filter by criteria
                    if h_index < min_h_index or paper_count < 3:
                        continue

                    # Extract info
                    info = extract_author_info(full_author)
                    name = info["name"]

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

                    if len(papers) < 2:  # Need at least 2 papers with abstracts
                        continue

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
                    reviewers_added += 1

                    logger.info(f"  ✅ [{reviewers_added}/{target_count}] Added: {name} (h-index: {h_index}, papers: {len(papers)})")

                except Exception as e:
                    logger.debug(f"Error processing author {author_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            continue

    return reviewers_added


def main():
    parser = argparse.ArgumentParser(
        description="Build reviewer database using author search (more reliable)"
    )
    parser.add_argument(
        "--db-path",
        default="data/reviewers.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key (HIGHLY RECOMMENDED)"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=1000,
        help="Target number of reviewers"
    )
    parser.add_argument(
        "--min-h-index",
        type=int,
        default=5,
        help="Minimum h-index"
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    logger.info("="*70)
    logger.info("Semantic Reviewer Database Builder (Using Author Search API)")
    logger.info("="*70)

    if not api_key:
        logger.warning("\n⚠️  WARNING: No API key detected!")
        logger.warning("Without an API key, you'll hit rate limits quickly (100 req/5min)")
        logger.warning("Get FREE API key: https://www.semanticscholar.org/product/api")
        logger.warning("Then run: export SEMANTIC_SCHOLAR_API_KEY='your_key'\n")

        response = input("Continue without API key? (y/N): ")
        if response.lower() != 'y':
            logger.info("Exiting. Get an API key and try again!")
            return 1

    # Initialize
    client = SemanticScholarClient(api_key=api_key, rate_limit=10 if api_key else 1)

    # Remove old database
    db_path = Path(args.db_path)
    if db_path.exists():
        logger.info(f"\n⚠️  Database already exists: {db_path}")
        response = input("Delete and rebuild? (y/N): ")
        if response.lower() == 'y':
            db_path.unlink()
            logger.info("Deleted old database")
        else:
            logger.info("Keeping existing database, will add new reviewers")

    db = ReviewerDatabase(args.db_path)

    # Search queries - these will search for AUTHORS, not papers
    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "bioinformatics",
        "computational biology",
        "artificial intelligence",
        "data mining"
    ]

    logger.info(f"\nUsing {len(queries)} search queries")
    logger.info(f"This will search for AUTHORS directly (not papers)")
    logger.info(f"Estimated time: {args.target_count // 20}-{args.target_count // 10} minutes\n")

    try:
        added = build_from_author_search(
            client=client,
            db=db,
            queries=queries,
            target_count=args.target_count,
            min_h_index=args.min_h_index
        )

        # Print results
        stats = db.get_stats()
        logger.info("\n" + "="*70)
        logger.info("✅ Database Build Complete!")
        logger.info("="*70)
        logger.info(f"Total reviewers: {stats['total_reviewers']}")
        logger.info(f"Active reviewers: {stats['active_reviewers']}")
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"Average h-index: {stats['avg_h_index']:.1f}")
        logger.info("="*70)

        if stats['total_reviewers'] >= 100:
            logger.info("\n✅ Database ready! Next step:")
            logger.info("   python scripts/build_faiss_index.py")
        else:
            logger.warning(f"\n⚠️  Only {stats['total_reviewers']} reviewers added")
            logger.warning("Consider running again or getting an API key for better results")

    except KeyboardInterrupt:
        logger.info("\n\n⏸️  Build interrupted by user")
        stats = db.get_stats()
        logger.info(f"Partial build: {stats['total_reviewers']} reviewers")
        return 0

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
