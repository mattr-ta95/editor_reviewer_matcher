#!/usr/bin/env python3
"""Debug version - shows why authors are rejected"""

import sys
import os
from pathlib import Path
import argparse
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.semantic_scholar_client import SemanticScholarClient, extract_author_info, extract_paper_info
from src.database.models import Reviewer, Paper, Affiliation
from src.database.operations import ReviewerDatabase
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_debug(client, db, query, max_authors=10):
    """Debug why authors aren't being added"""

    logger.info(f"\n{'='*70}")
    logger.info(f"DEBUG: Searching for '{query}'")
    logger.info('='*70)

    authors = client.search_authors(query=query, fields=[], limit=max_authors)
    logger.info(f"Found {len(authors)} authors\n")

    for i, author_data in enumerate(authors[:max_authors], 1):
        author_id = author_data.get("authorId")
        author_name = author_data.get("name", "Unknown")

        logger.info(f"\n[{i}/{max_authors}] {author_name} (ID: {author_id})")

        if not author_id:
            logger.warning("  ❌ No author ID")
            continue

        try:
            # Get full details
            full_author = client.get_author_details(author_id)
            if not full_author:
                logger.warning("  ❌ Could not fetch author details")
                continue

            h_index = full_author.get("hIndex", 0)
            paper_count = full_author.get("paperCount", 0)

            logger.info(f"  📊 h-index: {h_index}, papers: {paper_count}")

            # Check filters
            if h_index < 5:
                logger.warning(f"  ❌ h-index too low ({h_index} < 5)")
                continue
            if paper_count < 3:
                logger.warning(f"  ❌ Too few papers ({paper_count} < 3)")
                continue

            logger.info("  ✅ Passed h-index and paper count filters")

            # Get papers
            current_year = datetime.now().year
            logger.info(f"  📄 Fetching papers from {current_year - 5}...")

            papers_data = client.get_author_papers(
                author_id=author_id,
                limit=20,
                year_min=current_year - 5
            )

            logger.info(f"  📄 Found {len(papers_data)} papers")

            # Count papers with abstracts
            papers_with_abstracts = [p for p in papers_data if p.get("abstract")]
            logger.info(f"  📝 {len(papers_with_abstracts)} have abstracts")

            if len(papers_with_abstracts) < 2:
                logger.warning(f"  ❌ Not enough papers with abstracts ({len(papers_with_abstracts)} < 2)")
                continue

            # Show sample papers
            logger.info("  📚 Sample papers:")
            for j, p in enumerate(papers_with_abstracts[:3], 1):
                title = p.get("title", "No title")[:60]
                year = p.get("year", "?")
                abstract_len = len(p.get("abstract", ""))
                logger.info(f"     {j}. [{year}] {title}... (abstract: {abstract_len} chars)")

            # Try to create reviewer
            info = extract_author_info(full_author)
            papers = []
            for paper_data in papers_with_abstracts:
                paper_info = extract_paper_info(paper_data)
                papers.append(Paper(**paper_info))

            active = any(p.year >= current_year - 3 for p in papers)
            affiliations = [Affiliation(**aff) for aff in info["affiliations"]]
            fields = set()
            for paper in papers:
                fields.update(paper.fields)

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

            db.add_reviewer(reviewer)
            logger.info(f"  ✅ SUCCESS! Added to database")
            return True

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return False


def main():
    parser = argparse.ArgumentParser(description="Debug database builder")
    parser.add_argument("--db-path", default="data/reviewers.db")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--query", default="machine learning")
    parser.add_argument("--count", type=int, default=10)

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    logger.info("="*70)
    logger.info("DATABASE BUILDER DEBUG MODE")
    logger.info("="*70)
    logger.info(f"Query: '{args.query}'")
    logger.info(f"Testing {args.count} authors")
    logger.info(f"API Key: {'Yes' if api_key else 'No (using public rate limits)'}")
    logger.info("")

    client = SemanticScholarClient(api_key=api_key, rate_limit=1)

    # Clean database
    db_path = Path(args.db_path)
    if db_path.exists():
        db_path.unlink()

    db = ReviewerDatabase(args.db_path)

    success = build_debug(client, db, args.query, args.count)

    # Show results
    stats = db.get_stats()
    logger.info(f"\n{'='*70}")
    logger.info("RESULTS")
    logger.info('='*70)
    logger.info(f"Reviewers added: {stats['total_reviewers']}")
    logger.info(f"Papers indexed: {stats['total_papers']}")

    if stats['total_reviewers'] == 0:
        logger.error("\n❌ No reviewers added!")
        logger.error("Common issues:")
        logger.error("  1. Authors don't have papers with abstracts")
        logger.error("  2. Papers are too old (need recent papers)")
        logger.error("  3. API rate limiting blocking requests")
        logger.error("\nSolution: Get API key from https://www.semanticscholar.org/product/api")


if __name__ == "__main__":
    main()
