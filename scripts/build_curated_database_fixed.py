#!/usr/bin/env python3
"""Build database with FIXED database locking (uses fresh DB each time)"""

import sys
import os
from pathlib import Path
import logging
import time
import sqlite3

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

# Curated list - VERIFIED researcher IDs
KNOWN_RESEARCHERS = [
    "1688882",      # Yann LeCun (verified works)
    "1699545",      # Yejin Choi (verified works)
    "3458736",      # Dirk Groeneveld (verified works)
    "2375931",      # (verified works)
    "1769686",      # (verified works)
]


def fetch_and_add_author(client, db, author_id, index, total):
    """Fetch author by ID and add to database"""
    try:
        logger.info(f"\n[{index}/{total}] Fetching author {author_id}...")

        author_data = client.get_author_details(author_id)
        if not author_data:
            logger.warning(f"  ❌ Could not fetch author")
            return False

        info = extract_author_info(author_data)
        name = info["name"]
        h_index = info["h_index"]

        logger.info(f"  📊 {name} (h-index: {h_index})")

        current_year = datetime.now().year
        papers_data = client.get_author_papers(
            author_id=author_id,
            limit=20,
            year_min=current_year - 5
        )

        papers = []
        for paper_data in papers_data:
            if paper_data.get("abstract"):
                paper_info = extract_paper_info(paper_data)
                papers.append(Paper(**paper_info))

        if len(papers) < 1:
            logger.warning(f"  ⚠️  No papers with abstracts")
            return False

        logger.info(f"  ✅ {len(papers)} papers with abstracts")

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
        logger.info(f"  ✅ Added to database!")

        return True

    except sqlite3.OperationalError as e:
        if "locked" in str(e):
            logger.error(f"  ❌ Database locked - waiting 2s and retrying...")
            time.sleep(2)
            try:
                db.add_reviewer(reviewer)
                logger.info(f"  ✅ Retry successful!")
                return True
            except:
                logger.error(f"  ❌ Still locked, skipping")
                return False
        else:
            logger.error(f"  ❌ Database error: {e}")
            return False
    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return False


def main():
    logger.info("="*70)
    logger.info("Building Database with Curated IDs (Lock-Safe Version)")
    logger.info("="*70)

    # Use timestamp to create unique DB name
    import time
    timestamp = int(time.time())
    db_path = f"data/reviewers_{timestamp}.db"
    
    logger.info(f"Using database: {db_path}")
    logger.info("(Will rename to reviewers.db at the end)")
    logger.info("")

    # Initialize
    client = SemanticScholarClient(api_key=None, rate_limit=1)
    db = ReviewerDatabase(db_path)

    logger.info(f"Target: {len(KNOWN_RESEARCHERS)} verified researchers")
    logger.info("")

    added = 0
    failed = 0

    for i, author_id in enumerate(KNOWN_RESEARCHERS, 1):
        success = fetch_and_add_author(client, db, author_id, i, len(KNOWN_RESEARCHERS))
        if success:
            added += 1
        else:
            failed += 1

        if i < len(KNOWN_RESEARCHERS):
            time.sleep(1.5)

    # Results
    stats = db.get_stats()
    logger.info("\n" + "="*70)
    logger.info("✅ Database Build Complete!")
    logger.info("="*70)
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Active reviewers: {stats['active_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']:.1f}")
    logger.info("="*70)

    if stats['total_reviewers'] > 0:
        # Rename to final name
        final_path = "data/reviewers.db"
        if Path(final_path).exists():
            Path(final_path).unlink()
        Path(db_path).rename(final_path)
        logger.info(f"\n✅ Renamed to: {final_path}")
        logger.info("Next step: python scripts/build_faiss_index.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⏸️  Interrupted")
        sys.exit(0)
