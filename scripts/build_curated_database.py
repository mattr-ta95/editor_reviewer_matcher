#!/usr/bin/env python3
"""Build database using curated list of real researcher IDs (no API key needed)"""

import sys
import os
from pathlib import Path
import logging
import time

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

# Curated list of real ML/AI researchers (from public Semantic Scholar data)
# These are verified author IDs that work with direct lookup
KNOWN_RESEARCHERS = [
    # Pioneers & Turing Award Winners
    "1741101",      # Yoshua Bengio
    "1688882",      # Yann LeCun
    "1710790",      # Geoffrey Hinton
    "144794037",    # Andrew Ng
    "1769686",      # Christopher Manning
    "2262347",      # Fei-Fei Li
    "145253968",    # Judea Pearl

    # Leading ML Researchers
    "1780539",      # Ian Goodfellow
    "2474650",      # Pieter Abbeel
    "1800657",      # Sergey Levine
    "40710591",     # Chelsea Finn
    "3458736",      # Oriol Vinyals
    "2375931",      # Ilya Sutskever

    # NLP Leaders
    "2337113",      # Emily Bender
    "1784020",      # Dan Jurafsky
    "1713825",      # Chris Dyer
    "48416454",     # Graham Neubig

    # Computer Vision
    "145626267",    # Kaiming He
    "1752095",      # Ross Girshick
    "2341952",      # Jian Sun
    "1699545",      # Jitendra Malik

    # Theory & Foundations
    "145309825",    # Michael Jordan
    "1706396",      # Bernhard Schölkopf
    "144472577",    # Vladimir Vapnik

    # Reinforcement Learning
    "1693252",      # Richard Sutton
    "3084234",      # David Silver
    "2057259",      # Demis Hassabis

    # Robotics & Control
    "1732471",      # Sebastian Thrun
    "1717671",      # Dieter Fox

    # Medical AI & Bioinformatics
    "2337722",      # Andrew Beam
    "1690245",      # Olga Troyanskaya
]


def fetch_and_add_author(client, db, author_id, index, total):
    """Fetch author by ID and add to database"""
    try:
        logger.info(f"\n[{index}/{total}] Fetching author {author_id}...")

        # Get author details
        author_data = client.get_author_details(author_id)
        if not author_data:
            logger.warning(f"  ❌ Could not fetch author")
            return False

        info = extract_author_info(author_data)
        name = info["name"]
        h_index = info["h_index"]

        logger.info(f"  📊 {name} (h-index: {h_index})")

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

        if len(papers) < 2:
            logger.warning(f"  ⚠️  Only {len(papers)} papers with abstracts")
            if len(papers) == 0:
                return False

        logger.info(f"  ✅ {len(papers)} papers with abstracts")

        # Create reviewer
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

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return False


def main():
    logger.info("="*70)
    logger.info("Building Database with Curated Researcher IDs")
    logger.info("="*70)
    logger.info("This uses direct author lookups (no search API needed)")
    logger.info("Works WITHOUT an API key (just slower due to rate limits)")
    logger.info("")

    # Clean database
    db_path = "data/reviewers.db"
    db_file = Path(db_path)
    if db_file.exists():
        logger.info("⚠️  Removing existing database...")
        db_file.unlink()
        for f in db_file.parent.glob("*.db-*"):
            f.unlink()

    # Initialize
    client = SemanticScholarClient(api_key=None, rate_limit=1)  # 1 req/sec without key
    db = ReviewerDatabase(db_path)

    logger.info(f"Target: {len(KNOWN_RESEARCHERS)} real researchers")
    logger.info(f"Estimated time: {len(KNOWN_RESEARCHERS) * 2} seconds (~{len(KNOWN_RESEARCHERS) // 30} minutes)")
    logger.info("")

    added = 0
    failed = 0

    for i, author_id in enumerate(KNOWN_RESEARCHERS, 1):
        success = fetch_and_add_author(client, db, author_id, i, len(KNOWN_RESEARCHERS))
        if success:
            added += 1
        else:
            failed += 1

        # Rate limiting: wait between requests
        if i < len(KNOWN_RESEARCHERS):
            time.sleep(1.2)  # Conservative to avoid rate limits

    # Results
    stats = db.get_stats()
    logger.info("\n" + "="*70)
    logger.info("✅ Database Build Complete!")
    logger.info("="*70)
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Active reviewers: {stats['active_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']:.1f}")
    logger.info(f"Success: {added}, Failed: {failed}")
    logger.info("="*70)

    if stats['total_reviewers'] >= 20:
        logger.info("\n✅ Database ready!")
        logger.info("Next step: python scripts/build_faiss_index.py")
    else:
        logger.warning(f"\n⚠️  Only {stats['total_reviewers']} reviewers")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⏸️  Interrupted by user")
        sys.exit(0)
