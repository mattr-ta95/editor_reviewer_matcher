#!/usr/bin/env python3
"""Convert JSON database to SQLite (run AFTER build_json_database.py)"""

import sys
import json
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import Reviewer, Paper, Affiliation
from src.database.operations import ReviewerDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    json_file = "data/processed/metadata/reviewers.json"

    if not Path(json_file).exists():
        logger.error(f"❌ {json_file} not found!")
        logger.error("Run: python scripts/build_json_database.py first")
        return 1

    # Load JSON
    logger.info(f"Loading {json_file}...")
    with open(json_file) as f:
        reviewers_data = json.load(f)

    logger.info(f"Found {len(reviewers_data)} reviewers")

    # Delete old SQLite database
    db_path = "data/reviewers.db"
    if Path(db_path).exists():
        Path(db_path).unlink()
        logger.info("Deleted old SQLite database")

    # Create new database
    db = ReviewerDatabase(db_path)

    # Add each reviewer
    for i, r_data in enumerate(reviewers_data, 1):
        try:
            # Convert dicts back to objects
            affiliations = [Affiliation(**aff) for aff in r_data["affiliations"]]
            papers = [Paper(**p) for p in r_data["recent_papers"]]

            reviewer = Reviewer(
                reviewer_id=r_data["reviewer_id"],
                name=r_data["name"],
                affiliations=affiliations,
                h_index=r_data["h_index"],
                publication_count=r_data["publication_count"],
                citation_count=r_data["citation_count"],
                fields=r_data["fields"],
                active=r_data["active"],
                recent_papers=papers,
                last_updated=r_data["last_updated"]
            )

            db.add_reviewer(reviewer)
            logger.info(f"[{i}/{len(reviewers_data)}] Added: {reviewer.name}")

        except Exception as e:
            logger.error(f"Error adding {r_data.get('name')}: {e}")
            continue

    stats = db.get_stats()
    logger.info("\n" + "="*70)
    logger.info("✅ SQLite Database Created!")
    logger.info("="*70)
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']:.1f}")
    logger.info("="*70)
    logger.info("\nNext step: python scripts/build_faiss_index.py")


if __name__ == "__main__":
    main()
