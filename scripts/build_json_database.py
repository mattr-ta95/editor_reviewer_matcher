#!/usr/bin/env python3
"""Build database using JSON (no SQLite, no locking issues!)"""

import sys
import os
from pathlib import Path
import logging
import time
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.semantic_scholar_client import SemanticScholarClient, extract_author_info, extract_paper_info
from src.database.models import Reviewer, Paper, Affiliation
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verified researcher IDs
KNOWN_RESEARCHERS = [
    "1688882",      # Yann LeCun
    "1699545",      # Yejin Choi
    "3458736",      # Dirk Groeneveld
    "2375931",      # Other verified
    "1741101",      # Oren Etzioni
]


def fetch_author(client, author_id):
    """Fetch a single author and return as dict"""
    try:
        author_data = client.get_author_details(author_id)
        if not author_data:
            return None

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
                papers.append(paper_info)  # Keep as dict

        if len(papers) < 1:
            logger.warning(f"  ⚠️  No papers with abstracts")
            return None

        logger.info(f"  ✅ {len(papers)} papers with abstracts")

        # Build reviewer dict
        reviewer_dict = {
            "reviewer_id": info["reviewer_id"],
            "name": info["name"],
            "affiliations": info["affiliations"],
            "h_index": info["h_index"],
            "publication_count": info["publication_count"],
            "citation_count": info["citation_count"],
            "fields": list(set(f for p in papers for f in p.get("fields", []))),
            "active": any(p["year"] >= current_year - 3 for p in papers),
            "recent_papers": papers,
            "last_updated": datetime.now().isoformat()
        }

        return reviewer_dict

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return None


def main():
    logger.info("="*70)
    logger.info("Building Database using JSON (No SQLite!)")
    logger.info("="*70)
    logger.info("Saving to: data/processed/metadata/reviewers.json")
    logger.info("")

    # Create directories
    Path("data/processed/metadata").mkdir(parents=True, exist_ok=True)

    # Initialize
    client = SemanticScholarClient(api_key=None, rate_limit=1)

    reviewers = []
    total_papers = 0

    logger.info(f"Fetching {len(KNOWN_RESEARCHERS)} verified researchers...\n")

    for i, author_id in enumerate(KNOWN_RESEARCHERS, 1):
        logger.info(f"[{i}/{len(KNOWN_RESEARCHERS)}] Fetching {author_id}...")

        reviewer = fetch_author(client, author_id)
        if reviewer:
            reviewers.append(reviewer)
            total_papers += len(reviewer["recent_papers"])
            logger.info(f"  ✅ Added!\n")
        else:
            logger.warning(f"  ❌ Skipped\n")

        if i < len(KNOWN_RESEARCHERS):
            time.sleep(1.5)

    # Save to JSON
    output_file = "data/processed/metadata/reviewers.json"
    with open(output_file, 'w') as f:
        json.dump(reviewers, f, indent=2)

    logger.info("="*70)
    logger.info("✅ Database Build Complete!")
    logger.info("="*70)
    logger.info(f"Total reviewers: {len(reviewers)}")
    logger.info(f"Total papers: {total_papers}")
    logger.info(f"Avg h-index: {sum(r['h_index'] for r in reviewers) / len(reviewers):.1f}")
    logger.info(f"Saved to: {output_file}")
    logger.info("="*70)

    if len(reviewers) > 0:
        logger.info("\n✅ SUCCESS! Now convert to SQLite:")
        logger.info("   python scripts/json_to_sqlite.py")
        logger.info("")
        logger.info("Or skip SQLite and go straight to building index:")
        logger.info("   python scripts/build_faiss_index_from_json.py")
    else:
        logger.error("\n❌ No reviewers fetched")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⏸️  Interrupted")
        sys.exit(0)
