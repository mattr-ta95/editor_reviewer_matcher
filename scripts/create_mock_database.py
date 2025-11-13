#!/usr/bin/env python3
"""Create a mock database with synthetic data for immediate testing"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.models import Reviewer, Paper, Affiliation
from src.database.operations import ReviewerDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_reviewers(count: int = 100) -> list:
    """Create mock reviewers for testing"""
    reviewers = []

    institutions = [
        "MIT", "Stanford University", "UC Berkeley", "CMU",
        "Oxford University", "Cambridge University", "ETH Zurich",
        "University of Toronto", "NYU", "Princeton"
    ]

    fields_sets = [
        ["Machine Learning", "Deep Learning"],
        ["Natural Language Processing", "Computational Linguistics"],
        ["Computer Vision", "Image Processing"],
        ["Bioinformatics", "Computational Biology"],
        ["Reinforcement Learning", "Robotics"],
        ["Neural Networks", "AI"]
    ]

    for i in range(count):
        # Generate mock data
        reviewer_id = f"mock_{i:04d}"
        name = f"Dr. Researcher {chr(65 + i % 26)}. {chr(65 + (i // 26) % 26)}."
        institution = institutions[i % len(institutions)]
        fields = fields_sets[i % len(fields_sets)]

        # Create papers
        papers = []
        for j in range(5 + (i % 10)):  # 5-15 papers per reviewer
            paper = Paper(
                paper_id=f"mock_paper_{i}_{j}",
                title=f"Research on {fields[0]}: Paper {j+1}",
                abstract=f"This paper presents novel approaches to {fields[0]} using advanced techniques. We demonstrate state-of-the-art results on benchmark datasets and provide theoretical analysis of our methods. The proposed approach shows significant improvements over previous work.",
                year=2020 + (j % 5),  # 2020-2024
                venue=["NeurIPS", "ICML", "CVPR", "ACL", "EMNLP"][j % 5],
                citations=10 + (i * 2) + (j * 5),
                fields=fields,
                authors=[name, f"Co-author {j+1}"]
            )
            papers.append(paper)

        # Create reviewer
        reviewer = Reviewer(
            reviewer_id=reviewer_id,
            name=name,
            affiliations=[
                Affiliation(
                    institution=institution,
                    country="USA" if i % 3 == 0 else "UK"
                )
            ],
            h_index=10 + (i % 50),  # 10-60
            publication_count=20 + (i % 100),
            citation_count=500 + (i * 50),
            fields=fields,
            active=True,
            recent_papers=papers,
            last_updated=datetime.now()
        )

        reviewers.append(reviewer)

    return reviewers


def main():
    logger.info("Creating mock database for testing...")

    # Create database
    db_path = "data/reviewers.db"
    db = ReviewerDatabase(db_path)

    # Generate and add reviewers
    reviewers = create_mock_reviewers(100)

    for i, reviewer in enumerate(reviewers, 1):
        db.add_reviewer(reviewer)
        if i % 20 == 0:
            logger.info(f"Added {i}/100 reviewers...")

    # Print stats
    stats = db.get_stats()
    logger.info("\n" + "="*60)
    logger.info("✅ Mock Database Created!")
    logger.info("="*60)
    logger.info(f"Total reviewers: {stats['total_reviewers']}")
    logger.info(f"Active reviewers: {stats['active_reviewers']}")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Average h-index: {stats['avg_h_index']}")
    logger.info("="*60)
    logger.info("\n⚠️  Note: This is MOCK DATA for testing only")
    logger.info("For production, use real data from Semantic Scholar")
    logger.info("\nNext step: python scripts/build_faiss_index.py")


if __name__ == "__main__":
    main()
