"""JSON-based database operations (no SQLite)"""

import json
from pathlib import Path
from typing import List, Optional
import logging

from .models import Reviewer, Affiliation, Paper
from datetime import datetime

logger = logging.getLogger(__name__)


class JSONReviewerDatabase:
    """Handles storage and retrieval of reviewer data from JSON"""

    def __init__(self, json_path: str = "data/processed/metadata/reviewers.json"):
        self.json_path = json_path
        self.reviewers = []
        self._load()

    def _load(self):
        """Load reviewers from JSON"""
        json_file = Path(self.json_path)
        if json_file.exists():
            with open(json_file) as f:
                reviewers_data = json.load(f)

            for r_data in reviewers_data:
                try:
                    # Convert dicts to objects
                    affiliations = [Affiliation(**aff) for aff in r_data["affiliations"]]
                    papers = [Paper(**p) for p in r_data["recent_papers"]]

                    last_updated = datetime.fromisoformat(r_data["last_updated"]) if r_data.get("last_updated") else datetime.now()

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
                        last_updated=last_updated
                    )
                    self.reviewers.append(reviewer)
                except Exception as e:
                    logger.error(f"Error loading reviewer: {e}")
                    continue

            logger.info(f"Loaded {len(self.reviewers)} reviewers from JSON")
        else:
            logger.warning(f"JSON file not found: {self.json_path}")

    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        """Retrieve a reviewer by ID"""
        for reviewer in self.reviewers:
            if reviewer.reviewer_id == reviewer_id:
                return reviewer
        return None

    def get_all_reviewers(self, active_only: bool = True) -> List[Reviewer]:
        """Get all reviewers"""
        if active_only:
            return [r for r in self.reviewers if r.active]
        return self.reviewers

    def get_stats(self) -> dict:
        """Get database statistics"""
        active_reviewers = [r for r in self.reviewers if r.active]

        total_papers = sum(len(r.recent_papers) for r in self.reviewers)
        avg_h_index = sum(r.h_index for r in active_reviewers) / len(active_reviewers) if active_reviewers else 0

        return {
            "total_reviewers": len(self.reviewers),
            "active_reviewers": len(active_reviewers),
            "total_papers": total_papers,
            "avg_h_index": avg_h_index
        }
