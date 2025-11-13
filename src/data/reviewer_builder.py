"""Build reviewer database from Semantic Scholar data"""

import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import yaml

from .semantic_scholar_client import SemanticScholarClient, extract_author_info, extract_paper_info
from ..database.models import Reviewer, Paper, Affiliation
from ..database.operations import ReviewerDatabase

logger = logging.getLogger(__name__)


class ReviewerDatabaseBuilder:
    """Build and populate reviewer database"""

    def __init__(
        self,
        db_path: str = "data/reviewers.db",
        api_key: Optional[str] = None,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize database builder

        Args:
            db_path: Path to SQLite database
            api_key: Semantic Scholar API key
            config_path: Path to configuration file
        """
        self.db = ReviewerDatabase(db_path)
        self.client = SemanticScholarClient(api_key=api_key)

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def build_from_fields(
        self,
        fields: List[str],
        min_h_index: int = 10,
        min_publications: int = 5,
        target_count: int = 10000
    ) -> int:
        """
        Build reviewer database from specified fields

        Args:
            fields: List of field names (e.g., ['machine learning', 'bioinformatics'])
            min_h_index: Minimum h-index for inclusion
            min_publications: Minimum publication count
            target_count: Target number of reviewers

        Returns:
            Number of reviewers added
        """
        logger.info(f"Building reviewer database from fields: {fields}")
        logger.info(f"Target: {target_count} reviewers with h-index >= {min_h_index}")

        reviewers_added = 0

        for field in fields:
            if reviewers_added >= target_count:
                break

            logger.info(f"\nProcessing field: {field}")

            # Get top authors in this field
            authors = self.client.get_top_authors_by_field(
                field=field,
                min_h_index=min_h_index,
                min_papers=min_publications,
                limit=target_count - reviewers_added
            )

            logger.info(f"Found {len(authors)} qualifying authors in {field}")

            # Process each author
            for i, author_data in enumerate(authors):
                try:
                    reviewer = self._process_author(author_data)
                    if reviewer:
                        self.db.add_reviewer(reviewer)
                        reviewers_added += 1

                        if reviewers_added % 100 == 0:
                            logger.info(f"Progress: {reviewers_added}/{target_count} reviewers added")

                        if reviewers_added >= target_count:
                            break

                except Exception as e:
                    logger.error(f"Error processing author {author_data.get('name', 'Unknown')}: {e}")
                    continue

        logger.info(f"\nDatabase build complete! Added {reviewers_added} reviewers")
        return reviewers_added

    def _process_author(self, author_data: dict) -> Optional[Reviewer]:
        """
        Process author data and create Reviewer object

        Args:
            author_data: Raw author data from Semantic Scholar

        Returns:
            Reviewer object or None if invalid
        """
        # Extract basic info
        info = extract_author_info(author_data)

        # Get papers
        author_id = info["reviewer_id"]
        if not author_id:
            return None

        # Fetch recent papers
        current_year = datetime.now().year
        papers_data = self.client.get_author_papers(
            author_id=author_id,
            limit=20,
            year_min=current_year - 5  # Last 5 years
        )

        # Convert to Paper objects
        papers = []
        for paper_data in papers_data:
            if paper_data.get("abstract"):  # Only include papers with abstracts
                paper_info = extract_paper_info(paper_data)
                papers.append(Paper(**paper_info))

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

        return reviewer

    def update_reviewer(self, reviewer_id: str) -> bool:
        """
        Update an existing reviewer's data

        Args:
            reviewer_id: Semantic Scholar author ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch latest data
            author_data = self.client.get_author_details(reviewer_id)
            if not author_data:
                logger.warning(f"Could not fetch data for reviewer {reviewer_id}")
                return False

            # Process and update
            reviewer = self._process_author(author_data)
            if reviewer:
                self.db.add_reviewer(reviewer)
                logger.info(f"Updated reviewer: {reviewer.name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error updating reviewer {reviewer_id}: {e}")
            return False

    def build_from_paper_search(
        self,
        queries: List[str],
        target_count: int = 10000
    ) -> int:
        """
        Build reviewer database by searching for papers and extracting authors

        Args:
            queries: List of search queries
            target_count: Target number of reviewers

        Returns:
            Number of reviewers added
        """
        logger.info("Building reviewer database from paper searches")

        author_ids_seen = set()
        reviewers_added = 0

        for query in queries:
            if reviewers_added >= target_count:
                break

            logger.info(f"\nSearching papers for: {query}")

            # Search for papers
            papers = self.client.search_papers(
                query=query,
                limit=500,
                year_min=datetime.now().year - 5
            )

            logger.info(f"Found {len(papers)} papers")

            # Extract unique authors
            for paper in papers:
                if "authors" in paper:
                    for author in paper["authors"]:
                        author_id = author.get("authorId")

                        if not author_id or author_id in author_ids_seen:
                            continue

                        author_ids_seen.add(author_id)

                        try:
                            # Fetch author details
                            author_data = self.client.get_author_details(author_id)
                            if not author_data:
                                continue

                            # Check if meets criteria
                            h_index = author_data.get("hIndex", 0)
                            paper_count = author_data.get("paperCount", 0)

                            min_h = self.config["selection"]["min_h_index"]
                            min_pubs = self.config["selection"]["min_publications"]

                            if h_index >= min_h and paper_count >= min_pubs:
                                reviewer = self._process_author(author_data)
                                if reviewer:
                                    self.db.add_reviewer(reviewer)
                                    reviewers_added += 1

                                    if reviewers_added % 100 == 0:
                                        logger.info(f"Progress: {reviewers_added}/{target_count}")

                                    if reviewers_added >= target_count:
                                        return reviewers_added

                        except Exception as e:
                            logger.error(f"Error processing author {author_id}: {e}")
                            continue

        logger.info(f"\nBuild complete! Added {reviewers_added} reviewers")
        return reviewers_added

    def get_database_stats(self) -> dict:
        """Get statistics about the database"""
        return self.db.get_stats()
