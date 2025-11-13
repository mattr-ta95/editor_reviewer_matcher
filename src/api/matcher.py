"""Core matching logic for reviewer recommendations"""

import numpy as np
from typing import List, Optional, Tuple
import logging
from datetime import datetime
import uuid

from ..database.models import Reviewer, ManuscriptQuery, ReviewerRecommendation
from ..database.operations import ReviewerDatabase
from ..data.embedding_generator import SPECTER2EmbeddingGenerator
from ..search.vector_index import FAISSVectorIndex
from ..search.ranker import ReviewerRanker
from ..conflicts.detector import ConflictDetector

logger = logging.getLogger(__name__)


class ReviewerMatcher:
    """Main class for matching manuscripts to reviewers"""

    def __init__(
        self,
        db_path: str,
        index_path: str,
        model_name: str = "allenai/specter2_base",
        config: Optional[dict] = None
    ):
        """
        Initialize matcher

        Args:
            db_path: Path to reviewer database
            index_path: Path to FAISS index
            model_name: SPECTER2 model name
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        logger.info("Loading database...")
        self.db = ReviewerDatabase(db_path)

        logger.info("Loading embedding generator...")
        self.embedding_generator = SPECTER2EmbeddingGenerator(model_name=model_name)

        logger.info("Loading vector index...")
        self.vector_index = FAISSVectorIndex()
        try:
            self.vector_index.load(index_path)
        except FileNotFoundError:
            logger.warning(f"Index not found at {index_path}. Will need to build index.")

        # Initialize ranker
        ranking_config = self.config.get("ranking", {})
        self.ranker = ReviewerRanker(
            similarity_weight=ranking_config.get("similarity_weight", 0.70),
            recency_weight=ranking_config.get("recency_weight", 0.15),
            velocity_weight=ranking_config.get("velocity_weight", 0.10),
            diversity_weight=ranking_config.get("diversity_weight", 0.05),
            min_similarity=ranking_config.get("min_similarity_threshold", 0.50)
        )

        # Initialize conflict detector
        conflicts_config = self.config.get("conflicts", {})
        self.conflict_detector = ConflictDetector(
            recent_collaboration_years=conflicts_config.get("recent_collaboration_years", 3),
            institution_history_years=conflicts_config.get("institution_history_years", 5)
        )

        logger.info("Matcher initialized successfully")

    def search(
        self,
        abstract: str,
        title: Optional[str] = None,
        authors: Optional[List[dict]] = None,
        field: Optional[str] = None,
        min_h_index: int = 5,
        max_results: int = 20,
        include_conflicts: bool = False
    ) -> dict:
        """
        Search for matching reviewers

        Args:
            abstract: Manuscript abstract
            title: Manuscript title
            authors: List of author dictionaries
            field: Field of study
            min_h_index: Minimum h-index
            max_results: Maximum number of results
            include_conflicts: Include reviewers with conflicts

        Returns:
            Dictionary with search results and metadata
        """
        start_time = datetime.now()

        # Generate query ID
        query_id = str(uuid.uuid4())

        # Generate embedding for query
        logger.info("Generating query embedding...")
        query_text = f"{title} [SEP] {abstract}" if title else abstract
        query_embedding = self.embedding_generator.encode(query_text, normalize=True)

        # Search vector index
        logger.info("Searching vector index...")
        k = min(100, self.vector_index.index.ntotal) if self.vector_index.index else 100
        reviewer_ids, similarity_scores = self.vector_index.search(query_embedding, k=k)

        # Retrieve reviewers from database
        logger.info(f"Retrieving {len(reviewer_ids)} candidate reviewers...")
        reviewers = []
        valid_scores = []
        for reviewer_id, score in zip(reviewer_ids, similarity_scores):
            reviewer = self.db.get_reviewer(reviewer_id)
            if reviewer:
                reviewers.append(reviewer)
                valid_scores.append(score)

        total_candidates = len(reviewers)

        # Filter by criteria
        logger.info("Filtering by criteria...")
        reviewers = self.ranker.filter_by_criteria(
            reviewers,
            min_h_index=min_h_index,
            min_publications=self.config.get("filtering", {}).get("min_publications", 3),
            max_inactivity_years=self.config.get("filtering", {}).get("max_inactivity_years", 3)
        )

        # Update scores to match filtered reviewers
        valid_scores = valid_scores[:len(reviewers)]

        # Create manuscript query for conflict detection
        manuscript = ManuscriptQuery(
            title=title or "",
            abstract=abstract,
            authors=authors or []
        )

        # Detect conflicts
        logger.info("Detecting conflicts...")
        if not include_conflicts:
            non_conflicted, conflicted = self.conflict_detector.filter_conflicted_reviewers(
                manuscript,
                reviewers,
                exclude_high_severity=True,
                exclude_medium_severity=False
            )
            reviewers = non_conflicted
            # Update scores
            valid_scores = valid_scores[:len(reviewers)]
            conflicts_detected = len(conflicted)
        else:
            conflicts_detected = 0

        # Rank reviewers
        logger.info("Ranking reviewers...")
        recommendations = self.ranker.rank_reviewers(
            reviewers,
            valid_scores,
            max_results=max_results
        )

        # Add conflict info to each recommendation
        for rec in recommendations:
            conflict_info = self.conflict_detector.detect_conflicts(manuscript, rec.reviewer)
            rec.conflict_info = conflict_info

        # Calculate statistics
        match_scores = [rec.match_score for rec in recommendations]
        h_indices = [rec.reviewer.h_index for rec in recommendations]
        institutions = set()
        for rec in recommendations:
            aff = rec.reviewer.get_current_affiliation()
            if aff:
                institutions.add(aff.institution)

        # Calculate processing time
        processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Build response
        response = {
            "query_id": query_id,
            "results": recommendations,
            "metadata": {
                "total_candidates": total_candidates,
                "filtered_count": len(reviewers),
                "conflicts_detected": conflicts_detected,
                "processing_time_ms": processing_time_ms,
                "model_version": "specter2_base"
            },
            "statistics": {
                "avg_match_score": float(np.mean(match_scores)) if match_scores else 0.0,
                "avg_h_index": float(np.mean(h_indices)) if h_indices else 0.0,
                "institutions_represented": len(institutions)
            }
        }

        logger.info(f"Search complete in {processing_time_ms}ms. Found {len(recommendations)} recommendations")

        return response

    def get_reviewer_details(self, reviewer_id: str) -> Optional[dict]:
        """
        Get detailed information about a reviewer

        Args:
            reviewer_id: Reviewer ID

        Returns:
            Reviewer details dictionary or None
        """
        reviewer = self.db.get_reviewer(reviewer_id)
        if not reviewer:
            return None

        # Extract coauthors from papers
        coauthors = set()
        for paper in reviewer.recent_papers:
            coauthors.update(paper.authors)
        coauthors.discard(reviewer.name)  # Remove self

        # Extract expertise keywords from fields and paper titles
        keywords = set(reviewer.fields)
        for paper in reviewer.recent_papers[:10]:  # Top 10 papers
            # Simple keyword extraction from titles
            title_words = paper.title.lower().split()
            keywords.update([w for w in title_words if len(w) > 5])

        return {
            "reviewer_id": reviewer.reviewer_id,
            "name": reviewer.name,
            "affiliations": [
                {
                    "institution": aff.institution,
                    "department": aff.department,
                    "country": aff.country
                }
                for aff in reviewer.affiliations
            ],
            "h_index": reviewer.h_index,
            "publication_count": reviewer.publication_count,
            "citation_count": reviewer.citation_count,
            "fields": reviewer.fields,
            "recent_papers": [
                {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "citations": paper.citations,
                    "doi": paper.doi,
                    "url": paper.url
                }
                for paper in reviewer.recent_papers[:20]
            ],
            "coauthors": sorted(list(coauthors))[:10],
            "expertise_keywords": sorted(list(keywords))[:20]
        }

    def get_stats(self) -> dict:
        """Get system statistics"""
        db_stats = self.db.get_stats()
        index_stats = self.vector_index.get_stats()

        # Estimate index size (rough estimate)
        index_size_mb = 0.0
        if index_stats.get("total_vectors"):
            # 768 dimensions * 4 bytes (float32) * number of vectors / 1024^2
            index_size_mb = (768 * 4 * index_stats["total_vectors"]) / (1024 ** 2)

        return {
            "total_reviewers": db_stats["total_reviewers"],
            "active_reviewers": db_stats["active_reviewers"],
            "fields_covered": self.config.get("database", {}).get("fields", []),
            "avg_h_index": db_stats["avg_h_index"],
            "total_papers_indexed": db_stats["total_papers"],
            "index_size_mb": round(index_size_mb, 2),
            "last_updated": datetime.now().isoformat()
        }
