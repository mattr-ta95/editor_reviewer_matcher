"""Ranking algorithm for reviewer recommendations"""

import numpy as np
from typing import List, Dict
from datetime import datetime
import logging

from ..database.models import Reviewer, ReviewerRecommendation, Paper

logger = logging.getLogger(__name__)


class ReviewerRanker:
    """Ranks reviewers based on multiple factors"""

    def __init__(
        self,
        similarity_weight: float = 0.70,
        recency_weight: float = 0.15,
        velocity_weight: float = 0.10,
        diversity_weight: float = 0.05,
        min_similarity: float = 0.50
    ):
        """
        Initialize ranker with weights

        Args:
            similarity_weight: Weight for semantic similarity
            recency_weight: Weight for publication recency
            velocity_weight: Weight for publication velocity
            diversity_weight: Weight for diversity bonus
            min_similarity: Minimum similarity threshold
        """
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.velocity_weight = velocity_weight
        self.diversity_weight = diversity_weight
        self.min_similarity = min_similarity

        # Validate weights sum to 1.0
        total = similarity_weight + recency_weight + velocity_weight + diversity_weight
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, not 1.0. Normalizing...")
            self.similarity_weight /= total
            self.recency_weight /= total
            self.velocity_weight /= total
            self.diversity_weight /= total

    def calculate_recency_score(self, reviewer: Reviewer) -> float:
        """
        Calculate publication recency score

        Papers in last 12 months: score = 1.0
        Papers in last 24 months: score = 0.7
        Papers in last 36 months: score = 0.4
        Older: score = 0.0

        Args:
            reviewer: Reviewer object

        Returns:
            Recency score (0-1)
        """
        if not reviewer.recent_papers:
            return 0.0

        current_year = datetime.now().year

        # Get most recent publication year
        most_recent_year = max(paper.year for paper in reviewer.recent_papers)
        years_ago = current_year - most_recent_year

        if years_ago <= 1:
            return 1.0
        elif years_ago <= 2:
            return 0.7
        elif years_ago <= 3:
            return 0.4
        else:
            return 0.0

    def calculate_velocity_score(self, reviewer: Reviewer, years: int = 3) -> float:
        """
        Calculate publication velocity score

        10+ papers/year: score = 1.0
        5-9 papers/year: score = 0.7
        2-4 papers/year: score = 0.4
        <2 papers/year: score = 0.2

        Args:
            reviewer: Reviewer object
            years: Number of years to consider

        Returns:
            Velocity score (0-1)
        """
        if not reviewer.recent_papers:
            return 0.2

        current_year = datetime.now().year
        recent_papers = [
            p for p in reviewer.recent_papers
            if p.year >= current_year - years
        ]

        papers_per_year = len(recent_papers) / years

        if papers_per_year >= 10:
            return 1.0
        elif papers_per_year >= 5:
            return 0.7
        elif papers_per_year >= 2:
            return 0.4
        else:
            return 0.2

    def calculate_diversity_bonus(
        self,
        reviewer: Reviewer,
        existing_recommendations: List[Reviewer]
    ) -> float:
        """
        Calculate diversity bonus based on institution diversity

        Args:
            reviewer: Reviewer being evaluated
            existing_recommendations: List of already recommended reviewers

        Returns:
            Diversity bonus (0-1)
        """
        if not existing_recommendations:
            return 0.0

        # Get current institution
        current_aff = reviewer.get_current_affiliation()
        if not current_aff:
            return 0.5  # Neutral if no affiliation

        reviewer_institution = current_aff.institution.lower()

        # Count how many existing recommendations share institution
        same_institution = 0
        for rec in existing_recommendations:
            rec_aff = rec.get_current_affiliation()
            if rec_aff and rec_aff.institution.lower() == reviewer_institution:
                same_institution += 1

        # Apply penalty if overrepresented
        if len(existing_recommendations) >= 10:
            proportion = same_institution / len(existing_recommendations)
            if proportion > 0.3:  # More than 30% from same institution
                return -0.5
            elif proportion > 0.2:
                return -0.2

        return 0.0

    def calculate_final_score(
        self,
        similarity_score: float,
        recency_score: float,
        velocity_score: float,
        diversity_bonus: float
    ) -> float:
        """
        Calculate final ranking score

        Args:
            similarity_score: Semantic similarity score
            recency_score: Publication recency score
            velocity_score: Publication velocity score
            diversity_bonus: Diversity bonus/penalty

        Returns:
            Final weighted score
        """
        score = (
            similarity_score * self.similarity_weight +
            recency_score * self.recency_weight +
            velocity_score * self.velocity_weight +
            diversity_bonus * self.diversity_weight
        )
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def rank_reviewers(
        self,
        reviewers: List[Reviewer],
        similarity_scores: List[float],
        max_results: int = 20
    ) -> List[ReviewerRecommendation]:
        """
        Rank reviewers and create recommendations

        Args:
            reviewers: List of candidate reviewers
            similarity_scores: Corresponding similarity scores
            max_results: Maximum number of results

        Returns:
            List of ranked ReviewerRecommendation objects
        """
        if len(reviewers) != len(similarity_scores):
            raise ValueError("Number of reviewers must match number of similarity scores")

        # Filter by minimum similarity
        candidates = [
            (reviewer, score)
            for reviewer, score in zip(reviewers, similarity_scores)
            if score >= self.min_similarity
        ]

        if not candidates:
            logger.warning("No candidates meet minimum similarity threshold")
            return []

        # Calculate scores for all candidates
        recommendations = []
        existing_recommendations = []

        for reviewer, sim_score in candidates:
            # Calculate component scores
            recency = self.calculate_recency_score(reviewer)
            velocity = self.calculate_velocity_score(reviewer)
            diversity = self.calculate_diversity_bonus(reviewer, existing_recommendations)

            # Calculate final score
            final_score = self.calculate_final_score(sim_score, recency, velocity, diversity)

            # Get expertise evidence (top papers by citation)
            evidence_papers = sorted(
                reviewer.recent_papers,
                key=lambda p: p.citations,
                reverse=True
            )[:5]

            # Create recommendation
            rec = ReviewerRecommendation(
                rank=0,  # Will be set after sorting
                reviewer=reviewer,
                match_score=final_score,
                similarity_score=sim_score,
                recency_score=recency,
                velocity_score=velocity,
                diversity_bonus=diversity,
                expertise_evidence=evidence_papers
            )

            recommendations.append(rec)
            existing_recommendations.append(reviewer)

        # Sort by match score
        recommendations.sort(key=lambda x: x.match_score, reverse=True)

        # Assign ranks and limit results
        for i, rec in enumerate(recommendations[:max_results], 1):
            rec.rank = i

        return recommendations[:max_results]

    def filter_by_criteria(
        self,
        reviewers: List[Reviewer],
        min_h_index: int = 5,
        min_publications: int = 3,
        max_inactivity_years: int = 3,
        require_recent: bool = True
    ) -> List[Reviewer]:
        """
        Filter reviewers by quality criteria

        Args:
            reviewers: List of reviewers to filter
            min_h_index: Minimum h-index
            min_publications: Minimum publication count
            max_inactivity_years: Maximum years since last publication
            require_recent: Require publications in last N years

        Returns:
            Filtered list of reviewers
        """
        current_year = datetime.now().year
        filtered = []

        for reviewer in reviewers:
            # Check h-index
            if reviewer.h_index < min_h_index:
                continue

            # Check publication count
            if reviewer.publication_count < min_publications:
                continue

            # Check activity
            if require_recent and reviewer.recent_papers:
                most_recent = max(p.year for p in reviewer.recent_papers)
                if current_year - most_recent > max_inactivity_years:
                    continue

            filtered.append(reviewer)

        return filtered
