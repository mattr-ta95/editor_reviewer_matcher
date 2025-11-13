"""Conflict of interest detection for peer review"""

import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher
import logging

from ..database.models import Reviewer, ManuscriptQuery, ConflictInfo, Affiliation

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detects conflicts of interest between manuscript authors and reviewers"""

    def __init__(
        self,
        recent_collaboration_years: int = 3,
        institution_history_years: int = 5
    ):
        """
        Initialize conflict detector

        Args:
            recent_collaboration_years: Years to consider for recent collaboration
            institution_history_years: Years to consider for institutional conflicts
        """
        self.recent_collaboration_years = recent_collaboration_years
        self.institution_history_years = institution_history_years

    def detect_conflicts(
        self,
        manuscript: ManuscriptQuery,
        reviewer: Reviewer
    ) -> ConflictInfo:
        """
        Detect all types of conflicts

        Args:
            manuscript: Manuscript query with author information
            reviewer: Reviewer to check

        Returns:
            ConflictInfo object with detected conflicts
        """
        conflicts = []

        # Check co-authorship
        coauthor_conflict = self._check_coauthorship(manuscript, reviewer)
        if coauthor_conflict:
            conflicts.append(coauthor_conflict)

        # Check institutional affiliation
        institution_conflict = self._check_institution(manuscript, reviewer)
        if institution_conflict:
            conflicts.append(institution_conflict)

        # Create ConflictInfo
        if conflicts:
            # Take highest severity conflict
            highest_severity = self._get_highest_severity(conflicts)
            return ConflictInfo(
                has_conflict=True,
                conflict_type=highest_severity["type"],
                severity=highest_severity["severity"],
                evidence=highest_severity["evidence"],
                year=highest_severity.get("year"),
                details={"all_conflicts": conflicts}
            )
        else:
            return ConflictInfo(has_conflict=False)

    def _check_coauthorship(
        self,
        manuscript: ManuscriptQuery,
        reviewer: Reviewer
    ) -> Optional[Dict]:
        """
        Check for co-authorship conflicts

        Args:
            manuscript: Manuscript query
            reviewer: Reviewer to check

        Returns:
            Conflict dictionary or None
        """
        if not manuscript.authors or not reviewer.recent_papers:
            return None

        # Normalize manuscript author names
        manuscript_authors = [
            self._normalize_name(author.get("name", ""))
            for author in manuscript.authors
        ]

        # Check all reviewer papers for co-authorship
        for paper in reviewer.recent_papers:
            for paper_author in paper.authors:
                normalized_paper_author = self._normalize_name(paper_author)

                # Check if any manuscript author matches
                for ms_author in manuscript_authors:
                    if self._names_match(ms_author, normalized_paper_author):
                        return {
                            "type": "coauthorship",
                            "severity": "high",
                            "evidence": f"Co-authored '{paper.title}' ({paper.year})",
                            "year": paper.year
                        }

        return None

    def _check_institution(
        self,
        manuscript: ManuscriptQuery,
        reviewer: Reviewer
    ) -> Optional[Dict]:
        """
        Check for institutional conflicts

        Args:
            manuscript: Manuscript query
            reviewer: Reviewer to check

        Returns:
            Conflict dictionary or None
        """
        if not manuscript.authors or not reviewer.affiliations:
            return None

        # Get manuscript affiliations
        manuscript_institutions = []
        for author in manuscript.authors:
            aff = author.get("affiliation", "")
            if aff:
                manuscript_institutions.append(self._normalize_institution(aff))

        if not manuscript_institutions:
            return None

        # Get reviewer's current affiliation
        current_aff = reviewer.get_current_affiliation()
        if current_aff:
            reviewer_institution = self._normalize_institution(current_aff.institution)

            # Check for match
            for ms_inst in manuscript_institutions:
                if self._institutions_match(ms_inst, reviewer_institution):
                    return {
                        "type": "same_institution",
                        "severity": "high",
                        "evidence": f"Same institution: {current_aff.institution}"
                    }

        return None

    def _normalize_name(self, name: str) -> str:
        """
        Normalize author name for comparison

        Args:
            name: Raw name string

        Returns:
            Normalized name
        """
        # Convert to lowercase
        name = name.lower().strip()

        # Remove common titles
        titles = ["dr", "prof", "professor", "mr", "mrs", "ms", "miss"]
        for title in titles:
            name = re.sub(rf'\b{title}\.?\s*', '', name, flags=re.IGNORECASE)

        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)

        # Remove periods and commas
        name = name.replace('.', '').replace(',', '')

        return name.strip()

    def _normalize_institution(self, institution: str) -> str:
        """
        Normalize institution name for comparison

        Args:
            institution: Raw institution string

        Returns:
            Normalized institution name
        """
        # Convert to lowercase
        inst = institution.lower().strip()

        # Common abbreviations
        replacements = {
            'university': 'univ',
            'institute': 'inst',
            'technology': 'tech',
            'college': 'coll',
            '&': 'and'
        }

        for full, abbr in replacements.items():
            inst = inst.replace(full, abbr)

        # Remove common words
        remove_words = ['the', 'of', 'at', 'in']
        for word in remove_words:
            inst = re.sub(rf'\b{word}\b', '', inst)

        # Remove extra whitespace
        inst = re.sub(r'\s+', ' ', inst)

        return inst.strip()

    def _names_match(self, name1: str, name2: str, threshold: float = 0.85) -> bool:
        """
        Check if two names match using fuzzy matching

        Args:
            name1: First name (normalized)
            name2: Second name (normalized)
            threshold: Similarity threshold (0-1)

        Returns:
            True if names match
        """
        # Exact match
        if name1 == name2:
            return True

        # Check if one is substring of other (handles initials)
        if name1 in name2 or name2 in name1:
            return True

        # Fuzzy match
        similarity = SequenceMatcher(None, name1, name2).ratio()
        return similarity >= threshold

    def _institutions_match(self, inst1: str, inst2: str, threshold: float = 0.80) -> bool:
        """
        Check if two institutions match using fuzzy matching

        Args:
            inst1: First institution (normalized)
            inst2: Second institution (normalized)
            threshold: Similarity threshold (0-1)

        Returns:
            True if institutions match
        """
        # Exact match
        if inst1 == inst2:
            return True

        # Check if one is substring of other (handles department vs university)
        if inst1 in inst2 or inst2 in inst1:
            return True

        # Fuzzy match
        similarity = SequenceMatcher(None, inst1, inst2).ratio()
        return similarity >= threshold

    def _get_highest_severity(self, conflicts: List[Dict]) -> Dict:
        """
        Get conflict with highest severity

        Args:
            conflicts: List of conflict dictionaries

        Returns:
            Conflict with highest severity
        """
        severity_order = {"high": 3, "medium": 2, "low": 1}

        return max(
            conflicts,
            key=lambda c: severity_order.get(c.get("severity", "low"), 0)
        )

    def filter_conflicted_reviewers(
        self,
        manuscript: ManuscriptQuery,
        reviewers: List[Reviewer],
        exclude_high_severity: bool = True,
        exclude_medium_severity: bool = False
    ) -> tuple[List[Reviewer], List[tuple[Reviewer, ConflictInfo]]]:
        """
        Filter out reviewers with conflicts

        Args:
            manuscript: Manuscript query
            reviewers: List of reviewers to check
            exclude_high_severity: Exclude high severity conflicts
            exclude_medium_severity: Exclude medium severity conflicts

        Returns:
            Tuple of (non-conflicted reviewers, conflicted reviewers with info)
        """
        non_conflicted = []
        conflicted = []

        for reviewer in reviewers:
            conflict_info = self.detect_conflicts(manuscript, reviewer)

            if conflict_info.has_conflict:
                # Check if we should exclude
                exclude = False
                if exclude_high_severity and conflict_info.severity == "high":
                    exclude = True
                if exclude_medium_severity and conflict_info.severity == "medium":
                    exclude = True

                if exclude:
                    conflicted.append((reviewer, conflict_info))
                else:
                    non_conflicted.append(reviewer)
            else:
                non_conflicted.append(reviewer)

        logger.info(
            f"Filtered {len(reviewers)} reviewers: "
            f"{len(non_conflicted)} non-conflicted, {len(conflicted)} conflicted"
        )

        return non_conflicted, conflicted


def check_reviewer_conflict(
    manuscript_authors: List[Dict],
    reviewer_name: str,
    reviewer_papers: List[Dict],
    reviewer_affiliation: Optional[str] = None
) -> bool:
    """
    Simple function to check if reviewer has conflict with manuscript

    Args:
        manuscript_authors: List of author dictionaries
        reviewer_name: Reviewer's name
        reviewer_papers: Reviewer's papers
        reviewer_affiliation: Reviewer's current affiliation

    Returns:
        True if conflict detected
    """
    detector = ConflictDetector()

    # Create temporary objects
    from ..database.models import ManuscriptQuery, Reviewer, Paper, Affiliation

    manuscript = ManuscriptQuery(authors=manuscript_authors)

    papers = [Paper(**p) for p in reviewer_papers]
    affiliations = []
    if reviewer_affiliation:
        affiliations = [Affiliation(institution=reviewer_affiliation)]

    reviewer = Reviewer(
        reviewer_id="temp",
        name=reviewer_name,
        affiliations=affiliations,
        recent_papers=papers
    )

    conflict_info = detector.detect_conflicts(manuscript, reviewer)
    return conflict_info.has_conflict
