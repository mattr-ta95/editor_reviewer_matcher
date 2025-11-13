"""Tests for conflict detection"""

import pytest
from src.conflicts.detector import ConflictDetector
from src.database.models import Reviewer, ManuscriptQuery, Paper, Affiliation


@pytest.fixture
def detector():
    """Create conflict detector"""
    return ConflictDetector()


def test_name_normalization(detector):
    """Test name normalization"""
    assert detector._normalize_name("Dr. John Smith") == "john smith"
    assert detector._normalize_name("Prof. Jane Doe") == "jane doe"
    assert detector._normalize_name("J. Smith, PhD") == "j smith phd"


def test_institution_normalization(detector):
    """Test institution normalization"""
    mit1 = detector._normalize_institution("Massachusetts Institute of Technology")
    mit2 = detector._normalize_institution("MIT")

    # Both should be normalized similarly
    assert "inst" in mit1 or "tech" in mit1


def test_names_match(detector):
    """Test name matching"""
    # Exact match
    assert detector._names_match("john smith", "john smith")

    # Initials
    assert detector._names_match("j smith", "john smith")

    # Different names
    assert not detector._names_match("john smith", "jane doe")


def test_coauthorship_detection(detector):
    """Test co-authorship conflict detection"""
    manuscript = ManuscriptQuery(
        title="Test Paper",
        abstract="Test abstract",
        authors=[
            {"name": "John Doe", "affiliation": "MIT"},
            {"name": "Jane Smith", "affiliation": "Stanford"}
        ]
    )

    # Reviewer who co-authored with manuscript author
    reviewer = Reviewer(
        reviewer_id="test1",
        name="Alice Johnson",
        recent_papers=[
            Paper(
                paper_id="p1",
                title="Previous Work",
                abstract="Abstract",
                year=2022,
                authors=["Alice Johnson", "John Doe"]
            )
        ]
    )

    conflict = detector.detect_conflicts(manuscript, reviewer)
    assert conflict.has_conflict
    assert conflict.conflict_type == "coauthorship"


def test_institution_conflict(detector):
    """Test institutional conflict detection"""
    manuscript = ManuscriptQuery(
        title="Test Paper",
        abstract="Test abstract",
        authors=[
            {"name": "John Doe", "affiliation": "Stanford University"}
        ]
    )

    reviewer = Reviewer(
        reviewer_id="test2",
        name="Alice Johnson",
        affiliations=[
            Affiliation(
                institution="Stanford University",
                end_date=None  # Current position
            )
        ],
        recent_papers=[]
    )

    conflict = detector.detect_conflicts(manuscript, reviewer)
    assert conflict.has_conflict
    assert conflict.conflict_type == "same_institution"


def test_no_conflict(detector):
    """Test when there is no conflict"""
    manuscript = ManuscriptQuery(
        title="Test Paper",
        abstract="Test abstract",
        authors=[
            {"name": "John Doe", "affiliation": "MIT"}
        ]
    )

    reviewer = Reviewer(
        reviewer_id="test3",
        name="Alice Johnson",
        affiliations=[
            Affiliation(institution="Stanford University")
        ],
        recent_papers=[
            Paper(
                paper_id="p1",
                title="Different Work",
                abstract="Abstract",
                year=2022,
                authors=["Alice Johnson", "Bob Brown"]
            )
        ]
    )

    conflict = detector.detect_conflicts(manuscript, reviewer)
    assert not conflict.has_conflict
