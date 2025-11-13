"""Data models for reviewer and paper entities"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json


@dataclass
class Affiliation:
    """Represents an institutional affiliation"""
    institution: str
    department: Optional[str] = None
    country: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None  # None means current position

    def to_dict(self) -> dict:
        return {
            "institution": self.institution,
            "department": self.department,
            "country": self.country,
            "start_date": self.start_date,
            "end_date": self.end_date
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Affiliation':
        return cls(**data)


@dataclass
class Paper:
    """Represents a scientific paper"""
    paper_id: str
    title: str
    abstract: str
    year: int
    venue: Optional[str] = None
    citations: int = 0
    fields: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "citations": self.citations,
            "fields": self.fields,
            "authors": self.authors,
            "doi": self.doi,
            "url": self.url
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Paper':
        return cls(**data)


@dataclass
class Reviewer:
    """Represents a potential peer reviewer"""
    reviewer_id: str
    name: str
    affiliations: List[Affiliation] = field(default_factory=list)
    email: Optional[str] = None
    orcid: Optional[str] = None
    h_index: int = 0
    publication_count: int = 0
    citation_count: int = 0
    fields: List[str] = field(default_factory=list)
    active: bool = True
    recent_papers: List[Paper] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert reviewer to dictionary for storage"""
        return {
            "reviewer_id": self.reviewer_id,
            "name": self.name,
            "affiliations": [aff.to_dict() for aff in self.affiliations],
            "email": self.email,
            "orcid": self.orcid,
            "h_index": self.h_index,
            "publication_count": self.publication_count,
            "citation_count": self.citation_count,
            "fields": self.fields,
            "active": self.active,
            "recent_papers": [paper.to_dict() for paper in self.recent_papers],
            "embedding": self.embedding,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Reviewer':
        """Create reviewer from dictionary"""
        data = data.copy()

        # Convert affiliations
        if "affiliations" in data:
            data["affiliations"] = [Affiliation.from_dict(aff) for aff in data["affiliations"]]

        # Convert papers
        if "recent_papers" in data:
            data["recent_papers"] = [Paper.from_dict(paper) for paper in data["recent_papers"]]

        # Convert datetime
        if "last_updated" in data and data["last_updated"]:
            if isinstance(data["last_updated"], str):
                data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        return cls(**data)

    def get_current_affiliation(self) -> Optional[Affiliation]:
        """Get the current (most recent) affiliation"""
        current = [aff for aff in self.affiliations if aff.end_date is None]
        return current[0] if current else None

    def has_published_recently(self, years: int = 3) -> bool:
        """Check if reviewer has published in the last N years"""
        from datetime import datetime
        current_year = datetime.now().year
        return any(paper.year >= current_year - years for paper in self.recent_papers)


@dataclass
class ManuscriptQuery:
    """Represents a manuscript query for reviewer matching"""
    manuscript_id: Optional[str] = None
    title: str = ""
    abstract: str = ""
    authors: List[dict] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    field: Optional[str] = None
    submission_date: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "manuscript_id": self.manuscript_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "keywords": self.keywords,
            "field": self.field,
            "submission_date": self.submission_date.isoformat() if self.submission_date else None
        }


@dataclass
class ConflictInfo:
    """Represents conflict of interest information"""
    has_conflict: bool
    conflict_type: Optional[str] = None
    severity: str = "low"
    evidence: Optional[str] = None
    year: Optional[int] = None
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "has_conflict": self.has_conflict,
            "type": self.conflict_type,
            "severity": self.severity,
            "evidence": self.evidence,
            "year": self.year,
            "details": self.details
        }


@dataclass
class ReviewerRecommendation:
    """Represents a reviewer recommendation with scoring"""
    rank: int
    reviewer: Reviewer
    match_score: float
    similarity_score: float
    recency_score: float
    velocity_score: float = 0.0
    diversity_bonus: float = 0.0
    expertise_evidence: List[Paper] = field(default_factory=list)
    conflict_info: Optional[ConflictInfo] = None

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "reviewer": {
                "id": self.reviewer.reviewer_id,
                "name": self.reviewer.name,
                "affiliation": self.reviewer.get_current_affiliation().institution
                    if self.reviewer.get_current_affiliation() else "Unknown",
                "h_index": self.reviewer.h_index,
                "publication_count": self.reviewer.publication_count,
                "fields": self.reviewer.fields
            },
            "match_score": round(self.match_score, 3),
            "similarity_score": round(self.similarity_score, 3),
            "recency_score": round(self.recency_score, 3),
            "velocity_score": round(self.velocity_score, 3),
            "diversity_bonus": round(self.diversity_bonus, 3),
            "expertise_evidence": [paper.to_dict() for paper in self.expertise_evidence[:5]],
            "conflict_info": self.conflict_info.to_dict() if self.conflict_info else None
        }
