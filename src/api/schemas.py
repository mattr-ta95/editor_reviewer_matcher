"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


# Request Schemas

class AuthorInput(BaseModel):
    """Author information input"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class SearchRequest(BaseModel):
    """Request schema for reviewer search"""
    abstract: str = Field(..., min_length=100, max_length=5000, description="Manuscript abstract")
    title: Optional[str] = Field(None, max_length=500, description="Manuscript title")
    authors: List[AuthorInput] = Field(default_factory=list, description="Manuscript authors")
    field: Optional[str] = Field(None, description="Field of study (e.g., 'cs.AI', 'bio.MB')")
    exclude_keywords: List[str] = Field(default_factory=list, description="Keywords to exclude")
    min_h_index: int = Field(5, ge=0, le=100, description="Minimum h-index")
    max_results: int = Field(20, ge=1, le=50, description="Maximum number of results")
    include_conflicts: bool = Field(False, description="Include reviewers with potential conflicts")

    @validator('abstract')
    def validate_abstract(cls, v):
        if len(v.strip()) < 100:
            raise ValueError("Abstract must be at least 100 characters")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "abstract": "We present a novel deep learning approach for protein structure prediction using attention-based transformers. Our model achieves state-of-the-art accuracy on the CASP14 benchmark and demonstrates improved performance on challenging protein families.",
                "title": "Transformer-based Protein Structure Prediction",
                "authors": [
                    {"name": "Jane Doe", "affiliation": "Stanford University"}
                ],
                "field": "bio.MB",
                "max_results": 10
            }
        }


# Response Schemas

class AffiliationResponse(BaseModel):
    """Affiliation information"""
    institution: str
    department: Optional[str] = None
    country: Optional[str] = None


class PaperResponse(BaseModel):
    """Paper information"""
    paper_title: str
    year: int
    venue: Optional[str] = None
    citations: int
    relevance_score: Optional[float] = None


class ConflictResponse(BaseModel):
    """Conflict of interest information"""
    has_conflict: bool
    type: Optional[str] = None
    severity: Optional[str] = None
    evidence: Optional[str] = None
    year: Optional[int] = None


class ReviewerInfo(BaseModel):
    """Basic reviewer information"""
    id: str
    name: str
    affiliation: str
    h_index: int
    publication_count: int
    fields: List[str]


class ReviewerRecommendationResponse(BaseModel):
    """Single reviewer recommendation"""
    rank: int
    reviewer: ReviewerInfo
    match_score: float
    similarity_score: float
    recency_score: float
    velocity_score: float = 0.0
    diversity_bonus: float = 0.0
    expertise_evidence: List[PaperResponse]
    conflict_info: Optional[ConflictResponse] = None


class SearchMetadata(BaseModel):
    """Search metadata"""
    total_candidates: int
    filtered_count: int
    conflicts_detected: int
    processing_time_ms: int
    model_version: str


class SearchStatistics(BaseModel):
    """Search result statistics"""
    avg_match_score: float
    avg_h_index: float
    institutions_represented: int


class SearchResponse(BaseModel):
    """Response schema for reviewer search"""
    query_id: str
    results: List[ReviewerRecommendationResponse]
    metadata: SearchMetadata
    statistics: SearchStatistics


# Reviewer Details Schemas

class DetailedPaperResponse(BaseModel):
    """Detailed paper information"""
    title: str
    abstract: Optional[str] = None
    year: int
    venue: Optional[str] = None
    citations: int
    doi: Optional[str] = None
    url: Optional[str] = None


class ReviewerDetailsResponse(BaseModel):
    """Detailed reviewer information"""
    reviewer_id: str
    name: str
    affiliations: List[AffiliationResponse]
    h_index: int
    publication_count: int
    citation_count: int
    fields: List[str]
    recent_papers: List[DetailedPaperResponse]
    coauthors: List[str]
    expertise_keywords: List[str]


# Health and Stats Schemas

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database_status: str
    index_loaded: bool
    model_loaded: bool
    reviewer_count: int
    last_updated: Optional[str] = None
    uptime_seconds: int


class StatsResponse(BaseModel):
    """Database statistics response"""
    total_reviewers: int
    active_reviewers: int
    fields_covered: List[str]
    avg_h_index: float
    total_papers_indexed: int
    index_size_mb: float
    last_updated: Optional[str] = None


# Error Schemas

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    status_code: int
