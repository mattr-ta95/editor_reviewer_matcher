"""FastAPI application for Semantic Reviewer Matching System"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime
import time

from .schemas import (
    SearchRequest,
    SearchResponse,
    ReviewerDetailsResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    ReviewerRecommendationResponse,
    ReviewerInfo,
    PaperResponse,
    ConflictResponse,
    SearchMetadata,
    SearchStatistics
)
from .matcher import ReviewerMatcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.getenv("CONFIG_PATH", "config/config.yaml")
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.warning(f"Config file not found at {config_path}, using defaults")
    config = {}

# Initialize FastAPI app
app = FastAPI(
    title=config.get("api", {}).get("title", "Semantic Reviewer Matching System"),
    version=config.get("api", {}).get("version", "1.0.0"),
    description=config.get("api", {}).get("description", "REST API for matching manuscripts to peer reviewers"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Application state
app_start_time = time.time()
matcher = None


@app.on_event("startup")
async def startup_event():
    """Initialize matcher on startup"""
    global matcher

    try:
        db_path = os.getenv("DATABASE_PATH", "data/reviewers.db")
        index_path = os.getenv("FAISS_INDEX_PATH", "data/indices/specter2.index")
        model_name = os.getenv("MODEL_NAME", "allenai/specter2_base")

        logger.info("Initializing reviewer matcher...")
        matcher = ReviewerMatcher(
            db_path=db_path,
            index_path=index_path,
            model_name=model_name,
            config=config
        )
        logger.info("Matcher initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize matcher: {e}")
        matcher = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Semantic Reviewer Matching System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Search"],
    status_code=status.HTTP_200_OK
)
async def search_reviewers(request: SearchRequest):
    """
    Search for matching reviewers based on manuscript abstract

    - **abstract**: Manuscript abstract (100-5000 chars, required)
    - **title**: Manuscript title (optional)
    - **authors**: List of manuscript authors (optional)
    - **field**: Field of study (optional)
    - **min_h_index**: Minimum h-index (default: 5)
    - **max_results**: Maximum number of results (default: 20, max: 50)
    - **include_conflicts**: Include reviewers with conflicts (default: false)
    """
    if matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matcher not initialized. System may still be starting up."
        )

    try:
        # Convert authors to dict list
        authors = [author.dict() for author in request.authors] if request.authors else []

        # Perform search
        result = matcher.search(
            abstract=request.abstract,
            title=request.title,
            authors=authors,
            field=request.field,
            min_h_index=request.min_h_index,
            max_results=request.max_results,
            include_conflicts=request.include_conflicts
        )

        # Convert recommendations to response format
        recommendations = []
        for rec in result["results"]:
            reviewer_info = ReviewerInfo(
                id=rec.reviewer.reviewer_id,
                name=rec.reviewer.name,
                affiliation=rec.reviewer.get_current_affiliation().institution
                    if rec.reviewer.get_current_affiliation() else "Unknown",
                h_index=rec.reviewer.h_index,
                publication_count=rec.reviewer.publication_count,
                fields=rec.reviewer.fields
            )

            expertise_evidence = [
                PaperResponse(
                    paper_title=paper.title,
                    year=paper.year,
                    venue=paper.venue or "",
                    citations=paper.citations
                )
                for paper in rec.expertise_evidence
            ]

            conflict_info = None
            if rec.conflict_info and rec.conflict_info.has_conflict:
                conflict_info = ConflictResponse(
                    has_conflict=rec.conflict_info.has_conflict,
                    type=rec.conflict_info.conflict_type,
                    severity=rec.conflict_info.severity,
                    evidence=rec.conflict_info.evidence,
                    year=rec.conflict_info.year
                )

            recommendations.append(
                ReviewerRecommendationResponse(
                    rank=rec.rank,
                    reviewer=reviewer_info,
                    match_score=rec.match_score,
                    similarity_score=rec.similarity_score,
                    recency_score=rec.recency_score,
                    velocity_score=rec.velocity_score,
                    diversity_bonus=rec.diversity_bonus,
                    expertise_evidence=expertise_evidence,
                    conflict_info=conflict_info
                )
            )

        response = SearchResponse(
            query_id=result["query_id"],
            results=recommendations,
            metadata=SearchMetadata(**result["metadata"]),
            statistics=SearchStatistics(**result["statistics"])
        )

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/api/v1/reviewers/{reviewer_id}",
    response_model=ReviewerDetailsResponse,
    tags=["Reviewers"]
)
async def get_reviewer_details(reviewer_id: str):
    """
    Get detailed information about a specific reviewer

    - **reviewer_id**: Semantic Scholar author ID or reviewer ID
    """
    if matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matcher not initialized"
        )

    try:
        details = matcher.get_reviewer_details(reviewer_id)

        if details is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reviewer not found: {reviewer_id}"
            )

        return ReviewerDetailsResponse(**details)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving reviewer details: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint

    Returns system status and component health
    """
    uptime = int(time.time() - app_start_time)

    if matcher is None:
        return HealthResponse(
            status="unhealthy",
            database_status="disconnected",
            index_loaded=False,
            model_loaded=False,
            reviewer_count=0,
            uptime_seconds=uptime
        )

    try:
        # Check database
        db_stats = matcher.db.get_stats()
        db_connected = db_stats["total_reviewers"] > 0

        # Check index
        index_loaded = matcher.vector_index.index is not None

        # Check model
        model_loaded = matcher.embedding_generator.model is not None

        status_str = "healthy" if (db_connected and index_loaded and model_loaded) else "degraded"

        return HealthResponse(
            status=status_str,
            database_status="connected" if db_connected else "disconnected",
            index_loaded=index_loaded,
            model_loaded=model_loaded,
            reviewer_count=db_stats["total_reviewers"],
            last_updated=datetime.now().isoformat(),
            uptime_seconds=uptime
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            database_status="error",
            index_loaded=False,
            model_loaded=False,
            reviewer_count=0,
            uptime_seconds=uptime
        )


@app.get(
    "/api/v1/stats",
    response_model=StatsResponse,
    tags=["System"]
)
async def get_stats():
    """
    Get database and system statistics

    Returns information about reviewers, papers, and index
    """
    if matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Matcher not initialized"
        )

    try:
        stats = matcher.get_stats()
        return StatsResponse(**stats)

    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
