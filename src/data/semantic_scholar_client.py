"""Semantic Scholar API client for retrieving author and paper data"""

import requests
import time
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """Client for interacting with Semantic Scholar API"""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 100):
        """
        Initialize Semantic Scholar client

        Args:
            api_key: Optional API key for higher rate limits
            rate_limit: Requests per second (default 100 for free tier)
        """
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({"x-api-key": api_key})

    def _rate_limit_wait(self):
        """Implement rate limiting"""
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to the API with error handling"""
        self._rate_limit_wait()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            elif response.status_code == 404:
                logger.debug(f"Resource not found: {endpoint}")
                return None
            else:
                logger.error(f"HTTP error: {e}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def search_authors(self, query: str, fields: List[str], limit: int = 100) -> List[Dict]:
        """
        Search for authors in specific fields

        Args:
            query: Search query (e.g., field name or keywords)
            fields: List of field codes (e.g., ['Computer Science'])
            limit: Maximum number of results

        Returns:
            List of author data dictionaries
        """
        endpoint = "author/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex"
        }

        if fields:
            params["fields"] += ",papers.title,papers.year,papers.citationCount"

        data = self._make_request(endpoint, params)

        if data and "data" in data:
            return data["data"]
        return []

    def get_author_details(self, author_id: str) -> Optional[Dict]:
        """
        Get detailed information about an author

        Args:
            author_id: Semantic Scholar author ID

        Returns:
            Author data dictionary
        """
        endpoint = f"author/{author_id}"
        params = {
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex,papers,papers.title,papers.abstract,papers.year,papers.venue,papers.citationCount,papers.authors,papers.fieldsOfStudy"
        }

        return self._make_request(endpoint, params)

    def get_author_papers(
        self,
        author_id: str,
        limit: int = 50,
        offset: int = 0,
        year_min: Optional[int] = None
    ) -> List[Dict]:
        """
        Get papers by an author

        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers to retrieve
            offset: Offset for pagination
            year_min: Minimum publication year

        Returns:
            List of paper dictionaries
        """
        endpoint = f"author/{author_id}/papers"
        params = {
            "limit": limit,
            "offset": offset,
            "fields": "paperId,title,abstract,year,venue,citationCount,authors,fieldsOfStudy,externalIds"
        }

        if year_min:
            params["year"] = f"{year_min}-"

        data = self._make_request(endpoint, params)

        if data and "data" in data:
            return data["data"]
        return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """
        Get detailed information about a paper

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Paper data dictionary
        """
        endpoint = f"paper/{paper_id}"
        params = {
            "fields": "paperId,title,abstract,year,venue,citationCount,authors,fieldsOfStudy,externalIds"
        }

        return self._make_request(endpoint, params)

    def search_papers(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 100,
        year_min: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for papers

        Args:
            query: Search query
            fields: Filter by fields of study
            limit: Maximum number of results
            year_min: Minimum publication year

        Returns:
            List of paper dictionaries
        """
        endpoint = "paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,year,venue,citationCount,authors,fieldsOfStudy"
        }

        if fields:
            params["fieldsOfStudy"] = ",".join(fields)

        if year_min:
            params["year"] = f"{year_min}-"

        data = self._make_request(endpoint, params)

        if data and "data" in data:
            return data["data"]
        return []

    def get_top_authors_by_field(
        self,
        field: str,
        min_h_index: int = 10,
        min_papers: int = 5,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get top authors in a specific field

        Args:
            field: Field of study (e.g., 'machine learning', 'bioinformatics')
            min_h_index: Minimum h-index
            min_papers: Minimum number of papers
            limit: Maximum number of authors

        Returns:
            List of author dictionaries
        """
        # Search for highly cited papers in the field
        papers = self.search_papers(
            query=field,
            fields=[field],
            limit=limit
        )

        # Extract unique authors
        authors_dict = {}
        for paper in papers:
            if "authors" in paper:
                for author in paper["authors"]:
                    author_id = author.get("authorId")
                    if author_id and author_id not in authors_dict:
                        authors_dict[author_id] = author

        logger.info(f"Found {len(authors_dict)} unique authors in field '{field}'")

        # Get detailed info for each author
        detailed_authors = []
        for i, author_id in enumerate(authors_dict.keys()):
            if i >= limit:
                break

            author_details = self.get_author_details(author_id)
            if author_details:
                h_index = author_details.get("hIndex", 0)
                paper_count = author_details.get("paperCount", 0)

                if h_index >= min_h_index and paper_count >= min_papers:
                    detailed_authors.append(author_details)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(authors_dict)} authors")

        return detailed_authors


def extract_author_info(author_data: Dict) -> Dict:
    """
    Extract and normalize author information from S2 API response

    Args:
        author_data: Raw author data from Semantic Scholar

    Returns:
        Normalized author dictionary
    """
    affiliations = []
    if "affiliations" in author_data and author_data["affiliations"]:
        for aff in author_data["affiliations"]:
            if isinstance(aff, dict):
                affiliations.append({
                    "institution": aff.get("name", "Unknown"),
                    "department": None,
                    "country": None,
                    "start_date": None,
                    "end_date": None
                })
            elif isinstance(aff, str):
                affiliations.append({
                    "institution": aff,
                    "department": None,
                    "country": None,
                    "start_date": None,
                    "end_date": None
                })

    return {
        "reviewer_id": author_data.get("authorId", ""),
        "name": author_data.get("name", "Unknown"),
        "affiliations": affiliations,
        "h_index": author_data.get("hIndex", 0),
        "publication_count": author_data.get("paperCount", 0),
        "citation_count": author_data.get("citationCount", 0),
        "papers": author_data.get("papers", [])
    }


def extract_paper_info(paper_data: Dict) -> Dict:
    """
    Extract and normalize paper information from S2 API response

    Args:
        paper_data: Raw paper data from Semantic Scholar

    Returns:
        Normalized paper dictionary
    """
    authors = []
    if "authors" in paper_data and paper_data["authors"]:
        authors = [author.get("name", "Unknown") for author in paper_data["authors"]]

    fields = []
    if "fieldsOfStudy" in paper_data and paper_data["fieldsOfStudy"]:
        fields = paper_data["fieldsOfStudy"]

    # Get DOI if available
    doi = None
    if "externalIds" in paper_data and paper_data["externalIds"]:
        doi = paper_data["externalIds"].get("DOI")

    return {
        "paper_id": paper_data.get("paperId", ""),
        "title": paper_data.get("title", ""),
        "abstract": paper_data.get("abstract", ""),
        "year": paper_data.get("year", 0),
        "venue": paper_data.get("venue", ""),
        "citations": paper_data.get("citationCount", 0),
        "fields": fields,
        "authors": authors,
        "doi": doi,
        "url": f"https://www.semanticscholar.org/paper/{paper_data.get('paperId', '')}"
    }
