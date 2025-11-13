"""Database operations for storing and retrieving reviewers"""

import sqlite3
import json
import numpy as np
from typing import List, Optional
from pathlib import Path
import logging

from .models import Reviewer, Affiliation, Paper

logger = logging.getLogger(__name__)


class ReviewerDatabase:
    """Handles storage and retrieval of reviewer data"""

    def __init__(self, db_path: str = "data/reviewers.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Reviewers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviewers (
                reviewer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                orcid TEXT,
                h_index INTEGER DEFAULT 0,
                publication_count INTEGER DEFAULT 0,
                citation_count INTEGER DEFAULT 0,
                fields TEXT,
                active BOOLEAN DEFAULT 1,
                affiliations TEXT,
                recent_papers TEXT,
                last_updated TEXT
            )
        """)

        # Papers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                year INTEGER,
                venue TEXT,
                citations INTEGER DEFAULT 0,
                fields TEXT,
                authors TEXT,
                doi TEXT,
                url TEXT
            )
        """)

        # Reviewer-Paper relationship
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviewer_papers (
                reviewer_id TEXT,
                paper_id TEXT,
                FOREIGN KEY (reviewer_id) REFERENCES reviewers(reviewer_id),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id),
                PRIMARY KEY (reviewer_id, paper_id)
            )
        """)

        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviewer_fields ON reviewers(fields)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviewer_h_index ON reviewers(h_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_year ON papers(year)")

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def add_reviewer(self, reviewer: Reviewer):
        """Add or update a reviewer in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO reviewers
            (reviewer_id, name, email, orcid, h_index, publication_count,
             citation_count, fields, active, affiliations, recent_papers, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reviewer.reviewer_id,
            reviewer.name,
            reviewer.email,
            reviewer.orcid,
            reviewer.h_index,
            reviewer.publication_count,
            reviewer.citation_count,
            json.dumps(reviewer.fields),
            reviewer.active,
            json.dumps([aff.to_dict() for aff in reviewer.affiliations]),
            json.dumps([paper.to_dict() for paper in reviewer.recent_papers]),
            reviewer.last_updated.isoformat() if reviewer.last_updated else None
        ))

        # Add papers
        for paper in reviewer.recent_papers:
            self.add_paper(paper)
            cursor.execute("""
                INSERT OR IGNORE INTO reviewer_papers (reviewer_id, paper_id)
                VALUES (?, ?)
            """, (reviewer.reviewer_id, paper.paper_id))

        conn.commit()
        conn.close()
        logger.debug(f"Added/updated reviewer: {reviewer.name}")

    def add_paper(self, paper: Paper):
        """Add a paper to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO papers
            (paper_id, title, abstract, year, venue, citations, fields, authors, doi, url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper.paper_id,
            paper.title,
            paper.abstract,
            paper.year,
            paper.venue,
            paper.citations,
            json.dumps(paper.fields),
            json.dumps(paper.authors),
            paper.doi,
            paper.url
        ))

        conn.commit()
        conn.close()

    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        """Retrieve a reviewer by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM reviewers WHERE reviewer_id = ?", (reviewer_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        # Parse the row
        reviewer_data = {
            "reviewer_id": row[0],
            "name": row[1],
            "email": row[2],
            "orcid": row[3],
            "h_index": row[4],
            "publication_count": row[5],
            "citation_count": row[6],
            "fields": json.loads(row[7]) if row[7] else [],
            "active": bool(row[8]),
            "affiliations": [Affiliation.from_dict(aff) for aff in json.loads(row[9])] if row[9] else [],
            "recent_papers": [Paper.from_dict(paper) for paper in json.loads(row[10])] if row[10] else [],
            "last_updated": row[11]
        }

        conn.close()
        return Reviewer.from_dict(reviewer_data)

    def get_all_reviewers(self, active_only: bool = True) -> List[Reviewer]:
        """Get all reviewers from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM reviewers"
        if active_only:
            query += " WHERE active = 1"

        cursor.execute(query)
        rows = cursor.fetchall()

        reviewers = []
        for row in rows:
            try:
                reviewer_data = {
                    "reviewer_id": row[0],
                    "name": row[1],
                    "email": row[2],
                    "orcid": row[3],
                    "h_index": row[4],
                    "publication_count": row[5],
                    "citation_count": row[6],
                    "fields": json.loads(row[7]) if row[7] else [],
                    "active": bool(row[8]),
                    "affiliations": [Affiliation.from_dict(aff) for aff in json.loads(row[9])] if row[9] else [],
                    "recent_papers": [Paper.from_dict(paper) for paper in json.loads(row[10])] if row[10] else [],
                    "last_updated": row[11]
                }
                reviewers.append(Reviewer.from_dict(reviewer_data))
            except Exception as e:
                logger.error(f"Error parsing reviewer {row[0]}: {e}")
                continue

        conn.close()
        return reviewers

    def get_stats(self) -> dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM reviewers")
        total_reviewers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM reviewers WHERE active = 1")
        active_reviewers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM papers")
        total_papers = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(h_index) FROM reviewers WHERE active = 1")
        avg_h_index = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_reviewers": total_reviewers,
            "active_reviewers": active_reviewers,
            "total_papers": total_papers,
            "avg_h_index": round(avg_h_index, 2)
        }
