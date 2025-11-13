"""Streamlit web interface for Semantic Reviewer Matching System"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import json

# Page configuration
st.set_page_config(
    page_title="Semantic Reviewer Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_BASE_URL = "http://localhost:8000"


def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None


def search_reviewers(
    abstract: str,
    title: str = "",
    authors: List[Dict] = None,
    field: str = None,
    min_h_index: int = 5,
    max_results: int = 20
) -> Dict:
    """Call API to search for reviewers"""
    payload = {
        "abstract": abstract,
        "title": title,
        "authors": authors or [],
        "min_h_index": min_h_index,
        "max_results": max_results,
        "include_conflicts": False
    }

    if field:
        payload["field"] = field

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/search",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def parse_authors(author_text: str) -> List[Dict]:
    """Parse author text input into list of dictionaries"""
    if not author_text.strip():
        return []

    authors = []
    for line in author_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Parse "Name, Affiliation" format
        if ',' in line:
            parts = line.split(',', 1)
            authors.append({
                "name": parts[0].strip(),
                "affiliation": parts[1].strip() if len(parts) > 1 else ""
            })
        else:
            authors.append({"name": line.strip()})

    return authors


def display_reviewer_card(rec: Dict, rank: int):
    """Display a reviewer recommendation card"""
    reviewer = rec["reviewer"]
    match_score = rec["match_score"]
    similarity = rec["similarity_score"]
    h_index = reviewer["h_index"]

    # Color code match score
    if match_score >= 0.80:
        score_color = "🟢"
    elif match_score >= 0.65:
        score_color = "🟡"
    else:
        score_color = "⚪"

    # Display card
    with st.expander(f"**{rank}. {reviewer['name']}** {score_color} Match: {match_score:.1%}", expanded=(rank <= 3)):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**Affiliation:** {reviewer['affiliation']}")
            if reviewer.get('fields'):
                st.write(f"**Fields:** {', '.join(reviewer['fields'][:3])}")

        with col2:
            st.metric("H-Index", h_index)
            st.metric("Publications", reviewer['publication_count'])

        with col3:
            st.metric("Semantic Similarity", f"{similarity:.1%}")
            st.metric("Recency Score", f"{rec['recency_score']:.1%}")

        # Conflict warning
        if rec.get('conflict_info') and rec['conflict_info'].get('has_conflict'):
            conflict = rec['conflict_info']
            st.warning(f"⚠️ **Potential Conflict:** {conflict.get('evidence', 'Conflict detected')}")

        # Expertise evidence
        if rec.get('expertise_evidence'):
            st.write("**Representative Publications:**")
            for paper in rec['expertise_evidence'][:3]:
                st.write(f"- {paper['paper_title']} ({paper['year']}) - {paper['citations']} citations")


def create_visualizations(results: List[Dict]):
    """Create visualizations for search results"""
    if not results:
        return

    # Prepare data
    df = pd.DataFrame([
        {
            "Rank": rec["rank"],
            "Name": rec["reviewer"]["name"],
            "Match Score": rec["match_score"],
            "Similarity": rec["similarity_score"],
            "H-Index": rec["reviewer"]["h_index"],
            "Publications": rec["reviewer"]["publication_count"],
            "Institution": rec["reviewer"]["affiliation"]
        }
        for rec in results
    ])

    col1, col2 = st.columns(2)

    with col1:
        # Match score distribution
        fig1 = px.histogram(
            df,
            x="Match Score",
            nbins=20,
            title="Match Score Distribution",
            labels={"Match Score": "Match Score", "count": "Number of Reviewers"}
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # H-Index distribution
        fig2 = px.box(
            df,
            y="H-Index",
            title="H-Index Distribution",
            points="all"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Institution diversity
    institution_counts = df['Institution'].value_counts().head(10)
    if len(institution_counts) > 0:
        fig3 = px.bar(
            x=institution_counts.values,
            y=institution_counts.index,
            orientation='h',
            title="Top Institutions Represented",
            labels={"x": "Number of Reviewers", "y": "Institution"}
        )
        st.plotly_chart(fig3, use_container_width=True)


def export_to_csv(results: List[Dict]) -> str:
    """Export results to CSV"""
    data = []
    for rec in results:
        reviewer = rec["reviewer"]
        data.append({
            "Rank": rec["rank"],
            "Name": reviewer["name"],
            "Affiliation": reviewer["affiliation"],
            "H-Index": reviewer["h_index"],
            "Publications": reviewer["publication_count"],
            "Match Score": f"{rec['match_score']:.3f}",
            "Similarity Score": f"{rec['similarity_score']:.3f}",
            "Fields": ", ".join(reviewer.get("fields", []))
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# Main App
def main():
    st.title("🔍 Semantic Reviewer Matching System")
    st.markdown("Find qualified peer reviewers for your manuscript using AI-powered semantic matching")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Search Parameters")

        min_h_index = st.slider(
            "Minimum H-Index",
            min_value=0,
            max_value=50,
            value=5,
            help="Filter reviewers by minimum h-index"
        )

        max_results = st.slider(
            "Number of Results",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of reviewers to return"
        )

        field = st.selectbox(
            "Field (Optional)",
            ["", "cs.AI", "cs.LG", "cs.CL", "cs.CV", "bio.MB", "bio.GN"],
            help="Filter by specific field"
        )

        st.markdown("---")

        # Statistics
        st.header("📊 Database Stats")
        stats = get_api_stats()
        if stats:
            st.metric("Total Reviewers", stats["total_reviewers"])
            st.metric("Active Reviewers", stats["active_reviewers"])
            st.metric("Avg H-Index", f"{stats['avg_h_index']:.1f}")
        else:
            st.warning("API not available")

    # Main content
    tab1, tab2 = st.tabs(["🔍 Search", "📖 About"])

    with tab1:
        # Input form
        st.header("Manuscript Information")

        title = st.text_input(
            "Title (Optional)",
            placeholder="Enter manuscript title...",
            help="Manuscript title helps improve matching accuracy"
        )

        abstract = st.text_area(
            "Abstract *",
            height=200,
            placeholder="Paste your manuscript abstract here (minimum 100 characters)...",
            help="The abstract is the primary input for semantic matching"
        )

        abstract_length = len(abstract)
        if abstract_length > 0:
            if abstract_length < 100:
                st.warning(f"⚠️ Abstract too short: {abstract_length}/100 characters minimum")
            else:
                st.success(f"✓ Abstract length: {abstract_length} characters")

        authors_text = st.text_area(
            "Authors (Optional)",
            height=100,
            placeholder="Enter one author per line in format: Name, Affiliation\nExample:\nJohn Doe, MIT\nJane Smith, Stanford University",
            help="Authors are used for conflict of interest detection"
        )

        # Search button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            search_button = st.button("🔍 Find Reviewers", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("Clear", use_container_width=True)

        if clear_button:
            st.rerun()

        # Perform search
        if search_button:
            if len(abstract) < 100:
                st.error("❌ Abstract must be at least 100 characters long")
            else:
                authors = parse_authors(authors_text)

                with st.spinner("🔍 Searching for reviewers..."):
                    results = search_reviewers(
                        abstract=abstract,
                        title=title,
                        authors=authors,
                        field=field if field else None,
                        min_h_index=min_h_index,
                        max_results=max_results
                    )

                if results:
                    st.success(f"✅ Found {len(results['results'])} matching reviewers in {results['metadata']['processing_time_ms']}ms")

                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Candidates Evaluated", results['metadata']['total_candidates'])
                    with col2:
                        st.metric("Avg Match Score", f"{results['statistics']['avg_match_score']:.1%}")
                    with col3:
                        st.metric("Institutions", results['statistics']['institutions_represented'])

                    # Visualizations
                    st.markdown("---")
                    st.header("📊 Results Overview")
                    create_visualizations(results['results'])

                    # Export options
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = export_to_csv(results['results'])
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            "reviewer_recommendations.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    with col2:
                        json_data = json.dumps(results, indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_data,
                            "reviewer_recommendations.json",
                            "application/json",
                            use_container_width=True
                        )

                    # Display results
                    st.markdown("---")
                    st.header("👥 Recommended Reviewers")

                    for rec in results['results']:
                        display_reviewer_card(rec, rec['rank'])

    with tab2:
        st.header("About This System")
        st.markdown("""
        ### 🎯 Overview
        The Semantic Reviewer Matching System uses AI to automatically match manuscripts
        with qualified peer reviewers based on semantic similarity of research expertise.

        ### 🔬 How It Works
        1. **Semantic Analysis**: Your manuscript abstract is analyzed using SPECTER2,
           a transformer model trained on scientific papers
        2. **Vector Search**: The system searches through thousands of potential reviewers
           using FAISS vector similarity search
        3. **Multi-Factor Ranking**: Reviewers are ranked based on:
           - Semantic similarity (70%)
           - Publication recency (15%)
           - Publication velocity (10%)
           - Institution diversity (5%)
        4. **Conflict Detection**: Automatically detects co-authorship and institutional conflicts

        ### 📈 Success Metrics
        - **Search Speed**: <3 seconds for 20 recommendations
        - **Relevance**: ≥70% of top-5 rated "highly relevant" by editors
        - **Conflict Detection**: 100% accuracy on co-author detection

        ### 🔧 Technology Stack
        - **Embeddings**: SPECTER2 (allenai/specter2_base)
        - **Vector Search**: FAISS
        - **API**: FastAPI
        - **Interface**: Streamlit
        - **Data Source**: Semantic Scholar API

        ### 📚 Citation
        If you use this system in your research, please cite:
        ```
        Semantic Reviewer Matching System (SRMS)
        Version 1.0.0 MVP
        ```

        ### 🤝 Support
        For issues or questions, please contact the development team.
        """)


if __name__ == "__main__":
    main()
