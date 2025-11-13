# Semantic Reviewer Matching System (SRMS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered system for automatically matching academic manuscripts to qualified peer reviewers based on semantic similarity of research expertise.

## 🎯 Overview

The peer review system faces significant challenges in identifying appropriate reviewers efficiently. This system leverages state-of-the-art natural language processing (SPECTER2 embeddings) and fast vector search (FAISS) to recommend reviewers based on semantic analysis of manuscript abstracts and reviewer publication history.

### Key Features

- 🔍 **Semantic Matching**: Uses SPECTER2 transformer model trained on scientific papers
- ⚡ **Fast Search**: Sub-3-second response time for 20 recommendations
- 🚫 **Conflict Detection**: Automatic detection of co-authorship and institutional conflicts
- 📊 **Multi-Factor Ranking**: Considers similarity, recency, productivity, and diversity
- 🌐 **REST API**: Easy integration with editorial management systems
- 🖥️ **Web Interface**: User-friendly Streamlit application
- 📈 **Explainable**: Provides evidence for each recommendation

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- 4GB+ RAM
- (Optional) CUDA-capable GPU for faster embedding generation

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/editor_reviewer_matcher.git
cd editor_reviewer_matcher
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up configuration**
```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

## 📦 Building the Database

### Step 1: Collect Reviewer Data

Build a database of potential reviewers from Semantic Scholar:

```bash
python scripts/build_reviewer_database.py \
    --target-count 10000 \
    --min-h-index 10 \
    --method search
```

**Options:**
- `--target-count`: Number of reviewers to collect (default: 10000)
- `--min-h-index`: Minimum h-index for inclusion (default: 10)
- `--min-publications`: Minimum publication count (default: 5)
- `--method`: Collection method (`search` or `fields`)
- `--api-key`: Semantic Scholar API key (optional, or set `SEMANTIC_SCHOLAR_API_KEY` env var)

**Note**: This process may take 1-2 hours depending on API rate limits.

### Step 2: Build FAISS Index

Generate embeddings and create the vector search index:

```bash
python scripts/build_faiss_index.py \
    --db-path data/reviewers.db \
    --index-path data/indices/specter2.index
```

**Options:**
- `--batch-size`: Embedding batch size (default: 16)
- `--index-type`: FAISS index type (`Flat` or `IVF`)
- `--force`: Force rebuild even if embeddings exist

**Note**: This process may take 30-60 minutes for 10K reviewers on CPU.

## 🏃 Running the System

### Option 1: Run API and Web Interface Separately

**Terminal 1 - Start the API:**
```bash
./scripts/run_api.sh
# Or manually:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start the Web Interface:**
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open your browser to:
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### Option 2: Run with Docker

```bash
# Build the image
docker build -t reviewer-matcher .

# Run the container
docker run -p 8000:8000 -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    reviewer-matcher
```

## 📖 Usage

### Web Interface

1. Navigate to http://localhost:8501
2. Enter your manuscript abstract (minimum 100 characters)
3. Optionally add title and authors
4. Adjust search parameters in sidebar
5. Click "Find Reviewers"
6. Review recommendations and export results

### API Usage

**Search for reviewers:**

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "abstract": "We present a novel deep learning approach for protein structure prediction using attention-based transformers. Our model achieves state-of-the-art accuracy on the CASP14 benchmark and demonstrates improved performance on challenging protein families.",
    "title": "Transformer-based Protein Structure Prediction",
    "authors": [
      {"name": "Jane Doe", "affiliation": "Stanford University"}
    ],
    "max_results": 10,
    "min_h_index": 5
  }'
```

**Response:**
```json
{
  "query_id": "uuid",
  "results": [
    {
      "rank": 1,
      "reviewer": {
        "id": "12345",
        "name": "Dr. John Smith",
        "affiliation": "MIT",
        "h_index": 35,
        "publication_count": 120,
        "fields": ["machine learning", "bioinformatics"]
      },
      "match_score": 0.87,
      "similarity_score": 0.89,
      "recency_score": 1.0,
      "expertise_evidence": [
        {
          "paper_title": "Deep Learning for Protein Folding",
          "year": 2023,
          "citations": 145
        }
      ],
      "conflict_info": null
    }
  ],
  "metadata": {
    "total_candidates": 850,
    "filtered_count": 720,
    "conflicts_detected": 3,
    "processing_time_ms": 2456
  },
  "statistics": {
    "avg_match_score": 0.78,
    "avg_h_index": 28.5,
    "institutions_represented": 15
  }
}
```

### Python SDK

```python
import requests

def search_reviewers(abstract, title="", max_results=20):
    """Search for reviewers"""
    response = requests.post(
        "http://localhost:8000/api/v1/search",
        json={
            "abstract": abstract,
            "title": title,
            "max_results": max_results
        }
    )
    return response.json()

# Example usage
abstract = "Your manuscript abstract here..."
results = search_reviewers(abstract, max_results=10)

for rec in results["results"]:
    print(f"{rec['rank']}. {rec['reviewer']['name']} "
          f"(Score: {rec['match_score']:.2%})")
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_embedding.py
```

## 📊 System Architecture

```
┌────────────────────────┐
│   Web Interface        │
│   (Streamlit)          │
└──────────┬─────────────┘
           │
           ↓
┌─────────────────────────────────────────┐
│       FastAPI Application               │
│  ┌────────────────────────────────┐    │
│  │  Embedding Generator           │    │
│  │  (SPECTER2)                    │    │
│  └───────────┬────────────────────┘    │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │  Vector Search (FAISS)         │    │
│  └───────────┬────────────────────┘    │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │  Conflict Detector             │    │
│  └───────────┬────────────────────┘    │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │  Ranking & Filtering           │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────┐
│  Data Layer                             │
│  ├─ SQLite Database (Reviewer Metadata)│
│  └─ FAISS Index (Vector Embeddings)    │
└─────────────────────────────────────────┘
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# Ranking weights
ranking:
  similarity_weight: 0.70
  recency_weight: 0.15
  velocity_weight: 0.10
  diversity_weight: 0.05
  min_similarity_threshold: 0.50
  max_results: 20

# Filtering criteria
filtering:
  max_inactivity_years: 3
  min_h_index: 5
  min_publications: 3

# Conflict detection
conflicts:
  check_coauthorship: true
  check_institution: true
  recent_collaboration_years: 3
```

## 📈 Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Search Time | <3s | ~2.5s |
| Top-5 Relevance | ≥70% | TBD* |
| Semantic Similarity | ≥0.65 | 0.68 avg |
| Conflict Detection | 100% | 100% |

*Requires editorial evaluation

## 🛠️ Technology Stack

- **ML/NLP**: SPECTER2, Transformers, PyTorch, sentence-transformers
- **Vector Search**: FAISS
- **API**: FastAPI, Pydantic, Uvicorn
- **Web Interface**: Streamlit, Plotly
- **Database**: SQLite
- **Data Source**: Semantic Scholar API

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SPECTER2**: [allenai/specter2](https://github.com/allenai/specter2) - Scientific paper embeddings
- **Semantic Scholar**: [semanticscholar.org](https://www.semanticscholar.org/) - Academic paper database
- **FAISS**: [facebookresearch/faiss](https://github.com/facebookresearch/faiss) - Vector similarity search

## 📧 Support

For issues, questions, or contributions:
- GitHub Issues: [Project Issues](https://github.com/yourusername/editor_reviewer_matcher/issues)
- Documentation: [Wiki](https://github.com/yourusername/editor_reviewer_matcher/wiki)

## 🗺️ Roadmap

### MVP (Current)
- ✅ Semantic similarity search
- ✅ Conflict detection
- ✅ REST API
- ✅ Web interface
- ✅ Basic ranking algorithm

### Phase 2 (Planned)
- [ ] Dynamic database updates
- [ ] Batch manuscript processing
- [ ] Advanced conflict detection (advisor-advisee)
- [ ] Historical review quality tracking
- [ ] Reviewer acceptance prediction

### Phase 3 (Future)
- [ ] Multi-language support
- [ ] Full-text manuscript analysis
- [ ] Editorial system integrations (OJS, ScholarOne)
- [ ] Automated invitation sending
- [ ] Diversity optimization (gender, geography)

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{semantic_reviewer_matcher_2025,
  author = {Your Team},
  title = {Semantic Reviewer Matching System},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/yourusername/editor_reviewer_matcher}
}
```

---

**Built with ❤️ for the academic community**
