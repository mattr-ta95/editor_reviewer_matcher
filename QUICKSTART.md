# Quick Start Guide

This guide will help you get the Semantic Reviewer Matching System up and running in 30 minutes.

## Prerequisites

- Python 3.10+ installed
- 8GB RAM minimum
- Internet connection (for downloading models and data)
- ~10GB free disk space

## Step-by-Step Setup

### 1. Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/editor_reviewer_matcher.git
cd editor_reviewer_matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Reviewer Database (10-15 minutes)

**Option A: Quick Test (1000 reviewers, ~10 minutes)**
```bash
python scripts/build_reviewer_database.py \
    --target-count 1000 \
    --min-h-index 5 \
    --method search
```

**Option B: Full Database (10000 reviewers, ~1-2 hours)**
```bash
python scripts/build_reviewer_database.py \
    --target-count 10000 \
    --min-h-index 10 \
    --method search
```

**Note**: For faster collection, get a free Semantic Scholar API key:
1. Register at https://www.semanticscholar.org/product/api
2. Set environment variable: `export SEMANTIC_SCHOLAR_API_KEY=your_key`

### 3. Build Search Index (5-10 minutes)

```bash
python scripts/build_faiss_index.py \
    --db-path data/reviewers.db \
    --index-path data/indices/specter2.index \
    --batch-size 16
```

**Note**: First run will download the SPECTER2 model (~1GB). Subsequent runs will use cached model.

### 4. Start the System (1 minute)

**Terminal 1 - API Server:**
```bash
./scripts/run_api.sh
```

Wait for: "Application startup complete"

**Terminal 2 - Web Interface:**
```bash
streamlit run app/streamlit_app.py
```

### 5. Test the System (5 minutes)

Open your browser to: http://localhost:8501

**Try this example:**

1. Paste this abstract:
```
We propose a novel approach for protein structure prediction using deep learning.
Our model leverages attention mechanisms and graph neural networks to capture both
local and global structural patterns. We demonstrate state-of-the-art performance
on the CASP14 benchmark, achieving improved accuracy on challenging protein families.
The model is trained end-to-end on a large dataset of experimentally determined
structures and shows strong generalization to unseen protein sequences.
```

2. Click "Find Reviewers"

3. You should see 20 ranked reviewers with:
   - Match scores
   - H-indices
   - Representative publications
   - Institution diversity

## Troubleshooting

### "Cannot connect to API"
- Make sure API server is running (Terminal 1)
- Check http://localhost:8000/docs loads
- Verify no firewall blocking port 8000

### "Database not found"
- Complete Step 2 first
- Check `data/reviewers.db` exists
- Verify database has reviewers:
  ```bash
  sqlite3 data/reviewers.db "SELECT COUNT(*) FROM reviewers;"
  ```

### "Index not found"
- Complete Step 3 first
- Check `data/indices/specter2.index` exists

### "Out of memory"
- Reduce batch size: `--batch-size 8`
- Use smaller database: `--target-count 1000`
- Close other applications

### Model download slow/failing
- Model size: ~500MB
- Downloads to: `~/.cache/huggingface/`
- Manual download:
  ```python
  from transformers import AutoModel
  AutoModel.from_pretrained('allenai/specter2_base')
  ```

## Next Steps

Once running:

1. **Explore the API**: http://localhost:8000/docs
2. **Read full documentation**: See README.md
3. **Customize ranking**: Edit `config/config.yaml`
4. **Run tests**: `pytest tests/`
5. **Try batch searches**: Use the API programmatically

## Production Deployment

For production use:

1. Use larger database (10K+ reviewers)
2. Set up proper monitoring
3. Configure authentication
4. Use environment-specific configs
5. Deploy with Docker

See README.md for detailed deployment instructions.

## Support

- Issues: https://github.com/yourusername/editor_reviewer_matcher/issues
- Documentation: See README.md and docs/
- API Docs: http://localhost:8000/docs (when running)

## Success Checklist

- [ ] Dependencies installed
- [ ] Database built (contains >100 reviewers)
- [ ] FAISS index created
- [ ] API server starts successfully
- [ ] Web interface loads
- [ ] Example search returns results
- [ ] Can export results to CSV

If all checked, you're ready to use the system! 🎉
