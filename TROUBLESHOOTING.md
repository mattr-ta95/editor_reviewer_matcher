# Troubleshooting Guide

## Problem: Semantic Scholar API Errors

### Symptoms
- Getting HTTP 500 errors from API
- Rate limiting (429 errors)
- No papers found
- Build process taking too long

### Root Cause
The Semantic Scholar public API has limitations:
1. **Rate limits**: 100 requests per 5 minutes (public), 5000 with API key
2. **Server instability**: Paper search endpoint can return 500 errors
3. **Search API issues**: `/paper/search` endpoint is less reliable than direct lookups

## Solutions

### Option 1: Use Mock Database (FASTEST - 30 seconds) ⚡

**Best for:** Immediate testing, development, demos

```bash
python scripts/create_mock_database.py
```

This creates 100 synthetic reviewers with realistic data.

**Pros:**
- ✅ Works instantly (30 seconds)
- ✅ No API calls needed
- ✅ Predictable data

**Cons:**
- ❌ Not real researchers
- ❌ Can't be used for actual recommendations

---

### Option 2: Use Test Database (RECOMMENDED) ✨

**Best for:** Testing with real data

```bash
python scripts/create_test_database.py --count 50
```

Uses direct author lookups (bypasses broken search API).

**Pros:**
- ✅ Real researchers
- ✅ More reliable
- ✅ Works without API key

**Cons:**
- ⏱️ Takes 5-10 minutes for 50 reviewers

---

### Option 3: Get Free API Key 🔑

**Best for:** Production (1000+ reviewers)

1. Get key: https://www.semanticscholar.org/product/api
2. Set it: `export SEMANTIC_SCHOLAR_API_KEY="your_key"`
3. Build: `python scripts/build_database_safe.py --target-count 1000`

**Benefits:**
- 🚀 50x higher rate limits
- ⏱️ 1000 reviewers in 30-60 minutes

---

## Quick Start Recommendations

### For YOU Right Now (Based on Your Error)

Since you're seeing HTTP 500 errors from the paper search API, use **Option 1** for immediate testing:

```bash
# Stop the current build (Ctrl+C if still running)

# Create mock database (30 seconds)
python scripts/create_mock_database.py

# Build index (5-10 minutes)
python scripts/build_faiss_index.py

# Test the system
./scripts/run_api.sh &
streamlit run app/streamlit_app.py
```

Then visit http://localhost:8501 and try a search!

---

## Common Errors

### "HTTP 500 Server Error"
→ Use Option 1 or 2 above

### "Rate limit exceeded"
→ Get API key or use Option 1

### "ModuleNotFoundError"
→ Run: `pip install -r requirements.txt`

### "No papers found"
→ This is your current issue - use Option 1 or 2

---

## Verification Steps

After creating database:
```bash
# Check database exists and has data
sqlite3 data/reviewers.db "SELECT COUNT(*) FROM reviewers;"

# Should show: 100 (mock) or 50 (test)
```

After building index:
```bash
# Check index exists
ls -lh data/indices/specter2.index

# Should see a file ~50-100MB
```

---

## Need Help?

See QUICKSTART.md for full setup guide
