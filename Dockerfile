# Multi-stage Dockerfile for Reviewer Matcher System

# Stage 1: Build
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Download SPECTER2 model at build time
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('allenai/specter2_base'); \
    AutoTokenizer.from_pretrained('allenai/specter2_base')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/indices data/processed/embeddings data/raw/reviewers data/raw/papers logs

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run both API and Streamlit
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
