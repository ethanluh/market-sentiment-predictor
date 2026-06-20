# Serving image for the market-sentiment-predictor FastAPI app.
# Installs the full runtime (FinBERT/torch are needed for live sentiment), so
# the image is large; see docs/deploy.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code (data/, models/, tests are excluded via .dockerignore).
COPY src/ ./src/

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Pre-download FinBERT at build time so it is baked into the image; otherwise the
# weights download from Hugging Face on the first /predict call (~30-60s latency).
# Runs as appuser so the cache lands in a home dir the runtime can read.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN python -c "from transformers import BertForSequenceClassification, BertTokenizer; \
    BertTokenizer.from_pretrained('ProsusAI/finbert'); \
    BertForSequenceClassification.from_pretrained('ProsusAI/finbert')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# src/pipeline/api.py's __main__ runs uvicorn on 0.0.0.0:8000.
CMD ["python", "-m", "src.pipeline.api"]
