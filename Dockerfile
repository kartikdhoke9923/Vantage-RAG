# Vantage RAG — production image (single Render web service).
# The widget, admin API and RAG pipeline all live here.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime (pulled by chromadb) needs libgomp for OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app ./app
COPY config ./config
COPY ui/widget ./ui/widget
COPY data ./data

# Render injects PORT; default 8000 for local docker runs.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]