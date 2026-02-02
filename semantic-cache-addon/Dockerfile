ARG BUILD_FROM
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    jq && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    sentence-transformers \
    torch \
    numpy \
    python-multipart \
    watchdog \
    aiofiles \
    rank_bm25 \
    optimum[openvino] \
    nncf \
    "transformers<4.45" \
    einops

# Copy application files
COPY app.py /app.py
COPY cache_types.py /cache_types.py
COPY cache_loader.py /cache_loader.py
COPY embedding.py /embedding.py
COPY bm25_index.py /bm25_index.py
COPY file_watcher.py /file_watcher.py
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD [ "/run.sh" ]
