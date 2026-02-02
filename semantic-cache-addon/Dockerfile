ARG BUILD_FROM
FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    jq && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    watchdog>=4.0.0 \
    sentence-transformers>=2.2.0 \
    rank-bm25>=0.2.2 \
    aiohttp>=3.9.0 \
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
