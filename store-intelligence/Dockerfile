FROM python:3.11-slim

LABEL maintainer="store-intelligence"
LABEL description="Store Intelligence API for Apex Retail"

# System deps for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY store_layout.json .

# Create data directory for SQLite and events
RUN mkdir -p /data
ENV DATABASE_URL="sqlite:////data/store_intelligence.db"
ENV POS_CSV_PATH="/data/pos_transactions.csv"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
