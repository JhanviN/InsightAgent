# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install minimal build dependencies and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get purge -y --auto-remove build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies with minimal footprint
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -t /install \
    && find /install -type f -name "*.pyc" -delete

# Copy only necessary application files
COPY main.py ingest.py query.py ./
COPY faiss_index /app/faiss_index

# Stage 2: Runner
FROM python:3.10-slim

WORKDIR /app

# Copy installed Python packages and ensure executables are in PATH
COPY --from=builder /install /usr/local/lib/python3.10/site-packages
RUN ln -s /usr/local/lib/python3.10/site-packages/bin/uvicorn /usr/local/bin/uvicorn || true

# Copy application source code and prebuilt FAISS index
COPY --from=builder /app /app

# Ensure FAISS index directory exists
RUN mkdir -p /app/faiss_index

# Declare persistent volume for FAISS index
VOLUME ["/app/faiss_index"]

# Set environment variables
ENV PORT=8000

# Expose the port
EXPOSE $PORT

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]