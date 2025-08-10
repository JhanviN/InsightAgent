# Stage 1: builder
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy entire application code including FAISS index from repo
COPY . .

# Stage 2: runner
FROM python:3.10-slim-buster

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code including FAISS index
COPY --from=builder /app /app

# Ensure FAISS index directory exists
RUN mkdir -p /app/faiss_index

# Declare volume for FAISS index persistence
VOLUME ["/app/faiss_index"]

# Set default port
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Start the application
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
