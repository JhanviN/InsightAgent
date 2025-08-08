# Stage 1: builder
FROM python:3.10-slim-buster AS builder

WORKDIR /app
RUN mkdir -p /app/faiss_index

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runner
FROM python:3.10-slim-buster

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code (including .env file)
COPY . .

# Set default port
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Use environment variable for port in the command
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"