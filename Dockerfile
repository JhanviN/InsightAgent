# Stage 1: builder
FROM python:3.10-slim-buster AS builder

WORKDIR /app
RUN mkdir -p /app/faiss_index
VOLUME /app/faiss_index
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runner
FROM python:3.10-slim-buster

COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
