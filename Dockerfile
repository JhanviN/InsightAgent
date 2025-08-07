# Stage 1: builder
FROM python:3.10-slim-buster AS builder

WORKDIR /app
RUN mkdir -p /app/faiss_index

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runner
FROM python:3.10-slim-buster

# Accept Railway environment variables
ARG AUTH_TOKEN
ARG GROQ_API_KEY
ARG PORT

# Make them available in the container's environment
ENV AUTH_TOKEN=${AUTH_TOKEN}
ENV GROQ_API_KEY=${GROQ_API_KEY}
ENV PORT=${PORT}

COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .

# Use PORT variable in case it's dynamically assigned
EXPOSE ${PORT}

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
