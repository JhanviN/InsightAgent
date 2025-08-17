# InsightAgent: Advanced RAG-Powered Document Analysis API

**InsightAgent** is a high-performance FastAPI-based application designed for intelligent document analysis, leveraging Retrieval-Augmented Generation (RAG) techniques to process insurance-related documents with precision and efficiency. Built with LangChain for orchestration, FAISS for vector stores, and Sentence-Transformers for embeddings, InsightAgent delivers context-aware query responses by combining semantic search, text chunking, and LLM integration. Optimized for scalability, the project features a prebuilt persistent FAISS index, enabling fast vector retrieval and reducing latency for real-world applications. Deployed on Railway, InsightAgent exemplifies modern AI engineering, with a Docker image streamlined to under 1 GB for seamless deployment.

## ✨ Project Highlights
- **RAG Architecture**: Implements Retrieval-Augmented Generation for accurate, context-enhanced responses using FAISS vector stores and MMR (Maximal Marginal Relevance) retrieval.
- **Efficient Vector Search**: Utilizes FAISS (CPU-optimized) for lightning-fast similarity searches on prebuilt indices, supporting large-scale document querying.
- **Advanced Text Processing**: Employs `RecursiveCharacterTextSplitter` for chunking with quality filtering (e.g., alpha ratio, minimum word count) to ensure relevant, high-quality segments.
- **AI-Driven Querying**: Integrates Groq LLM with custom prompts for nuanced answers, including fallback strategies for comprehensive information extraction.

## 📋 Workflow Overview
InsightAgent follows a streamlined RAG pipeline to transform raw documents into actionable insights:

### 1. Document Ingestion
- Downloads documents from URLs with robust error handling and size limits (max 100MB).
- Extracts text using PyMuPDF, enriched with metadata like page numbers, word counts, and insurance flags (e.g., sub-limit, PPN exceptions).

### 2. Text Chunking
- Uses `RecursiveCharacterTextSplitter` to split documents into chunks (1500 characters, 400 overlap).
- Applies quality filters (e.g., minimum word count, alpha ratio > 0.3) and scores chunks (e.g., for numbers, currency, percentages) to prioritize high-value segments, limiting to 1000 chunks.

### 3. Vector Store Creation
- Generates embeddings with `all-MiniLM-L6-v2` from Sentence-Transformers.
- Builds a FAISS vector store for semantic similarity search, cached on disk for efficiency.

### 4. Semantic Search and Querying
- Preprocesses queries with synonym expansion (e.g., "deductible" to "deductible out-of-pocket amount").
- Retrieves relevant chunks using MMR to ensure diversity and relevance (up to 6 documents).
- Augments responses with Groq LLM using dual prompts for thorough extraction, with semantic caching for repeated queries.

### 5. API Response
- Serves results via FastAPI endpoints, including answers, source metadata, and quality scores.
- Logs performance to Google Sheets for monitoring.

This workflow achieves low latency (~13–15s first request, ~8s cached) while maintaining high accuracy through quality chunking and RAG.

## 🚀 Quick Start

### Prerequisites
- Docker (latest version recommended).
- Git (for cloning and version control).
- Railway account (for deployment).
- Optional: Python 3.10 (for local development or debugging).

### Installation

#### Clone the Repository
```bash
git clone https://github.com/your-username/InsightAgent.git
cd InsightAgent


Build the Docker Image
Ensure the faiss_index directory (containing prebuilt FAISS files like index.faiss and index.pkl) is in the project root.

docker build -t insight-agent .

Environment Variables
Set these in a .env file or your environment:

GROQ_API_KEY: Your Groq API key for LLM access.
AUTH_TOKEN: Secure token for API authentication (e.g., your-secret-token-123).

☁️ Deployment to Railway

Deployed on Railway with a persistent faiss_index volume for scalability.

📐 Project Architecture
textInsightAgent/
├── Dockerfile              # Multi-stage build for ~800MB image
├── main.py                 # FastAPI server and endpoints
├── ingest.py               # Document ingestion and FAISS indexing
├── query.py                # Query processing with LLM and retrieval
├── logging_service.py      # Custom logging to Google Sheets
├── requirements.txt        # Dependency list
├── faiss_index/            # Prebuilt FAISS index files
├── .dockerignore           # Excludes bloat (e.g., .git, .venv)
└── .env                    # Environment variables (optional)
