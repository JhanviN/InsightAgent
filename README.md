# InsightAgent: Advanced RAG-Powered Document Analysis API

**InsightAgent** is a high-performance FastAPI-based application designed for intelligent document analysis, leveraging **Retrieval-Augmented Generation (RAG)** techniques to process insurance-related documents with precision and efficiency.  

Built with **LangChain** for orchestration, **FAISS** for vector stores, and **Sentence-Transformers** for embeddings, InsightAgent delivers context-aware query responses by combining semantic search, text chunking, and LLM integration.  

Optimized for scalability, the project features a **prebuilt persistent FAISS index**, enabling fast vector retrieval and reducing latency for real-world applications. Deployed on **Railway**, InsightAgent exemplifies modern AI engineering, with a **Docker image under 1 GB** for seamless deployment.

---

## ✨ Project Highlights
- ⚡ **RAG Architecture**: Retrieval-Augmented Generation for accurate, context-enhanced responses using FAISS vector stores and MMR retrieval.  
- 🔍 **Efficient Vector Search**: FAISS (CPU-optimized) for lightning-fast similarity searches on prebuilt indices.  
- 📄 **Advanced Text Processing**: `RecursiveCharacterTextSplitter` with quality filters (alpha ratio, min word count) for relevant segments.  
- 🤖 **AI-Driven Querying**: Groq LLM with custom prompts, fallback strategies, and semantic caching.  

---

## 📋 Workflow Overview
InsightAgent follows a streamlined RAG pipeline to transform raw documents into actionable insights:

### 1. Document Ingestion
- Downloads documents from URLs with robust error handling and size limits (max 100MB).  
- Extracts text using PyMuPDF with metadata (page numbers, word counts, insurance flags like sub-limit, PPN exceptions).  

### 2. Text Chunking
- Uses `RecursiveCharacterTextSplitter` (1500 characters, 400 overlap).  
- Applies quality filters (min word count, alpha ratio > 0.3).  
- Scores chunks for numbers, currency, percentages → prioritizes high-value segments (max 1000 chunks).  

### 3. Vector Store Creation
- Embeddings via **`all-MiniLM-L6-v2`** (Sentence-Transformers).  
- FAISS vector store built and cached on disk.  

### 4. Semantic Search and Querying
- Query preprocessing with synonym expansion (e.g., *deductible → out-of-pocket amount*).  
- Retrieval with **MMR** (max 6 documents).  
- Augmentation with Groq LLM + dual prompts.  
- Semantic caching for repeated queries.  

### 5. API Response
- Results served via **FastAPI endpoints**, including answers, metadata, and quality scores.  
- Performance logs exported to Google Sheets.  

⏱️ Latency: ~13–15s first request, ~8s cached.  

---

## 🚀 Quick Start

### Prerequisites
- [Docker](https://www.docker.com/) (latest recommended).  
- [Git](https://git-scm.com/) (for cloning).  
- [Railway](https://railway.app/) account (for deployment).   

---

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/InsightAgent.git
cd InsightAgent

```

Build the Docker Image
Ensure the faiss_index directory (containing prebuilt FAISS files like index.faiss and index.pkl) is in the project root.

```bash
docker build -t insight-agent .
```
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
