from fastapi import FastAPI, Header, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from ingest import process_document_from_url, warmup_embeddings
from query import analyze_query_with_vectorstore_fast, process_queries_batch
import gc

load_dotenv()
EXPECTED_TOKEN = os.getenv("AUTH_TOKEN")

executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="DocProcessor")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting InsightAgent API...")
    print("🔥 Warming up embeddings model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, warmup_embeddings)
    print("✅ API ready!")
    yield
    print("🛑 Shutting down...")
    executor.shutdown(wait=True)
    gc.collect()

app = FastAPI(
    title="InsightAgent - Document Query API",
    version="2.0",
    description="High-performance document analysis with LLM-powered insights",
    lifespan=lifespan
)

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def verify_auth(authorization: str) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = authorization[len("Bearer "):].strip()
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")
    return token

@app.get("/")
def read_root():
    return {
        "service": "InsightAgent API",
        "status": "running",
        "version": "2.0",
        "features": [
            "Smart document chunking",
            "Cached embeddings",
            "Sequential query processing",
            "Insurance domain optimization"
        ]
    }

@app.get("/health")
def health_check():
    import multiprocessing
    children = multiprocessing.active_children()
    return {
        "status": "ok",
        "workers": [str(child) for child in children],
    }

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(request: QueryRequest, authorization: str = Header(...)):
    start_time = time.time()
    verify_auth(authorization)
    vectorstore = None
    try:
        print(f"🚀 Processing {len(request.questions)} questions")
        loop = asyncio.get_event_loop()
        print("📄 Processing document...")
        doc_start = time.time()
        vectorstore = await loop.run_in_executor(executor, process_document_from_url, request.documents)
        doc_time = time.time() - doc_start
        print(f"✅ Document processed in {doc_time:.1f}s")
        print("🔍 Processing questions...")
        query_start = time.time()
        answers = await loop.run_in_executor(executor, process_queries_batch, vectorstore, request.questions)
        query_time = time.time() - query_start
        print(f"✅ Questions processed in {query_time:.1f}s")
        total_time = time.time() - start_time
        print(f"🎉 Request completed in {total_time:.1f}s")
        print(f"📊 Breakdown - Doc: {doc_time:.1f}s, Query: {query_time:.1f}s")
        return QueryResponse(answers=answers)
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.1f}s: {str(e)}")
        return QueryResponse(answers=[f"Processing error: {str(e)}"] * len(request.questions))
    finally:
        if vectorstore:
            from ingest import cleanup_vectorstore
            cleanup_vectorstore(vectorstore)
        gc.collect()

@app.get("/api/v1/stats")
def get_performance_stats():
    return {
        "optimizations": [
            "Smart chunking (1500 chars)",
            "Higher overlap (400 chars)",
            "MMR-only retrieval",
            "Sequential query processing",
            "Insurance-specific prompts",
            "Cached LLM connections"
        ],
        "expected_performance": {
            "document_processing": "5-10s",
            "query_processing": "1-2s per question",
            "sequential_10_questions": "10-20s for queries"
        },
        "model_info": {
            "embeddings": "all-MiniLM-L6-v2",
            "llm": "openai/gpt-oss-120b",
            "chunk_size": 1500,
            "overlap": 400
        }
    }

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting InsightAgent API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )