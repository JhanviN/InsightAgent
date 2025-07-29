from fastapi import FastAPI, Header, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import json
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from ingest import process_document_from_url, warmup_embeddings
from query import analyze_query_with_vectorstore_fast, analyze_multiple_queries_fast

load_dotenv()
EXPECTED_TOKEN = os.getenv("AUTH_TOKEN")

# Global thread pool
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="DocProcessor")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting InsightAgent API...")
    print("🔥 Warming up embeddings model...")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, warmup_embeddings)
    
    print("✅ API ready!")
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    executor.shutdown(wait=True)

# Initialize FastAPI with lifespan
app = FastAPI(
    title="InsightAgent - Document Query API",
    version="2.0",
    description="High-performance document analysis with LLM-powered insights",
    lifespan=lifespan
)

router = APIRouter()

# Request/Response models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    # processing_time: float
    # total_questions: int
    # status: str = "success"

# Auth helper
def verify_auth(authorization: str) -> str:
    """Verify bearer token"""
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
            "Batch query processing",
            "Insurance domain optimization"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "embeddings": "loaded",
        "workers": executor._threads,
        "timestamp": time.time()
    }

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(
    request: QueryRequest,
    authorization: str = Header(...)
):
    """
    Main endpoint: Process document and answer questions
    Optimized for accuracy and speed
    """
    start_time = time.time()
    
    # Authentication
    print(authorization)
    verify_auth(authorization)
    
    try:
        print(f"🚀 Processing {len(request.questions)} questions")
        
        # Step 1: Process document (CPU intensive)
        loop = asyncio.get_event_loop()
        
        print("📄 Processing document...")
        doc_start = time.time()
        vectorstore = await loop.run_in_executor(
            executor,
            process_document_from_url,
            request.documents
        )
        doc_time = time.time() - doc_start
        print(f"✅ Document processed in {doc_time:.1f}s")
        
        # Step 2: Process questions in batch (more efficient)
        print("🔍 Processing questions...")
        query_start = time.time()
        
        raw_answers = await loop.run_in_executor(
            executor,
            analyze_multiple_queries_fast,
            request.questions,
            vectorstore
        )
        
        query_time = time.time() - query_start
        print(f"✅ Questions processed in {query_time:.1f}s")
        
        # Step 3: Clean and format answers
        answers = []
        for result in raw_answers:
            if isinstance(result, str):
                # Try to parse JSON response
                try:
                    parsed = json.loads(result)
                    clean_answer = parsed.get("justification", result)
                except json.JSONDecodeError:
                    clean_answer = result
            else:
                clean_answer = str(result)
            
            # Basic cleaning
            clean_answer = clean_answer.strip()
            if clean_answer.startswith("Answer:"):
                clean_answer = clean_answer[7:].strip()
            
            answers.append(clean_answer)
        
        total_time = time.time() - start_time
        
        print(f"🎉 Request completed in {total_time:.1f}s")
        print(f"📊 Breakdown - Doc: {doc_time:.1f}s, Query: {query_time:.1f}s")
        
        return QueryResponse(
            answers=answers,
            # processing_time=round(total_time, 2),
            # total_questions=len(request.questions),
            # status="success"
        )
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.1f}s: {str(e)}")
        
        # Return partial response for debugging
        return QueryResponse(
            answers=[f"Processing error: {str(e)}"] * len(request.questions),
            # processing_time=round(error_time, 2),
            # total_questions=len(request.questions),
            # status="error"
        )

@router.post("/hackrx/run-simple")
async def process_simple(
    request: QueryRequest,
    authorization: str = Header(...)
):
    """
    Simplified endpoint matching exact expected format
    """
    verify_auth(authorization)
    
    try:
        loop = asyncio.get_event_loop()
        
        # Process document
        vectorstore = await loop.run_in_executor(
            executor,
            process_document_from_url,
            request.documents
        )
        
        # Process questions
        raw_answers = await loop.run_in_executor(
            executor,
            analyze_multiple_queries_fast,
            request.questions,
            vectorstore
        )
        
        # Clean answers
        answers = []
        for result in raw_answers:
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    answer = parsed.get("justification", result)
                except json.JSONDecodeError:
                    answer = result
            else:
                answer = str(result)
            
            # Basic cleanup
            answer = answer.strip()
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            answers.append(answer)
        
        return {"answers": answers}
        
    except Exception as e:
        print(f"❌ Simple endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Performance monitoring endpoint
@app.get("/api/v1/stats")
def get_performance_stats():
    return {
        "optimizations": [
            "Smaller chunks (800 chars) for accuracy",
            "Higher overlap (200) for context",
            "Score threshold retrieval",
            "Batch query processing",
            "Insurance-specific prompts",
            "Cached LLM connections"
        ],
        "expected_performance": {
            "document_processing": "10-25s",
            "query_processing": "1-3s per question",
            "batch_10_questions": "15-35s total"
        },
        "model_info": {
            "embeddings": "all-MiniLM-L6-v2",
            "llm": "llama-3.1-8b-instant",
            "chunk_size": 800,
            "overlap": 200
        }
    }

# Test endpoint
@app.post("/api/v1/test")
async def test_endpoint():
    """Quick test endpoint"""
    try:
        # Test with sample URL
        test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        test_questions = ["What is the grace period for premium payment?"]
        
        loop = asyncio.get_event_loop()
        vectorstore = await loop.run_in_executor(
            executor,
            process_document_from_url,
            test_url
        )
        
        answers = await loop.run_in_executor(
            executor,
            analyze_multiple_queries_fast,
            test_questions,
            vectorstore
        )
        
        return {
            "status": "success",
            "test_answer": answers[0][:100] + "...",
            "message": "API is working correctly"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "message": "Test failed"
        }

# Include router
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting InsightAgent API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable for production
        workers=1      # Single worker for consistency
    )