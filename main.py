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
import ray
from ingest import process_document_from_url, warmup_embeddings
from query import analyze_query_with_vectorstore_fast, process_queries_batch

load_dotenv()
EXPECTED_TOKEN = os.getenv("AUTH_TOKEN")

# Global thread pool
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="DocProcessor")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    ray.init(num_cpus=4, ignore_reinit_error=True)
    print("🚀 Starting InsightAgent API...")
    print("🔥 Warming up embeddings model...")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, warmup_embeddings)
    
    print("✅ API ready!")
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    executor.shutdown(wait=True)
    ray.shutdown()
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

# @router.post("/hackrx/run", response_model=QueryResponse)
# async def process_document_queries(
#     request: QueryRequest,
#     authorization: str = Header(...)
# ):
#     """
#     Main endpoint: Process document and answer questions
#     Optimized for accuracy with sequential processing
#     """
#     start_time = time.time()
    
#     # Authentication
#     verify_auth(authorization)
    
#     try:
#         print(f"🚀 Processing {len(request.questions)} questions")
        
#         # Step 1: Process document (CPU intensive)
#         loop = asyncio.get_event_loop()
        
#         print("📄 Processing document...")
#         doc_start = time.time()
#         vectorstore = await loop.run_in_executor(
#             executor,
#             process_document_from_url,
#             request.documents
#         )
#         doc_time = time.time() - doc_start
#         print(f"✅ Document processed in {doc_time:.1f}s")
        
#         # Step 2: Process questions sequentially
#         print("🔍 Processing questions...")
#         query_start = time.time()
        
#         answers = []
#         for i, question in enumerate(request.questions):
#             print(f"  Processing question {i+1}/{len(request.questions)}")
            
#             # Process each question individually
#             raw_answer = await loop.run_in_executor(
#                 executor,
#                 analyze_query_with_vectorstore_fast,
#                 question,
#                 vectorstore,
#                 False  # cleanup_after parameter
#             )
            
#             # Clean and format answer
#             if isinstance(raw_answer, str):
#                 # Try to parse JSON response
#                 try:
#                     parsed = json.loads(raw_answer)
#                     clean_answer = parsed.get("justification", raw_answer)
#                 except json.JSONDecodeError:
#                     clean_answer = raw_answer
#             else:
#                 clean_answer = str(raw_answer)
            
#             # Basic cleaning
#             clean_answer = clean_answer.strip()
#             if clean_answer.startswith("Answer:"):
#                 clean_answer = clean_answer[7:].strip()
            
#             answers.append(clean_answer)
        
#         query_time = time.time() - query_start
#         print(f"✅ Questions processed in {query_time:.1f}s")
        
#         total_time = time.time() - start_time
        
#         print(f"🎉 Request completed in {total_time:.1f}s")
#         print(f"📊 Breakdown - Doc: {doc_time:.1f}s, Query: {query_time:.1f}s")
        
#         return QueryResponse(answers=answers)
        
#     except Exception as e:
#         error_time = time.time() - start_time
#         print(f"❌ Error after {error_time:.1f}s: {str(e)}")
        
#         # Return partial response for debugging
#         return QueryResponse(
#             answers=[f"Processing error: {str(e)}"] * len(request.questions)
#         )

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(request: QueryRequest, authorization: str = Header(...)):
    start_time = time.time()
    verify_auth(authorization)
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
# Performance monitoring endpoint
@app.get("/api/v1/stats")
def get_performance_stats():
    return {
        "optimizations": [
            "Smaller chunks (800 chars) for accuracy",
            "Higher overlap (200) for context",
            "Score threshold retrieval",
            "Sequential query processing",
            "Insurance-specific prompts",
            "Cached LLM connections"
        ],
        "expected_performance": {
            "document_processing": "10-25s",
            "query_processing": "1-3s per question",
            "sequential_10_questions": "10-30s for queries"
        },
        "model_info": {
            "embeddings": "all-MiniLM-L6-v2",
            "llm": "llama-3.1-8b-instant",
            "chunk_size": 800,
            "overlap": 200
        }
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