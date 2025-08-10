from fastapi import FastAPI, Header, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from ingest import process_document_from_url, warmup_embeddings, load_persistent_vectorstore
from query import  process_queries_batch, load_query_cache, save_query_cache
from logging_service import request_logger, generate_request_id, log_api_request
import gc

load_dotenv()
EXPECTED_TOKEN = os.getenv("AUTH_TOKEN")

executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="DocProcessor")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting InsightAgent API...")
    print("🔥 Warming up embeddings model...")
    print("📋 Loading persistent FAISS index and query cache...")
    
    loop = asyncio.get_event_loop()
    
    # Initialize persistent systems
    await loop.run_in_executor(executor, load_persistent_vectorstore)
    await loop.run_in_executor(executor, load_query_cache)
    await loop.run_in_executor(executor, warmup_embeddings)
    
    print("✅ API ready with persistent storage!")
    yield
    
    print("🛑 Shutting down...")
    print("💾 Saving query cache...")
    
    # Save cache before shutdown
    await loop.run_in_executor(executor, save_query_cache)
    
    executor.shutdown(wait=True)
    request_logger.close()  # Clean up logging resources
    gc.collect()
    print("✅ Shutdown complete")

app = FastAPI(
    title="InsightAgent - Document Query API",
    version="2.1",
    description="High-performance document analysis with persistent storage and query caching",
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
        "version": "2.1",
        "features": [
            "Persistent FAISS vector store",
            "Query result caching",
            "Smart document chunking",
            "Cached embeddings",
            "Sequential query processing",
            "Insurance domain optimization",
            "Request logging to Google Sheets"
        ]
    }

@app.get("/health")
def health_check():
    import multiprocessing
    from query import _persistent_query_cache
    from ingest import _processed_documents
    
    children = multiprocessing.active_children()
    return {
        "status": "ok",
        "workers": [str(child) for child in children],
        "logging_enabled": request_logger.sheet is not None,
        "persistent_storage": {
            "faiss_index_exists": os.path.exists("./faiss_index"),
            "query_cache_size": len(_persistent_query_cache),
            "processed_documents": len(_processed_documents)
        }
    }

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_document_queries(request: QueryRequest, authorization: str = Header(...)):
    start_time = time.time()
    request_id = generate_request_id()
    
    print(f"🚀 Processing request {request_id} with {len(request.questions)} questions")
    
    verify_auth(authorization)
    vectorstore = None
    doc_time = 0
    query_time = 0
    success = True
    error_message = ""
    answers = []
    cache_hits = 0
    
    try:
        loop = asyncio.get_event_loop()
        
        print("📄 Processing document...")
        doc_start = time.time()
        vectorstore = await loop.run_in_executor(executor, process_document_from_url, request.documents)
        doc_time = time.time() - doc_start
        print(f"✅ Document processed in {doc_time:.1f}s")
        
        print("🔍 Processing questions with caching...")
        query_start = time.time()
        answers = await loop.run_in_executor(executor, process_queries_batch, vectorstore, request.questions)
        query_time = time.time() - query_start
        print(f"✅ Questions processed in {query_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"🎉 Request {request_id} completed in {total_time:.1f}s")
        print(f"📊 Breakdown - Doc: {doc_time:.1f}s, Query: {query_time:.1f}s")
        
        # Log successful request
        await log_api_request(
            request_id=request_id,
            document_url=request.documents,
            questions=request.questions,
            answers=answers,
            total_time=total_time,
            doc_time=doc_time,
            query_time=query_time,
            success=True
        )
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        success = False
        error_message = str(e)
        error_time = time.time() - start_time
        
        print(f"❌ Request {request_id} failed after {error_time:.1f}s: {error_message}")
        
        # Create error answers
        answers = [f"Processing error: {error_message}"] * len(request.questions)
        
        # Log failed request
        await log_api_request(
            request_id=request_id,
            document_url=request.documents,
            questions=request.questions,
            answers=answers,
            total_time=error_time,
            doc_time=doc_time,
            query_time=query_time,
            success=False,
            error_message=error_message
        )
        
        return QueryResponse(answers=answers)
        
    finally:
        # Don't cleanup persistent vectorstore, just clean memory
        gc.collect()

@app.get("/api/v1/stats")
def get_performance_stats():
    from query import _persistent_query_cache
    from ingest import _processed_documents
    
    return {
        "optimizations": [
            "Persistent FAISS vector store",
            "Query result caching",
            "Smart chunking (1500 chars)",
            "Higher overlap (400 chars)",
            "MMR-only retrieval",
            "Sequential query processing",
            "Insurance-specific prompts",
            "Cached LLM connections",
            "Request logging to Google Sheets"
        ],
        "expected_performance": {
            "document_processing": "5-10s (instant if cached)",
            "query_processing": "0.1-2s per question (instant if cached)",
            "sequential_10_questions": "1-20s for queries (depending on cache hits)"
        },
        "model_info": {
            "embeddings": "all-MiniLM-L6-v2",
            "llm": "openai/gpt-oss-120b",
            "chunk_size": 1500,
            "overlap": 400
        },
        "persistent_storage": {
            "faiss_index_path": "./faiss_index",
            "query_cache_path": "./query_cache.json",
            "cached_queries": len(_persistent_query_cache),
            "processed_documents": len(_processed_documents)
        },
        "logging": {
            "enabled": request_logger.sheet is not None,
            "storage": "Google Sheets"
        }
    }

@router.get("/cache/stats")
async def get_cache_stats(authorization: str = Header(...)):
    """Get cache statistics (requires authentication)"""
    verify_auth(authorization)
    
    from query import _persistent_query_cache
    from ingest import _processed_documents
    
    return {
        "query_cache": {
            "total_entries": len(_persistent_query_cache),
            "cache_file_exists": os.path.exists("./query_cache.json"),
            "recent_queries": list(_persistent_query_cache.keys())[-5:] if _persistent_query_cache else []
        },
        "document_cache": {
            "total_processed": len(_processed_documents),
            "faiss_index_exists": os.path.exists("./faiss_index"),
            "recent_documents": [doc_info.get('url', 'Unknown')[:50] + '...' 
                              for doc_info in list(_processed_documents.values())[-3:]]
        },
        "storage_info": {
            "faiss_index_size_mb": round(
                sum(os.path.getsize(os.path.join("./faiss_index", f)) 
                    for f in os.listdir("./faiss_index") 
                    if os.path.isfile(os.path.join("./faiss_index", f))) / (1024*1024), 2
            ) if os.path.exists("./faiss_index") else 0,
            "query_cache_size_kb": round(
                os.path.getsize("./query_cache.json") / 1024, 2
            ) if os.path.exists("./query_cache.json") else 0
        }
    }

@router.post("/cache/clear")
async def clear_caches(authorization: str = Header(...)):
    """Clear all caches (requires authentication)"""
    verify_auth(authorization)
    
    try:
        from query import _persistent_query_cache, save_query_cache
        from ingest import _processed_documents, save_document_metadata
        
        # Clear query cache
        old_query_count = len(_persistent_query_cache)
        _persistent_query_cache.clear()
        save_query_cache()
        
        # Clear document metadata (but keep FAISS index)
        old_doc_count = len(_processed_documents)
        _processed_documents.clear()
        save_document_metadata()
        
        return {
            "success": True,
            "cleared": {
                "queries": old_query_count,
                "document_metadata": old_doc_count
            },
            "note": "FAISS index preserved - only metadata cleared"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# New endpoint to get recent logs (optional)
@router.get("/logs/recent")
async def get_recent_logs(authorization: str = Header(...)):
    """Get recent request logs (requires authentication)"""
    verify_auth(authorization)
    
    if not request_logger.sheet:
        return {"error": "Logging not enabled"}
    
    try:
        # Get the last 10 rows (excluding header)
        all_records = request_logger.sheet.get_all_records()
        recent_logs = all_records[-10:] if len(all_records) > 10 else all_records
        return {"recent_logs": recent_logs, "total_count": len(all_records)}
    except Exception as e:
        return {"error": f"Failed to fetch logs: {str(e)}"}

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting InsightAgent API server with persistent storage...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )