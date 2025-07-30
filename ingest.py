import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
import threading
from typing import Optional
import gc

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Optimized parameters for accuracy vs speed
CHUNK_SIZE = 1100           # Smaller for better accuracy
CHUNK_OVERLAP = 300        # Higher overlap for context preservation
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNKS = 400          # Limit for memory efficiency

# Global cache with thread safety
_embeddings_cache = None
_embeddings_lock = threading.Lock()

def get_embeddings():
    """Thread-safe cached embeddings"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        with _embeddings_lock:
            if _embeddings_cache is None:
                print(f"🔧 Loading embeddings: {EMBED_MODEL}")
                _embeddings_cache = HuggingFaceEmbeddings(
                    model_name=EMBED_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
    return _embeddings_cache

def cleanup_vectorstore(vectorstore):
    """Clean up vector store from memory"""
    try:
        if vectorstore is not None:
            # Clear FAISS index
            if hasattr(vectorstore, 'index'):
                del vectorstore.index
            
            # Clear document store
            if hasattr(vectorstore, 'docstore'):
                vectorstore.docstore.clear() if hasattr(vectorstore.docstore, 'clear') else None
                del vectorstore.docstore
            
            # Clear index to docstore mapping
            if hasattr(vectorstore, 'index_to_docstore_id'):
                vectorstore.index_to_docstore_id.clear()
                del vectorstore.index_to_docstore_id
            
            del vectorstore
            
        # Force garbage collection
        gc.collect()
        print("🧹 Vector store cleaned from memory")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {str(e)}")

def cleanup_chunks(chunks):
    """Clean up chunks from memory"""
    try:
        if chunks:
            for chunk in chunks:
                if hasattr(chunk, 'page_content'):
                    del chunk.page_content
                if hasattr(chunk, 'metadata'):
                    chunk.metadata.clear()
                    del chunk.metadata
            chunks.clear()
            del chunks
        
        gc.collect()
        print("🧹 Chunks cleaned from memory")
        
    except Exception as e:
        print(f"⚠️ Chunk cleanup warning: {str(e)}")

def download_document(url: str) -> str:
    """Optimized document download with proper error handling"""
    print("📥 Downloading document...")
    start = time.time()
    
    try:
        with requests.Session() as session:
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/1.0)'
            })
            
            response = session.get(url, stream=True, timeout=20)
            response.raise_for_status()
            
            # Determine file extension
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            if not ext:
                content_type = response.headers.get('content-type', '')
                ext = '.pdf' if 'pdf' in content_type else '.docx'
            
            # Stream to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                temp_path = temp_file.name
            
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            print(f"✅ Downloaded {size_mb:.1f}MB in {time.time()-start:.1f}s")
            return temp_path
            
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def load_document(file_path: str):
    """Load document with content filtering"""
    print("📄 Loading document...")
    start = time.time()
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            loader = PyPDFLoader(file_path)  # Default to PDF
        
        docs = loader.load()
        
        # Filter meaningful content
        filtered_docs = []
        for doc in docs:
            content = doc.page_content.strip()
            if len(content) > 100 and not content.isspace():
                # Clean content
                doc.page_content = ' '.join(content.split())
                filtered_docs.append(doc)
        
        print(f"✅ Loaded {len(filtered_docs)} pages in {time.time()-start:.1f}s")
        return filtered_docs
        
    except Exception as e:
        raise Exception(f"Document loading failed: {str(e)}")

def smart_chunk_documents(documents):
    """Smart chunking with accuracy focus"""
    print("✂️ Smart chunking...")
    start = time.time()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        keep_separator=True,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    # Quality filtering
    quality_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if (len(content) > 50 and 
            len(content.split()) > 8 and  # At least 8 words
            not content.lower().startswith(('page', 'chapter', 'section'))):
            quality_chunks.append(chunk)
    
    # Limit chunks for performance
    if len(quality_chunks) > MAX_CHUNKS:
        print(f"⚠️ Limiting to {MAX_CHUNKS} best chunks")
        quality_chunks = quality_chunks[:MAX_CHUNKS]
    
    print(f"✅ Created {len(quality_chunks)} chunks in {time.time()-start:.1f}s")
    return quality_chunks

def create_vectorstore(chunks):
    """Create optimized vector store"""
    print(f"🧠 Creating vector store from {len(chunks)} chunks...")
    start = time.time()
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"✅ Vector store ready in {time.time()-start:.1f}s")
    return vectorstore

def process_document_from_url(document_url: str, cleanup_after_use: bool = True, return_chunks: bool = False):
    """Main processing pipeline with optional cleanup"""
    total_start = time.time()
    temp_file = None
    chunks = None
    documents = None
    
    try:
        # Ensure embeddings are loaded
        get_embeddings()
        
        # Pipeline steps
        temp_file = download_document(document_url)
        documents = load_document(temp_file)
        chunks = smart_chunk_documents(documents)
        vectorstore = create_vectorstore(chunks)
        
        total_time = time.time() - total_start
        print(f"🎉 TOTAL TIME: {total_time:.1f}s")
        print(f"📊 Rate: {len(chunks)/total_time:.1f} chunks/sec")
        
        # Optional cleanup of intermediate data
        if cleanup_after_use:
            # Clean up documents (no longer needed)
            if documents:
                for doc in documents:
                    if hasattr(doc, 'page_content'):
                        del doc.page_content
                    if hasattr(doc, 'metadata'):
                        doc.metadata.clear()
                documents.clear()
                del documents
                documents = None
            
            # Keep chunks reference for manual cleanup later
            # Don't clean chunks here as they're still referenced in vectorstore
        
        # Return format based on parameters
        if return_chunks:
            return vectorstore, chunks
        else:
            return vectorstore
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        # Cleanup on error
        if chunks:
            cleanup_chunks(chunks)
        if documents:
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    del doc.page_content
            documents.clear()
        raise
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                print("🧹 Cleaned up temp file")
            except:
                pass

def process_and_query_with_cleanup(document_url: str, query_function, *query_args):
    """
    Process document, run queries, then clean up everything
    
    Args:
        document_url: URL to process
        query_function: Function to call for querying (from query.py)
        *query_args: Arguments to pass to query_function
    
    Returns:
        Query results
    """
    vectorstore = None
    chunks = None
    
    try:
        # Process document
        vectorstore, chunks = process_document_from_url(document_url, cleanup_after_use=False)
        
        # Run queries
        print("🔍 Running queries...")
        query_start = time.time()
        
        # Call the query function with vectorstore and other args
        results = query_function(vectorstore, *query_args)
        
        query_time = time.time() - query_start
        print(f"✅ Queries completed in {query_time:.1f}s")
        
        return results
        
    finally:
        # Always cleanup, even if queries fail
        print("🧹 Starting cleanup...")
        
        if chunks:
            cleanup_chunks(chunks)
        
        if vectorstore:
            cleanup_vectorstore(vectorstore)
        
        # Final garbage collection
        gc.collect()
        print("✅ Complete cleanup finished")

def warmup_embeddings():
    """Initialize embeddings model"""
    print("🔥 Warming up embeddings...")
    start = time.time()
    
    embeddings = get_embeddings()
    # Process dummy text to initialize
    embeddings.embed_documents(["Insurance policy coverage and benefits."])
    
    print(f"✅ Warmup done in {time.time()-start:.1f}s")

# Legacy compatibility
def load_documents_from_directory(data_path):
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return loader.load()
