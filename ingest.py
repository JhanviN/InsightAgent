import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
import threading
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Optimized parameters for accuracy vs speed
CHUNK_SIZE = 800           # Smaller for better accuracy
CHUNK_OVERLAP = 200        # Higher overlap for context preservation
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNKS = 150          # Limit for memory efficiency

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
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
                )
    return _embeddings_cache

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

def process_document_from_url(document_url: str):
    """Main processing pipeline"""
    total_start = time.time()
    temp_file = None
    
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
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
        raise
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                print("🧹 Cleaned up temp file")
            except:
                pass

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

def save_to_faiss(chunks, embeddings, index_path):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

if __name__ == "__main__":
    print("🚀 OPTIMIZED PROCESSING TEST")
    warmup_embeddings()
    
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        vectorstore = process_document_from_url(test_url)
        
        # Test retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents("grace period premium payment")
        print(f"🔍 Found {len(docs)} relevant chunks")
        print("✅ SUCCESS!")
        
    except Exception as e:
        print(f"❌ Error: {e}")