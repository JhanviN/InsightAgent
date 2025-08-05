
import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
import threading
from typing import Optional
import gc
import hashlib
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF
from langchain.schema import Document

load_dotenv()

# Keep your perfect chunking parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNKS = 1000

# Global cache with thread safety
_embeddings_cache = None
_embeddings_lock = threading.Lock()

def get_embeddings():
    """Thread-safe cached embeddings"""
    global _embeddings_cache
    
    if _embeddings_cache is None:
        with _embeddings_lock:
            if _embeddings_cache is None:
                _embeddings_cache = HuggingFaceEmbeddings(
                    model_name=EMBED_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
    return _embeddings_cache

def download_document(url: str) -> str:
    """Streamlined document download"""
    try:
        with requests.Session() as session:
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Simple file extension detection
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if not ext:
                content_type = response.headers.get('content-type', '').lower()
                ext = '.pdf' if 'pdf' in content_type else '.docx' if 'word' in content_type else '.pdf'
            
            # Stream to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                return temp_file.name
            
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def clean_content(text: str) -> str:
    """Single-pass content cleaning"""
    if not text:
        return ""
    
    # Single regex pass for cleanup
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces
    text = re.sub(r'\s+([\.,:;!?])', r'\1', text)  # Space before punctuation
    
    return text.strip()

def load_document(file_path: str):
    """Optimized single-pass document loading with PyMuPDF"""
    ext = os.path.splitext(file_path)[1].lower()
    docs = []
    
    try:
        if ext == '.pdf':
            # Single pass with PyMuPDF only
            pdf_document = fitz.open(file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text()
                
                # Extract tables efficiently
                has_tables = False
                try:
                    tables = page.find_tables()
                    if tables:
                        has_tables = True
                        table_content = "\n".join([
                            " | ".join(str(cell) for cell in table.extract()[row] if cell)
                            for table in tables for row in range(len(table.extract()))
                        ])
                        text += "\n" + table_content
                except:
                    # Fallback if table extraction fails
                    pass
                
                if len(text.strip()) < 50:
                    continue
                
                cleaned_content = clean_content(text)
                
                doc = Document(
                    page_content=cleaned_content,
                    metadata={
                        'page': page_num + 1,
                        'source_file': os.path.basename(file_path),
                        'has_table': has_tables,
                        'has_sub_limit': bool(re.search(r'sub-limit|\%\s*of\s*SI', text, re.IGNORECASE))
                    }
                )
                docs.append(doc)
            
            pdf_document.close()
        else:
            # DOCX processing
            loader = Docx2txtLoader(file_path)
            raw_docs = loader.load()
            
            for i, doc in enumerate(raw_docs):
                content = doc.page_content.strip()
                if len(content) < 50:
                    continue
                
                cleaned_content = clean_content(content)
                doc.page_content = cleaned_content
                doc.metadata.update({
                    'page': i + 1,
                    'source_file': os.path.basename(file_path),
                    'has_sub_limit': bool(re.search(r'sub-limit|\%\s*of\s*SI', content, re.IGNORECASE))
                })
                docs.append(doc)
        
        return docs
        
    except Exception as e:
        raise Exception(f"Document loading failed: {str(e)}")

def chunk_documents(documents):
    """Optimized chunking with your perfect settings"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " | ", " - "],
        keep_separator=True,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    quality_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) < 30 or len(content.split()) < 5:
            continue
        if content.lower().startswith(('page ', 'chapter ', 'section ')):
            continue
        quality_chunks.append(chunk)
    
    if len(quality_chunks) > MAX_CHUNKS:
        quality_chunks = quality_chunks[:MAX_CHUNKS]
    
    return quality_chunks

def create_vectorstore(chunks):
    """Optimized single-pass vector store creation"""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def process_document_from_url(document_url: str, cleanup_after_use: bool = True):
    """Streamlined document processing pipeline"""
    temp_file = None
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = f"/app/faiss_index/{url_hash}"
    
    os.makedirs("/app/faiss_index", exist_ok=True)
    
    try:
        temp_file = download_document(document_url)
        documents = load_document(temp_file)
        chunks = chunk_documents(documents)
        if not chunks:
            raise Exception("No chunks created")
        vectorstore = create_vectorstore(chunks)
        try:
            vectorstore.save_local(cache_path)
        except Exception:
            pass
        return vectorstore
    except Exception as e:
        raise Exception(f"Processing failed: {str(e)}")
    finally:
        if cleanup_after_use and temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

def warmup_embeddings():
    """Efficient warmup"""
    embeddings = get_embeddings()
    sample_texts = [
        "Insurance policy coverage and benefits",
        "Deductible amount and premium payment",
        "Claims processing and reimbursement"
    ]
    embeddings.embed_documents(sample_texts)
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=600,
            request_timeout=15
        )
        llm.invoke("What is an insurance deductible?")
    except Exception:
        pass
