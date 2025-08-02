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

load_dotenv()

# Keep your perfect chunking parameters
CHUNK_SIZE = 1500           # Your perfect setting
CHUNK_OVERLAP = 400        # Your perfect setting
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNKS = 500          # Increased slightly for better coverage

# Global cache with thread safety
_embeddings_cache = None
_embeddings_lock = threading.Lock()

def get_embeddings():
    """Thread-safe cached embeddings with enhanced configuration"""
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
    """Enhanced document download with better error handling"""
    print("📥 Downloading document...")
    start = time.time()
    
    try:
        with requests.Session() as session:
            # Enhanced headers for better compatibility
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            })
            
            response = session.get(url, stream=True, timeout=30)  # Increased timeout
            response.raise_for_status()
            
            # Better file extension detection
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if not ext:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    ext = '.pdf'
                elif 'word' in content_type or 'officedocument' in content_type:
                    ext = '.docx'
                else:
                    ext = '.pdf'  # Default to PDF
            
            # Stream to temporary file with better error handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        total_size += len(chunk)
                        
                        # Safety check for very large files
                        if total_size > 100 * 1024 * 1024:  # 100MB limit
                            raise Exception("File too large (>100MB)")
                
                temp_path = temp_file.name
            
            size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            print(f"✅ Downloaded {size_mb:.1f}MB in {time.time()-start:.1f}s")
            return temp_path
            
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def enhanced_content_cleaning(text: str) -> str:
    """Enhanced content cleaning for better text quality"""
    if not text:
        return ""
    
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks -> double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces -> single space
    
    # Remove common OCR artifacts and noise
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\$\%\@\#\&\*\+\=\<\>\|\~\`]', ' ', text)
    
    # Fix common spacing issues
    text = re.sub(r'\s+([\.,:;!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([\.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentence end
    
    # Remove very short "sentences" that are likely noise
    sentences = text.split('.')
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence.split()) > 2:  # At least 10 chars and 2 words
            clean_sentences.append(sentence)
    
    return '. '.join(clean_sentences) if clean_sentences else text

def load_document(file_path: str):
    """Enhanced document loading with better content filtering"""
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
        
        # Enhanced content filtering and cleaning
        filtered_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            
            # Skip very short or empty pages
            if len(content) < 50:
                continue
            
            # Skip pages that are mostly non-text (tables of numbers, etc.)
            word_count = len(content.split())
            if word_count < 10:
                continue
            
            # Enhanced content cleaning
            cleaned_content = enhanced_content_cleaning(content)
            
            # Skip if cleaning removed too much content
            if len(cleaned_content) < len(content) * 0.3:  # Less than 30% remained
                cleaned_content = content  # Use original
            
            # Update document with cleaned content
            doc.page_content = cleaned_content
            
            # Enhanced metadata
            doc.metadata.update({
                'page': i + 1,
                'char_count': len(cleaned_content),
                'word_count': len(cleaned_content.split()),
                'source_file': os.path.basename(file_path)
            })
            
            filtered_docs.append(doc)
        
        print(f"✅ Loaded {len(filtered_docs)} pages in {time.time()-start:.1f}s")
        return filtered_docs
        
    except Exception as e:
        raise Exception(f"Document loading failed: {str(e)}")

def smart_chunk_documents(documents):
    """Enhanced smart chunking with your perfect settings"""
    print("✂️ Smart chunking with enhanced quality control...")
    start = time.time()
    
    # Use your perfect settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],  # Added comma separator
        keep_separator=True,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enhanced quality filtering
    quality_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        
        # Basic quality checks
        if len(content) < 30:  # Minimum length
            continue
            
        word_count = len(content.split())
        if word_count < 5:  # Minimum word count
            continue
        
        # Skip chunks that are mostly headers or navigation
        content_lower = content.lower()
        skip_patterns = [
            r'^\s*(page\s+\d+|chapter\s+\d+|section\s+\d+)\s*',
            r'^\s*(table\s+of\s+contents|index|bibliography)\s*',
            r'^\s*\d+\s*,  # Just numbers',
            r'^\s*[ivxlcdm]+\s*,  # Roman numerals only'
        ]
        
        should_skip = any(re.match(pattern, content_lower) for pattern in skip_patterns)
        if should_skip:
            continue
        
        # Skip chunks with too high ratio of numbers/symbols to words
        alpha_chars = sum(1 for c in content if c.isalpha())
        total_chars = len(content)
        if total_chars > 0 and (alpha_chars / total_chars) < 0.3:  # Less than 30% letters
            continue
        
        # Enhance chunk metadata for better retrieval
        chunk.metadata.update({
            'chunk_length': len(content),
            'word_count': word_count,
            'alpha_ratio': alpha_chars / total_chars if total_chars > 0 else 0,
            'has_numbers': bool(re.search(r'\d', content)),
            'has_currency': bool(re.search(r'[\$£€¥]', content)),
            'has_percentage': bool(re.search(r'\d+%', content)),
        })
        
        quality_chunks.append(chunk)
    
    # Sort by quality indicators for better chunk selection
    def chunk_quality_score(chunk):
        """Score chunk quality for prioritization"""
        score = 0
        metadata = chunk.metadata
        
        # Length-based scoring (sweet spot around your chunk size)
        length = metadata.get('chunk_length', 0)
        if 800 <= length <= 1200:  # Around your perfect CHUNK_SIZE
            score += 10
        elif 600 <= length <= 1400:
            score += 5
        
        # Content quality indicators
        if metadata.get('has_numbers', False):
            score += 3  # Numbers often indicate important data
        if metadata.get('has_currency', False):
            score += 5  # Financial information is often key
        if metadata.get('has_percentage', False):
            score += 4  # Percentages are important in insurance
        
        # Text density (higher alpha ratio = better text quality)
        alpha_ratio = metadata.get('alpha_ratio', 0)
        score += int(alpha_ratio * 10)
        
        return score
    
    # Sort chunks by quality (best first)
    quality_chunks.sort(key=chunk_quality_score, reverse=True)
    
    # Limit chunks but keep the best ones
    if len(quality_chunks) > MAX_CHUNKS:
        print(f"⚠️ Limiting to {MAX_CHUNKS} highest quality chunks (from {len(quality_chunks)})")
        quality_chunks = quality_chunks[:MAX_CHUNKS]
    
    print(f"✅ Created {len(quality_chunks)} quality chunks in {time.time()-start:.1f}s")
    return quality_chunks

def create_vectorstore(chunks):
    """Enhanced vector store creation with optimization"""
    print(f"🧠 Creating enhanced vector store from {len(chunks)} chunks...")
    start = time.time()
    
    embeddings = get_embeddings()
    
    # Create vector store with batching for stability
    batch_size = 50  # Process in smaller batches
    all_vectorstores = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        if i == 0:
            # Create initial vectorstore
            vectorstore = FAISS.from_documents(batch_chunks, embeddings)
        else:
            # Create batch vectorstore and merge
            batch_vectorstore = FAISS.from_documents(batch_chunks, embeddings)
            vectorstore.merge_from(batch_vectorstore)
            
            # Clean up batch vectorstore
            del batch_vectorstore
            gc.collect()
    
    print(f"✅ Enhanced vector store ready in {time.time()-start:.1f}s")
    return vectorstore

def process_document_from_url(document_url: str, cleanup_after_use: bool = True, return_chunks: bool = False):
    """Enhanced document processing with caching and quality improvements"""
    total_start = time.time()
    temp_file = None
    chunks = None
    documents = None
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = f"/app/faiss_index/{url_hash}"
    
    # Create cache directory
    os.makedirs("/app/faiss_index", exist_ok=True)
    embeddings = get_embeddings()
    
    # Check for cached vectorstore
    if os.path.exists(cache_path):
        try:
            vectorstore = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
            print(f"✅ Loaded cached vectorstore in {time.time() - total_start:.1f}s")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}, processing fresh...")
    
    try:
        # Enhanced processing pipeline
        temp_file = download_document(document_url)
        documents = load_document(temp_file)
        chunks = smart_chunk_documents(documents)
        vectorstore = create_vectorstore(chunks)
        
        # Save to cache for future use
        try:
            vectorstore.save_local(cache_path)
            print(f"✅ Saved vectorstore to cache: {cache_path}")
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
        
        total_time = time.time() - total_start
        print(f"🎉 TOTAL PROCESSING TIME: {total_time:.1f}s")
        print(f"📊 Rate: {len(chunks)/total_time:.1f} chunks/sec")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Enhanced pipeline error: {str(e)}")
        raise
    finally:
        # Cleanup intermediate data
        if cleanup_after_use:
            if chunks:
                cleanup_chunks(chunks)
            if documents:
                for doc in documents:
                    if hasattr(doc, 'page_content'):
                        del doc.page_content
                    if hasattr(doc, 'metadata'):
                        doc.metadata.clear()
                documents.clear()
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print("🧹 Cleaned up temp file")
                except:
                    pass

def process_and_query_with_cleanup(document_url: str, query_function, *query_args):
    """
    Enhanced process document, run queries, then clean up everything
    
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
        # Process document with enhanced pipeline
        vectorstore = process_document_from_url(document_url, cleanup_after_use=False)
        
        # Run queries
        print("🔍 Running enhanced queries...")
        query_start = time.time()
        
        # Call the query function with vectorstore and other args
        results = query_function(vectorstore, *query_args)
        
        query_time = time.time() - query_start
        print(f"✅ Enhanced queries completed in {query_time:.1f}s")
        
        return results
        
    finally:
        # Always cleanup, even if queries fail
        print("🧹 Starting enhanced cleanup...")
        
        if vectorstore:
            cleanup_vectorstore(vectorstore)
        
        # Final garbage collection
        gc.collect()
        print("✅ Enhanced cleanup finished")

def warmup_embeddings():
    """Enhanced warmup with insurance-specific content"""
    print("🔥 Warming up embeddings and LLM with insurance content...")
    start = time.time()
    
    # Warm up embeddings with insurance-domain content
    embeddings = get_embeddings()
    insurance_samples = [
        "Insurance policy coverage and benefits for medical expenses",
        "Deductible amount and premium payment requirements",
        "Claims processing and reimbursement procedures",
        "Exclusions and limitations of coverage terms",
        "Copayment and coinsurance responsibilities"
    ]
    embeddings.embed_documents(insurance_samples)
    
    # Warm up LLM with insurance query
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,  # Match enhanced settings
            max_tokens=600,   # Match enhanced settings
            request_timeout=15
        )
        llm.invoke("Warmup query: What is an insurance policy deductible and how does it work?")
    except Exception as e:
        print(f"⚠️ LLM warmup failed: {e}")
    
    print(f"✅ Enhanced warmup completed in {time.time() - start:.1f}s")

def validate_document_quality(file_path: str) -> dict:
    """Validate document quality before processing"""
    try:
        file_size = os.path.getsize(file_path)
        
        # Quick content sample
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        
        # Load first few pages for quality check
        docs = loader.load()
        sample_content = ""
        for doc in docs[:3]:  # First 3 pages
            sample_content += doc.page_content
        
        # Quality metrics
        word_count = len(sample_content.split())
        char_count = len(sample_content)
        alpha_ratio = sum(1 for c in sample_content if c.isalpha()) / max(char_count, 1)
        
        quality_info = {
            'file_size_mb': file_size / (1024 * 1024),
            'sample_word_count': word_count,
            'sample_char_count': char_count,
            'alpha_ratio': alpha_ratio,
            'estimated_quality': 'high' if alpha_ratio > 0.5 and word_count > 100 else 'medium' if alpha_ratio > 0.3 else 'low'
        }
        
        print(f"📊 Document quality: {quality_info['estimated_quality']} (alpha ratio: {alpha_ratio:.2f})")
        return quality_info
        
    except Exception as e:
        print(f"⚠️ Quality validation failed: {e}")
        return {'estimated_quality': 'unknown'}

# Legacy compatibility
def load_documents_from_directory(data_path):
    """Enhanced directory loading with quality filtering"""
    from langchain_community.document_loaders import DirectoryLoader
    
    loader = DirectoryLoader(
        data_path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    
    # Apply enhanced content cleaning to directory docs
    enhanced_docs = []
    for doc in docs:
        cleaned_content = enhanced_content_cleaning(doc.page_content)
        if len(cleaned_content) > 50:  # Filter out very short content
            doc.page_content = cleaned_content
            enhanced_docs.append(doc)
    
    print(f"✅ Enhanced directory loading: {len(enhanced_docs)} quality documents")
    return enhanced_docs