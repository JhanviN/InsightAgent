import os
import requests
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
import threading
import gc
import hashlib
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import fitz
from pathlib import Path

load_dotenv()

# Keep your perfect chunking parameters
CHUNK_SIZE = 1500           # Your perfect setting
CHUNK_OVERLAP = 400        # Your perfect setting
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNKS = 1000          # Increased slightly for better coverage

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
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive'
            })
            
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            parsed_url = urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if not ext:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    ext = '.pdf'
                elif 'word' in content_type or 'officedocument' in content_type:
                    ext = '.docx'
                else:
                    ext = '.pdf'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        total_size += len(chunk)
                        if total_size > 100 * 1024 * 1024:
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
    
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\$\%\@\#\&\*\+\=\<\>\|\~\`]', ' ', text)
    text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
    text = re.sub(r'([\.!?])\s*([A-Z])', r'\1 \2', text)
    
    sentences = text.split('.')
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence.split()) > 2:
            clean_sentences.append(sentence)
    
    return '. '.join(clean_sentences) if clean_sentences else text

def load_document(file_path: str):
    print("📄 Loading document...")
    start = time.time()
    ext = os.path.splitext(file_path)[1].lower()
    docs = []
    try:
        if ext == '.pdf':
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf):
                    text = page.get_text("text") or ""
                    if len(text.strip()) < 50:
                        continue
                    cleaned_content = enhanced_content_cleaning(text)
                    if len(cleaned_content) < len(text) * 0.3:
                        cleaned_content = text
                    doc = Document(
                        page_content=cleaned_content,
                        metadata={
                            'page': page_num + 1,
                            'char_count': len(cleaned_content),
                            'word_count': len(cleaned_content.split()),
                            'source_file': os.path.basename(file_path),
                            'has_table': False,
                            'has_sub_limit': bool(re.search(r'sub-limit|\%\s*of\s*SI', text, re.IGNORECASE)),
                            'has_exception': any(term in text.lower() for term in [
                                'preferred provider network', 'ppn', 'exemption', 'listed procedure', 'not apply'
                            ])
                        }
                    )
                    for plan in ['Plan A', 'Plan B', 'Plan C']:
                        if plan in text:
                            doc.metadata['plan'] = plan
                            break
                    if doc.metadata.get('has_exception'):
                        doc.metadata['exception_type'] = 'PPN' if 'ppn' in text.lower() or 'preferred provider network' in text.lower() else 'Other'
                    docs.append(doc)
        else:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages in {time.time()-start:.1f}s")
        return docs
    except Exception as e:
        raise Exception(f"Document loading failed: {str(e)}")

def smart_chunk_documents(documents):
    print("✂️ Smart chunking with enhanced quality control...")
    start = time.time()
    chunk_size = CHUNK_SIZE
    for doc in documents:
        if doc.metadata.get('has_table', False):
            chunk_size = min(CHUNK_SIZE, 1000)
            break
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " | ", " - "],
        keep_separator=True,
        add_start_index=True
    )
    
    chunks = splitter.split_documents(documents)
    
    quality_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if len(content) < 30:
            continue
        word_count = len(content.split())
        if word_count < 5:
            continue
        content_lower = content.lower()
        skip_patterns = [
            r'^\s*(page\s+\d+|chapter\s+\d+|section\s+\d+)\s*',
            r'^\s*(table\s+of\s+contents|index|bibliography)\s*',
            r'^\s*\d+\s*$',
            r'^\s*[ivxlcdm]+\s*$',
        ]
        should_skip = any(re.match(pattern, content_lower) for pattern in skip_patterns)
        if should_skip:
            continue
        alpha_chars = sum(1 for c in content if c.isalpha())
        total_chars = len(content)
        if total_chars > 0 and (alpha_chars / total_chars) < 0.3:
            continue
        chunk.metadata.update({
            'chunk_length': len(content),
            'word_count': word_count,
            'alpha_ratio': alpha_chars / total_chars if total_chars > 0 else 0,
            'has_numbers': bool(re.search(r'\d', content)),
            'has_currency': bool(re.search(r'[\$£€¥]', content)),
            'has_percentage': bool(re.search(r'\d+%', content)),
        })
        quality_chunks.append(chunk)
    
    def chunk_quality_score(chunk):
        score = 0
        metadata = chunk.metadata
        length = metadata.get('chunk_length', 0)
        if 800 <= length <= 1200:
            score += 10
        elif 600 <= length <= 1400:
            score += 5
        if metadata.get('has_numbers', False):
            score += 3
        if metadata.get('has_currency', False):
            score += 5
        if metadata.get('has_percentage', False):
            score += 4
        alpha_ratio = metadata.get('alpha_ratio', 0)
        score += int(alpha_ratio * 10)
        return score
    
    quality_chunks.sort(key=chunk_quality_score, reverse=True)
    if len(quality_chunks) > MAX_CHUNKS:
        print(f"⚠️ Limiting to {MAX_CHUNKS} highest quality chunks (from {len(quality_chunks)})")
        quality_chunks = quality_chunks[:MAX_CHUNKS]
    
    print(f"✅ Created {len(quality_chunks)} quality chunks in {time.time()-start:.1f}s")
    return quality_chunks

def create_vectorstore(chunks):
    start_time = time.time()
    print(f"🧠 Creating vector store from {len(chunks)} chunks...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector store ready in {time.time() - start_time:.1f}s")
    return vectorstore

def process_document_from_url(document_url: str, cleanup_after_use: bool = True):
    total_start = time.time()
    temp_file = None
    chunks = None
    documents = None
    cache_base = Path(os.getenv("CACHE_DIR", "./faiss_index"))
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = cache_base / url_hash
    embeddings = get_embeddings()
    try:
        cache_base.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                print("✅ Loading cached vectorstore...")
                cache_start = time.time()
                vectorstore = FAISS.load_local(str(cache_path), embeddings, allow_dangerous_deserialization=True)
                print(f"✅ Loaded cached vectorstore in {time.time() - cache_start:.1f}s")
                return vectorstore
            except Exception as e:
                print(f"⚠️ Cache load failed: {str(e)}, regenerating...")
        temp_file = download_document(document_url)
        print(f"📂 Temp file: {temp_file}, Size: {os.path.getsize(temp_file) / (1024 * 1024):.1f}MB")
        documents = load_document(temp_file)
        print(f"📜 Loaded {len(documents)} documents")
        for i, doc in enumerate(documents[:3]):
            print(f"Doc {i+1}: {doc.page_content[:100]}... Metadata: {doc.metadata}")
        chunks = smart_chunk_documents(documents)
        print(f"✂️ Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1}: {chunk.page_content[:100]}... Metadata: {chunk.metadata}")
        if not chunks:
            raise Exception("No chunks created, check document loading or chunking")
        vectorstore = create_vectorstore(chunks)
        print(f"🧠 Vector store created with {vectorstore.index.ntotal} vectors")
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(str(cache_path))
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
            gc.collect()

def process_and_query_with_cleanup(document_url: str, query_function, *query_args):
    vectorstore = None
    try:
        vectorstore = process_document_from_url(document_url, cleanup_after_use=False)
        print("🔍 Running enhanced queries...")
        query_start = time.time()
        results = query_function(vectorstore, *query_args)
        query_time = time.time() - query_start
        print(f"✅ Enhanced queries completed in {query_time:.1f}s")
        return results
    finally:
        if vectorstore:
            cleanup_vectorstore(vectorstore)
        gc.collect()
        print("✅ Enhanced cleanup finished")

def warmup_embeddings():
    print("🔥 Warming up embeddings and LLM with insurance content...")
    start = time.time()
    embeddings = get_embeddings()
    insurance_samples = [
        "Insurance policy coverage and benefits for medical expenses",
        "Deductible amount and premium payment requirements",
        "Claims processing and reimbursement procedures",
        "Exclusions and limitations of coverage terms",
        "Copayment and coinsurance responsibilities"
    ]
    embeddings.embed_documents(insurance_samples)
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=600,
            request_timeout=15
        )
        llm.invoke("Warmup query: What is an insurance policy deductible and how does it work?")
    except Exception as e:
        print(f"⚠️ LLM warmup failed: {e}")
    print(f"✅ Enhanced warmup completed in {time.time() - start:.1f}s")

def validate_document_quality(file_path: str) -> dict:
    try:
        file_size = os.path.getsize(file_path)
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = Docx2txtLoader(file_path)
        docs = loader.load()
        sample_content = ""
        for doc in docs[:3]:
            sample_content += doc.page_content
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