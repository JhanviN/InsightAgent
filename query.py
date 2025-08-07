import os
import time
from dotenv import load_dotenv
import gc
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import re
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"

# Global caches
_llm_cache = None
_query_cache = {}
_cache_embeddings = None

def get_llm():
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=MODEL_NAME,
            temperature=0.1,
            max_tokens=600,
            request_timeout=15
        )
    return _llm_cache

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

def get_query_embeddings():
    global _cache_embeddings
    if _cache_embeddings is None:
        _cache_embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
        )
    return _cache_embeddings

def preprocess_query(query: str) -> str:
    query = ' '.join(query.split())
    expansions = {
        r'\bdeductible\b': 'deductible out-of-pocket amount',
        r'\bpremium\b': 'premium cost payment fee',
        r'\bcoverage\b': 'coverage benefit protection covered',
        r'\bclaim\b': 'claim reimbursement payout settlement',
        r'\bexclusion\b': 'exclusion excluded not covered',
        r'\bcopay\b': 'copay co-pay copayment co-payment',
        r'\bpolicy\b': 'policy contract agreement',
        r'\blimit\b': 'limit maximum cap ceiling',
        r'\bwhen\b': 'when what time period effective date',
        r'\bhow much\b': 'amount cost price value sum total',
        r'\bwhat is\b': 'definition meaning explanation details'
    }
    expanded_query = query
    for pattern, expansion in expansions.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded_query += f" {expansion}"
    return expanded_query

ENHANCED_INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert insurance policy analyst. Your job is to find ANY relevant information in the context to answer the question.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

CRITICAL INSTRUCTIONS:
1. READ EVERY SENTENCE in the context carefully - look for ANY information related to the question
2. Look for:
   - Exact matches for terms in the question
   - Related concepts and synonyms 
   - Numbers, amounts, percentages, dates, time periods
   - Conditions, requirements, or qualifying statements
   - Partial information that addresses part of the question
3. Even if information is not perfectly formatted or directly stated, extract what IS available
4. If you find relevant details, provide a complete answer with specific information from the context
5. Include amounts, percentages, conditions, timeframes, and any qualifying details
6. ONLY say "Information not found in the document" if you have thoroughly examined every sentence and found absolutely nothing relevant
7. Give exact to the point answers.
8. Try your best to find answers from the provided context
9. Reply in only 2-3 sentences.
10.Any special character/formatting not allowed in responses

Be thorough and extract every relevant detail available in the context.

ANSWER:"""
)

AGGRESSIVE_SEARCH_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""DOCUMENT CONTENT: {context}

QUESTION: {question}

TASK: Find ANY information in the document content that relates to this question. Look for:
- Direct answers
- Partial information 
- Related terms or concepts
- Numbers, dates, amounts
- Conditions or requirements

Extract and provide ALL relevant details found. If absolutely nothing relates to the question, then say "Information not found in the document."

RESPONSE:"""
)

def enhanced_retrieval_with_multiple_strategies(vectorstore, query_text: str, max_docs: int = 6) -> List[Document]:
    all_docs = []
    seen_content = set()
    try:
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": max_docs, "lambda_mult": 0.3}
        )
        docs_mmr = retriever_mmr.invoke(query_text)
        for doc in docs_mmr:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    except Exception as e:
        print(f"MMR retrieval failed: {e}")
    return all_docs[:max_docs]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def invoke_llm_with_retry(llm, prompt):
    return llm.invoke(prompt)

def analyze_query_with_vectorstore_fast(query_text: str, vectorstore, cleanup_after=True) -> str:
    try:
        query_emb = get_query_embeddings().embed_query(query_text)
        for cached_query, (cached_emb, cached_answer) in _query_cache.items():
            similarity = sum(a * b for a, b in zip(query_emb, cached_emb))
            if similarity > 0.95:
                print(f"✅ Cache hit for query: {query_text}")
                return cached_answer
        llm = get_llm()
        processed_query = preprocess_query(query_text)
        docs = enhanced_retrieval_with_multiple_strategies(vectorstore, query_text, max_docs=6)
        if not docs:
            print("⚠️ No documents retrieved")
            answer = "No relevant information found in the document."
            _query_cache[query_text] = (query_emb, answer)
            return answer
        print(f"📄 Retrieved {len(docs)} documents for analysis")
        context = "\n\n".join([doc.page_content for doc in docs])
        first_prompt = ENHANCED_INSURANCE_PROMPT.format(context=context, question=query_text)
        response = invoke_llm_with_retry(llm, first_prompt)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if answer.startswith("ANSWER:"):
            answer = answer[7:].strip()
        if (not answer or len(answer) < 20 or "not found" in answer.lower() or 
            "not mentioned" in answer.lower() or "does not contain" in answer.lower()):
            print("🔄 First attempt unsuccessful, trying aggressive search...")
            aggressive_prompt = AGGRESSIVE_SEARCH_PROMPT.format(context=context, question=query_text)
            response2 = invoke_llm_with_retry(llm, aggressive_prompt)
            answer2 = response2.content.strip() if hasattr(response2, 'content') else str(response2).strip()
            if answer2.startswith("RESPONSE:"):
                answer2 = answer2[9:].strip()
            if answer2 and len(answer2) > len(answer) and "not found" not in answer2.lower():
                answer = answer2
        _query_cache[query_text] = (query_emb, answer)
        return answer
    except Exception as e:
        print(f"❌ Enhanced query processing error: {str(e)}")
        return f"Error processing query: {str(e)}"
    finally:
        if cleanup_after:
            gc.collect()

def analyze_query_with_sources_fast(query_text: str, vectorstore, cleanup_after=True) -> dict:
    try:
        docs = enhanced_retrieval_with_multiple_strategies(vectorstore, query_text, max_docs=5)
        answer = analyze_query_with_vectorstore_fast(query_text, vectorstore, cleanup_after=False)
        sources = []
        for i, doc in enumerate(docs[:4]):
            sources.append({
                "content": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                "page": doc.metadata.get("page", f"chunk_{i+1}"),
                "relevance_rank": i+1,
                "content_length": len(doc.page_content)
            })
        result = {
            "answer": answer,
            "sources": sources,
            "query": query_text,
            "num_sources": len(docs),
            "retrieval_method": "mmr_only"
        }
        if cleanup_after:
            gc.collect()
        return result
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "query": query_text,
            "num_sources": 0
        }

def score_answer_quality(answer: str, question: str) -> int:
    if not answer or len(answer.strip()) < 10:
        return 0
    score = 30
    negative_phrases = [
        "not found", "not mentioned", "does not contain",
        "unable to determine", "not specified", "no information",
        "cannot be determined", "not available"
    ]
    if any(phrase in answer.lower() for phrase in negative_phrases):
        return 5
    if len(answer.split()) > 20:
        score += 20
    if any(char in answer for char in ['$', '%']) or re.search(r'\d+', answer):
        score += 25
    insurance_terms = [
        'policy', 'coverage', 'deductible', 'premium', 'benefit',
        'claim', 'exclusion', 'copay', 'limit', 'effective'
    ]
    found_terms = sum(1 for term in insurance_terms if term in answer.lower())
    score += found_terms * 5
    if any(word in answer.lower() for word in ['must', 'required', 'eligible', 'covered', 'applies']):
        score += 10
    return min(score, 100)

def cleanup_chain_components(chain=None, retriever=None):
    try:
        if chain:
            if hasattr(chain, 'combine_documents_chain'):
                del chain.combine_documents_chain
            if hasattr(chain, 'retriever'):
                del chain.retriever
            del chain
        if retriever:
            if hasattr(retriever, 'vectorstore'):
                del retriever.vectorstore
            del retriever
        gc.collect()
        print("🧹 Chain components cleaned")
    except Exception as e:
        print(f"⚠️ Chain cleanup warning: {str(e)}")

def query_with_auto_cleanup(vectorstore, query_text: str) -> str:
    return analyze_query_with_vectorstore_fast(query_text, vectorstore, cleanup_after=True)

def validate_answer_quality(answer: str, question: str) -> bool:
    score = score_answer_quality(answer, question)
    return score > 25

def test_single_query_performance(vectorstore, test_question: str):
    print(f"🧪 Testing single question: {test_question[:50]}...")
    start = time.time()
    answer = query_with_auto_cleanup(vectorstore, test_question)
    total_time = time.time() - start
    quality_score = score_answer_quality(answer, test_question)
    print(f"📊 Single Query Performance:")
    print(f"   Time taken: {total_time:.2f}s")
    print(f"   Quality score: {quality_score}/100")
    print(f"   Answer: {answer}")
    return answer

def analyze_query(query_text: str) -> str:
    embeddings = None
    vector_db = None
    try:
        embeddings = get_embeddings()
        vector_db = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return analyze_query_with_vectorstore_fast(query_text, vector_db, cleanup_after=True)
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if vector_db:
            try:
                if hasattr(vector_db, 'index'):
                    del vector_db.index
                if hasattr(vector_db, 'docstore'):
                    vector_db.docstore.clear() if hasattr(vector_db.docstore, 'clear') else None
                del vector_db
            except:
                pass
        if embeddings:
            try:
                del embeddings
            except:
                pass
        gc.collect()

INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert insurance policy analyst. Use the provided context from the document to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Respond clearly and concisely in 2-3 sentences maximum.
- If the answer is not found in the context, reply with: "Information not found in the document."
- Avoid unnecessary elaboration, filler, or repeated context.
- Use normal punctuation; do not include escape characters or special formatting.

Answer:"""
)

def process_single_query_example(vectorstore, question: str):
    print(f"🔍 Processing: {question}")
    answer = query_with_auto_cleanup(vectorstore, question)
    print(f"✅ Answer: {answer}")
    print("🧹 Memory cleaned automatically")
    return answer

def process_queries_batch(vectorstore, questions: List[str], batch_size: int = 5) -> List[str]:
    start = time.time()
    print(f"🔍 Processing {len(questions)} questions sequentially...")
    answers = []
    for i, question in enumerate(questions):
        print(f"  Processing question {i+1}/{len(questions)}")
        answer = analyze_query_with_vectorstore_fast(question, vectorstore, cleanup_after=False)
        answers.append(answer)
    print(f"✅ Sequential processing completed in {time.time() - start:.1f}s")
    return answers