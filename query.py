import os
import json
import time
from dotenv import load_dotenv
import gc
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor
import re
from typing import List, Dict, Any

load_dotenv()

# Configuration - Enhanced for accuracy
MODEL_NAME = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"

# Global LLM cache
_llm_cache = None

def get_llm():
    """Cached LLM with enhanced settings for accuracy"""
    global _llm_cache
    
    if _llm_cache is None:
        _llm_cache = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=MODEL_NAME,
            temperature=0.1,  # Slightly higher for nuanced responses
            max_tokens=600,   # Increased for detailed answers
            request_timeout=15  # Longer timeout for complex queries
        )
    return _llm_cache

def get_embeddings():
    """Get embeddings model with enhanced configuration"""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}  # Removed show_progress_bar
    )

def preprocess_query(query: str) -> str:
    """Enhanced query preprocessing for better retrieval"""
    # Clean whitespace
    query = ' '.join(query.split())
    
    # Insurance domain synonym expansion
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
    
    # Apply expansions to improve semantic matching
    expanded_query = query
    for pattern, expansion in expansions.items():
        if re.search(pattern, query, re.IGNORECASE):
            # Add expansion terms without replacing original
            expanded_query += f" {expansion}"
    
    return expanded_query

# ENHANCED PROMPT - More aggressive about finding information
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

# Fallback prompt for second attempts
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

def enhanced_retrieval_with_multiple_strategies(vectorstore, query_text: str, max_docs: int = 8) -> List[Document]:
    """Multi-strategy retrieval for maximum coverage"""
    all_docs = []
    seen_content = set()
    
    # Strategy 1: MMR for diversity (your current approach)
    try:
        retriever_mmr = vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 4, "lambda_mult": 0.3}  # More diversity
        )
        docs_mmr = retriever_mmr.get_relevant_documents(query_text)
        for doc in docs_mmr:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    except Exception as e:
        print(f"MMR retrieval failed: {e}")
    
    # Strategy 2: Lower threshold similarity 
    try:
        retriever_sim = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6,
                "score_threshold": 0.1,  # Much lower threshold
                "fetch_k": 12  # Cast wider net
            }
        )
        docs_sim = retriever_sim.get_relevant_documents(query_text)
        for doc in docs_sim:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    except Exception as e:
        print(f"Similarity threshold failed: {e}")
        # Fallback to basic similarity
        try:
            retriever_basic = vectorstore.as_retriever(search_kwargs={"k": 6})
            docs_basic = retriever_basic.get_relevant_documents(query_text)
            for doc in docs_basic:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        except Exception as e2:
            print(f"Basic retrieval failed: {e2}")
    
    # Strategy 3: Try with preprocessed query if original didn't get enough results
    if len(all_docs) < 4:
        try:
            processed_query = preprocess_query(query_text)
            if processed_query != query_text:
                retriever_processed = vectorstore.as_retriever(search_kwargs={"k": 4})
                docs_processed = retriever_processed.get_relevant_documents(processed_query)
                for doc in docs_processed:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
        except Exception as e:
            print(f"Processed query retrieval failed: {e}")
    
    return all_docs[:max_docs]

def analyze_query_with_vectorstore_fast(query_text: str, vectorstore, cleanup_after=True) -> str:
    """Enhanced query analysis with multiple attempts and fallbacks"""
    try:
        llm = get_llm()
        
        # Preprocess query
        processed_query = preprocess_query(query_text)
        
        # Enhanced retrieval
        docs = enhanced_retrieval_with_multiple_strategies(vectorstore, query_text, max_docs=6)
        
        if not docs:
            print("⚠️ No documents retrieved")
            return "No relevant information found in the document."
        
        print(f"📄 Retrieved {len(docs)} documents for analysis")
        
        # First attempt with enhanced prompt
        try:
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Direct LLM call for more control
            first_prompt = ENHANCED_INSURANCE_PROMPT.format(
                context=context,
                question=query_text
            )
            
            response = llm.invoke(first_prompt)
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
            
            # Clean up answer format
            if answer.startswith("ANSWER:"):
                answer = answer[7:].strip()
            elif answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            # Check if we got a meaningful answer
            if (not answer or 
                len(answer) < 20 or 
                "not found" in answer.lower() or 
                "not mentioned" in answer.lower() or
                "does not contain" in answer.lower()):
                
                print("🔄 First attempt unsuccessful, trying aggressive search...")
                
                # Second attempt with aggressive prompt and more context
                aggressive_prompt = AGGRESSIVE_SEARCH_PROMPT.format(
                    context=context,
                    question=query_text
                )
                
                response2 = llm.invoke(aggressive_prompt)
                if hasattr(response2, 'content'):
                    answer2 = response2.content.strip()
                else:
                    answer2 = str(response2).strip()
                
                # Clean up second answer
                if answer2.startswith("RESPONSE:"):
                    answer2 = answer2[9:].strip()
                
                # Use second answer if it's better
                if (answer2 and 
                    len(answer2) > len(answer) and 
                    "not found" not in answer2.lower()):
                    answer = answer2
            
            return answer
            
        except Exception as e:
            print(f"❌ Enhanced query processing error: {str(e)}")
            return f"Error processing query: {str(e)}"
        
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return f"Error processing query: {str(e)}"
    
    finally:
        if cleanup_after:
            gc.collect()

# Enhanced batch processing
async def analyze_multiple_queries_fast(questions: list, vectorstore, cleanup_after=True) -> list:
    """Enhanced batch processing with better accuracy"""
    async def process_question(question):
        start = time.time()
        try:
            # Use enhanced single query processing
            answer = analyze_query_with_vectorstore_fast(question, vectorstore, cleanup_after=False)
            print(f"✅ [Q{question[:30]}...] Done in {time.time() - start:.1f}s")
            return answer
        except Exception as e:
            print(f"❌ [Q{question[:30]}...] Error: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    # Process in smaller batches for stability
    answers = []
    batch_size = 2  # Smaller batches for accuracy
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        tasks = [process_question(q) for q in batch]
        batch_answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        batch_answers = [str(ans) if isinstance(ans, Exception) else ans for ans in batch_answers]
        answers.extend(batch_answers)
        
        # Brief pause between batches
        if i + batch_size < len(questions):
            await asyncio.sleep(0.3)
    
    if cleanup_after:
        gc.collect()
    
    return answers

def analyze_query_with_sources_fast(query_text: str, vectorstore, cleanup_after=True) -> dict:
    """Enhanced query with detailed source information"""
    try:
        # Use enhanced retrieval
        docs = enhanced_retrieval_with_multiple_strategies(vectorstore, query_text, max_docs=5)
        
        # Get enhanced answer
        answer = analyze_query_with_vectorstore_fast(query_text, vectorstore, cleanup_after=False)
        
        # Detailed source information
        sources = []
        for i, doc in enumerate(docs[:4]):  # Show top 4 sources
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
            "retrieval_method": "enhanced_multi_strategy"
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
    """Enhanced answer quality scoring"""
    if not answer or len(answer.strip()) < 10:
        return 0
    
    score = 30  # Base score
    
    # Negative indicators
    negative_phrases = [
        "not found", "not mentioned", "does not contain", 
        "unable to determine", "not specified", "no information",
        "cannot be determined", "not available"
    ]
    
    if any(phrase in answer.lower() for phrase in negative_phrases):
        return 5  # Very low score for "not found" responses
    
    # Positive scoring factors
    if len(answer.split()) > 20:
        score += 20  # Detailed response
    
    # Contains specific data
    if any(char in answer for char in ['$', '%']) or re.search(r'\d+', answer):
        score += 25  # Contains numbers/amounts
    
    # Insurance-specific terms
    insurance_terms = [
        'policy', 'coverage', 'deductible', 'premium', 'benefit', 
        'claim', 'exclusion', 'copay', 'limit', 'effective'
    ]
    found_terms = sum(1 for term in insurance_terms if term in answer.lower())
    score += found_terms * 5
    
    # Completeness indicators
    if any(word in answer.lower() for word in ['must', 'required', 'eligible', 'covered', 'applies']):
        score += 10
    
    return min(score, 100)

# Maintain compatibility with existing interface
def cleanup_chain_components(chain=None, retriever=None):
    """Clean up chain and retriever components"""
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
    """Wrapper function that automatically cleans up after single query"""
    return analyze_query_with_vectorstore_fast(query_text, vectorstore, cleanup_after=True)

def batch_query_with_auto_cleanup(vectorstore, questions: list) -> list:
    """Wrapper function that automatically cleans up after batch queries"""
    return asyncio.run(analyze_multiple_queries_fast(questions, vectorstore, cleanup_after=True))

def validate_answer_quality(answer: str, question: str) -> bool:
    """Enhanced answer quality validation"""
    score = score_answer_quality(answer, question)
    return score > 25  # More lenient threshold

def test_query_performance(vectorstore, test_questions: list):
    """Enhanced performance testing with detailed metrics"""
    print(f"🧪 Testing with {len(test_questions)} questions...")
    start = time.time()
    
    answers = batch_query_with_auto_cleanup(vectorstore, test_questions)
    
    total_time = time.time() - start
    avg_time = total_time / len(test_questions)
    
    # Detailed quality analysis
    quality_scores = [score_answer_quality(ans, test_questions[i]) for i, ans in enumerate(answers)]
    avg_score = sum(quality_scores) / len(quality_scores)
    high_quality_count = sum(1 for score in quality_scores if score > 70)
    medium_quality_count = sum(1 for score in quality_scores if 40 <= score <= 70)
    low_quality_count = sum(1 for score in quality_scores if score < 40)
    
    print(f"📊 Enhanced Performance Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg per question: {avg_time:.1f}s")
    print(f"   Average quality score: {avg_score:.1f}/100")
    print(f"   High quality (70+): {high_quality_count}/{len(test_questions)} ({high_quality_count/len(test_questions)*100:.1f}%)")
    print(f"   Medium quality (40-70): {medium_quality_count}/{len(test_questions)} ({medium_quality_count/len(test_questions)*100:.1f}%)")
    print(f"   Low quality (<40): {low_quality_count}/{len(test_questions)} ({low_quality_count/len(test_questions)*100:.1f}%)")
    print(f"   Throughput: {len(test_questions)/total_time:.1f} q/sec")
    
    return answers

# Legacy support with enhanced accuracy
def analyze_query(query_text: str) -> str:
    """Legacy function for local FAISS index with enhanced accuracy"""
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

# Keep your existing prompt for backward compatibility
INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert insurance policy analyst. Use the provided context from the document to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Respond clearly and concisely in 2-3 sentences maximum.
- If the answer is not found in the context, reply with: "Information not found in the document."
- Avoid unnecessary elaboration, filler, or repeated context.
- Use normal punctuation; do not include escape characters or special formatting.

Answer:
"""
)
# Example usage functions that demonstrate the cleanup pattern
# def process_single_query_example(vectorstore, question: str):
#     """Example: Process single query with automatic cleanup"""
#     print(f"🔍 Processing: {question}")
    
#     answer = query_with_auto_cleanup(vectorstore, question)
    
#     print(f"✅ Answer: {answer}")
#     print("🧹 Memory cleaned automatically")
    
#     return answer

# def process_batch_queries_example(vectorstore, questions: list):
#     """Example: Process multiple queries with automatic cleanup"""
#     print(f"🔍 Processing {len(questions)} questions...")
    
#     answers = batch_query_with_auto_cleanup(vectorstore, questions)
    
#     print(f"✅ Processed {len(answers)} answers")
#     print("🧹 Memory cleaned automatically")
    
#     return answers
