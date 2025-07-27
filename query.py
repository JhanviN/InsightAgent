import os
import json
import time
from dotenv import load_dotenv
import gc

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

load_dotenv()

# Configuration
MODEL_NAME = "llama-3.1-8b-instant"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index"

# Global LLM cache
_llm_cache = None

def get_llm():
    """Cached LLM with optimized settings"""
    global _llm_cache
    
    if _llm_cache is None:
        _llm_cache = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=MODEL_NAME,
            temperature=0.0,  # Deterministic for accuracy
            max_tokens=512,   # Concise answers
            request_timeout=10
        )
    return _llm_cache

def get_embeddings():
    """Get embeddings model"""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def cleanup_chain_components(chain=None, retriever=None):
    """Clean up chain and retriever components"""
    try:
        if chain:
            # Clear chain components
            if hasattr(chain, 'combine_documents_chain'):
                del chain.combine_documents_chain
            if hasattr(chain, 'retriever'):
                del chain.retriever
            del chain
        
        if retriever:
            # Clear retriever
            if hasattr(retriever, 'vectorstore'):
                del retriever.vectorstore
            del retriever
        
        gc.collect()
        print("🧹 Chain components cleaned")
        
    except Exception as e:
        print(f"⚠️ Chain cleanup warning: {str(e)}")

# Optimized prompt for insurance/legal documents
INSURANCE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert insurance policy analyst. Based on the policy document context below, provide a precise and accurate answer.

Context from policy document:
{context}

Question: {question}

Instructions:
- Answer directly and specifically based only on the provided context
- Include relevant policy terms, conditions, and time periods
- If the context doesn't contain the answer, state "Information not found in the provided context"
- Be concise but complete
- You have to reply in paragraphs/sentences, feel free to use punctuation marks but avoid using escape characters and special characters

Answer:"""
)

def analyze_query_with_vectorstore_fast(query_text: str, vectorstore, cleanup_after=True) -> str:
    """
    Optimized query analysis with enhanced retrieval and cleanup
    """
    chain = None
    retriever = None
    
    try:
        llm = get_llm()
        
        # Enhanced retrieval with better scoring
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4,  # Increase to retrieve more chunks
                "score_threshold": 0.2,  # Lower threshold for broader recall
                "fetch_k": 12  # Increase candidates for better filtering
            }
        )
        
        # Try retrieval, fallback to basic if score threshold fails
        try:
            docs = retriever.get_relevant_documents(query_text)
        except:
            # Fallback to basic similarity
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query_text)
        
        if not docs:
            return "No relevant information found in the document."
        
        # Build chain with custom prompt
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": INSURANCE_PROMPT},
            return_source_documents=False
        )
        
        # Execute query
        response = chain.invoke({"query": query_text})
        answer = response["result"].strip()
        
        # Clean up answer
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return answer
        
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return f"Error processing query: {str(e)}"
    
    finally:
        # Cleanup chain components after query
        if cleanup_after:
            cleanup_chain_components(chain, retriever)

def analyze_multiple_queries_fast(questions: list, vectorstore, cleanup_after=True) -> list:
    """
    Batch process multiple questions efficiently with cleanup
    """
    chain = None
    retriever = None
    
    try:
        llm = get_llm()
        
        # Setup retriever once
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.2, "fetch_k": 12}
            )
        except:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create chain once
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": INSURANCE_PROMPT},
            return_source_documents=False
        )
        
        answers = []
        total_start = time.time()
        
        for i, question in enumerate(questions, 1):
            start = time.time()
            print(f"🔍 [{i}/{len(questions)}] Processing: {question}...")
            
            try:
                # Retrieve documents to get context
                docs = retriever.get_relevant_documents(question)
                context = "\n".join([doc.page_content for doc in docs])
                print(f"\nContext for question ', {context}...\n")
                
                response = chain.invoke({"query": question})
                answer = response["result"].strip()
                
                # Clean answer
                if answer.startswith("Answer:"):
                    answer = answer[7:].strip()
                
                answers.append(answer)
                print(f"✅ [{i}] Done in {time.time()-start:.1f}s")
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                answers.append(error_msg)
                print(f"❌ [{i}] Error: {str(e)}")
        
        total_time = time.time() - total_start
        print(f"📊 Batch completed: {len(questions)} questions in {total_time:.1f}s")
        print(f"⚡ Average: {total_time/len(questions):.1f}s per question")
        
        return answers
        
    except Exception as e:
        print(f"❌ Batch error: {str(e)}")
        return [f"Batch processing error: {str(e)}"] * len(questions)
    
    finally:
        # Cleanup after batch processing
        if cleanup_after:
            cleanup_chain_components(chain, retriever)

def analyze_query_with_sources_fast(query_text: str, vectorstore, cleanup_after=True) -> dict:
    """Query with source information for debugging, with cleanup"""
    chain = None
    retriever = None
    
    try:
        llm = get_llm()
        
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.2}
            )
        except:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": INSURANCE_PROMPT},
            return_source_documents=True
        )
        
        response = chain.invoke({"query": query_text})
        
        # Extract source info
        sources = []
        if "source_documents" in response:
            for i, doc in enumerate(response["source_documents"][:2]):
                sources.append({
                    "content": doc.page_content[:150] + "...",
                    "page": doc.metadata.get("page", f"chunk_{i+1}")
                })
        
        answer = response["result"].strip()
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query_text
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "query": query_text
        }
    
    finally:
        # Cleanup after source query
        if cleanup_after:
            cleanup_chain_components(chain, retriever)

def query_with_auto_cleanup(vectorstore, query_text: str) -> str:
    """
    Wrapper function that automatically cleans up after single query
    """
    return analyze_query_with_vectorstore_fast(query_text, vectorstore, cleanup_after=True)

def batch_query_with_auto_cleanup(vectorstore, questions: list) -> list:
    """
    Wrapper function that automatically cleans up after batch queries
    """
    return analyze_multiple_queries_fast(questions, vectorstore, cleanup_after=True)

# Legacy support with cleanup
def analyze_query(query_text: str) -> str:
    """Legacy function for local FAISS index with cleanup"""
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
        # Clean up loaded components
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

def validate_answer_quality(answer: str, question: str) -> bool:
    """Basic answer quality validation"""
    if not answer or len(answer.strip()) < 10:
        return False
    
    error_indicators = [
        "error processing",
        "information not found",
        "context doesn't contain",
        "unable to determine"
    ]
    
    return not any(indicator in answer.lower() for indicator in error_indicators)

def test_query_performance(vectorstore, test_questions: list):
    """Performance testing utility with cleanup"""
    print(f"🧪 Testing with {len(test_questions)} questions...")
    start = time.time()
    
    # Use batch processing with cleanup
    answers = analyze_multiple_queries_fast(test_questions, vectorstore, cleanup_after=True)
    
    total_time = time.time() - start
    avg_time = total_time / len(test_questions)
    
    # Quality metrics
    quality_count = sum(1 for i, ans in enumerate(answers) 
                       if validate_answer_quality(ans, test_questions[i]))
    
    print(f"📊 Performance Results:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg per question: {avg_time:.1f}s")
    print(f"   Quality answers: {quality_count}/{len(test_questions)}")
    print(f"   Throughput: {len(test_questions)/total_time:.1f} q/sec")
    
    return answers

# Example usage functions that demonstrate the cleanup pattern
def process_single_query_example(vectorstore, question: str):
    """Example: Process single query with automatic cleanup"""
    print(f"🔍 Processing: {question}")
    
    answer = query_with_auto_cleanup(vectorstore, question)
    
    print(f"✅ Answer: {answer}")
    print("🧹 Memory cleaned automatically")
    
    return answer

def process_batch_queries_example(vectorstore, questions: list):
    """Example: Process multiple queries with automatic cleanup"""
    print(f"🔍 Processing {len(questions)} questions...")
    
    answers = batch_query_with_auto_cleanup(vectorstore, questions)
    
    print(f"✅ Processed {len(answers)} answers")
    print("🧹 Memory cleaned automatically")
    
    return answers

if __name__ == "__main__":
    print("🚀 Query Engine Test with Cleanup")
    
    test_questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?", 
        "Does this policy cover maternity expenses?",
        "What is the waiting period for cataract surgery?",
        "What is the No Claim Discount offered?"
    ]
    
    embeddings = None
    vector_db = None
    
    try:
        embeddings = get_embeddings()
        vector_db = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        answers = test_query_performance(vector_db, test_questions)
        
        print("\n📋 Sample Results:")
        for i, (q, a) in enumerate(zip(test_questions[:3], answers[:3])):
            print(f"\nQ{i+1}: {q}")
            print(f"A{i+1}: {a[:120]}...")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("Run ingest.py first to create the index")
    
    finally:
        # Final cleanup
        print("🧹 Final cleanup...")
        
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
        print("✅ All cleanup completed")