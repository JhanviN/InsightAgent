import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# -----------------------------
# Configurations
# -----------------------------
INDEX_PATH = "faiss_index"  # Folder containing vector DB
MODEL_NAME = "llama3-8b-8192"  # Groq-supported model
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# -----------------------------
# LLM (Groq)
# -----------------------------
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=MODEL_NAME,
        temperature=0.1
    )

# -----------------------------
# Custom Prompt Template
# -----------------------------
CLAUSE_DECISION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal and insurance domain expert AI assistant.

Given the following unstructured document context and a user's query, determine:

1. Whether the condition/procedure is approved under the policy or not.
2. The payout amount if applicable.
3. Which clauses or sections support this decision.

Respond strictly in this structured JSON format:
{{
    "decision": "approved" | "rejected" | "not_sure",
    "amount": "<amount or null>",
    "justification": "<brief explanation>",
    "matched_clauses": ["<clause snippet 1>", "<clause snippet 2>", ...]
}}

Use only the provided context to justify your answer.

Context:
{context}

Query:
{question}
"""
)

# -----------------------------
# Core Inference Logic
# -----------------------------
def analyze_query(query_text: str) -> dict:
    # Load vector DB
    embeddings = get_embeddings()
    vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Load Groq LLM
    llm = get_llm()

    # Build RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": CLAUSE_DECISION_PROMPT},
        return_source_documents=True
    )

    # Run chain and return structured response
    response = chain.invoke({"query": query_text})
    return response["result"]

# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":
    print("🧾 Decision Agent — Ask document-based questions (e.g., claims, coverage, clauses). Type 'exit' to quit.")
    while True:
        user_query = input("\n💬 Query: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
        result = analyze_query(user_query)
        print(f"\n📄 Structured Response:\n{result}")
