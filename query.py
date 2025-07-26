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
<<<<<<< Updated upstream
INDEX_PATH = "faiss_index"  # Folder containing vector DB
MODEL_NAME = "llama3-8b-8192"  # Groq-supported model
=======
INDEX_PATH = "faiss_index"
MODEL_NAME = "llama-3.3-70b-versatile"
>>>>>>> Stashed changes
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# -----------------------------
# Load ONCE (important for latency)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=MODEL_NAME,
    temperature=0.1
)

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


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": CLAUSE_DECISION_PROMPT},
    return_source_documents=True
)

# -----------------------------
# Core Logic
# -----------------------------
def analyze_query(query_text: str) -> dict:
    response = qa_chain.invoke({"query": query_text})
    return response["result"]

# -----------------------------
# CLI (for testing)
# -----------------------------
if __name__ == "__main__":
    print("🧾 Decision Agent — Ask document-based questions. Type 'exit' to quit.")
    while True:
        user_query = input("\n💬 Query: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
        result = analyze_query(user_query)
        print(f"\n📄 Structured Response:\n{result}")
