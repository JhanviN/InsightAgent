import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Configurable Parameters
# -----------------------------
INDEX_PATH = "faiss_index"

# -----------------------------
# Embedding Model
# -----------------------------
def get_embeddings():
    model_name = "BAAI/bge-base-en-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name)

# -----------------------------
# Groq LLM
# -----------------------------
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",  # You can also try: "llama3-8b-8192"
        temperature=0.1
    )

# -----------------------------
# Prompt Template
# -----------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
)

# -----------------------------
# Ask Question
# -----------------------------
def ask_question(query_text):
    # Load vector store
    embeddings = get_embeddings()
    db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # Search top k documents
    docs = db.similarity_search(query_text, k=3)

    # Load QA chain
    llm = get_llm()
    chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)
    
    # Get response
    response = chain.run(input_documents=docs, question=query_text)
    return response

# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":
    print("🔍 Ask InsightAgent anything from your documents! Type 'exit' to quit.")
    while True:
        query = input("\n💬 Question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = ask_question(query)
        print(f"🧠 Answer: {answer}")
