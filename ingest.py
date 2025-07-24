import os
from dotenv import load_dotenv

# ✅ Updated imports to avoid deprecation
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Configurable Parameters
# -----------------------------
DATA_PATH = "data"
INDEX_PATH = "faiss_index"

# -----------------------------
# Load Documents
# -----------------------------
def load_documents(data_path):
    print(f"Loading documents from: {data_path}")
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    print(f"Loaded {len(docs)} documents.")
    return docs

# -----------------------------
# Split Documents into Chunks
# -----------------------------
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# -----------------------------
# Create Embeddings
# -----------------------------
def get_embeddings():
    print("Using Hugging Face Embeddings: BAAI/bge-base-en-v1.5")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# -----------------------------
# Save to Vector Store
# -----------------------------
def save_to_faiss(chunks, embeddings, index_path):
    print("Generating vector embeddings and saving index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Index saved to '{index_path}'.")

# -----------------------------
# Main Flow
# -----------------------------
if __name__ == "__main__":
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    save_to_faiss(chunks, embeddings, INDEX_PATH)
