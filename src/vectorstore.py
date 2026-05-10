import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "faiss_store"

# Embedding model converts text chunks into vectors for similarity search
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Splits long documents into overlapping chunks so context isn't lost at boundaries
_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def build_vectorstore(documents):
    """Chunk documents, embed them, and save a FAISS index to disk."""
    chunks = _splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, _embeddings)
    store.save_local(PERSIST_DIR)
    return store


def load_vectorstore():
    """Load the saved FAISS index from disk."""
    return FAISS.load_local(PERSIST_DIR, _embeddings, allow_dangerous_deserialization=True)


def vectorstore_exists():
    """Return True if a FAISS index has been built and saved."""
    return os.path.exists(os.path.join(PERSIST_DIR, "index.faiss"))
