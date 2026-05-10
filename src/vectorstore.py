import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "faiss_store"

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def build_vectorstore(documents):
    """Chunk documents, embed them, and save a FAISS index to disk."""
    chunks = _splitter.split_documents(documents)
    store = FAISS.from_documents(chunks, _get_embeddings())
    store.save_local(PERSIST_DIR)
    return store


def load_vectorstore():
    """Load the saved FAISS index from disk."""
    return FAISS.load_local(PERSIST_DIR, _get_embeddings(), allow_dangerous_deserialization=True)


def vectorstore_exists():
    """Return True if a FAISS index has been built and saved."""
    return os.path.exists(os.path.join(PERSIST_DIR, "index.faiss"))
