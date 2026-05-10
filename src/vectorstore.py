import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSIST_DIR = "faiss_store"

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


class GeminiEmbeddings(Embeddings):
    """Calls Google's v1 embedding API directly — avoids the v1beta SDK issue."""

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-embedding-001:embedContent"
        )

    def _embed(self, text: str) -> list[float]:
        resp = requests.post(
            self.url,
            params={"key": self.api_key},
            json={"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]}},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GeminiEmbeddings()
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
