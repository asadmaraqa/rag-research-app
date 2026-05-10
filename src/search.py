from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vectorstore import load_vectorstore
from src.prompts import get_prompt

load_dotenv()

_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
_chain = get_prompt("rag-search") | _llm


def _format_history(history: list) -> str:
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


def ask(query: str, chat_history: list = None, top_k: int = 5) -> dict:
    """Find the most relevant chunks for query and return answer + source metadata."""
    store = load_vectorstore()
    trace = []

    docs = store.similarity_search(query, k=top_k)
    trace.append({
        "node": "retrieve",
        "label": "Retriever",
        "detail": f"Found {len(docs)} chunks from vector store",
        "icon": "🔍",
    })

    context = "\n\n".join(d.page_content for d in docs)
    if not context:
        return {"answer": "No relevant documents found.", "sources": [], "trace": trace}

    answer = _chain.invoke({
        "context": context,
        "question": query,
        "chat_history": _format_history(chat_history),
    }).content
    trace.append({
        "node": "generate",
        "label": "Answer Generator",
        "detail": f"Generated answer using {len(docs)} document chunks",
        "icon": "💬",
    })

    seen = set()
    sources = []
    for d in docs:
        meta = d.metadata
        print(f"[DEBUG] chunk metadata: {meta}")
        key = str(meta)
        if key not in seen:
            seen.add(key)
            sources.append(meta)

    return {"answer": answer, "sources": sources, "trace": trace}
