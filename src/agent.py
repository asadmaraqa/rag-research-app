"""
Agentic RAG using LangGraph — Option 1: Always retrieve first.

Instead of guessing whether to search documents, we always retrieve first
and let the grader decide if the chunks are actually useful.
If no relevant chunks are found after retries, we fall back to general knowledge.

Graph flow:

  START → rewrite_query → retrieve_documents → grade_documents → generate_answer → END
                                                      │
                                                      ├──→ rewrite_query (retry if no relevant docs, max 2x)
                                                      │
                                                      └──→ answer_from_general_knowledge (after retries exhausted)
"""

from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from src.vectorstore import load_vectorstore
from src.prompts import get_prompt

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ── Agent State ───────────────────────────────────────────────────────────────
# This dict is passed between every node in the graph.
# Each node reads what it needs and writes back its output.

class AgentState(TypedDict):
    question: str                  # original user question
    chat_history: str              # formatted prior conversation turns
    rewritten_query: str           # improved version for better retrieval
    retrieved_docs: List[Document] # chunks returned from FAISS
    relevant_docs: List[Document]  # chunks confirmed relevant by grader
    answer: str                    # final answer shown to the user
    sources: list                  # metadata of the chunks used
    trace: list                    # step-by-step log of nodes that fired


# ── Node 1: Query Rewriter ────────────────────────────────────────────────────
# Rewrites the user's question into a cleaner search query so FAISS
# finds more relevant chunks (e.g. removes filler words, sharpens intent).

_rewriter_chain = get_prompt("rag-query-rewriter") | llm

def rewrite_query(state: AgentState) -> AgentState:
    """Turn the raw question into an optimised retrieval query."""
    response = _rewriter_chain.invoke({"question": state["question"], "chat_history": state["chat_history"]})
    rewritten_query = response.content.strip()
    entry = {
        "node": "rewrite_query",
        "label": "Query Rewriter",
        "detail": f'Rewrote to: "{rewritten_query[:90]}"',
        "icon": "✏️",
    }
    return {**state, "rewritten_query": rewritten_query, "trace": [*state.get("trace", []), entry]}


# ── Node 2: Retriever ─────────────────────────────────────────────────────────
# Searches the FAISS vector store using the rewritten query
# and returns the top-k most similar document chunks.

def retrieve_documents(state: AgentState) -> AgentState:
    """Search the FAISS index for the most relevant document chunks."""
    vectorstore = load_vectorstore()
    retrieved_docs = vectorstore.similarity_search(state["rewritten_query"], k=5)
    entry = {
        "node": "retrieve_documents",
        "label": "Retriever",
        "detail": f"Found {len(retrieved_docs)} chunks from vector store",
        "icon": "🔍",
    }
    return {**state, "retrieved_docs": retrieved_docs, "trace": [*state.get("trace", []), entry]}


# ── Node 3: Document Grader ───────────────────────────────────────────────────
# Grades all chunks in one LLM call instead of one call per chunk.
# This avoids rapid-fire API requests that cause connection resets.

_grader_chain = get_prompt("rag-document-grader") | llm

def grade_documents(state: AgentState) -> AgentState:
    """Filter retrieved chunks in one LLM call, keeping only relevant ones."""

    retrieved_docs = state["retrieved_docs"]
    if not retrieved_docs:
        return {**state, "relevant_docs": []}

    # Format all chunks into a numbered list for a single grading call
    numbered_chunks = "\n\n".join(
        f"[{i+1}] {doc.page_content[:500]}"  # cap at 500 chars to stay within token limits
        for i, doc in enumerate(retrieved_docs)
    )

    # Ask Gemini to grade all chunks at once
    response = _grader_chain.invoke({
        "question": state["question"],
        "chunks": numbered_chunks
    }).content.strip().lower()

    # Parse which chunk numbers Gemini said are relevant
    relevant_docs = []
    if response != "none":
        for part in response.split(","):
            part = part.strip()
            if part.isdigit():
                index = int(part) - 1  # convert 1-based to 0-based
                if 0 <= index < len(retrieved_docs):
                    relevant_docs.append(retrieved_docs[index])

    entry = {
        "node": "grade_documents",
        "label": "Document Grader",
        "detail": f"{len(relevant_docs)} of {len(retrieved_docs)} chunks passed relevance check",
        "icon": "✅" if relevant_docs else "❌",
    }
    return {**state, "relevant_docs": relevant_docs, "trace": [*state.get("trace", []), entry]}


# ── Node 4a: Answer from Documents ───────────────────────────────────────────
# Uses the relevant chunks as context and asks Gemini to generate an answer
# grounded in those chunks.

_answer_chain = get_prompt("rag-answer-generator") | llm

def generate_answer(state: AgentState) -> AgentState:
    """Generate an answer grounded in the relevant document chunks."""
    context = "\n\n".join(doc.page_content for doc in state["relevant_docs"])

    answer = _answer_chain.invoke({
        "context": context,
        "question": state["question"],
        "chat_history": state["chat_history"],
    }).content

    # Build one source entry per relevant chunk, including a content snippet
    sources = [
        {**document.metadata, "snippet": document.page_content[:200]}
        for document in state["relevant_docs"]
    ]

    entry = {
        "node": "generate_answer",
        "label": "Answer Generator",
        "detail": f"Generated answer grounded in {len(state['relevant_docs'])} document chunks",
        "icon": "💬",
    }
    return {**state, "answer": answer, "sources": sources, "trace": [*state.get("trace", []), entry]}


# ── Node 4b: Answer from General Knowledge ────────────────────────────────────
# Only reached when no relevant chunks were found after all retries.
# Gemini answers from its own training knowledge — no documents involved.

_general_chain = get_prompt("rag-general-fallback") | llm

def answer_from_general_knowledge(state: AgentState) -> AgentState:
    """Answer the question using the LLM's general knowledge (no documents)."""
    answer = _general_chain.invoke({"question": state["question"], "chat_history": state["chat_history"]}).content
    entry = {
        "node": "answer_from_general_knowledge",
        "label": "General Knowledge Fallback",
        "detail": "No relevant docs found after retries — answered from LLM training data",
        "icon": "🧠",
    }
    return {**state, "answer": answer, "sources": [], "trace": [*state.get("trace", []), entry]}


# ── Routing Condition ─────────────────────────────────────────────────────────
# Called after grading — decides whether to answer, retry, or fall back.

def decide_answer_or_fallback(state: AgentState) -> str:
    """After grading, either generate an answer or fall back to general knowledge."""
    if state["relevant_docs"]:
        return "generate_answer"
    return "answer_from_general_knowledge"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_agent():
    """Assemble the LangGraph agent and compile it into a runnable."""
    graph = StateGraph(AgentState)

    # Register every node (each is a function that takes and returns AgentState)
    graph.add_node("rewrite_query",                rewrite_query)
    graph.add_node("retrieve_documents",           retrieve_documents)
    graph.add_node("grade_documents",              grade_documents)
    graph.add_node("generate_answer",              generate_answer)
    graph.add_node("answer_from_general_knowledge", answer_from_general_knowledge)

    # Always start with rewriting the query — no routing needed
    graph.set_entry_point("rewrite_query")

    # Retrieval pipeline: rewrite → retrieve → grade → branch
    graph.add_edge("rewrite_query",      "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")

    # After grading: answer, retry, or fall back to general knowledge
    graph.add_conditional_edges("grade_documents", decide_answer_or_fallback)

    # Both answer nodes lead to END
    graph.add_edge("generate_answer",               END)
    graph.add_edge("answer_from_general_knowledge", END)

    return graph.compile()


# Compile once at import time so the app reuses the same instance
agent = build_agent()


# ── Public Interface ──────────────────────────────────────────────────────────

def _format_history(history: list) -> str:
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines) + "\n\n"


def ask(question: str, chat_history: list = None) -> dict:
    """
    Run the agentic RAG pipeline for a question.
    Returns a dict with 'answer' and 'sources'.
    """
    initial_state: AgentState = {
        "question":        question,
        "chat_history":    _format_history(chat_history),
        "rewritten_query": "",
        "retrieved_docs":  [],
        "relevant_docs":   [],
        "answer":          "",
        "sources":         [],
        "trace":           [],
    }

    final_state = agent.invoke(initial_state)
    return {
        "answer":  final_state["answer"],
        "sources": final_state["sources"],
        "trace":   final_state.get("trace", []),
    }
