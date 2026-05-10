"""
ReAct (Reasoning + Acting) RAG Agent.

Unlike the fixed-pipeline Agentic RAG, the LLM here drives every decision:

  Thought → Action (tool call) → Observation (result) → repeat → Final Answer

The model freely decides how many times to search, which tool to use, and when
it has enough information to stop — no hardcoded graph edges.

Tools available:
  search_documents — searches the FAISS vector store built from uploaded files
  web_search       — live DuckDuckGo search for information not in the documents
"""

import contextvars
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from src.vectorstore import load_vectorstore, vectorstore_exists

load_dotenv()

# ── LLM & external tools ──────────────────────────────────────────────────────

llm  = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
_ddg = DuckDuckGoSearchRun()

# ── Thread-safe source collector ──────────────────────────────────────────────
# Each ask() call sets its own list here so concurrent requests don't mix up
# source metadata collected inside search_documents.

_sources_ctx: contextvars.ContextVar[list] = contextvars.ContextVar(
    "react_sources", default=None
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded documents for chunks relevant to the query.
    Use this first for any question that might be answered by the uploaded files."""
    if not vectorstore_exists():
        return "No documents have been uploaded yet."

    store = load_vectorstore()
    docs  = store.similarity_search(query, k=5)
    if not docs:
        return "No relevant chunks found in the uploaded documents."

    # Accumulate source metadata into the per-call context list
    sources = _sources_ctx.get()
    if sources is not None:
        seen = {str(s) for s in sources}
        for d in docs:
            key = str(d.metadata)
            if key not in seen:
                seen.add(key)
                sources.append(d.metadata)

    return "\n\n".join(f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs))


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for up-to-date or general information
    not covered by the uploaded documents."""
    return _ddg.run(query)


# ── Build agent (compiled once at import) ────────────────────────────────────

react_agent = create_react_agent(llm, [search_documents, web_search])


# ── Public interface ──────────────────────────────────────────────────────────

def ask(question: str) -> dict:
    """
    Run the ReAct agent and return answer, sources, and a step-by-step trace.

    Streams all LangGraph events so every Thought / Action / Observation is
    captured in the trace list, which the frontend animates step by step.
    """
    doc_sources: list = []
    token = _sources_ctx.set(doc_sources)   # bind this call's source list

    trace: list        = []
    final_answer: str  = ""
    web_was_used: bool = False

    try:
        for event in react_agent.stream(
            {"messages": [("human", question)]},
            stream_mode="updates",
        ):
            for _node, output in event.items():
                for msg in output.get("messages", []):
                    tool_calls = getattr(msg, "tool_calls", None)

                    # ── AI chose to call a tool ──────────────────────────────
                    if tool_calls:
                        # Some models emit a reasoning string before tool calls
                        if getattr(msg, "content", ""):
                            trace.append({
                                "node":   "thought",
                                "label":  "Thought",
                                "detail": str(msg.content)[:200],
                                "icon":   "💭",
                            })
                        for tc in tool_calls:
                            args      = tc.get("args", {})
                            query_arg = str(args.get("query", args))[:100]
                            is_web    = tc["name"] == "web_search"
                            if is_web:
                                web_was_used = True
                            trace.append({
                                "node":   f"action_{tc['name']}",
                                "label":  f"Action: {tc['name']}",
                                "detail": f'Query: "{query_arg}"',
                                "icon":   "🌐" if is_web else "📄",
                            })

                    # ── Tool returned a result (Observation) ─────────────────
                    elif msg.__class__.__name__ == "ToolMessage":
                        content = str(msg.content)
                        short   = (content[:150] + "…") if len(content) > 150 else content
                        trace.append({
                            "node":   f"observation_{getattr(msg, 'name', 'tool')}",
                            "label":  "Observation",
                            "detail": short,
                            "icon":   "👁️",
                        })

                    # ── Final AI answer (no more tool calls) ─────────────────
                    elif msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
                        final_answer = msg.content
                        trace.append({
                            "node":   "final_answer",
                            "label":  "Final Answer",
                            "detail": "Agent reasoned enough to produce an answer",
                            "icon":   "✅",
                        })

    finally:
        _sources_ctx.reset(token)

    sources = list(doc_sources)
    if web_was_used:
        sources.append({"source": "DuckDuckGo Web Search", "query": question})

    return {"answer": final_answer, "sources": sources, "trace": trace}
