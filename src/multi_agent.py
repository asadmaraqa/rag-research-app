"""
Multi-Agent RAG System using LangGraph.

Three specialized agents work together, coordinated by an orchestrator:

  ┌─────────────────────────────────────────────────────────┐
  │                    Orchestrator                         │
  │   reads the question and decides which agents to use    │
  └───────────┬─────────────────────────┬───────────────────┘
              │                         │
              ▼                         ▼
       RAG Agent                   Web Agent
  searches uploaded docs        searches the internet
              │                         │
              └────────────┬────────────┘
                           ▼
                      Synthesizer
               combines both results into
                    one final answer
"""

from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

from src.agent import ask as rag_ask
from src.vectorstore import vectorstore_exists
from src.prompts import get_prompt

load_dotenv()

# ── Shared LLM ────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ── Web Search Tool ───────────────────────────────────────────────────────────
# DuckDuckGo requires no API key — searches the live web

web_search_tool = DuckDuckGoSearchRun()

# ── Orchestrator State ────────────────────────────────────────────────────────
# Shared state passed between all agents in the multi-agent graph.
# Each agent reads what it needs and writes back its result.

class OrchestratorState(TypedDict):
    question: str          # original user question
    chat_history: str      # formatted history for synthesizer prompt
    chat_history_raw: list # raw history list passed to inner RAG pipeline
    use_rag: bool          # orchestrator decided: call the RAG agent?
    use_web: bool          # orchestrator decided: call the Web agent?
    rag_answer: str        # answer returned by the RAG agent
    rag_sources: list      # source metadata returned by the RAG agent
    web_answer: str        # search results returned by the Web agent
    final_answer: str      # synthesized answer shown to the user
    sources: list          # final sources shown in the UI
    trace: list            # step-by-step log of nodes that fired


# ── Node 1: Orchestrator ──────────────────────────────────────────────────────
# Reads the question and decides which agents to call.
# It can call RAG only, Web only, or both.

_web_chain = get_prompt("multi-agent-web-decision") | llm


def orchestrate(state: OrchestratorState) -> OrchestratorState:
    """
    Enable RAG whenever a vector store exists, then ask the LLM whether
    live web results are also needed. If RAG runs but finds nothing relevant,
    run_rag_agent will flip use_web=True as a fallback.
    """
    question = state["question"]
    trace    = list(state.get("trace", []))

    use_rag = vectorstore_exists()
    trace.append({
        "node":   "orchestrate_rag_probe",
        "label":  "DB Probe",
        "detail": "Vector store found — RAG enabled" if use_rag else "No vector store found — RAG skipped",
        "icon":   "🗄️",
    })

    # LLM decides if web is also needed
    web_response = _web_chain.invoke({"question": question}).content.strip().lower()
    use_web      = web_response.startswith("yes")

    # Fallback: if neither fired, use web
    if not use_rag and not use_web:
        use_web = True

    agents_used = []
    if use_rag: agents_used.append("RAG (documents)")
    if use_web: agents_used.append("Web (DuckDuckGo)")
    trace.append({
        "node":   "orchestrate",
        "label":  "Orchestrator Decision",
        "detail": f"Will use: {', '.join(agents_used)}",
        "icon":   "🎯",
    })

    return {**state, "use_rag": use_rag, "use_web": use_web, "trace": trace}


# ── Node 2: RAG Agent ─────────────────────────────────────────────────────────
# Calls the existing single-agent RAG pipeline from src/agent.py.
# Searches the FAISS vector store built from uploaded documents.

def run_rag_agent(state: OrchestratorState) -> OrchestratorState:
    """Call the RAG agent to search uploaded documents."""
    if not state["use_rag"] or not vectorstore_exists():
        entry = {
            "node": "run_rag_agent",
            "label": "RAG Agent",
            "detail": "Skipped — orchestrator did not request document search",
            "icon": "⏭️",
        }
        return {**state, "rag_answer": "", "rag_sources": [], "trace": [*state.get("trace", []), entry]}

    result = rag_ask(state["question"], chat_history=state.get("chat_history_raw"))
    # Merge the inner RAG pipeline's trace steps into our trace
    inner_trace = [{"node": f"rag__{s['node']}", "label": f"  ↳ {s['label']}", "detail": s["detail"], "icon": s["icon"]}
                   for s in result.get("trace", [])]

    found_chunks = bool(result.get("sources"))
    # If RAG found no relevant chunks the question is not about the documents —
    # enable web so the web agent picks it up instead.
    use_web = state.get("use_web") or not found_chunks

    entry = {
        "node": "run_rag_agent",
        "label": "RAG Agent",
        "detail": (
            f"Found {len(result['sources'])} relevant chunk(s) in documents"
            if found_chunks else
            "No relevant chunks found — falling back to web"
        ),
        "icon": "📄" if found_chunks else "⚠️",
    }
    return {
        **state,
        "rag_answer":  result["answer"] if found_chunks else "",
        "rag_sources": result["sources"],
        "use_web":     use_web,
        "trace": [*state.get("trace", []), entry, *inner_trace],
    }


# ── Node 3: Web Agent ─────────────────────────────────────────────────────────
# Searches DuckDuckGo for live web results about the question.
# Useful for recent events, facts not in the uploaded documents, etc.

def run_web_agent(state: OrchestratorState) -> OrchestratorState:
    """Call the Web agent to search the internet."""
    if not state["use_web"]:
        entry = {
            "node": "run_web_agent",
            "label": "Web Agent",
            "detail": "Skipped — orchestrator did not request web search",
            "icon": "⏭️",
        }
        return {**state, "web_answer": "", "trace": [*state.get("trace", []), entry]}

    web_results = web_search_tool.run(state["question"])
    entry = {
        "node": "run_web_agent",
        "label": "Web Agent",
        "detail": f"Searched DuckDuckGo — got {len(web_results)} chars of results",
        "icon": "🌐",
    }
    return {**state, "web_answer": web_results, "trace": [*state.get("trace", []), entry]}


# ── Node 4: Synthesizer ───────────────────────────────────────────────────────
# Combines the RAG answer and web results into one coherent final answer.
# If only one agent ran, it just cleans up that result.

_synthesizer_chain = get_prompt("multi-agent-synthesizer") | llm

def synthesize(state: OrchestratorState) -> OrchestratorState:
    """Combine RAG and web results into one final answer."""
    final_answer = _synthesizer_chain.invoke({
        "question":    state["question"],
        "chat_history": state.get("chat_history", ""),
        "rag_answer":  state["rag_answer"] or "No document results.",
        "web_answer":  state["web_answer"] or "No web results.",
    }).content

    # Build sources: document sources + web source (if web was actually used)
    sources = list(state["rag_sources"])
    if state.get("web_answer"):
        sources.append({"source": "DuckDuckGo Web Search", "query": state["question"]})

    used = []
    if state["rag_answer"]: used.append("documents")
    if state["web_answer"]:  used.append("web")
    entry = {
        "node": "synthesize",
        "label": "Synthesizer",
        "detail": f"Combined {' + '.join(used) if used else 'no'} results into final answer",
        "icon": "🔗",
    }
    return {**state, "final_answer": final_answer, "sources": sources, "trace": [*state.get("trace", []), entry]}


# ── Routing Condition ─────────────────────────────────────────────────────────
# After orchestration, decide whether to start with RAG or jump straight to web.

def decide_first_agent(state: OrchestratorState) -> str:
    """Start with RAG agent if needed, otherwise go straight to web agent."""
    return "run_rag_agent" if state["use_rag"] else "run_web_agent"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_multi_agent():
    """Assemble the multi-agent graph and compile it into a runnable."""
    graph = StateGraph(OrchestratorState)

    # Register all agent nodes
    graph.add_node("orchestrate",    orchestrate)
    graph.add_node("run_rag_agent",  run_rag_agent)
    graph.add_node("run_web_agent",  run_web_agent)
    graph.add_node("synthesize",     synthesize)

    # Orchestrator is always the entry point
    graph.set_entry_point("orchestrate")

    # After orchestration: go to RAG agent or skip straight to web
    graph.add_conditional_edges("orchestrate", decide_first_agent)

    # RAG always feeds into web (web node skips itself if use_web is False)
    graph.add_edge("run_rag_agent", "run_web_agent")

    # Both agents feed into the synthesizer
    graph.add_edge("run_web_agent", "synthesize")

    # Synthesizer is the final step
    graph.add_edge("synthesize", END)

    return graph.compile()


# Compile once at import time
multi_agent = build_multi_agent()


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
    Run the multi-agent pipeline for a question.
    Returns a dict with 'answer' and 'sources'.
    """
    initial_state: OrchestratorState = {
        "question":         question,
        "chat_history":     _format_history(chat_history),
        "chat_history_raw": chat_history or [],
        "use_rag":          False,
        "use_web":          False,
        "rag_answer":       "",
        "rag_sources":      [],
        "web_answer":       "",
        "final_answer":     "",
        "sources":          [],
        "trace":            [],
    }

    final_state = multi_agent.invoke(initial_state)
    return {
        "answer":  final_state["final_answer"],
        "sources": final_state["sources"],
        "trace":   final_state.get("trace", []),
    }
