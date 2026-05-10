# RAG Research

A document Q&A web app with four retrieval pipelines built on LangGraph, FAISS, and Gemini.

Upload PDFs or text files, then ask questions. The app searches your documents and returns grounded answers with source citations and a step-by-step trace of how the answer was produced.

## Pipelines

| Mode | Description |
|---|---|
| **Traditional** | Simple FAISS similarity search → generate answer |
| **Single Agent** | Agentic RAG: rewrites query → retrieves → grades chunks → generates (falls back to LLM general knowledge if no relevant docs found) |
| **Multi-Agent** | Orchestrator decides whether to use RAG, web search (DuckDuckGo), or both → synthesizer combines results |
| **ReAct** | LLM drives its own Thought → Action → Observation loop, choosing tools freely until it has enough to answer |

All modes include an input guardrail (blocks harmful/injection queries) and an output guardrail (flags hallucinations before the answer reaches the user).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_api_key   # optional, for tracing
LANGSMITH_TRACING=true                     # optional
```

## Running

```bash
python app.py
```

Opens at `http://localhost:5001`.

## Usage

1. Click **Upload** and select one or more PDF or text files — the app chunks and indexes them into a local FAISS store.
2. Select a pipeline mode from the UI.
3. Type a question and submit — the answer, sources, and agent trace are shown.

## Project Structure

```
app.py                  Flask API + UI routing
src/
  data_loader.py        Loads and chunks uploaded documents
  vectorstore.py        Builds and loads the FAISS index (HuggingFace embeddings)
  search.py             Traditional RAG pipeline
  agent.py              Single-agent LangGraph pipeline
  multi_agent.py        Multi-agent LangGraph pipeline
  react_agent.py        ReAct LangGraph pipeline
  guardrails.py         Input/output safety checks via Gemini
  prompts.py            Prompt templates shared across pipelines
templates/              Jinja2 HTML templates
uploads/                Uploaded files (git-ignored)
faiss_store/            Persisted FAISS index (git-ignored)
eval/                   Evaluation scripts
```

## Models & Tools

- **LLM**: Gemini 2.0 Flash (`langchain-google-genai`)
- **Embeddings**: `all-MiniLM-L6-v2` via HuggingFace
- **Vector store**: FAISS (local, persisted to `faiss_store/`)
- **Web search**: DuckDuckGo (no API key required)
- **Orchestration**: LangGraph
- **Tracing**: LangSmith (optional)
