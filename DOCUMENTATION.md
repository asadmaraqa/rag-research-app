# RAG Research — Full System Documentation

## Table of Contents

1. [What This App Does](#1-what-this-app-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Tech Stack](#3-tech-stack)
4. [How a Request Flows End-to-End](#4-how-a-request-flows-end-to-end)
5. [Document Ingestion Pipeline](#5-document-ingestion-pipeline)
6. [Vector Store — FAISS + Gemini Embeddings](#6-vector-store--faiss--gemini-embeddings)
7. [The Four RAG Pipelines](#7-the-four-rag-pipelines)
   - [Traditional RAG](#71-traditional-rag)
   - [Agentic RAG (Single Agent)](#72-agentic-rag-single-agent)
   - [Multi-Agent RAG](#73-multi-agent-rag)
   - [ReAct Agent](#74-react-agent)
8. [Prompt Registry](#8-prompt-registry)
9. [Guardrails — Input & Output Safety](#9-guardrails--input--output-safety)
10. [Conversation Memory](#10-conversation-memory)
11. [Authentication](#11-authentication)
12. [Frontend](#12-frontend)
13. [Evaluation Suite](#13-evaluation-suite)
14. [File Structure Reference](#14-file-structure-reference)
15. [Environment Variables](#15-environment-variables)

---

## 1. What This App Does

RAG Research is a web application that lets you upload your own documents (PDFs, Word files, spreadsheets, text files, JSON) and then ask questions about them in a chat interface. The app uses four different AI pipeline strategies — from a simple retrieve-then-answer to a full multi-agent system with web search — so you can compare how each approach handles the same question.

Every answer is grounded in your uploaded content. The app never makes up sources: if the documents don't contain the answer, it tells you so or falls back to the AI's general knowledge transparently.

---

## 2. High-Level Architecture

```
Browser (index.html)
    │
    │  POST /chat  { query, mode }
    ▼
Flask (app.py)
    │
    ├── Input Guardrail (guardrails.py)
    │
    ├── Session: load chat_history
    │
    ├──► Traditional RAG    (src/search.py)
    ├──► Agentic RAG        (src/agent.py)       ─── LangGraph graph
    ├──► Multi-Agent RAG    (src/multi_agent.py) ─── LangGraph graph
    └──► ReAct Agent        (src/react_agent.py) ─── LangGraph prebuilt
              │
              ▼
         FAISS Vector Store  ◄──  Gemini Embedding API
              │
              ▼
         Gemini 2.0 Flash (LLM)
              │
    ├── Output Guardrail (guardrails.py)
    │
    ├── Session: save chat_history
    │
    └── JSON response  { answer, sources, trace, mode }
```

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| Production server | Gunicorn |
| LLM | Google Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Embeddings | Google Gemini Embedding API (`gemini-embedding-001`) |
| Vector store | FAISS (Facebook AI Similarity Search, CPU version) |
| Agent / graph framework | LangGraph |
| LLM orchestration | LangChain |
| Document loaders | LangChain community loaders (PDF, DOCX, CSV, XLSX, JSON, TXT) |
| Evaluation | LangSmith |
| Web search | DuckDuckGo (no API key required) |

---

## 4. How a Request Flows End-to-End

1. **User types a question** in the browser and clicks Send.
2. The frontend sends a `POST /chat` request with `{ query, mode }`.
3. `app.py` receives it and runs the **input guardrail** — if the query is harmful or an injection attempt, it returns a `400` error immediately.
4. The **conversation history** for this browser session is loaded from the Flask session.
5. The request is routed to whichever pipeline matches the selected mode. The pipeline receives the question and the full history.
6. The pipeline retrieves relevant chunks from FAISS, generates an answer using Gemini, and returns `{ answer, sources, trace }`.
7. The **output guardrail** checks the answer for relevance and faithfulness. If it fails, a warning is attached to the response.
8. The new question and answer are appended to the session history (capped at 20 messages = 10 exchanges).
9. The JSON response is returned to the browser, which renders the answer in the chat and animates the pipeline trace in the sidebar.

---

## 5. Document Ingestion Pipeline

**File:** `src/data_loader.py`

When you click **Upload & Index**, all files in the `uploads/` folder are processed:

```
Raw file
  │
  ▼
LangChain document loader (per file type)
  │   PDF → PyPDFLoader
  │   TXT → TextLoader
  │   CSV → CSVLoader
  │   DOCX → Docx2txtLoader
  │   XLSX → UnstructuredExcelLoader
  │   JSON → JSONLoader
  ▼
Text cleaning
  │   strips control characters
  │   removes copyright notices, page numbers
  │   collapses excess whitespace
  ▼
Minimum length filter (< 50 chars → dropped)
  │
  ▼
build_vectorstore() in src/vectorstore.py
  │
  ▼
RecursiveCharacterTextSplitter
  chunk_size=1000 chars, overlap=200 chars
  │
  ▼
Gemini Embedding API  →  FAISS index
  │
  ▼
Saved to disk at faiss_store/
```

The overlap of 200 characters ensures context is not lost at chunk boundaries — a sentence that spans two chunks can still be retrieved from either side.

---

## 6. Vector Store — FAISS + Gemini Embeddings

**File:** `src/vectorstore.py`

**Embeddings** are generated by calling Google's `gemini-embedding-001` model directly via the REST API. Each chunk of text is converted into a high-dimensional float vector that represents its meaning.

**FAISS** stores all these vectors on disk (`faiss_store/index.faiss`). When a user asks a question, the question itself is embedded the same way, then FAISS finds the top-K most similar chunks using approximate nearest-neighbour search — this is the "retrieval" step of RAG.

The FAISS index persists across server restarts. Once documents are indexed, you don't need to re-upload them. Uploading new documents rebuilds the entire index from scratch.

---

## 7. The Four RAG Pipelines

### 7.1 Traditional RAG

**File:** `src/search.py` | **Mode name:** `traditional`

The simplest possible approach. No agents, no routing decisions.

```
Question + History
      │
      ▼
FAISS similarity_search (top 5 chunks)
      │
      ▼
All chunks concatenated into context
      │
      ▼
Gemini (rag-search prompt) → Answer
```

**Best for:** Fast answers on straightforward questions. This mode never retries and never searches the web.

---

### 7.2 Agentic RAG (Single Agent)

**File:** `src/agent.py` | **Mode name:** `single`

A LangGraph state machine with 5 nodes. The LLM is used not only to generate the final answer but also to rewrite the query and grade whether the retrieved chunks are actually useful.

```
START
  │
  ▼
rewrite_query
  Gemini rewrites the user's question into a
  keyword-rich search query. Uses conversation
  history so references like "that book" resolve
  correctly.
  │
  ▼
retrieve_documents
  FAISS similarity_search on the rewritten query (top 5)
  │
  ▼
grade_documents
  Gemini evaluates all chunks in a single call.
  Returns a comma-separated list of relevant chunk
  numbers, e.g. "1,3" or "none".
  │
  ├── relevant_docs found ──► generate_answer ──► END
  │                             Gemini answers using
  │                             only the relevant chunks
  │
  └── no relevant docs ──────► answer_from_general_knowledge ──► END
                                 Gemini answers from its
                                 own training knowledge
```

**AgentState** carries these fields across nodes:
- `question` — original user question
- `chat_history` — formatted conversation history
- `rewritten_query` — improved query for FAISS
- `retrieved_docs` — raw chunks from FAISS
- `relevant_docs` — chunks that passed grading
- `answer` — final answer text
- `sources` — one entry per relevant chunk that passed grading; each entry contains the chunk's file/page metadata plus a `snippet` field (first 200 chars of the chunk text)
- `trace` — step-by-step log

**Best for:** Better retrieval quality than Traditional RAG, especially for vague or conversational questions. Slower due to two extra LLM calls (rewriter + grader).

---

### 7.3 Multi-Agent RAG

**File:** `src/multi_agent.py` | **Mode name:** `multi`

A LangGraph graph with four specialized agents coordinated by an orchestrator. Documents and the live web can both be searched in the same request.

```
START
  │
  ▼
orchestrate
  ┌─ Step 1: use_rag = True whenever a FAISS index exists.
  │
  └─ Step 2: ask Gemini "does this need live web data?"
     yes → use_web = True
     If neither fired → force use_web = True
  │
  ▼ (conditional edge)
  ├── use_rag=True ──► run_rag_agent
  │                     Calls the full Agentic RAG
  │                     pipeline (src/agent.py) with
  │                     conversation history.
  │
  │                     If no relevant chunks are found:
  │                       • rag_answer is cleared
  │                       • use_web is flipped to True
  │                         so the web agent covers the gap
  │                       │
  │                       ▼
  └── (always) ──────► run_web_agent
                        If use_web=False, skips itself.
                        Otherwise: DuckDuckGo search.
                          │
                          ▼
                       synthesize
                        Gemini combines the RAG answer
                        and web results into one final
                        answer. Uses conversation history.
                          │
                          ▼
                         END
```

**OrchestratorState** carries:
- `question`, `chat_history`, `chat_history_raw`
- `use_rag`, `use_web`
- `rag_answer`, `rag_sources`, `web_answer`
- `final_answer`, `sources`, `trace`

**Best for:** Questions that mix document knowledge with up-to-date facts (e.g. "What does our policy say about X, and what are the latest regulations?"). Most powerful but slowest due to the most LLM calls.

---

### 7.4 ReAct Agent

**File:** `src/react_agent.py` | **Mode name:** `react`

Uses LangGraph's `create_react_agent` prebuilt. The LLM itself decides at every step what to do next — there are no hardcoded graph edges beyond the tool loop.

```
Human message (+ conversation history)
      │
      ▼
┌─────────────────────────────┐
│  Gemini (Thought)           │
│  Decides: call a tool,      │
│  or produce final answer?   │
└──────────┬──────────────────┘
           │
     tool call?
     │         │
     ▼         ▼
search_documents   web_search
(FAISS, top 5)    (DuckDuckGo)
     │         │
     └────┬────┘
          │ Observation
          ▼
     Back to Thought
     (repeats until
      agent says done)
          │
          ▼
     Final Answer
```

**Tools:**
- `search_documents(query)` — searches the FAISS vector store. Gemini's docstring tells it to use this first.
- `web_search(query)` — searches DuckDuckGo for live data.

**Source tracking:** A `contextvars.ContextVar` is used so each concurrent request maintains its own source list without cross-request contamination.

**Conversation history** is passed as actual message objects prepended to the messages list, so the LLM's native multi-turn context handles it naturally.

**Best for:** Complex, multi-step research questions where the answer requires multiple searches or reasoning between tool calls. Most flexible; the number of steps is not fixed.

---

## 8. Prompt Registry

**File:** `src/prompts.py`

All prompts are defined in a single dict `PROMPT_TEMPLATES` and retrieved by name via `get_prompt("name")`, which returns a LangChain `ChatPromptTemplate`. This makes prompts easy to find, version-control, and change in one place.

| Prompt name | Used in | Purpose |
|---|---|---|
| `rag-search` | Traditional RAG | Answer using retrieved context |
| `rag-query-rewriter` | Agentic RAG | Rewrite question into a better search query |
| `rag-document-grader` | Agentic RAG | Filter chunks by relevance |
| `rag-answer-generator` | Agentic RAG | Generate answer from relevant chunks |
| `rag-general-fallback` | Agentic RAG | Answer from general knowledge when no docs match |
| `multi-agent-web-decision` | Multi-Agent | Decide if live web search is needed |
| `multi-agent-synthesizer` | Multi-Agent | Merge RAG answer and web results |

Every prompt that generates an answer includes a `{chat_history}` variable. When the conversation is new, this is an empty string and has no effect. As the conversation grows, it prepends a formatted summary of prior turns so the model understands references and can give consistent follow-up answers.

---

## 9. Guardrails — Input & Output Safety

**File:** `src/guardrails.py`

Two LLM-based guardrails run on every request, using Gemini as a judge.

### Input Guardrail (`validate_input`)

Runs before the pipeline. Checks for:
- Harmful, abusive, or illegal content
- Prompt injection attempts (e.g. "ignore previous instructions")
- Queries completely unrelated to document research

Returns `(is_safe: bool, reason: str)`. If `is_safe=False`, the request is rejected with HTTP 400 before any expensive pipeline work happens.

Hard limits applied before the LLM check:
- Query shorter than 2 characters → blocked
- Query longer than 2000 characters → blocked

**Fail-open:** if the guardrail LLM call itself fails (network error, etc.), the request is allowed through. This prevents the guardrail from becoming a point of failure.

### Output Guardrail (`validate_output`)

Runs after the pipeline generates an answer. Checks for:
- **Relevance:** does the answer address the question?
- **Faithfulness:** does the answer stay within what the source context supports?
- **Safety:** does the answer contain harmful content?

Returns `(is_ok: bool, issue: str)`. If `is_ok=False`, a `guardrail_warning` field is added to the response. The answer is still shown to the user but the warning is attached. Only the first 600 characters of context are sent to the judge to keep the check fast.

---

## 10. Conversation Memory

**Files:** `app.py` (session management), all pipeline `ask()` functions

The app maintains per-session conversation memory using Flask's server-side session (stored in a signed cookie).

**How it works:**

1. On each `/chat` request, `session["chat_history"]` is loaded — a list of `{"role": "user"/"assistant", "content": "..."}` dicts.
2. The list is passed to whichever pipeline is active.
3. Each pipeline formats the history as a readable block:
   ```
   Conversation so far:
   Human: What is chapter 3 about?
   Assistant: Chapter 3 covers risk management frameworks...
   ```
4. This block is injected into prompts via the `{chat_history}` variable.
5. After the answer is generated, the new question+answer pair is appended and the list is trimmed to the last 20 messages (10 full exchanges) to control token usage.
6. The session is saved back to the cookie.

**Why the query rewriter gets history too:** Questions like "tell me more about that" or "what did it say about chapter 2?" are ambiguous without context. The rewriter uses history to resolve these references before searching FAISS, so retrieval quality stays high throughout a conversation.

**New Chat button:** Sends `POST /clear`, which wipes `chat_history` from the session. The UI resets the chat box without a page reload.

---

## 11. Authentication

**File:** `app.py`

The entire app (except `/login`) is protected by a password gate. The password is set via the `APP_PASSWORD` environment variable (default: `testItsMe92` for development).

- `GET /login` — renders the login form
- `POST /login` — checks the submitted password; sets `session["logged_in"] = True` on success
- `GET /logout` — clears the session and redirects to login
- All other routes use the `@login_required` decorator, which redirects to `/login` if the session flag is missing

The Flask secret key (used to sign the session cookie) is set via the `SECRET_KEY` environment variable. The default value must be changed before deploying to production.

---

## 12. Frontend

**File:** `templates/index.html`

A single-page application built with plain HTML, CSS, and vanilla JavaScript — no frontend framework.

### Layout

```
┌─────────────────────────────────────────────┐
│               Header + New Chat             │
├─────────────────────────────────────────────┤
│              Upload Card                    │
│  Drag-and-drop area | Upload & Index button │
├─────────────────────────────────────────────┤
│              Mode Selector                  │
│  Traditional | Agentic | Multi-Agent | ReAct│
├──────────────────────────┬──────────────────┤
│    Chat Column           │  Pipeline Trace  │
│  ┌─────────────────────┐ │  Sidebar         │
│  │  chat messages      │ │                  │
│  │  (bubbles)          │ │  Animated step   │
│  └─────────────────────┘ │  list for each   │
│  [ input box ] [ Send ]  │  pipeline run    │
└──────────────────────────┴──────────────────┘
```

### Key JavaScript functions

| Function | What it does |
|---|---|
| `setMode(mode)` | Switches the active pipeline mode, updates the UI badge |
| `uploadFiles()` | POSTs selected files to `/upload`, shows indexing status |
| `sendQuery()` | POSTs the question to `/chat`, shows typing indicator, renders the response |
| `appendMsg(text, role, sources)` | Adds a chat bubble; if sources are present, adds a collapsible source list where each item shows the file/page metadata and a 200-character snippet of the actual chunk text that was used |
| `showTrace(steps, mode)` | Animates the pipeline trace steps in the sidebar with a staggered 260ms delay per step |
| `newChat()` | POSTs to `/clear`, resets the chat box and trace sidebar |

### Pipeline trace sidebar

Every pipeline returns a `trace` array — a list of steps in the order they fired. Each step has:
- `icon` — emoji shown on the left
- `label` — bold step name
- `detail` — description of what happened (e.g. "Found 4 chunks from vector store")
- `node` — internal identifier; steps prefixed with `rag__` are indented to show they belong to the inner RAG sub-pipeline

Steps animate in one at a time (260ms apart) to give a sense of the pipeline running in real-time.

---

## 13. Evaluation Suite

**File:** `eval/run_evals.py`

An offline evaluation runner that uses LangSmith to score pipeline quality against a golden Q&A dataset.

### Running it

```bash
python -m eval.run_evals                 # Traditional RAG
python -m eval.run_evals --mode single   # Agentic RAG
python -m eval.run_evals --mode multi    # Multi-Agent RAG
python -m eval.run_evals --mode react    # ReAct Agent
```

### How it works

1. A **golden dataset** of 5 reference Q&A pairs is uploaded to LangSmith on first run (reused on subsequent runs).
2. The selected pipeline is run against each question.
3. Two **LLM-as-judge evaluators** score each answer:
   - **Correctness (0–1):** does the answer match the expected answer?
   - **Faithfulness (0–1):** does the answer avoid hallucinating content beyond the context?
4. Scores are averaged and printed. Full results are visible in the LangSmith project dashboard.

The golden dataset is intentionally general (about RAG concepts) so it works without uploading domain-specific documents. Replace the `GOLDEN_DATASET` list in `eval/run_evals.py` with your own Q&A pairs for meaningful production evaluations.

---

## 14. File Structure Reference

```
rag-research/
│
├── app.py                  # Flask app, routes, session management
├── gunicorn.conf.py        # Production server config
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── data_loader.py      # Load, clean, and prepare documents
│   ├── vectorstore.py      # FAISS index build/load, Gemini embeddings
│   ├── prompts.py          # Centralized prompt registry
│   ├── guardrails.py       # Input and output safety checks
│   ├── search.py           # Traditional RAG pipeline
│   ├── agent.py            # Agentic RAG (LangGraph state machine)
│   ├── multi_agent.py      # Multi-Agent RAG (LangGraph orchestrator)
│   └── react_agent.py      # ReAct Agent (LangGraph prebuilt)
│
├── eval/
│   └── run_evals.py        # LangSmith offline evaluation runner
│
├── templates/
│   ├── index.html          # Main chat UI
│   └── login.html          # Password gate
│
├── uploads/                # Uploaded files (created at runtime)
└── faiss_store/            # Persisted FAISS index (created at runtime)
```

---

## 15. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Google API key for Gemini LLM and embedding calls |
| `SECRET_KEY` | Yes (production) | `change-me-in-production` | Flask session signing key |
| `APP_PASSWORD` | No | `testItsMe92` | Password to access the app |
| `LANGCHAIN_API_KEY` | For evals only | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | For evals only | `rag-research` | LangSmith project name |
| `PORT` | No | `5001` | Port for the development server |

Set these in a `.env` file at the project root. The app loads it automatically via `python-dotenv`.

Example `.env`:
```
GOOGLE_API_KEY=your-google-api-key-here
SECRET_KEY=a-long-random-string-here
APP_PASSWORD=your-chosen-password
```
