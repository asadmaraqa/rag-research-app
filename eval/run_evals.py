"""
LangSmith Evaluation Runner for the RAG Research app.

Runs the RAG pipeline against a golden Q&A dataset and scores each answer using
two LLM-as-judge evaluators:
  - correctness  : does the answer match the expected answer?
  - faithfulness : does the answer avoid hallucinating content not in context?

Usage:
  cd /path/to/rag-research
  python -m eval.run_evals                       # uses "traditional" pipeline
  python -m eval.run_evals --mode single         # single-agent RAG
  python -m eval.run_evals --mode multi          # multi-agent RAG
  python -m eval.run_evals --mode react          # ReAct agent

Results appear in LangSmith under the project defined by LANGCHAIN_PROJECT.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# Ensure project root is on sys.path when run as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ── Golden dataset ────────────────────────────────────────────────────────────
# These Q&A pairs are used when no LangSmith dataset exists yet.
# They are intentionally general so the eval works without uploaded documents.
# Replace or extend with domain-specific pairs once you have real documents.

GOLDEN_DATASET = [
    {
        "question": "What is retrieval-augmented generation (RAG)?",
        "expected": (
            "RAG is a technique that combines a retrieval step — fetching relevant "
            "documents from a knowledge base — with a language model that uses those "
            "documents as context to generate a grounded answer."
        ),
    },
    {
        "question": "What is the difference between dense and sparse retrieval?",
        "expected": (
            "Dense retrieval uses vector embeddings to measure semantic similarity, "
            "while sparse retrieval (e.g. BM25) uses keyword overlap. "
            "Dense retrieval handles synonyms better; sparse retrieval is faster and "
            "more precise for exact-match queries."
        ),
    },
    {
        "question": "What is a vector store?",
        "expected": (
            "A vector store is a database optimised for storing and searching high-dimensional "
            "embedding vectors. It supports approximate nearest-neighbour search so you can "
            "quickly find the most semantically similar documents to a query."
        ),
    },
    {
        "question": "What does a document grader do in an agentic RAG pipeline?",
        "expected": (
            "A document grader evaluates retrieved chunks and filters out those that are "
            "not relevant to the question, so only useful context reaches the answer generator."
        ),
    },
    {
        "question": "What is LangSmith used for?",
        "expected": (
            "LangSmith is an LLMOps platform for tracing, debugging, evaluating, and "
            "monitoring LLM applications. It logs every LLM call, tool use, and chain "
            "step so you can inspect and improve your pipelines."
        ),
    },
]

DATASET_NAME = "rag-research-golden-dataset"

# ── LLM judge ────────────────────────────────────────────────────────────────

_judge_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

_correctness_prompt = ChatPromptTemplate.from_template(
    "You are an expert evaluator. Compare the ACTUAL answer to the EXPECTED answer "
    "for the given question. Score from 0 to 1:\n"
    "  1.0 = correct and complete\n"
    "  0.5 = partially correct\n"
    "  0.0 = wrong or irrelevant\n\n"
    "Question: {question}\n"
    "Expected: {expected}\n"
    "Actual: {actual}\n\n"
    "Respond with ONLY a number between 0 and 1."
)

_faithfulness_prompt = ChatPromptTemplate.from_template(
    "You are a faithfulness checker. Does the ACTUAL answer contain claims that go "
    "beyond or contradict what can be reasonably inferred from the QUESTION context?\n"
    "Score from 0 to 1:\n"
    "  1.0 = fully grounded, no hallucinations\n"
    "  0.5 = minor unsupported claims\n"
    "  0.0 = significant hallucination\n\n"
    "Question: {question}\n"
    "Actual answer: {actual}\n\n"
    "Respond with ONLY a number between 0 and 1."
)


def _parse_score(raw: str) -> float:
    import re
    match = re.search(r"[\d.]+", raw.strip())
    if match:
        return min(1.0, max(0.0, float(match.group())))
    return 0.5


def correctness_evaluator(run, example) -> dict:
    question = example.inputs.get("question", "")
    expected = example.outputs.get("expected", "")
    actual   = (run.outputs or {}).get("answer", "")
    raw = (_correctness_prompt | _judge_llm).invoke({
        "question": question,
        "expected": expected,
        "actual": actual,
    }).content
    return {"key": "correctness", "score": _parse_score(raw)}


def faithfulness_evaluator(run, example) -> dict:
    question = example.inputs.get("question", "")
    actual   = (run.outputs or {}).get("answer", "")
    raw = (_faithfulness_prompt | _judge_llm).invoke({
        "question": question,
        "actual": actual,
    }).content
    return {"key": "faithfulness", "score": _parse_score(raw)}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def ensure_dataset(client: Client) -> str:
    """Create the golden dataset on LangSmith if it doesn't exist yet."""
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME in existing:
        print(f"Dataset '{DATASET_NAME}' already exists — reusing.")
        return DATASET_NAME

    print(f"Creating dataset '{DATASET_NAME}' on LangSmith…")
    dataset = client.create_dataset(DATASET_NAME, description="Golden Q&A pairs for RAG eval")
    client.create_examples(
        inputs=[{"question": r["question"]} for r in GOLDEN_DATASET],
        outputs=[{"expected": r["expected"]} for r in GOLDEN_DATASET],
        dataset_id=dataset.id,
    )
    print(f"  Added {len(GOLDEN_DATASET)} examples.")
    return DATASET_NAME


# ── Pipeline targets ──────────────────────────────────────────────────────────

def _make_target(mode: str):
    if mode == "single":
        from src.agent import ask
    elif mode == "multi":
        from src.multi_agent import ask
    elif mode == "react":
        from src.react_agent import ask
    else:
        from src.search import ask

    def target(inputs: dict) -> dict:
        return ask(inputs["question"])

    target.__name__ = f"rag-{mode}"
    return target


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LangSmith evals for RAG Research")
    parser.add_argument(
        "--mode",
        choices=["traditional", "single", "multi", "react"],
        default="traditional",
        help="Which pipeline to evaluate (default: traditional)",
    )
    args = parser.parse_args()

    client = Client()
    dataset_name = ensure_dataset(client)
    target = _make_target(args.mode)

    print(f"\nRunning evaluation — mode={args.mode}, dataset={dataset_name}")
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_evaluator, faithfulness_evaluator],
        experiment_prefix=f"rag-{args.mode}",
        metadata={"mode": args.mode},
    )

    print("\n── Results ─────────────────────────────────────────────")
    scores: dict[str, list[float]] = {}
    for r in results:
        for fb in r.get("feedback", []):
            scores.setdefault(fb.key, []).append(fb.score)

    for metric, vals in scores.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {metric:20s}  avg={avg:.2f}  ({len(vals)} examples)")

    project = os.getenv("LANGCHAIN_PROJECT", "rag-research")
    print(f"\nFull results in LangSmith project: {project}")


if __name__ == "__main__":
    main()
