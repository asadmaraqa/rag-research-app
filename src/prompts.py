"""
Centralized prompt registry.

All prompts are defined here. Import with:
  from src.prompts import get_prompt
  prompt = get_prompt("rag-search")

Version control is via git — commit this file to track prompt changes.
"""

from langchain_core.prompts import ChatPromptTemplate

PROMPT_TEMPLATES: dict[str, str] = {
    "rag-search": (
        "You are a helpful assistant. If the context below is relevant to the question, "
        "use it to answer. If not, answer from your general knowledge. "
        "For greetings, respond naturally.\n\n"
        "{chat_history}"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
    "rag-query-rewriter": (
        "Rewrite the question below into a concise, keyword-rich search query "
        "that will retrieve the most relevant document chunks. "
        "Use the conversation history to resolve references like 'that', 'it', 'them'.\n\n"
        "{chat_history}"
        "Question: {question}\n\nSearch query:"
    ),
    "rag-document-grader": (
        "You are a relevance grader. Given the question and the numbered chunks below, "
        "reply with only the numbers of the chunks that are relevant to the question.\n"
        "Format: comma-separated numbers, e.g. '1,3' or 'none' if nothing is relevant.\n\n"
        "Question: {question}\n\n"
        "Chunks:\n{chunks}"
    ),
    "rag-answer-generator": (
        "You are a helpful assistant. Answer the question using the context below.\n\n"
        "{chat_history}"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
    "rag-general-fallback": (
        "You are a helpful assistant. Answer the question below.\n\n"
        "{chat_history}"
        "Question: {question}"
    ),
    "multi-agent-rag-decision": (
        "You are a routing agent. The user has uploaded research documents into a vector store. "
        "Does the question below ask about something that is likely to be found in uploaded documents "
        "(e.g. specific research content, details from files, document-specific topics)? "
        "Answer only 'yes' or 'no'.\n\nQuestion: {question}"
    ),
    "multi-agent-web-decision": (
        "Does the question below require up-to-date information from the web "
        "(e.g. recent news, current prices, live data)? "
        "Answer only 'yes' or 'no'.\n\nQuestion: {question}"
    ),
    "multi-agent-synthesizer": (
        "You are a helpful assistant. Synthesize the information below into a single, "
        "clear, and complete answer to the question.\n\n"
        "{chat_history}"
        "Question: {question}\n\n"
        "Answer from uploaded documents:\n{rag_answer}\n\n"
        "Answer from web search:\n{web_answer}\n\n"
        "Final answer:"
    ),
}


def get_prompt(name: str) -> ChatPromptTemplate:
    template = PROMPT_TEMPLATES.get(name)
    if template is None:
        raise KeyError(f"Unknown prompt name: '{name}'. Available: {list(PROMPT_TEMPLATES)}")
    return ChatPromptTemplate.from_template(template)
