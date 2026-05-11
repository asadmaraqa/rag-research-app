import re
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# Map glob patterns to their matching LangChain document loader
FILE_EXTENSION_TO_LOADER = {
    "**/*.pdf":  PyPDFLoader,
    "**/*.txt":  TextLoader,
    "**/*.csv":  CSVLoader,
    "**/*.xlsx": UnstructuredExcelLoader,
    "**/*.docx": Docx2txtLoader,
    "**/*.json": JSONLoader,
}

# Boilerplate phrases to strip from every chunk (case-insensitive)
BOILERPLATE_PATTERNS = [
    r"confidential[\s\-–—]*do not distribute",
    r"all rights reserved",
    r"page\s+\d+\s+of\s+\d+",
    r"©.{0,60}",
]
BOILERPLATE_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

MIN_CHUNK_LENGTH = 50  # chunks shorter than this are too noisy to be useful


def _clean_text(raw_text: str) -> str:
    """Remove noise from extracted text before it enters the vector store."""
    # Strip control characters and unicode garbage (e.g. \x00, )
    cleaned_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f-￿]", " ", raw_text)
    # Remove boilerplate patterns like copyright notices and page numbers
    cleaned_text = BOILERPLATE_REGEX.sub(" ", cleaned_text)
    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    # Collapse runs of spaces and tabs within lines
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)
    return cleaned_text.strip()


def load_all_documents(data_dir: str):
    """Load, clean, and return (docs, chunks_per_file) for all supported documents in data_dir."""
    data_directory = Path(data_dir).resolve()
    all_documents = []
    chunks_per_file = {}  # filename → chunk count

    for glob_pattern, DocumentLoader in FILE_EXTENSION_TO_LOADER.items():
        for file_path in data_directory.glob(glob_pattern):
            try:
                loaded_documents = DocumentLoader(str(file_path)).load()
                for document in loaded_documents:
                    document.page_content = _clean_text(document.page_content)
                non_empty_documents = [
                    document for document in loaded_documents
                    if len(document.page_content) > MIN_CHUNK_LENGTH
                ]
                all_documents.extend(non_empty_documents)
                chunks_per_file[file_path.name] = len(non_empty_documents)
            except Exception as load_error:
                print(f"[WARN] Could not load {file_path}: {load_error}")

    print(f"[INFO] Loaded {len(all_documents)} document chunks after cleaning.")
    return all_documents, chunks_per_file
