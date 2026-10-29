"""Load PDFs / text files, split into chunks, and store them in the vector DB.

CLI usage:
    python -m app.ingest data/sample_docs
    python -m app.ingest path/to/file.pdf
"""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore import add_documents

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest_path(path: str | Path) -> int:
    """Ingest one file or every supported file in a folder. Returns chunk count."""
    path = Path(path)
    files = (
        [p for p in sorted(path.rglob("*")) if p.suffix.lower() in SUPPORTED]
        if path.is_dir()
        else [path]
    )
    total = 0
    for f in files:
        chunks = split_documents(load_file(f))
        total += add_documents(chunks)
        print(f"Ingested {f.name}: {len(chunks)} chunks")
    return total


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m app.ingest <file-or-folder>")
    print(f"Done. {ingest_path(sys.argv[1])} chunks stored.")
