"""Vector database layer: Hugging Face embeddings + Chroma."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

COLLECTION = "documents"


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().embedding_model)


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def add_documents(docs: list[Document]) -> int:
    """Embed and store chunks. Returns number of chunks added."""
    if not docs:
        return 0
    get_vectorstore().add_documents(docs)
    return len(docs)


def search(query: str, k: int | None = None) -> list[Document]:
    k = k or get_settings().top_k
    return get_vectorstore().similarity_search(query, k=k)


def count() -> int:
    return get_vectorstore()._collection.count()
