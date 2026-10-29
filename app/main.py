"""FastAPI service exposing the RAG agent.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.graph import ask  # noqa: E402
from app.ingest import SUPPORTED, ingest_path  # noqa: E402
from app.vectorstore import count  # noqa: E402

# Optional LangSmith tracing (reads LANGSMITH_* env vars automatically)
if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

app = FastAPI(title="RAG Chatbot", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    thread_id: str = "default"


class Source(BaseModel):
    source: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": get_settings().llm_provider,
        "chunks_indexed": count(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if count() == 0:
        raise HTTPException(400, "No documents indexed yet. Upload a file via /upload first.")
    try:
        return ChatResponse(**ask(req.question, req.thread_id))
    except Exception as exc:  # surface provider errors (bad key, rate limit) clearly
        raise HTTPException(500, f"Agent error: {exc}") from exc


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type. Use: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = ingest_path(dest)
    return {"filename": dest.name, "chunks_added": chunks}
