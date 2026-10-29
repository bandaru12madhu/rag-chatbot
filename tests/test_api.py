from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


def test_health(monkeypatch):
    monkeypatch.setattr(main, "count", lambda: 7)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 7


def test_chat_requires_indexed_documents(monkeypatch):
    monkeypatch.setattr(main, "count", lambda: 0)
    resp = client.post("/chat", json={"question": "hi"})
    assert resp.status_code == 400


def test_chat_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(main, "count", lambda: 3)
    monkeypatch.setattr(
        main,
        "ask",
        lambda q, t: {"answer": "RAG grounds answers.", "sources": [{"source": "a.pdf", "page": 0}]},
    )
    resp = client.post("/chat", json={"question": "What is RAG?", "thread_id": "t1"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "RAG grounds answers."
    assert resp.json()["sources"][0]["source"] == "a.pdf"


def test_upload_rejects_bad_extension():
    resp = client.post("/upload", files={"file": ("evil.exe", b"x")})
    assert resp.status_code == 400
