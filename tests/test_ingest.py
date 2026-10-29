from langchain_core.documents import Document

from app.ingest import load_file, split_documents


def test_load_markdown_sets_source(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nSome content about RAG.", encoding="utf-8")
    docs = load_file(f)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.md"


def test_split_documents_creates_overlapping_chunks():
    text = "word " * 1000  # ~5000 characters
    chunks = split_documents([Document(page_content=text, metadata={"source": "x.txt"})])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 800 for c in chunks)
    assert all(c.metadata["source"] == "x.txt" for c in chunks)


def test_unsupported_file_type_raises(tmp_path):
    f = tmp_path / "image.png"
    f.write_bytes(b"not text")
    try:
        load_file(f)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")
