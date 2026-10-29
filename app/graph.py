"""LangGraph RAG agent: retrieve -> generate, with per-thread conversation memory."""
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.llm import get_llm
from app.vectorstore import search

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. Cite sources like [source.pdf, p.3] after the claims they support. "
    "If the context does not contain the answer, say you don't know."
)


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]


def _format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = d.metadata.get("source", "unknown")
        if page is not None:
            label += f", p.{int(page) + 1}"
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def retrieve(state: RAGState) -> dict:
    question = state["messages"][-1].content
    return {"context": search(question)}


def generate(state: RAGState) -> dict:
    context = _format_context(state["context"])
    prompt = [
        SystemMessage(content=f"{SYSTEM_PROMPT}\n\nContext:\n{context}"),
        *state["messages"],
    ]
    response = get_llm().invoke(prompt)
    return {"messages": [response]}


def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, thread_id: str = "default") -> dict:
    """Ask a question. Returns {"answer": str, "sources": [{"source", "page"}]}."""
    result = get_graph().invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
    sources, seen = [], set()
    for d in result["context"]:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return {"answer": answer.content, "sources": sources}
