"""Streamlit chat UI that talks to the FastAPI backend.

Run:  streamlit run ui/streamlit_app.py
"""
import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot")
st.title("Chat with your documents")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"])
    if uploaded and st.button("Index document"):
        with st.spinner("Embedding and storing..."):
            resp = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=300,
            )
        if resp.ok:
            st.success(f"Indexed {resp.json()['chunks_added']} chunks")
        else:
            st.error(resp.json().get("detail", resp.text))
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/chat",
                json={"question": question, "thread_id": st.session_state.thread_id},
                timeout=120,
            )
        if resp.ok:
            data = resp.json()
            text = data["answer"]
            if data["sources"]:
                refs = ", ".join(
                    f"{s['source']}" + (f" p.{s['page'] + 1}" if s["page"] is not None else "")
                    for s in data["sources"]
                )
                text += f"\n\n*Sources: {refs}*"
        else:
            text = f"Error: {resp.json().get('detail', resp.text)}"
        st.markdown(text)
    st.session_state.messages.append({"role": "assistant", "content": text})
