# Agentic AI: A Short Introduction

Agentic AI refers to systems where a large language model (LLM) does more than
answer a single prompt. An agent can plan steps, call tools, observe the results,
and decide what to do next until a goal is reached.

## Core building blocks

- **LLM**: the reasoning engine that decides what to do.
- **Tools**: functions the agent can call, such as web search, a calculator, or a database query.
- **Memory**: short-term memory keeps the current conversation; long-term memory stores facts across sessions.
- **Orchestration**: frameworks like LangGraph model the agent as a graph of nodes and edges, which makes loops, branching, and retries explicit.

## Retrieval-Augmented Generation (RAG)

RAG gives an LLM access to private or fresh knowledge. Documents are split into
chunks, each chunk is converted into an embedding (a list of numbers capturing
meaning), and the embeddings are stored in a vector database such as Chroma or FAISS.
At question time the system embeds the question, finds the most similar chunks,
and passes them to the LLM as context so the answer is grounded and can cite sources.

## Human-in-the-loop

For risky actions such as sending emails or issuing refunds, a graph can pause
before the action and wait for a human to approve, edit, or reject it. LangGraph
supports this through checkpointing and interrupts.

## Evaluation

Agents should be tested like software. A good evaluation set contains questions
with expected answers, and tools like LangSmith can score accuracy, latency,
and cost across different models and prompts.
