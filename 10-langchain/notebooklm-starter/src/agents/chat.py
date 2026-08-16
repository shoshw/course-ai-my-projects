"""The conversational chat agent."""

import os
import re
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.globals import set_debug
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from core.sources import format_docs
from core.store import SourceStore, store

if os.getenv("NOTEBOOKLM_DEBUG") == "1":
    set_debug(True)

SYSTEM_PROMPT = (
    "You are the chat assistant in NotebookLM, a grounded research assistant. "
    "Answer only using information retrieved from the user's sources - never rely on "
    "outside knowledge. Use the `list_sources` tool to see which sources exist and "
    "whether they are active. Use the `search_sources` tool whenever the user asks a "
    "question about their sources, to retrieve the relevant passages to ground your "
    "answer on; cite passages with their bracketed number (e.g. [1]). Use the "
    "`get_source` tool when you need a specific source's full text (e.g. the user asks "
    "about a document by name, or the retrieved snippets aren't enough context). If the "
    "sources don't contain the answer, say so instead of guessing. Be helpful, clear, "
    "and concise."
)

_SOURCE_TAG_RE = re.compile(r"\(source: (.*?)\)")


def make_tools_(store: SourceStore) -> list:
    """Build the retrieval tools bound to a given SourceStore."""

    @tool
    def search_sources(query: str) -> str:
        """Search the uploaded sources for passages relevant to a query.

        Use this to ground an answer in the user's documents. Returns the matching
        passages, each tagged with the source it came from.
        """
        docs = store.search(query)
        return format_docs(docs) if docs else "No matching passages found."

    @tool
    def list_sources() -> str:
        """List the available sources (id, name, and whether they're active)."""
        sources = store.list()
        if not sources:
            return "No sources have been uploaded yet."
        return "\n".join(
            f"- id={s.id} name={s.name!r} active={s.active}" for s in sources
        )

    @tool
    def get_source(source_id: str) -> str:
        """Fetch the full text of one source by its id.

        Use this when the retrieved passages aren't enough context, or the user asks
        about a specific document directly.
        """
        source = store.get(source_id)
        if source is None:
            return f"No source found with id {source_id!r}."
        return f"Source {source.name!r}:\n{source.content}"

    return [search_sources, list_sources, get_source]


checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
    tools=make_tools_(store),
    checkpointer=checkpointer,
)


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)


def _cited_sources(messages) -> list[str]:
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
    )
    seen: list[str] = []
    for message in messages[last_human + 1 :]:
        if isinstance(message, ToolMessage) and message.name == "search_sources":
            for name in _SOURCE_TAG_RE.findall(str(message.content)):
                if name not in seen:
                    seen.append(name)
    return seen


def answer(message: str, thread_id: str) -> Answer:
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return Answer(text=state["messages"][-1].content, sources=_cited_sources(state["messages"]))
