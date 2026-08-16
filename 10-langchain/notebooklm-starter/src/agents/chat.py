"""The conversational chat agent."""

from dataclasses import dataclass

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = (
    "You are the chat assistant in NotebookLM, a grounded research assistant. "
    "Be helpful, clear, and concise."
)

checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


@dataclass
class Answer:
    text: str


def answer(message: str, thread_id: str) -> Answer:
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return Answer(text=state["messages"][-1].content)
