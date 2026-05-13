from agent.agent import Agent

__all__ = ["Agent", "tools"]


def stream_answer(query: str, history: list | None = None):
    """Placeholder stream for the Streamlit app; replace with LLM integration."""
    reply = (
        "[Placeholder] Wire an LLM here. "
        f"You asked: {query!r}. "
        "Tool implementations live in agent.tool_impl."
    )
    yield reply
