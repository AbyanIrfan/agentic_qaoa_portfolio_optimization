"""
LLM router — returns the correct LangChain chat model for a given backend name.
Supports homogeneous mode (same LLM for all nodes) and heterogeneous mode
(different LLM per node, configured in settings.NODE_LLM_MAP).
"""

from functools import lru_cache
from langchain_anthropic      import ChatAnthropic
from langchain_openai         import ChatOpenAI
from langchain_google_genai   import ChatGoogleGenerativeAI

from config.settings import LLM_BACKENDS, NODE_LLM_MAP


@lru_cache(maxsize=8)
def get_llm(name: str):
    """
    Returns a cached LangChain chat model instance.

    Args:
        name: "claude" | "gpt" | "gemini"

    Usage:
        llm = get_llm("claude")
        response = llm.invoke("your prompt")
    """
    model_id = LLM_BACKENDS.get(name)
    if model_id is None:
        raise ValueError(f"Unknown LLM backend: '{name}'. "
                         f"Choose from {list(LLM_BACKENDS.keys())}")

    if name == "claude":
        return ChatAnthropic(model=model_id, temperature=0)
    elif name == "gpt":
        return ChatOpenAI(model=model_id, temperature=0)
    elif name == "gemini":
        return ChatGoogleGenerativeAI(model=model_id, temperature=0, timeout=90, thinking_level="Medium")


def get_node_llm(node_name: str, llm_override: str = None):
    """
    Returns the LLM assigned to a specific pipeline node.

    If llm_override is "hetero", uses NODE_LLM_MAP for routing.
    If llm_override is a specific backend name, uses that for all nodes.
    """
    if llm_override and llm_override != "hetero":
        return get_llm(llm_override)

    assigned = NODE_LLM_MAP.get(node_name, "claude")
    return get_llm(assigned)


def get_response_text(response) -> str:
    """
    Normalize a LangChain chat response's .content into a plain string,
    regardless of provider quirks. ChatAnthropic/ChatOpenAI return a plain
    string in normal (non-thinking) mode; ChatGoogleGenerativeAI's newer
    reasoning models (e.g. gemini-3.1-pro-preview) return a list of content
    blocks instead (text blocks plus an internal reasoning signature block),
    even with default settings. Every node should call this instead of
    touching response.content directly, so a provider switch never crashes
    on response-shape differences.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)