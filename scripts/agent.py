# the tool wrapper and the graph

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import DEFAULT_CHAT_MODEL
from retrieval import KnowledgeBase, RetrievalResult, format_for_llm, format_for_llm

SYSTEM_PROMPT = """You answer questions about the HuggingFace ecosystem and the \
transformers library, grounded in retrieved documentation.

Use search_knowledge_base for anything factual about these libraries. Write a focused \
search query rather than passing the user's message through verbatim: resolve pronouns \
and references against the conversation, and use the specific API or concept names the \
user is really asking about.

If the returned passages do not answer the question, search again with a different \
phrasing before falling back to web_search. Use web_search only for genuinely recent \
information or topics outside the indexed documentation.

Answer only from retrieved passages. Cite them by their bracketed numbers. If the \
passages do not contain the answer, say so plainly rather than filling the gap."""

class State(TypedDict):
    """data carried through the graph. `add_messages` appends rather than overwrites."""
    messages = Annotated[list, add_messages]

def make_search_tool(kb: KnowledgeBase):
    """builds the one retrieval tool, closed over a KnowledgeBase."""

    @tool("search_knowledge_base", response_format="content_and_artifact")
    def search_knowledge_base(query: str) -> tuple[str, RetrievalResult]:
        """Search the indexed HuggingFace and transformers documentation.
        
        Args:
            query: A focused search query describing the information needed.
        """
        result = kb.retrieve(query)
        return format_for_llm(result.passages), result
    
    return search_knowledge_base

@tool("web_search")
def web_search(query: str) -> str:
    """Search the live web for recent or general information not in the documentation.
    
    Args:
        query: The search query.
    """
    from langchain_community.tools import BraveSearch

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Web search unavailable: BRAVE_API_KEY is not set."
    search = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 3})
    return search.run(query)

def build_graph(kb: KnowledgeBase, checkpointer=None):
    """compiles the agent graph. pass a durable checkpointer in the service layer."""
    tools = [make_search_tool(kb), web_search]
    llm = ChatOpenAI(model=DEFAULT_CHAT_MODEL, temperature=0).bind_tools(tools)

    def agent(state: State) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer or MemorySaver())