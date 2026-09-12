"""The graph, with a fake chat model. No key, no network, no tokens spent.

These exist because a typo in `State` — `=` where `:` belonged — left the graph with no
channels at all, and every retrieval test still passed. The retrieval half of this repo
is well covered; this is the half that was not.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage

import agent as agent_module
from agent import SYSTEM_PROMPT, build_graph
from retrieval import Passage, RetrievalResult


class FakeChat(FakeMessagesListChatModel):
    """answers from a fixed list and records what it was handed."""

    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


class FakeKB:
    def retrieve(self, query, **kwargs):
        return RetrievalResult(
            query=query, passages=[Passage(id="1", text="a chunk", source="alpha")]
        )


@pytest.fixture
def llm(monkeypatch):
    fake = FakeChat(responses=[AIMessage(content="answer"), AIMessage(content="answer")])
    fake.seen = []
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **kwargs: fake)
    return fake


@pytest.fixture
def graph(llm):
    return build_graph(FakeKB())


def test_history_accumulates_across_turns(graph):
    """The `add_messages` reducer, exercised. Without live channels this raises."""
    config = {"configurable": {"thread_id": "t"}}

    first = graph.invoke({"messages": [("user", "one")]}, config=config)
    assert len(first["messages"]) == 2

    second = graph.invoke({"messages": [("user", "two")]}, config=config)
    assert len(second["messages"]) == 4, "turn two must see turn one"
    assert [m.content for m in second["messages"]][0] == "one"


def test_threads_do_not_share_history(graph):
    graph.invoke({"messages": [("user", "one")]}, config={"configurable": {"thread_id": "a"}})
    other = graph.invoke(
        {"messages": [("user", "two")]}, config={"configurable": {"thread_id": "b"}}
    )
    assert len(other["messages"]) == 2


def test_system_prompt_is_prepended_every_turn(graph, llm):
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [("user", "one")]}, config=config)
    graph.invoke({"messages": [("user", "two")]}, config=config)

    assert len(llm.seen) == 2
    for turn in llm.seen:
        assert isinstance(turn[0], SystemMessage)
        assert turn[0].content == SYSTEM_PROMPT


def test_search_tool_returns_text_and_the_result_as_artifact():
    """`content_and_artifact` is what keeps metadata alive for citations."""
    tool = agent_module.make_search_tool(FakeKB())
    message = tool.invoke({
        "type": "tool_call", "name": "search_knowledge_base",
        "args": {"query": "pipelines"}, "id": "call_1",
    })

    assert "a chunk" in message.content, "the model sees rendered text"
    assert isinstance(message.artifact, RetrievalResult), "the handler sees the object"
    assert message.artifact.passages[0].source == "alpha"
