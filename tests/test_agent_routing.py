import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END
from scripts import main


def test_system_prompt_guidance():
    '''SYSTEM_PROMPT directs routing across both retrievers and web fallback.'''
    prompt = main.SYSTEM_PROMPT
    assert "retriever_hugging_face_documentation" in prompt
    assert "retriever_transformer_documentation" in prompt
    assert "web_search_tool" in prompt
    assert "Routing guidelines" in prompt


def test_route_function():
    '''route sends tool calls to 'tools' and final answers to END.'''
    # message with tool call
    call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "web_search_tool", "args": {"query": "test"}, "id": "call_1"}]
    )
    assert main.route({"messages": [call_msg]}) == "tools"

    # message with answer content only
    answer_msg = AIMessage(content="Here is the explanation of transformers.")
    assert main.route({"messages": [answer_msg]}) == END

    # invalid empty state
    with pytest.raises(ValueError, match="No messages found"):
        main.route({"messages": []})


def test_compile_graph_prepends_system_message(monkeypatch):
    '''compile_graph ensures the model is invoked with the system prompt.'''
    recorded_invocations = []

    class MockChatModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            recorded_invocations.append(messages)
            return AIMessage(content="Mocked response")

    monkeypatch.setattr(main, "ChatOpenAI", lambda **kwargs: MockChatModel())

    @tool
    def dummy_hf(query: str) -> str:
        '''dummy hf doc'''
        return "hf"

    @tool
    def dummy_transformer(query: str) -> str:
        '''dummy transformer doc'''
        return "transformer"

    graph = main.compile_graph(dummy_hf, dummy_transformer)
    config = {"configurable": {"thread_id": "test_session"}}

    # run a single turn
    graph.invoke({"messages": [HumanMessage(content="How does pipeline work?")]}, config=config)

    assert len(recorded_invocations) > 0
    first_messages = recorded_invocations[0]

    # verify first message is a SystemMessage with main.SYSTEM_PROMPT
    assert isinstance(first_messages[0], SystemMessage)
    assert first_messages[0].content == main.SYSTEM_PROMPT
    assert isinstance(first_messages[1], HumanMessage)
    assert first_messages[1].content == "How does pipeline work?"


def test_custom_system_prompt_override(monkeypatch):
    '''compile_graph accepts a custom system prompt override.'''
    recorded_invocations = []

    class MockChatModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            recorded_invocations.append(messages)
            return AIMessage(content="Custom response")

    monkeypatch.setattr(main, "ChatOpenAI", lambda **kwargs: MockChatModel())

    @tool
    def dummy_hf(query: str) -> str:
        '''dummy hf doc'''
        return "hf"

    @tool
    def dummy_transformer(query: str) -> str:
        '''dummy transformer doc'''
        return "transformer"

    custom_prompt = "Custom system instructions for routing."
    graph = main.compile_graph(dummy_hf, dummy_transformer, system_prompt=custom_prompt)
    config = {"configurable": {"thread_id": "custom_session"}}

    graph.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)

    assert len(recorded_invocations) > 0
    assert recorded_invocations[0][0].content == custom_prompt
