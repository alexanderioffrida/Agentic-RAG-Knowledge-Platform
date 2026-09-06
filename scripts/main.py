import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.tools import BraveSearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_qdrant import QdrantVectorStore

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
brave_key = os.getenv("BRAVE_API_KEY")

number_of_docs = 50

def preprocess_dataset(docs_list):
    '''this function processes our documents by splitting them into manageable chunks, 
    ensuring important context is preserved at the chunk boundaries through overlap.'''
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=50,
        disallowed_special=()
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits

def create_retriever(collection_name, doc_splits):
    vectorstore = QdrantVectorStore.from_documents(
        doc_splits,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=collection_name
    )
    return vectorstore.as_retriever()

def ingest():
    hugging_face_doc = HuggingFaceDatasetLoader("m-ric/huggingface_doc", "text")
    transformers_doc = HuggingFaceDatasetLoader("m-ric/transformers_documentation_en", "text")

    hf_splits = preprocess_dataset(hugging_face_doc.load()[:number_of_docs])
    transformer_splits = preprocess_dataset(transformers_doc.load()[:number_of_docs])

    hf_retriever = create_retriever("hf_docs", hf_splits)
    transformer_retriever = create_retriever("transformer_docs", transformer_splits)

    hf_retriever_tool = create_retriever_tool(
        hf_retriever,
        "retriever_hugging_face_documentation",
        "Search and return information about hugging face documentation, it includes the guide and Python code."
    )

    transformer_retriever_tool = create_retriever_tool(
        transformer_retriever,
        "retriever_transformer_documentation",
        "Search and return information specifically about transformers library"
    )

    return hf_retriever_tool, transformer_retriever_tool

class State(TypedDict):
    '''a state refers to the data or information stored and maintained at a specific point
    during the execution of a process or a series of operations.'''
    messages: Annotated[list, add_messages]

@tool("web_search_tool")
def search_tool(query: str) -> str:
    '''Search the live web for recent or general information not in the documentation.'''
    search = BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": 3})
    return search.run(query)

def route(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return END

def compile_graph(hf_retriever_tool, transformer_retriever_tool):
    tools = [hf_retriever_tool, transformer_retriever_tool, search_tool]
    tool_node = ToolNode(tools=tools)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge(START, "agent")
    return graph_builder.compile()

def run_agent(graph, user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():
    '''REPL'''
    hf_retriever_tool, transformer_retriever_tool = ingest()
    graph = compile_graph(hf_retriever_tool, transformer_retriever_tool)
    print("Ready. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        run_agent(graph, user_input)

if __name__ == "__main__":
    main()