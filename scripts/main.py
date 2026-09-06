import os
import re
from datetime import datetime, timezone
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
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    VectorParams
)

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
brave_key = os.getenv("BRAVE_API_KEY")

number_of_docs = 50
embedding_model = "text-embedding-3-small"

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

_client = None

def get_client():
    '''returns a single module-level QdrantClient, created on first use and reused.'''
    global _client
    if _client is None:
        _client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    return _client

def alias_for(base):
    '''assembles the alias from base name, model slug, and document count.'''
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_model.lower()).strip("_")
    return f"{base}__{slug}__n{number_of_docs}"

def ingest_collection(client, alias, dataset):
    '''creates the build collection, uploads splits, returns the build name.'''
    pass

def swap_alias(client, alias, build):
    '''points the alias at the completed build via one CreateAliasOperation. this uses
    the exact same alias name and leverages the upsert behavior to modify where it points.'''
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=build, alias_name=alias)
            )
        ]
    )

def cleanup_builds(client, alias):
    '''deletes every {alias}__build_* collection that is not the alias's current target.'''
    prefix = f"{alias}__build_"
    current = None
    for entry in client.get_aliases().aliases:
        if entry.alias_name == alias:
            current = entry.collection_name
            break
    
    for collection in client.get_collections().collections:
        if collection.name.startswith(prefix) and collection.name != current:
            print(f"-> Removing stale build collection '{collection.name}'...")
            client.delete_collection(collection.name)

def get_or_create_retriever(client, base, dataset):
    '''orchestrates cleanup, then load or build.'''
    pass


# def create_retriever(collection_name, doc_splits):
#     '''creates a fresh colletion, embeds docs, and uploads them to Qdrant.'''
#     vectorstore = QdrantVectorStore.from_documents(
#         doc_splits,
#         OpenAIEmbeddings(model=embedding_model),
#         url=qdrant_url,
#         api_key=qdrant_key,
#         collection_name=collection_name
#     )
#     return vectorstore.as_retriever()

# def get_retriever(collection_name):
#     '''connects directly to a pre-existing collection without embedding anything.'''
#     vectorstore = QdrantVectorStore.from_existing_collection(
#         embedding=OpenAIEmbeddings(model=embedding_model),
#         url=qdrant_url,
#         api_key=qdrant_key,
#         collection_name=collection_name
#     )
#     return vectorstore.as_retriever()

# def get_or_create_retriever(collection_name, dataset_name):
#     '''checks Qdrant first. loads if exists, otherwise downloads and ingests.'''
#     client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

#     if client.collection_exists(collection_name):
#         print(f"-> Collection '{collection_name}' found in Qdrant. Loading existing data...")
#         return get_retriever(collection_name)
#     else:
#         print(f"-> Collection '{collection_name}' NOT found. Downloading and ingesting '{dataset_name}'...")
#         loader = HuggingFaceDatasetLoader(dataset_name, "text")
#         splits = preprocess_dataset(loader.load()[:number_of_docs])
#         return create_retriever(collection_name, splits)

def ingest():
    hf_retriever = get_or_create_retriever("hf_docs", "m-ric/huggingface_doc")
    transformer_retriever = get_or_create_retriever("transformer_docs", "m-ric/transformers_documentation_en")

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

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def run_agent(graph, user_input: str, config: dict):
    for event in graph.stream({"messages": [("user", user_input)]}, config=config):
        if "agent" in event:
            content = event["agent"]["messages"][-1].content
            if content:
                print("Assistant:", content)

def main():
    '''REPL'''
    hf_retriever_tool, transformer_retriever_tool = ingest()
    graph = compile_graph(hf_retriever_tool, transformer_retriever_tool)
    config = {"configurable": {"thread_id": "cli_session"}}
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
        run_agent(graph, user_input, config)

if __name__ == "__main__":
    main()