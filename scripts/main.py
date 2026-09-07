import os
import re
import uuid
from datetime import datetime, timezone
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage
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

SYSTEM_PROMPT = """You are an expert AI assistant specializing in Hugging Face and machine learning.
You have three tools available:
- retriever_hugging_face_documentation: Search general Hugging Face ecosystem documentation, tutorials, Hub, and guides.
- retriever_transformer_documentation: Search documentation specifically for the Hugging Face Transformers library (models, pipelines, tokenizers, Trainer).
- web_search_tool: Search the live web for recent developments, external information, or topics outside Hugging Face documentation.

Routing guidelines:
1. Always prefer the documentation retrievers when answering technical questions about Hugging Face or Transformers.
2. Route queries about Transformer models, architectures, tokenizers, or specific transformers classes to retriever_transformer_documentation.
3. Route queries about other Hugging Face libraries, Hub, datasets, spaces, or general ecosystem workflows to retriever_hugging_face_documentation.
4. Fall back to web_search_tool only if the information cannot be found in the documentation retrievers or if the user explicitly asks about current events / external packages.
5. Ground your answers strictly in the retrieved information and explain the reasoning clearly."""

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
    print(f"-> Alias '{alias}' NOT found. Downloading and ingesting '{dataset}'...")
    loader = HuggingFaceDatasetLoader(dataset, "text")
    splits = preprocess_dataset(loader.load()[:number_of_docs])

    embeddings = OpenAIEmbeddings(model=embedding_model)
    dimension = len(embeddings.embed_query("dimension_probe"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    build = f"{alias}__build_{stamp}_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        build, VectorParams(size=dimension, distance=Distance.COSINE)
    )

    try:
        store = QdrantVectorStore(
            client=client, collection_name=build, embedding=embeddings
        )
        store.add_documents(splits)
        uploaded = client.count(build, exact=True).count # was originally client.get_collection(build).points_count
        if uploaded != len(splits):
            raise RuntimeError(
                f"expected {len(splits)} points in '{build}', found {uploaded}"
            )
    except BaseException:
        print(f"-> Ingest failed. Discarding incomplete build '{build}'...")
        client.delete_collection(build)
        raise

    print(f"-> Uploaded {len(splits)} chunks to '{build}'.")
    return build

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
    alias = alias_for(base)
    cleanup_builds(client, alias)
    
    if client.collection_exists(alias):
        print(f"-> Alias '{alias}' found in Qdrant. Loading existing index...")
    else:
        build = ingest_collection(client, alias, dataset)
        swap_alias(client, alias, build)
        cleanup_builds(client, alias)
        print(f"-> Alias '{alias}' now points at '{build}'.")
    
    store = QdrantVectorStore(
        client=client,
        collection_name=alias,
        embedding=OpenAIEmbeddings(model=embedding_model)
    )
    return store.as_retriever()

def ingest():
    client = get_client()
    hf_retriever = get_or_create_retriever(client, "hf_docs", "m-ric/huggingface_doc")
    transformer_retriever = get_or_create_retriever(client, "transformer_docs", "m-ric/transformers_documentation_en")

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

def compile_graph(hf_retriever_tool, transformer_retriever_tool, system_prompt: str = SYSTEM_PROMPT):
    tools = [hf_retriever_tool, transformer_retriever_tool, search_tool]
    tool_node = ToolNode(tools=tools)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: State):
        messages = state["messages"]
        if system_prompt:
            prompt_messages = [SystemMessage(content=system_prompt)] + [
                m for m in messages if not isinstance(m, SystemMessage)
            ]
        else:
            prompt_messages = messages
        return {"messages": [llm_with_tools.invoke(prompt_messages)]}

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
            message = event["agent"]["messages"][-1]
            for call in getattr(message, "tool_calls", []):
                print(f"-> Calling {call['name']}...")
            if message.content:
                print("Assistant:", message.content)

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