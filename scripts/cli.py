# the REPL (api.py later)

from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from agent import build_graph
from config import CORPORA, DEFAULT_EMBEDDING_MODEL, require_env
from index import get_client
from retrieval import KnowledgeBase

EXIT_WORDS = {"quit", "exit", "q"}

def run_turn(graph, user_input: str, config: dict) -> None:
    """streams one turn, announcing tool calls as they fire."""
    for event in graph.stream({"messages": [("user", user_input)]}, config=config):
        if "agent" in event:
            message = event["agent"]["messages"][-1]
            for call in getattr(message, "tool_calls", []):
                query = call.get("args", {}).get("query", "")
                print(f"-> {call['name']}({query!r})...")
            if message.content:
                print("Assistant:", message.content)

def main() -> None:
    load_dotenv()
    require_env("QDRANT_URL", "QDRANT_KEY", "OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    kb = KnowledgeBase(get_client(), CORPORA, embeddings).ensure_indexes()
    graph = build_graph(kb)

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
        if user_input.lower() in EXIT_WORDS:
            break
        run_turn(graph, user_input, config)

if __name__ == "__main__":
    main()