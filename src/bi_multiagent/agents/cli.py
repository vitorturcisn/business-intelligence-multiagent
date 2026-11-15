import argparse

from langchain_core.messages import AIMessageChunk, HumanMessage

from bi_multiagent.agents.single_agent_graph import (
    build_single_agent_graph,
)


def stream_turn(graph, question: str, thread_id: str) -> None:
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    print("\nAgente: ", end="", flush=True)

    for chunk, _metadata in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        if (
            isinstance(chunk, AIMessageChunk)
            and isinstance(chunk.content, str)
            and chunk.content
        ):
            print(chunk.content, end="", flush=True)

    print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Business Intelligence Multi-Agent - Agente individual"
    )
    parser.add_argument(
        "--thread-id",
        default="terminal-session",
        help="Identificador da conversa para persistir o histórico.",
    )
    args = parser.parse_args()

    graph = build_single_agent_graph()

    print("=" * 65)
    print("BUSINESS INTELLIGENCE AGENT - LANGGRAPH")
    print(f"Modelo: thread_id={args.thread_id}")
    print("Digite 'sair' para encerrar.")
    print("=" * 65)

    while True:
        try:
            question = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        if not question:
            continue

        stream_turn(graph, question, args.thread_id)


if __name__ == "__main__":
    main()
