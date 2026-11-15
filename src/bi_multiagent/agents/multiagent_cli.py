"""CLI multiagente com PostgreSQL, Human-in-the-Loop e concorrência."""

import argparse
import sys

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from bi_multiagent.coordination.supervisor_graph import (
    build_multiagent_graph,
)
from bi_multiagent.persistence.postgres import (
    postgres_checkpointer,
    postgres_thread_lock,
)


def configure_console_encoding():
    """Força UTF-8 na entrada e saída do terminal."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def pending_interrupt(snapshot):
    """Recupera o payload de um interrupt persistido."""
    if snapshot is None:
        return None

    for task in getattr(snapshot, "tasks", ()):
        for item in getattr(task, "interrupts", ()):
            return item.value

    return None


def ask_for_approval(payload: dict) -> dict:
    """Solicita autorização humana pelo terminal."""
    action = payload.get("action", {})

    print("\n" + "=" * 72)
    print("APROVAÇÃO HUMANA NECESSÁRIA")
    print("=" * 72)
    print("Ação:", action.get("description", "Ação pendente"))
    print("Destino:", action.get("target", "Não informado"))
    print("Solicitação:", payload.get("question", ""))

    print("\nPrévia do relatório:")
    print(payload.get("report_preview", "(sem prévia)"))
    print("\nAviso:", payload.get("warning", ""))

    choice = input("\nAutorizar a gravação? (aprovar/rejeitar): ")
    approved = choice.strip().casefold() in {
        "s",
        "sim",
        "a",
        "aprovar",
        "aprovado",
        "yes",
        "y",
    }

    comment = input("Observação da decisão (opcional): ").strip()

    return {
        "approved": approved,
        "comment": comment,
    }


def resolve_interrupts(graph, result, config):
    """Apresenta interrupções e retoma o grafo mantendo o lock externo."""
    while result.get("__interrupt__"):
        interruption = result["__interrupt__"][0]
        decision = ask_for_approval(interruption.value)

        result = graph.invoke(
            Command(resume=decision),
            config=config,
        )

    return result


def print_result(result):
    answer = result.get("answer")
    if answer:
        print("\n" + answer)

    action = result.get("action_result", {})
    if action:
        print("\nStatus da ação:", action.get("status"))
        if action.get("path"):
            print("Arquivo:", action["path"])


def main() -> None:
    configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Business Intelligence Multi-Agent System"
    )
    parser.add_argument(
        "--thread-id",
        default="multiagent-terminal",
        help="Identificador persistente da conversa.",
    )
    args = parser.parse_args()

    config = {
        "configurable": {
            "thread_id": args.thread_id,
        }
    }

    with postgres_checkpointer() as checkpointer:
        graph = build_multiagent_graph(
            checkpointer=checkpointer,
        )

        print("=" * 72)
        print("BUSINESS INTELLIGENCE MULTI-AGENT SYSTEM")
        print("Supervisor + Sales + Knowledge + Economic + Audit")
        print("Human-in-the-Loop: ativo")
        print("Checkpointer: PostgreSQL")
        print("Double-texting: advisory lock por thread_id")
        print(f"Thread: {args.thread_id}")
        print("Digite 'sair' para encerrar.")
        print("=" * 72)

        # Recupera uma aprovação pendente deixada por outro processo.
        try:
            with postgres_thread_lock(args.thread_id):
                snapshot = graph.get_state(config)
                payload = pending_interrupt(snapshot)

                if payload is not None:
                    print("\nFoi encontrada uma aprovação pendente nesta conversa.")
                    result = graph.invoke(
                        Command(resume=ask_for_approval(payload)),
                        config=config,
                    )
                    result = resolve_interrupts(
                        graph,
                        result,
                        config,
                    )
                    print_result(result)

        except RuntimeError as exc:
            print(f"\nNão foi possível iniciar: {exc}")
            return

        while True:
            try:
                question = input("\nVocê: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando.")
                break

            if question.casefold() in {"sair", "exit", "quit"}:
                print("Encerrando.")
                break

            if not question:
                continue

            try:
                # O lock permanece ativo durante a execução, a espera
                # pela decisão humana e a retomada do grafo.
                with postgres_thread_lock(args.thread_id):
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=question)]},
                        config=config,
                    )

                    result = resolve_interrupts(
                        graph,
                        result,
                        config,
                    )

                print_result(result)

            except RuntimeError as exc:
                print(f"\nInteração não processada: {exc}")


if __name__ == "__main__":
    main()
