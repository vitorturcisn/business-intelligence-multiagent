"""Verifica persist?ncia entre reconstru??es do pool e do grafo."""

from uuid import uuid4

from langchain_core.messages import HumanMessage

from bi_multiagent.coordination.supervisor_graph import (
    build_multiagent_graph,
)
from bi_multiagent.persistence.postgres import postgres_checkpointer
from bi_multiagent.schemas.contracts import SupervisorDecision


class FixedKnowledgeRouter:
    """Evita depender do LLM Supervisor durante este teste de persist?ncia."""

    def invoke(self, messages):
        return SupervisorDecision(
            agents=["knowledge"],
            rationale="Teste de persist?ncia PostgreSQL.",
        )


def message_contents(state) -> list[str]:
    return [
        message.content
        for message in state.values.get("messages", [])
        if isinstance(message, HumanMessage) and isinstance(message.content, str)
    ]


def main() -> None:
    thread_id = f"postgres-persistence-{uuid4().hex}"
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    first_question = "Quais indicadores de vendas est?o documentados?"
    second_question = "Qual a diferen?a entre receita bruta e receita l?quida?"

    print("THREAD_ID:", thread_id)

    # PRIMEIRA INST?NCIA
    # Cria uma conversa e fecha completamente o pool ao sair.
    print("\n1. Criando primeira inst?ncia do grafo...")

    with postgres_checkpointer(setup=True) as checkpointer:
        graph = build_multiagent_graph(
            router=FixedKnowledgeRouter(),
            checkpointer=checkpointer,
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content=first_question)]},
            config=config,
        )

        if result.get("audit", {}).get("passed") is not True:
            raise RuntimeError(f"Auditoria inicial n?o aprovada: {result.get('audit')}")

        first_state = graph.get_state(config)

        if first_question not in message_contents(first_state):
            raise RuntimeError("A primeira mensagem n?o foi encontrada no checkpoint.")

        print("Primeira mensagem gravada.")
        print("Agentes:", result.get("route"))
        print("Auditoria:", result.get("audit", {}).get("passed"))

    print("Primeiro pool encerrado.")

    # SEGUNDA INST?NCIA
    # Um novo pool e um novo grafo recuperam o mesmo thread_id.
    print("\n2. Reconstruindo pool e grafo...")

    with postgres_checkpointer() as checkpointer:
        graph = build_multiagent_graph(
            router=FixedKnowledgeRouter(),
            checkpointer=checkpointer,
        )

        restored_state = graph.get_state(config)
        restored_messages = message_contents(restored_state)

        if first_question not in restored_messages:
            raise RuntimeError("A conversa anterior n?o foi recuperada do PostgreSQL.")

        print("Checkpoint anterior recuperado.")
        print("Mensagens humanas recuperadas:", len(restored_messages))

        result = graph.invoke(
            {"messages": [HumanMessage(content=second_question)]},
            config=config,
        )

        if result.get("audit", {}).get("passed") is not True:
            raise RuntimeError(
                f"Auditoria da segunda intera??o n?o aprovada: {result.get('audit')}"
            )

        final_state = graph.get_state(config)
        final_messages = message_contents(final_state)

        if first_question not in final_messages:
            raise RuntimeError("A primeira intera??o foi perdida.")

        if second_question not in final_messages:
            raise RuntimeError("A segunda intera??o n?o foi persistida.")

        print("Segunda intera??o gravada.")
        print("Total de mensagens humanas:", len(final_messages))
        print("Auditoria:", result.get("audit", {}).get("passed"))

    print("\nPOSTGRES PERSISTENCE: TESTE APROVADO.")
    print("Thread preservada:", thread_id)


if __name__ == "__main__":
    main()
