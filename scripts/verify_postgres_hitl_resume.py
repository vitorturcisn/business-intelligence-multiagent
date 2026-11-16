"""Valida a persistência e a retomada de interrupções HITL no PostgreSQL.

O teste usa workers determinísticos para isolar o comportamento do Supervisor,
interrupt(), Command(resume=...) e do checkpointer PostgreSQL. Não chama Ollama,
LangSmith nem serviços externos de embeddings.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import bi_multiagent.coordination.supervisor_graph as supervisor_module
from bi_multiagent.persistence.postgres import postgres_checkpointer
from bi_multiagent.schemas.contracts import AgentResult, SupervisorDecision


class FixedKnowledgeRouter:
    """Roteador determinístico para não depender do LLM Supervisor."""

    def invoke(self, messages):
        return SupervisorDecision(
            agents=["knowledge"],
            rationale="Teste determinístico de retomada HITL no PostgreSQL.",
        )


class DeterministicKnowledgeWorker:
    """Substitui a recuperação real para testar somente o fluxo HITL/checkpoint."""

    def invoke(self, state):
        task = state["task"]
        result = AgentResult(
            request_id=task["request_id"],
            agent="knowledge",
            status="completed",
            summary="Indicadores de vendas documentados para o teste HITL.",
            evidence="Evidência determinística de documentação de negócio.",
            sources=["data/knowledge_base/business_glossary.md"],
        )
        return {"result": result.model_dump(mode="json")}


def build_test_graph(checkpointer):
    """Monta o Supervisor real com especialistas determinísticos."""
    worker_factory = DeterministicKnowledgeWorker
    supervisor_module.build_sales_analyst_graph = worker_factory
    supervisor_module.build_knowledge_analyst_graph = worker_factory
    supervisor_module.build_economic_analyst_graph = worker_factory

    return supervisor_module.build_multiagent_graph(
        router=FixedKnowledgeRouter(),
        checkpointer=checkpointer,
    )


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_pending_approval(thread_id: str, setup: bool = False) -> str:
    """Executa até interrupt(), valida checkpoint e encerra a conexão."""
    config = thread_config(thread_id)
    question = (
        "Quais indicadores de vendas estão documentados? "
        "Salve o relatório em arquivo."
    )

    with postgres_checkpointer(setup=setup) as checkpointer:
        graph = build_test_graph(checkpointer)
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        interrupts = result.get("__interrupt__", [])
        if not interrupts:
            raise AssertionError("O grafo não pausou aguardando aprovação humana.")

        interrupt_value = interrupts[0].value
        if interrupt_value.get("type") != "human_approval":
            raise AssertionError(
                f"Tipo de interrupção inesperado: {interrupt_value.get('type')}"
            )

        snapshot = graph.get_state(config)
        if "human_approval" not in snapshot.next:
            raise AssertionError(
                f"Checkpoint não preservou o nó de aprovação: {snapshot.next}"
            )

        if snapshot.values.get("approval_required") is not True:
            raise AssertionError("Estado não registra aprovação humana obrigatória.")

        request_id = snapshot.values["request_id"]
        print("Interrupção HITL gravada no PostgreSQL.")
        print("Nó pendente:", snapshot.next)
        print("Request ID preservado:", request_id)

    print("Conexão encerrada com aprovação pendente.")
    return request_id


def resume_after_reconnect(
    thread_id: str,
    request_id: str,
    approved: bool,
    root_dir: Path,
) -> dict:
    """Reconecta, recupera a interrupção e a retoma com a decisão humana."""
    config = thread_config(thread_id)

    with postgres_checkpointer() as checkpointer:
        graph = build_test_graph(checkpointer)
        restored = graph.get_state(config)

        if "human_approval" not in restored.next:
            raise AssertionError(
                "A interrupção HITL não foi recuperada após reconectar: "
                f"{restored.next}"
            )

        if restored.values.get("request_id") != request_id:
            raise AssertionError("O request_id mudou após a reconexão.")

        print("Interrupção recuperada em uma nova conexão.")
        print("Decisão humana enviada:", "APROVAR" if approved else "REJEITAR")

        result = graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "comment": (
                        "Aprovação após reconexão PostgreSQL."
                        if approved
                        else "Rejeição após reconexão PostgreSQL."
                    ),
                }
            ),
            config=config,
        )

    report_path = (
        root_dir / "outputs" / "approved_reports" / f"{request_id}.md"
    )

    if approved:
        action = result.get("action_result", {})
        if action.get("status") != "executed":
            raise AssertionError(f"A ação aprovada não foi executada: {action}")
        if action.get("authorized") is not True:
            raise AssertionError("A ação executada não está marcada como autorizada.")
        if not report_path.exists():
            raise AssertionError("O relatório aprovado não foi criado no disco.")
        if "Relatório aprovado" not in report_path.read_text(encoding="utf-8"):
            raise AssertionError("O conteúdo do relatório aprovado está incorreto.")
        print("Ação aprovada executada e relatório persistido.")
    else:
        action = result.get("action_result", {})
        if action.get("status") != "rejected":
            raise AssertionError(f"A rejeição não foi aplicada: {action}")
        if action.get("authorized") is not False:
            raise AssertionError("A rejeição não bloqueou a autorização.")
        if report_path.exists():
            raise AssertionError("Um relatório foi criado apesar da rejeição.")
        print("Ação rejeitada; nenhum relatório foi criado.")

    return action


def main() -> None:
    original_root = supervisor_module.ROOT_DIR

    with TemporaryDirectory(prefix="bi_multiagent_hitl_") as temp_dir:
        isolated_root = Path(temp_dir)
        supervisor_module.ROOT_DIR = isolated_root

        try:
            print("===== HITL POSTGRES: APROVAÇÃO APÓS RECONEXÃO =====")
            approved_thread = f"hitl-approved-{uuid4().hex}"
            approved_request = start_pending_approval(
                approved_thread,
                setup=True,
            )
            resume_after_reconnect(
                approved_thread,
                approved_request,
                approved=True,
                root_dir=isolated_root,
            )

            print("\n===== HITL POSTGRES: REJEIÇÃO APÓS RECONEXÃO =====")
            rejected_thread = f"hitl-rejected-{uuid4().hex}"
            rejected_request = start_pending_approval(
                rejected_thread,
                setup=False,
            )
            resume_after_reconnect(
                rejected_thread,
                rejected_request,
                approved=False,
                root_dir=isolated_root,
            )

            print("\nPOSTGRES HITL RESUME: TESTE APROVADO.")
            print("Aprovação e rejeição foram retomadas após reconexão.")
        finally:
            supervisor_module.ROOT_DIR = original_root


if __name__ == "__main__":
    main()
