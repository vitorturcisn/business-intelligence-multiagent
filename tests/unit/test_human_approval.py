from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import bi_multiagent.coordination.supervisor_graph as supervisor_module
from bi_multiagent.coordination.supervisor_graph import (
    _detect_report_export,
    build_multiagent_graph,
)
from bi_multiagent.schemas.contracts import SupervisorDecision


class FixedKnowledgeRouter:
    """Router determinístico para testar o fluxo sem depender do LLM."""

    def invoke(self, messages):
        return SupervisorDecision(
            agents=["knowledge"],
            rationale="Teste do checkpoint de aprovação humana.",
        )


def test_report_export_policy_requires_explicit_request():
    action = _detect_report_export(
        "Analise os indicadores e salve o relatório em arquivo."
    )

    assert action is not None
    assert action["type"] == "save_report"

    assert _detect_report_export("Quais são os indicadores de vendas?") is None

    assert _detect_report_export("Não salve o relatório.") is None


def test_human_approval_pauses_then_executes_or_rejects(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        supervisor_module,
        "ROOT_DIR",
        tmp_path,
    )

    graph = build_multiagent_graph(
        router=FixedKnowledgeRouter(),
        checkpointer=InMemorySaver(),
    )

    def start_request(thread_id):
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Quais indicadores de vendas estão documentados? "
                            "Salve o relatório em arquivo."
                        )
                    )
                ]
            },
            config=config,
        )

        assert result.get("__interrupt__")
        assert result["__interrupt__"][0].value["type"] == "human_approval"

        snapshot = graph.get_state(config)
        assert "human_approval" in snapshot.next

        return config, snapshot

    # Aprovação: o arquivo deve ser gravado.
    approved_thread = f"hitl-approved-{uuid4().hex}"
    approved_config, approved_snapshot = start_request(approved_thread)

    approved_result = graph.invoke(
        Command(
            resume={
                "approved": True,
                "comment": "Aprovado pelo teste automatizado.",
            }
        ),
        config=approved_config,
    )

    approved_request_id = approved_snapshot.values["request_id"]
    approved_file = (
        tmp_path / "outputs" / "approved_reports" / f"{approved_request_id}.md"
    )

    assert approved_result["action_result"]["status"] == "executed"
    assert approved_result["action_result"]["authorized"] is True
    assert approved_file.exists()
    assert "Relatório aprovado" in approved_file.read_text(encoding="utf-8")

    # Rejeição: nenhum arquivo deve ser gravado.
    rejected_thread = f"hitl-rejected-{uuid4().hex}"
    rejected_config, rejected_snapshot = start_request(rejected_thread)

    rejected_result = graph.invoke(
        Command(
            resume={
                "approved": False,
                "comment": "Rejeitado pelo teste automatizado.",
            }
        ),
        config=rejected_config,
    )

    rejected_request_id = rejected_snapshot.values["request_id"]
    rejected_file = (
        tmp_path / "outputs" / "approved_reports" / f"{rejected_request_id}.md"
    )

    assert rejected_result["action_result"]["status"] == "rejected"
    assert rejected_result["action_result"]["authorized"] is False
    assert not rejected_file.exists()
