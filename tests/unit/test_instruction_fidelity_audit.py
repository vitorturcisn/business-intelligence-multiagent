from __future__ import annotations

from bi_multiagent.coordination.instruction_fidelity import (
    audit_instruction_fidelity,
    infer_expected_agents,
)
from bi_multiagent.schemas.contracts import AgentResult


def result(agent: str, request_id: str = "r1", **overrides) -> AgentResult:
    payload = {
        "request_id": request_id,
        "agent": agent,
        "status": "completed",
        "summary": f"Resumo de {agent}.",
        "evidence": f"Evidência de {agent}.",
        "sources": [f"fonte-{agent}.md"],
        "warnings": [],
    }
    payload.update(overrides)
    return AgentResult.model_validate(payload)


def test_infer_numeric_sales_request():
    assert infer_expected_agents("Quanto foi a receita líquida total?") == ["sales"]
    assert infer_expected_agents("Qual a receita líquida?") == ["sales"]


def test_infer_conceptual_sales_request():
    assert infer_expected_agents("Qual é a fórmula da receita líquida?") == ["knowledge"]
    assert infer_expected_agents("Qual foi a fórmula da receita líquida?") == ["knowledge"]


def test_infer_combined_sales_and_knowledge_request():
    assert infer_expected_agents(
        "Qual foi a receita líquida total e qual é a fórmula?"
    ) == ["sales", "knowledge"]


def test_infer_selic_value_and_selic_source_requests():
    assert infer_expected_agents(
        "Qual foi a meta Selic entre 01/10/2026 e 15/11/2026?"
    ) == ["economic"]
    assert infer_expected_agents("Qual é a fonte documentada da série Selic?") == ["knowledge"]


def test_audit_passes_when_route_and_results_match_user_intent():
    sales = result("sales")
    knowledge = result("knowledge")
    answer = f"{sales.summary}\n{knowledge.summary}"
    audit = audit_instruction_fidelity(
        question="Qual foi a receita líquida total e qual é a fórmula?",
        answer=answer,
        selected_agents=["sales", "knowledge"],
        agent_results=[sales, knowledge],
        request_id="r1",
    )
    assert audit["status"] == "passed"
    assert audit["score"] == 1.0


def test_audit_flags_missing_required_specialist():
    sales = result("sales")
    audit = audit_instruction_fidelity(
        question="Qual foi a receita líquida total e qual é a fórmula?",
        answer=sales.summary,
        selected_agents=["sales"],
        agent_results=[sales],
        request_id="r1",
    )
    assert audit["status"] == "review_required"
    assert "clear_intent_routing" in audit["failed_checks"]
    assert "required_specialists_returned" in audit["failed_checks"]


def test_audit_flags_unnecessary_agent_for_conceptual_only_request():
    knowledge = result("knowledge")
    sales = result("sales")
    audit = audit_instruction_fidelity(
        question="Qual é a fórmula da receita líquida?",
        answer=f"{knowledge.summary}\n{sales.summary}",
        selected_agents=["sales", "knowledge"],
        agent_results=[sales, knowledge],
        request_id="r1",
    )
    assert audit["status"] == "review_required"
    assert "clear_intent_routing" in audit["failed_checks"]


def test_audit_requires_evidence_and_sources_for_completed_results():
    sales = result("sales", evidence="", sources=[])
    audit = audit_instruction_fidelity(
        question="Quanto foi a receita líquida total?",
        answer=sales.summary,
        selected_agents=["sales"],
        agent_results=[sales],
        request_id="r1",
    )
    assert audit["status"] == "review_required"
    assert "completed_results_have_provenance" in audit["failed_checks"]


def test_audit_checks_export_approval_and_negative_instruction():
    base = dict(
        answer="Relatório pronto.\nResumo de knowledge.",
        selected_agents=["knowledge"],
        agent_results=[result("knowledge")],
        request_id="r1",
    )
    approved_request = audit_instruction_fidelity(
        question="Analise os indicadores e salve o relatório em arquivo.",
        approval_required=True,
        pending_action={"type": "save_report"},
        **base,
    )
    negative_request = audit_instruction_fidelity(
        question="Analise os indicadores, mas não salve o relatório.",
        approval_required=False,
        pending_action={},
        **base,
    )
    assert approved_request["status"] == "passed"
    assert negative_request["status"] == "passed"


def test_audit_ignores_previous_thread_results():
    current = result("sales", request_id="r-current")
    previous = result("knowledge", request_id="r-previous")
    audit = audit_instruction_fidelity(
        question="Quanto foi a receita líquida total?",
        answer=current.summary,
        selected_agents=["sales"],
        agent_results=[previous, current],
        request_id="r-current",
    )
    assert audit["status"] == "passed"
    assert audit["current_result_count"] == 1


def test_audit_flags_empty_answer_and_missing_summary():
    sales = result("sales")
    audit = audit_instruction_fidelity(
        question="Quanto foi a receita líquida total?",
        answer="",
        selected_agents=["sales"],
        agent_results=[sales],
        request_id="r1",
    )
    assert audit["status"] == "review_required"
    assert "answer_nonempty" in audit["failed_checks"]
    assert "specialist_summaries_preserved" in audit["failed_checks"]


def test_audit_does_not_pass_when_no_specialist_result_exists():
    audit = audit_instruction_fidelity(
        question="Explique este pedido.",
        answer="",
        selected_agents=["knowledge"],
        agent_results=[],
        request_id="r1",
    )
    assert audit["status"] == "review_required"
    assert "at_least_one_specialist_result" in audit["failed_checks"]


class FixedRouter:
    def invoke(self, messages):
        from bi_multiagent.schemas.contracts import SupervisorDecision
        return SupervisorDecision(
            agents=["sales", "knowledge"],
            rationale="Integration test: combined numeric and conceptual intent.",
        )


class FakeSpecialistGraph:
    def __init__(self, agent: str):
        self.agent = agent

    def invoke(self, state):
        source = "data/sample/sales.db" if self.agent == "sales" else f"{self.agent}-source.md"
        item = AgentResult(
            request_id=state["request_id"],
            agent=self.agent,
            status="completed",
            summary=f"Resumo de {self.agent}.",
            evidence=f"Evidência de {self.agent}.",
            sources=[source],
        )
        return {"result": item.model_dump(mode="json")}


def test_supervisor_integrates_instruction_fidelity_audit(monkeypatch):
    import bi_multiagent.coordination.supervisor_graph as module
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(module, "build_sales_analyst_graph", lambda: FakeSpecialistGraph("sales"))
    monkeypatch.setattr(module, "build_knowledge_analyst_graph", lambda: FakeSpecialistGraph("knowledge"))
    monkeypatch.setattr(module, "build_economic_analyst_graph", lambda: FakeSpecialistGraph("economic"))

    graph = module.build_multiagent_graph(router=FixedRouter(), checkpointer=InMemorySaver())
    result_state = graph.invoke(
        {"messages": [HumanMessage(content="Qual foi a receita líquida total e qual é a fórmula?")]},
        config={"configurable": {"thread_id": "instruction-fidelity-integration"}},
    )

    audit = result_state["instruction_fidelity"]
    assert audit["status"] == "passed"
    assert audit["expected_agents"] == ["sales", "knowledge"]
    assert "AUDITORIA DE FIDELIDADE" in result_state["answer"]
