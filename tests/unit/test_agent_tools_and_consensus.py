from __future__ import annotations

from bi_multiagent.coordination.agent_tools import AgentTool
from bi_multiagent.coordination.weighted_consensus import (
    weighted_evidence_consensus,
)
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import bi_multiagent.coordination.supervisor_graph as supervisor_module
from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph
from bi_multiagent.schemas.contracts import AgentResult, AgentTask, SupervisorDecision


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.received = None

    def invoke(self, state):
        self.received = state
        return {"result": self.result.model_dump(mode="json")}


def make_result(
    agent: str,
    *,
    status: str = "completed",
    evidence: str = "evidence from the specialist",
    sources: list[str] | None = None,
    warnings: list[str] | None = None,
    request_id: str = "request-1",
) -> AgentResult:
    return AgentResult(
        request_id=request_id,
        agent=agent,
        status=status,
        summary=f"{agent} summary",
        evidence=evidence,
        sources=sources if sources is not None else [f"{agent}-source.md"],
        warnings=warnings or [],
    )


def test_agent_tool_exposes_structured_tool_and_invokes_subgraph():
    expected = make_result("knowledge")
    graph = FakeGraph(expected)
    agent_tool = AgentTool(
        agent="knowledge",
        objective="Retrieve documented business definitions.",
        graph=graph,
    )
    task = AgentTask(
        request_id="request-1",
        agent="knowledge",
        objective="Retrieve documented business definitions.",
        question="What is net revenue?",
    )

    actual = agent_tool.invoke(task)

    assert agent_tool.tool.name == "knowledge_agent_tool"
    assert actual == expected
    assert graph.received["request_id"] == "request-1"
    assert graph.received["question"] == "What is net revenue?"
    assert graph.received["task"]["agent"] == "knowledge"


def test_agent_tool_rejects_task_for_another_agent():
    graph = FakeGraph(make_result("knowledge"))
    agent_tool = AgentTool(
        agent="knowledge",
        objective="Retrieve business knowledge.",
        graph=graph,
    )
    task = AgentTask(
        request_id="request-1",
        agent="sales",
        objective="Calculate sales metrics.",
        question="What was net revenue?",
    )

    try:
        agent_tool.invoke(task)
    except ValueError as exc:
        assert "cannot execute task" in str(exc)
    else:
        raise AssertionError("A specialist tool must reject tasks for another agent.")


def test_weighted_consensus_accepts_complete_sourced_results():
    result = weighted_evidence_consensus(
        [make_result("sales"), make_result("knowledge")],
        ["sales", "knowledge"],
    )

    assert result["status"] == "accepted"
    assert result["evidence_support_score"] == 1.0
    assert result["missing_agents"] == []


def test_weighted_consensus_penalizes_missing_specialist():
    result = weighted_evidence_consensus(
        [make_result("sales")],
        ["sales", "knowledge"],
    )

    assert result["status"] == "review_required"
    assert result["missing_agents"] == ["knowledge"]
    assert result["evidence_support_score"] < result["acceptance_threshold"]


def test_weighted_consensus_requires_input_when_specialist_requests_it():
    result = weighted_evidence_consensus(
        [
            make_result(
                "economic",
                status="needs_input",
                evidence="",
                sources=["04_fontes_e_limitacoes.md"],
            )
        ],
        ["economic"],
    )

    assert result["status"] == "needs_input"


def test_weighted_consensus_rejects_failed_or_unsourced_evidence():
    failed = weighted_evidence_consensus(
        [make_result("sales", status="failed", evidence="", sources=[])],
        ["sales"],
    )
    unsourced = weighted_evidence_consensus(
        [make_result("sales", evidence="evidence", sources=[])],
        ["sales"],
    )

    assert failed["status"] == "insufficient_evidence"
    assert unsourced["status"] == "review_required"


def test_weighted_consensus_rejects_duplicate_results():
    result = weighted_evidence_consensus(
        [make_result("knowledge"), make_result("knowledge")],
        ["knowledge"],
    )

    assert result["status"] == "review_required"
    assert any("Duplicate" in reason for reason in result["reasons"])


class FakeSpecialistGraph:
    def __init__(self, agent: str):
        self.agent = agent

    def invoke(self, state):
        source = (
            "data/sample/sales.db"
            if self.agent == "sales"
            else f"{self.agent}-source.md"
        )
        result = AgentResult(
            request_id=state["request_id"],
            agent=self.agent,
            status="completed",
            summary=f"{self.agent} produced a test result.",
            evidence=f"Evidence from {self.agent}.",
            sources=[source],
        )
        return {"result": result.model_dump(mode="json")}


class FixedRouter:
    def invoke(self, messages):
        return SupervisorDecision(
            agents=["sales", "knowledge"],
            rationale="Integration test for tools and weighted evidence consensus.",
        )


def test_supervisor_uses_agent_tools_and_weighted_consensus(monkeypatch):
    monkeypatch.setattr(
        supervisor_module, "build_sales_analyst_graph",
        lambda: FakeSpecialistGraph("sales"),
    )
    monkeypatch.setattr(
        supervisor_module, "build_knowledge_analyst_graph",
        lambda: FakeSpecialistGraph("knowledge"),
    )
    monkeypatch.setattr(
        supervisor_module, "build_economic_analyst_graph",
        lambda: FakeSpecialistGraph("economic"),
    )

    graph = build_multiagent_graph(
        router=FixedRouter(),
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Quanto foi a receita líquida total e qual é a fórmula?"
                    )
                )
            ]
        },
        config={"configurable": {"thread_id": "agent-tools-consensus-test"}},
    )

    current_artifacts = [
        item for item in result["blackboard"]
        if item.get("request_id") == result["request_id"]
    ]
    tool_artifacts = [
        item for item in current_artifacts
        if item.get("artifact_type") == "agent_result"
    ]

    assert result["audit"]["passed"] is True
    assert result["weighted_consensus"]["status"] == "accepted"
    assert result["weighted_consensus"]["evidence_support_score"] == 1.0
    assert {item["tool_name"] for item in tool_artifacts} == {
        "sales_agent_tool", "knowledge_agent_tool"
    }
    assert all(
        item["execution_mode"] == "langchain_structured_tool"
        for item in tool_artifacts
    )
