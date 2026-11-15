from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from bi_multiagent.agents.specialized_agents import (
    build_economic_analyst_graph,
    build_knowledge_analyst_graph,
    build_sales_analyst_graph,
)
from bi_multiagent.coordination.supervisor_graph import (
    build_multiagent_graph,
    fallback_route,
)
from bi_multiagent.schemas.contracts import (
    AgentResult,
    AgentTask,
    SupervisorDecision,
)


class FakeRouter:
    """Substitui o LLM para testar a orquestração de forma determinística."""

    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.decision


def test_agent_task_contract():
    task = AgentTask(
        request_id="test-001",
        agent="sales",
        objective="Calcular receita líquida.",
        question="Qual a receita líquida?",
    )

    assert task.agent == "sales"
    assert task.request_id == "test-001"


def test_supervisor_rejects_duplicate_agents():
    try:
        SupervisorDecision(
            agents=["sales", "sales"],
            rationale="Teste",
        )
    except ValueError:
        return

    raise AssertionError("Agentes duplicados deveriam ser rejeitados.")


def test_fallback_routes_documented_definition_to_knowledge():
    route = fallback_route("Qual a diferença entre receita bruta e líquida?")

    assert route == ["knowledge"]


def test_fallback_routes_sales_question():
    route = fallback_route("Quanto foi a receita líquida total?")

    assert "sales" in route


def test_fallback_routes_selic_question():
    route = fallback_route("Qual foi a meta Selic entre 01/10/2026 e 11/11/2026?")

    assert route == ["economic"]


def test_sales_analyst_subgraph_executes_net_revenue_query():
    graph = build_sales_analyst_graph()

    result = graph.invoke(
        {
            "request_id": "sales-test",
            "question": "Qual foi a receita líquida total?",
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "sales"
    assert payload.status == "completed"
    assert "net_revenue" in payload.evidence
    assert "data/sample/sales.db" in payload.sources


def test_knowledge_analyst_subgraph_retrieves_sources():
    graph = build_knowledge_analyst_graph()

    result = graph.invoke(
        {
            "request_id": "knowledge-test",
            "question": ("Quais são os indicadores de vendas documentados?"),
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "knowledge"
    assert payload.status == "completed"
    assert "01_glossario_kpis.md" in payload.sources
    assert "Receita bruta" in payload.evidence


def test_economic_agent_requests_missing_period():
    graph = build_economic_analyst_graph()

    result = graph.invoke(
        {
            "request_id": "economic-test",
            "question": "Qual é a meta Selic?",
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "economic"
    assert payload.status == "needs_input"
    assert "datas inicial e final" in payload.summary


def test_supervisor_fans_out_to_two_agents_and_audits():
    router = FakeRouter(
        SupervisorDecision(
            agents=["sales", "knowledge"],
            rationale=(
                "A pergunta exige o cálculo comercial e "
                "a explicação da fórmula documentada."
            ),
        )
    )

    graph = build_multiagent_graph(
        router=router,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=("Qual a receita líquida total e como ela é calculada?")
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": "multiagent-fanout-test",
            }
        },
    )

    current_results = [
        item
        for item in result["agent_results"]
        if item["request_id"] == result["request_id"]
    ]

    assert router.calls == 1
    assert result["route"] == ["sales", "knowledge"]
    assert {item["agent"] for item in current_results} == {
        "sales",
        "knowledge",
    }
    assert result["audit"]["passed"] is True
    assert "net_revenue" in result["answer"]
    assert "01_glossario_kpis.md" in result["answer"]


def test_supervisor_uses_contract_and_blackboard():
    router = FakeRouter(
        SupervisorDecision(
            agents=["knowledge"],
            rationale="Pergunta conceitual.",
        )
    )

    graph = build_multiagent_graph(
        router=router,
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Quais indicadores estão documentados?")]},
        config={
            "configurable": {
                "thread_id": "blackboard-test",
            }
        },
    )

    current_artifacts = [
        item
        for item in result["blackboard"]
        if item.get("request_id") == result["request_id"]
    ]

    assert any(item["agent"] == "supervisor" for item in current_artifacts)
    assert any(item["agent"] == "knowledge" for item in current_artifacts)
    assert any(item["agent"] == "audit" for item in current_artifacts)


def test_multiagent_checkpointer_keeps_conversation_history():
    router = FakeRouter(
        SupervisorDecision(
            agents=["knowledge"],
            rationale="Teste de memória.",
        )
    )

    graph = build_multiagent_graph(
        router=router,
        checkpointer=InMemorySaver(),
    )

    config = {
        "configurable": {
            "thread_id": "multiagent-memory-test",
        }
    }

    first = graph.invoke(
        {"messages": [HumanMessage(content="Meu nome é Vítor.")]},
        config=config,
    )

    second = graph.invoke(
        {"messages": [HumanMessage(content="Qual é meu nome?")]},
        config=config,
    )

    assert any(
        getattr(message, "content", "") == "Meu nome é Vítor."
        for message in second["messages"]
    )
    assert first["request_id"] != second["request_id"]

    second_results = [
        item
        for item in second["agent_results"]
        if item["request_id"] == second["request_id"]
    ]

    assert all(item["request_id"] == second["request_id"] for item in second_results)
