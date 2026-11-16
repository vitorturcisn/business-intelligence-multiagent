from langchain_core.messages import AIMessage

from bi_multiagent.agents.specialized_agents import (
    build_knowledge_analyst_graph,
    build_sales_analyst_graph,
)
from bi_multiagent.schemas.contracts import AgentResult


class FakePlanner:
    """Planejador determinístico para testes do Text-to-Query."""

    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt: str):
        return AIMessage(content=self.content)


def test_sales_analyst_uses_validated_text_to_query_for_dimensions():
    planner = FakePlanner(
        '{"metric":"net_revenue","dimension":"category","sort":"desc","limit":5}'
    )

    graph = build_sales_analyst_graph(planner=planner)

    result = graph.invoke(
        {
            "request_id": "text-to-query-test",
            "question": "Mostre a receita líquida por categoria.",
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "sales"
    assert payload.status == "completed"
    assert "Text-to-Query" in payload.summary
    assert '"dimension": "category"' in payload.evidence
    assert "net_revenue" in payload.evidence
    assert "data/sample/sales.db" in payload.sources


def test_sales_analyst_rejects_invalid_text_to_query_plan():
    planner = FakePlanner(
        '{"metric":"drop_table","dimension":"category","sort":"desc","limit":5}'
    )

    graph = build_sales_analyst_graph(planner=planner)

    result = graph.invoke(
        {
            "request_id": "invalid-plan-test",
            "question": "Mostre a receita líquida por categoria.",
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "sales"
    assert payload.status == "failed"
    assert payload.warnings
    assert "data/sample/sales.db" in payload.sources


def test_knowledge_analyst_combines_faiss_and_documentary_retrieval(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_VECTOR_BACKEND", "faiss")
    def fake_retriever(_question: str, k: int = 4) -> str:
        assert k == 4
        return (
            "Resultado 1\n"
            "Fonte: data/knowledge_base/02_dicionario_dados.md\n"
            "Conteúdo: definição dos campos da tabela sales."
        )

    graph = build_knowledge_analyst_graph(retriever=fake_retriever)

    result = graph.invoke(
        {
            "request_id": "hybrid-retrieval-test",
            "question": ("Quais são os indicadores de vendas documentados?"),
        }
    )["result"]

    payload = AgentResult.model_validate(result)

    assert payload.agent == "knowledge"
    assert payload.status == "completed"
    assert "02_dicionario_dados.md" in payload.sources
    assert "01_glossario_kpis.md" in payload.sources
    assert "FAISS" in payload.evidence
    assert "Receita bruta" in payload.evidence
