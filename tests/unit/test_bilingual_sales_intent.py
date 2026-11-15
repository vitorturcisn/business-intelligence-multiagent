import pytest

from bi_multiagent.a2a.cards import build_agent_card
from bi_multiagent.a2a.registry import AGENT_SPECS
from bi_multiagent.agents.specialized_agents import build_sales_sql
from business_agent.agents.text_to_query import (
    build_query_prompt,
    is_dimensional_sales_query,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Calculate total net revenue.", "AS net_revenue"),
        ("Calculate total net sales.", "AS net_revenue"),
        ("Calculate gross revenue.", "AS gross_revenue"),
        ("Calculate gross sales.", "AS gross_revenue"),
        ("Calcule a receita líquida total.", "AS net_revenue"),
        ("Calcule a receita bruta total.", "AS gross_revenue"),
        ("How many orders are there?", "AS order_count"),
        ("Calculate units sold.", "AS units_sold"),
        ("Calculate total discounts.", "AS total_discounts"),
    ],
)
def test_deterministic_sales_sql_accepts_portuguese_and_english(
    question,
    expected,
):
    sql = build_sales_sql(question)

    assert sql is not None
    assert expected in sql
    assert "FROM sales" in sql


@pytest.mark.parametrize(
    "question",
    [
        "Rank net revenue by category.",
        "Compare sales by region.",
        "Top products by units sold.",
        "Ranking gross revenue by categories.",
        "Faça um ranking de receita líquida por categoria.",
        "Compare as vendas por região.",
        "Mostre os produtos mais vendidos.",
    ],
)
def test_text_to_query_recognizes_dimensional_questions_in_both_languages(
    question,
):
    assert is_dimensional_sales_query(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Explain the formula for net revenue by category.",
        "What is the difference between orders and units?",
        "Explique a fórmula de receita por categoria.",
    ],
)
def test_conceptual_questions_are_not_routed_to_text_to_query(question):
    assert is_dimensional_sales_query(question) is False


def test_query_prompt_documents_bilingual_catalog_and_sql_guardrail():
    prompt = build_query_prompt("Rank net revenue by category.")

    assert "Portuguese or English" in prompt
    assert "net_revenue" in prompt
    assert "category" in prompt
    assert "Never return SQL" in prompt
    assert "Rank net revenue by category." in prompt


def test_sales_agent_card_has_portuguese_and_english_examples():
    card = build_agent_card(
        "sales",
        f"http://127.0.0.1:{AGENT_SPECS['sales'].port}",
    )

    examples = list(card.skills[0].examples)

    assert "Calculate total net revenue." in examples
    assert "Calcule a receita líquida total." in examples
