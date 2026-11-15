import importlib
import json
import sqlite3
from collections import defaultdict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from bi_multiagent.agents.plan_execute_graph import build_plan_execute_graph
from bi_multiagent.agents.single_agent_graph import SYSTEM_PROMPT
from bi_multiagent.agents.specialized_agents import build_sales_sql
from bi_multiagent.settings import DATABASE_PATH
from bi_multiagent.tools.business_tools import (
    get_sales_schema,
    query_sales_database,
)
from business_agent.agents.text_to_query import build_query_prompt


class FixedModel:
    """Small deterministic model double for prompt-contract tests."""

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def _expected_net_revenue_from_database():
    """Calculate reference values from whichever SQLite seed is present."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            """
            SELECT category, quantity, unit_price, discount_pct
            FROM sales
            """
        ).fetchall()

    assert rows, f"A tabela sales está vazia: {DATABASE_PATH}"

    total = 0.0
    category_totals = defaultdict(float)

    for category, quantity, unit_price, discount_pct in rows:
        # discount_pct is stored as a decimal fraction: 0.10 means 10%.
        net_revenue = (
            float(quantity)
            * float(unit_price)
            * (1.0 - float(discount_pct))
        )
        total += net_revenue
        category_totals[category] += net_revenue

    return (
        round(total, 2),
        {category: round(value, 2) for category, value in category_totals.items()},
    )


def test_sales_schema_explains_discount_fraction():
    schema = json.loads(get_sales_schema.func())
    semantics = schema["column_semantics"]["discount_pct"]

    assert semantics["storage_format"] == "decimal_fraction"
    assert semantics["examples"]["0.10"] == "10%"
    assert "never divide discount_pct by 100" in semantics["warning"].lower()
    assert "1 - discount_pct" in semantics["net_revenue_formula"]


def test_react_system_prompt_explains_discount_fraction():
    prompt = SYSTEM_PROMPT.lower()

    assert "0.10 representa 10%" in prompt
    assert "nunca divida discount_pct por 100" in prompt
    assert "1 - discount_pct" in prompt


def test_text_to_query_prompt_explains_discount_fraction():
    prompt = build_query_prompt("Rank net revenue by category.")

    assert "0.10 represents 10%" in prompt
    assert "Never divide discount_pct by 100" in prompt
    assert "1 - discount_pct" in prompt


def test_query_tool_description_explains_discount_fraction():
    description = query_sales_database.description

    assert "0.10 represents 10%" in description
    assert "Never divide discount_pct by 100" in description


def test_deterministic_total_net_revenue_matches_database():
    sql = build_sales_sql("Calcule a receita líquida total.")

    assert sql is not None
    assert "1 - discount_pct" in sql
    assert "discount_pct / 100" not in sql

    result = json.loads(query_sales_database.func(sql))
    expected_total, _ = _expected_net_revenue_from_database()

    assert result["read_only"] is True
    # SQLite and Python may differ by one cent due to floating-point SUM order.
    assert result["rows"][0]["net_revenue"] == pytest.approx(
        expected_total, abs=0.02
    )


def test_deterministic_category_net_revenue_matches_database():
    sql = build_sales_sql("Calcule receita por categoria.")

    assert sql is not None
    assert "1 - discount_pct" in sql
    assert "discount_pct / 100" not in sql

    result = json.loads(query_sales_database.func(sql))
    values = {
        row["category"]: row["net_revenue"]
        for row in result["rows"]
    }
    _, expected_by_category = _expected_net_revenue_from_database()

    # SQLite's floating-point SUM can differ from Python's accumulation by
    # one cent for a category. The tolerance is limited to two cents.
    assert values == pytest.approx(expected_by_category, abs=0.02)


def test_plan_execute_planner_receives_discount_scale_rule(monkeypatch):
    # The graph has a documentary fast path that may skip the planner when
    # retrieved evidence already answers the question. Disable that branch
    # here so this test specifically inspects the prompt sent to the planner.
    plan_execute_module = importlib.import_module(
        "bi_multiagent.agents.plan_execute_graph"
    )
    monkeypatch.setattr(
        plan_execute_module,
        "_can_answer_from_evidence",
        lambda question, evidence: False,
    )

    planner = FixedModel('{"steps":[]}')
    synthesizer = FixedModel(
        AIMessage(content="No additional tool execution is required for this test.")
    )

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        checkpointer=InMemorySaver(),
    )

    graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Explain which tools are available in this BI system."
                )
            ]
        },
        config={"configurable": {"thread_id": "discount-fraction-prompt-test"}},
    )

    assert planner.prompts, "The planner was not invoked in this test."
    planner_prompt = planner.prompts[0]

    assert "query_sales_database" in planner_prompt
    assert "0.10 means 10%" in planner_prompt
    assert "Never divide discount_pct by 100" in planner_prompt
    assert "quantity * unit_price * (1 - discount_pct)" in planner_prompt
