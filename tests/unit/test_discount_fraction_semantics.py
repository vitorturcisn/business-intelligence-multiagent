import json

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from bi_multiagent.agents.plan_execute_graph import build_plan_execute_graph
from bi_multiagent.agents.single_agent_graph import SYSTEM_PROMPT
from bi_multiagent.agents.specialized_agents import build_sales_sql
from bi_multiagent.tools.business_tools import (
    get_sales_schema,
    query_sales_database,
)
from business_agent.agents.text_to_query import build_query_prompt


class FixedModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_sales_schema_explains_discount_fraction():
    schema = json.loads(get_sales_schema.func())
    semantics = schema["column_semantics"]["discount_pct"]

    assert semantics["storage_format"] == "decimal_fraction"
    assert semantics["examples"]["0.10"] == "10%"
    assert "Never divide" in semantics["warning"]
    assert "1 - discount_pct" in semantics["net_revenue_formula"]


def test_react_prompt_explains_discount_fraction():
    prompt = SYSTEM_PROMPT.lower()

    assert "0.10 representa 10%" in prompt
    assert "nunca divida discount_pct por 100" in prompt
    assert "1 - discount_pct" in prompt


def test_sql_tool_description_explains_discount_fraction():
    description = query_sales_database.description

    assert "0.10 represents 10%" in description
    assert "Never divide discount_pct by 100" in description


def test_text_to_query_prompt_explains_discount_fraction():
    prompt = build_query_prompt("Rank net revenue by category.")

    assert "0.10 represents 10%" in prompt
    assert "Never divide discount_pct by 100" in prompt
    assert "1 - discount_pct" in prompt


def test_deterministic_total_net_revenue_matches_database():
    sql = build_sales_sql("Calcule a receita l?quida total.")

    assert sql is not None
    assert "1 - discount_pct" in sql
    assert "discount_pct / 100" not in sql

    result = json.loads(query_sales_database.func(sql))

    assert result["read_only"] is True
    assert result["rows"][0]["net_revenue"] == 43270.0


def test_deterministic_category_net_revenue_matches_database():
    sql = build_sales_sql("Calcule receita por categoria.")

    assert sql is not None
    assert "1 - discount_pct" in sql
    assert "discount_pct / 100" not in sql

    result = json.loads(query_sales_database.func(sql))

    values = {row["category"]: row["net_revenue"] for row in result["rows"]}

    assert values == {
        "Eletr\u00f4nicos": 31255.0,
        "M\u00f3veis": 7650.0,
        "Acess\u00f3rios": 4365.0,
    }


def test_plan_execute_planner_receives_discount_scale_rule():
    planner = FixedModel('{"steps":[]}')
    synthesizer = FixedModel(AIMessage(content="No external evidence was required."))

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        checkpointer=InMemorySaver(),
    )

    graph.invoke(
        {"messages": [HumanMessage(content="What is total net revenue?")]},
        config={"configurable": {"thread_id": "discount-fraction-prompt-test"}},
    )

    planner_prompt = planner.prompts[0]

    assert "query_sales_database" in planner_prompt
    assert "0.10 means 10%" in planner_prompt
    assert "Never divide discount_pct by 100" in planner_prompt
    assert "Never write SQL." not in planner_prompt
