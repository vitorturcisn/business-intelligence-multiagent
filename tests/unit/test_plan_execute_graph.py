import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from bi_multiagent.agents import plan_execute_graph as plan_module
from bi_multiagent.agents.plan_execute_graph import (
    _can_answer_from_evidence,
    _fallback_sales_sql,
    _invoke_json_object,
    _requires_sales_query,
    build_plan_execute_graph,
)


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def invoke(self, prompt):
        self.inputs.append(prompt)
        if not self.responses:
            raise AssertionError("FakeModel has no remaining responses.")
        return self.responses.pop(0)


class FakeTool:
    name = "lookup"
    description = "Returns deterministic test evidence."

    def __init__(self):
        self.calls = []

    def invoke(self, arguments):
        self.calls.append(arguments)
        return {"evidence": f"Found: {arguments['query']}"}


def test_plan_execute_runs_tool_replans_and_synthesizes():
    planner = FakeModel(
        [
            json.dumps(
                {
                    "steps": [
                        {
                            "tool": "lookup",
                            "args": {"query": "business definition"},
                            "reason": "Retrieve the business definition.",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "decision": "continue",
                    "steps": [
                        {
                            "tool": "lookup",
                            "args": {"query": "calculation formula"},
                            "reason": "Retrieve the calculation formula.",
                        }
                    ],
                    "reason": "The formula is still missing.",
                }
            ),
            json.dumps(
                {
                    "decision": "final",
                    "steps": [],
                    "reason": "The evidence is sufficient.",
                }
            ),
        ]
    )

    synthesizer = FakeModel(
        [AIMessage(content="The answer is supported by both observations.")]
    )
    tool = FakeTool()

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        tools=[tool],
        checkpointer=InMemorySaver(),
        max_replans=2,
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Run the planned lookup operation.")]},
        config={"configurable": {"thread_id": "plan-execute-test"}},
    )

    assert len(tool.calls) == 2
    assert result["final_answer"] == ("The answer is supported by both observations.")
    assert len(result["observations"]) == 2
    assert result["replan_count"] == 1
    assert result["decision"] == "final"


def test_plan_execute_rejects_tool_outside_allowlist():
    planner = FakeModel(
        [
            json.dumps(
                {
                    "steps": [
                        {
                            "tool": "run_arbitrary_sql",
                            "args": {},
                            "reason": "This tool is not allowed.",
                        }
                    ]
                }
            )
        ]
    )

    synthesizer = FakeModel([AIMessage(content="Should not be called.")])

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        tools=[FakeTool()],
        checkpointer=InMemorySaver(),
    )

    with pytest.raises(ValueError, match="not part of the allowed tool catalog"):
        graph.invoke(
            {"messages": [HumanMessage(content="Run a safe lookup operation.")]},
            config={"configurable": {"thread_id": "plan-guardrail-test"}},
        )


def test_plan_execute_can_answer_without_tool_calls():
    planner = FakeModel([json.dumps({"steps": []})])
    synthesizer = FakeModel([AIMessage(content="No external evidence was required.")])

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        tools=[FakeTool()],
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content="Explain the meaning of this workflow.")]},
        config={"configurable": {"thread_id": "plan-no-tools-test"}},
    )

    assert result["observations"] == []
    assert result["final_answer"] == "No external evidence was required."


def test_formula_question_can_be_answered_from_retrieved_documents():
    question = "Qual \u00e9 a f\u00f3rmula de receita l\u00edquida?"
    evidence = (
        "[EVID\u00caNCIAS DOCUMENTAIS] "
        "Receita bruta por item = quantidade * unit_price. "
        "Desconto por item = receita bruta * discount_pct. "
        "Receita l\u00edquida = receita bruta - descontos."
    )

    assert _can_answer_from_evidence(question, evidence)


def test_formula_fast_path_skips_planner(monkeypatch):
    evidence = (
        "[EVID\u00caNCIAS DOCUMENTAIS] "
        "Receita bruta por item = quantidade * unit_price. "
        "Desconto por item = receita bruta * discount_pct. "
        "Receita l\u00edquida = receita bruta - descontos."
    )

    monkeypatch.setattr(
        plan_module,
        "retrieve_context",
        lambda state: {"evidence": evidence},
    )

    planner = FakeModel([])
    synthesizer = FakeModel(
        [
            AIMessage(
                content=("Receita l\u00edquida \u00e9 receita bruta menos descontos.")
            )
        ]
    )

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        tools=[FakeTool()],
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Qual \u00e9 a f\u00f3rmula de receita l\u00edquida?"
                )
            ]
        },
        config={"configurable": {"thread_id": "documentary-fast-path-test"}},
    )

    assert planner.inputs == []
    assert result["observations"] == []
    assert "receita bruta menos descontos" in result["final_answer"]


def test_planner_json_parser_retries_once_after_invalid_output():
    model = FakeModel(
        [
            "I cannot produce the requested format.",
            '{"steps":[]}',
        ]
    )

    result = _invoke_json_object(
        model,
        "Return JSON with an empty steps list.",
    )

    assert result == {"steps": []}
    assert len(model.inputs) == 2


def test_total_net_revenue_is_classified_as_sql_request():
    question = "Qual \u00e9 a receita l\u00edquida total de todas as vendas?"

    assert _requires_sales_query(question)
    assert not _requires_sales_query(
        "Qual \u00e9 a f\u00f3rmula da receita l\u00edquida?"
    )

    assert _fallback_sales_sql(question) == (
        "SELECT COALESCE("
        "SUM(quantity * unit_price * (1 - discount_pct)), 0"
        ") AS net_revenue FROM sales"
    )


def test_empty_planner_plan_forces_total_net_revenue_sql(monkeypatch):
    question = "Qual \u00e9 a receita l\u00edquida total de todas as vendas?"

    monkeypatch.setattr(
        plan_module,
        "retrieve_context",
        lambda state: {
            "evidence": (
                "[SCHEMA DO BANCO SQLITE] "
                "table sales: quantity, unit_price, discount_pct. "
                "[EVID\u00caNCIAS DOCUMENTAIS] "
                "Receita l\u00edquida por item usa "
                "quantity * unit_price * (1 - discount_pct)."
            )
        },
    )

    class FakeSalesQueryTool:
        name = "query_sales_database"
        description = "Executes read-only sales SQL."

        def __init__(self):
            self.args = {"sql": {"type": "string"}}
            self.calls = []

        def invoke(self, arguments):
            self.calls.append(arguments)
            return (
                '{"columns":["net_revenue"],'
                '"rows":[{"net_revenue":43270.0}],'
                '"returned_rows":1,"truncated":false,"read_only":true}'
            )

    sales_tool = FakeSalesQueryTool()
    planner = FakeModel([])
    synthesizer = FakeModel([])

    graph = build_plan_execute_graph(
        planner=planner,
        synthesizer=synthesizer,
        tools=[sales_tool],
        checkpointer=InMemorySaver(),
    )

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": "total-net-revenue-deterministic-test"}},
    )

    expected_sql = _fallback_sales_sql(question)

    assert sales_tool.calls == [{"sql": expected_sql}]
    assert planner.inputs == []
    assert synthesizer.inputs == []

    assert len(result["observations"]) == 1
    assert result["observations"][0]["tool"] == "query_sales_database"
    assert result["observations"][0]["status"] == "completed"

    assert "R$ 43.270,00" in result["final_answer"]
