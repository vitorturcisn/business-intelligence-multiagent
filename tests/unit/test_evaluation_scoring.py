from bi_multiagent.agents.single_agent_graph import SYSTEM_PROMPT
from scripts.evaluate_reasoning_architectures import (
    number_is_present,
    response_fidelity,
    trajectory_completeness,
)


def test_number_parser_accepts_brazilian_currency():
    assert number_is_present("R$ 31.255,00", 31255.0)
    assert number_is_present("R$ 7.650,00", 7650.0)
    assert number_is_present("R$ 4.365,00", 4365.0)


def test_number_parser_handles_previous_malformed_output():
    assert number_is_present("R$ 31.255.00", 31255.0)
    assert number_is_present("R$ 7.650.00", 7650.0)
    assert number_is_present("R$ 4.365.00", 4365.0)


def test_number_parser_does_not_accept_wrong_value():
    assert not number_is_present("R$ 31.256,00", 31255.0)


def test_react_prompt_requires_brazilian_currency_format():
    assert "FORMATACAO MONETARIA BRASILEIRA OBRIGATORIA" in SYSTEM_PROMPT
    assert "R$ 31.255,00" in SYSTEM_PROMPT


def test_complete_sales_trajectory_scores_one():
    result = trajectory_completeness(
        inputs={"case_id": "sales_by_category"},
        outputs={
            "answer": "Ranking de receita por categoria.",
            "trajectory": {
                "tool_steps": [{"tool": "query_sales_database", "status": "success"}],
                "tool_names": ["query_sales_database"],
                "node_sequence": [
                    "retrieve_context",
                    "agent",
                    "ToolNode",
                    "agent",
                    "END",
                ],
            },
            "metrics": {
                "failed_step_count": 0,
                "retrieval_context_chars": 5997,
            },
        },
        reference_outputs={
            "required_tool": "query_sales_database",
            "requires_context": False,
        },
    )

    assert result["score"] == 1.0
    assert "checks=5/5" in result["comment"]


def test_incomplete_trajectory_exposes_failed_check():
    result = trajectory_completeness(
        inputs={"case_id": "sales_by_category"},
        outputs={
            "answer": "Ranking de receita por categoria.",
            "trajectory": {
                "tool_steps": [{"tool": "query_sales_database", "status": "success"}],
                "tool_names": ["query_sales_database"],
                "node_sequence": ["retrieve_context", "agent"],
            },
            "metrics": {"failed_step_count": 0},
        },
        reference_outputs={
            "required_tool": "query_sales_database",
            "requires_context": False,
        },
    )

    assert result["score"] == 0.8
    assert "ends_at_END': False" in result["comment"]


def test_response_fidelity_accepts_singular_discount_for_plural_reference():
    answer = (
        "A f?rmula de receita l?quida ? calculada como: "
        "receita bruta por item = quantidade vezes pre?o unit?rio. "
        "Desconto por item = receita bruta vezes desconto_pct. "
        "Receita l?quida por item = receita bruta menos desconto. "
        "A f?rmula ? quantity * unit_price * (1 - discount_pct)."
    )

    result = response_fidelity(
        inputs={"case_id": "business_formula"},
        outputs={"answer": answer},
        reference_outputs={
            "expected_terms": ["receita bruta", "descontos"],
            "expected_numbers": [],
        },
    )

    assert result["score"] == 1.0
    assert "missing=[]" in result["comment"]
