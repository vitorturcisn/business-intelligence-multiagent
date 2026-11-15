from bi_multiagent.coordination.supervisor_graph import (
    enforce_routing_policy,
    fallback_route,
)


def test_formula_question_routes_only_to_knowledge():
    question = "Qual e a formula documentada para calcular a receita liquida?"
    assert fallback_route(question) == ["knowledge"]


def test_definition_difference_routes_only_to_knowledge():
    question = "Qual a diferenca entre receita bruta e receita liquida?"
    assert fallback_route(question) == ["knowledge"]


def test_mixed_question_routes_two_agents():
    question = "Qual foi a receita liquida total e como ela e calculada?"
    assert fallback_route(question) == ["sales", "knowledge"]


def test_policy_guard_removes_sales_from_conceptual_question():
    question = "Qual e a formula documentada para calcular a receita liquida?"
    assert enforce_routing_policy(question, ["sales"]) == ["knowledge"]


def test_policy_guard_adds_knowledge_to_mixed_question():
    question = "Qual foi a receita liquida total e como ela e calculada?"
    assert enforce_routing_policy(question, ["sales"]) == [
        "sales",
        "knowledge",
    ]


def test_ranking_question_stays_with_sales():
    question = "Mostre o ranking de receita liquida por categoria."
    assert fallback_route(question) == ["sales"]
