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

def test_policy_recognizes_question_form_numeric_sales_metric():
    question = "Qual é a receita líquida?"
    assert enforce_routing_policy(question, ["sales", "knowledge"]) == ["sales"]


def test_policy_removes_knowledge_for_numeric_sales_only():
    question = "Qual é a receita líquida total?"
    assert enforce_routing_policy(question, ["sales", "knowledge"]) == ["sales"]


def test_policy_defaults_unqualified_sales_question_to_knowledge():
    question = "Fale sobre as vendas."
    assert enforce_routing_policy(question, ["sales", "knowledge"]) == ["knowledge"]


def test_policy_removes_sales_for_conceptual_only_even_if_model_adds_extras():
    question = "Qual é a fórmula documentada da receita líquida?"
    assert enforce_routing_policy(
        question, ["sales", "knowledge", "economic"]
    ) == ["knowledge"]


def test_policy_keeps_both_agents_for_numeric_and_conceptual_sales():
    question = "Qual foi a receita líquida total e como ela é calculada?"
    assert enforce_routing_policy(question, ["sales"]) == ["sales", "knowledge"]


def test_policy_removes_knowledge_for_historical_selic_value():
    question = "Qual foi a meta Selic entre 01/10/2026 e 10/11/2026?"
    assert enforce_routing_policy(question, ["economic", "knowledge"]) == ["economic"]


def test_policy_routes_selic_source_documentation_to_knowledge():
    question = "Qual é a fonte documentada da série Selic?"
    assert enforce_routing_policy(question, ["economic", "knowledge"]) == ["knowledge"]
