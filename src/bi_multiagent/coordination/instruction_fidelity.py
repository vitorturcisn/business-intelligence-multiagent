"""Deterministic audit of instruction-to-execution fidelity for the Supervisor."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from bi_multiagent.schemas.contracts import AgentResult

_AGENT_ORDER = ("sales", "knowledge", "economic")


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _asks_for_report_export(question: str) -> bool:
    query = _normalize(question)
    tokens = set(re.findall(r"\w+", query))
    if any(
        first in tokens and second in tokens
        for first, second in (
            ("nao", "salve"),
            ("nao", "salvar"),
            ("nao", "exporte"),
            ("nao", "exportar"),
        )
    ):
        return False
    asks_report = bool(tokens.intersection({"relatorio", "arquivo"}))
    asks_save = bool(tokens.intersection({"salve", "salvar", "exporte", "exportar"}))
    return asks_report and asks_save


def infer_expected_agents(question: str) -> list[str]:
    """Infer only clear specialist requirements; return [] for ambiguous intent."""
    query = _normalize(question)
    expected: set[str] = set()

    sales_terms = (
        "receita", "faturamento", "venda", "vendas", "pedido", "pedidos",
        "unidade", "unidades", "quantidade", "produto", "categoria", "desconto",
        "sales", "revenue", "orders", "units sold",
    )
    conceptual_terms = (
        "definicao", "formula", "glossario", "documentado", "documentacao",
        "regra de negocio", "como calcula", "como calcular", "como e calculada",
        "como e calculado", "diferenca entre", "significado", "o que e",
        "explique", "explicacao", "documented", "definition", "formula",
    )
    numeric_terms = (
        "quanto", "quantos", "qual a receita", "qual o faturamento",
        "qual e a receita", "qual e o faturamento", "valor total", "receita total",
        "receita liquida total", "faturamento total", "total de vendas",
        "ranking", "top ", "mostre os valores", "mostre o ranking", "calcule",
        "compare", "unidades vendidas", "quantidade vendida", "numero de pedidos",
        "net revenue total", "total sales", "how many", "calculate", "rank",
    )

    has_sales = any(term in query for term in sales_terms)
    has_conceptual = any(term in query for term in conceptual_terms)
    has_numeric = any(term in query for term in numeric_terms)
    if "qual foi" in query and not has_conceptual:
        has_numeric = True
    if re.search(r"\bqual (?:e|foi) (?:a|o) (?:receita|faturamento|valor)\b", query):
        has_numeric = True

    if has_sales:
        if has_numeric:
            expected.add("sales")
        if has_conceptual or not has_numeric:
            expected.add("knowledge")

    if "selic" in query or "banco central" in query or "central bank" in query:
        asks_source_only = any(
            term in query
            for term in ("fonte", "documentado", "documentacao", "source", "documentation")
        ) and not has_numeric
        expected.add("knowledge" if asks_source_only else "economic")

    if not expected and any(
        term in query
        for term in ("indicadores documentados", "fontes do projeto", "glossario", "regras de negocio")
    ):
        expected.add("knowledge")

    return [agent for agent in _AGENT_ORDER if agent in expected]


def audit_instruction_fidelity(
    *,
    question: str,
    answer: str,
    selected_agents: Iterable[str],
    agent_results: Iterable[AgentResult | dict[str, Any]],
    request_id: str,
    approval_required: bool = False,
    pending_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit route, specialist execution, provenance, answer carry-through and HITL intent.

    The audit is a deterministic control check, not a factuality guarantee or
    calibrated quality score. Route requirements are enforced only when the
    question contains an unambiguous domain intent.
    """
    question_text = str(question or "")
    answer_text = str(answer or "")
    selected = list(dict.fromkeys(str(agent) for agent in selected_agents))
    expected = infer_expected_agents(question_text)

    normalized_results = [AgentResult.model_validate(item) for item in agent_results]
    current_results = [item for item in normalized_results if item.request_id == request_id]
    by_agent: dict[str, AgentResult] = {}
    duplicate_agents: list[str] = []
    for result in current_results:
        if result.agent in by_agent:
            duplicate_agents.append(result.agent)
        else:
            by_agent[result.agent] = result

    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        passed: bool,
        detail: str,
        *,
        applicable: bool = True,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "applicable": applicable,
                "passed": bool(passed) if applicable else None,
                "detail": detail,
            }
        )

    add_check(
        "answer_nonempty",
        bool(answer_text.strip()),
        "A síntese final deve conter uma resposta não vazia.",
    )
    add_check(
        "clear_intent_routing",
        set(selected) == set(expected),
        f"Esperados={expected}; selecionados={selected}.",
        applicable=bool(expected),
    )

    missing_results = [agent for agent in expected if agent not in by_agent]
    add_check(
        "required_specialists_returned",
        not missing_results,
        "Especialistas ausentes: " + (", ".join(missing_results) or "nenhum"),
        applicable=bool(expected),
    )

    failed_agents = [agent for agent, item in by_agent.items() if item.status == "failed"]
    add_check(
        "at_least_one_specialist_result",
        bool(current_results),
        "A solicitação deve produzir ao menos um resultado de especialista.",
    )
    add_check(
        "no_specialist_failure",
        not failed_agents,
        "Especialistas com falha: " + (", ".join(failed_agents) or "nenhum"),
    )

    incomplete_provenance = [
        item.agent
        for item in current_results
        if item.status == "completed" and (not item.evidence.strip() or not item.sources)
    ]
    add_check(
        "completed_results_have_provenance",
        not incomplete_provenance,
        "Resultados concluídos sem evidência ou fonte: "
        + (", ".join(incomplete_provenance) or "nenhum"),
    )

    summaries_missing = [
        item.agent
        for item in current_results
        if item.summary.strip() and item.summary.strip() not in answer_text
    ]
    add_check(
        "specialist_summaries_preserved",
        not summaries_missing,
        "Resumos não incorporados à síntese: "
        + (", ".join(summaries_missing) or "nenhum"),
        applicable=bool(current_results),
    )

    duplicate_agents = sorted(set(duplicate_agents))
    add_check(
        "no_duplicate_specialist_results",
        not duplicate_agents,
        "Resultados duplicados: " + (", ".join(duplicate_agents) or "nenhum"),
    )

    save_requested = _asks_for_report_export(question_text)
    action_type = str((pending_action or {}).get("type", ""))
    approval_policy_ok = (
        bool(approval_required) and action_type == "save_report"
        if save_requested
        else not bool(approval_required) and action_type != "save_report"
    )
    add_check(
        "explicit_export_requires_hitl",
        approval_policy_ok,
        (
            "Pedido explícito de exportação deve exigir aprovação humana."
            if save_requested
            else "Não deve existir exportação pendente sem pedido explícito."
        ),
    )

    applicable_checks = [item for item in checks if item["applicable"]]
    passed_count = sum(item["passed"] is True for item in applicable_checks)
    total_count = len(applicable_checks)
    score = round(passed_count / total_count, 4) if total_count else 1.0
    failures = [item["id"] for item in applicable_checks if item["passed"] is not True]
    status = "passed" if not failures else "review_required"

    return {
        "method": "instruction_fidelity_audit_v1",
        "status": status,
        "score": score,
        "passed_checks": passed_count,
        "total_checks": total_count,
        "intent_classification": "+".join(expected) if expected else "not_confidently_classified",
        "expected_agents": expected,
        "selected_agents": selected,
        "current_request_id": request_id,
        "current_result_count": len(current_results),
        "checks": checks,
        "failed_checks": failures,
    }
