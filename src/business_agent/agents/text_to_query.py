from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from business_agent.tools.sales_tool import get_sales_breakdown

DIMENSION_MARKERS = {
    "category": ("categoria", "categorias", "category", "categories"),
    "region": ("regiao", "regioes", "region", "regions"),
    "product": (
        "produto",
        "produtos",
        "item",
        "itens",
        "product",
        "products",
        "items",
    ),
}

SALES_MARKERS = (
    "receita",
    "faturamento",
    "desconto",
    "venda",
    "vendas",
    "pedido",
    "pedidos",
    "unidades",
    "quantidade",
    "mais vendidos",
    "revenue",
    "revenues",
    "sales",
    "sale",
    "discount",
    "discounts",
    "order",
    "orders",
    "units",
    "unit",
    "quantity",
    "best selling",
    "best-selling",
    "ranking",
    "rank",
    "top",
    "higher",
    "lower",
    "compare",
    "comparison",
)

CONCEPTUAL_MARKERS = (
    "o que e",
    "explique",
    "explicar",
    "diferenca entre",
    "como calcular",
    "formula",
    "definicao",
    "defina",
    "correto",
    "correta",
    "certo",
    "certa",
    "confere",
    "faz sentido",
    "verdadeiro",
    "verdadeira",
    "falso",
    "falsa",
    "incorreto",
    "incorreta",
    "what is",
    "explain",
    "explanation",
    "difference between",
    "how to calculate",
    "formula",
    "definition",
    "define",
    "does it make sense",
    "correct",
    "true",
    "false",
    "incorrect",
)

ALLOWED_METRICS = {
    "net_revenue",
    "gross_revenue",
    "total_discounts",
    "order_count",
    "units_sold",
}
ALLOWED_DIMENSIONS = {"category", "region", "product"}
ALLOWED_SORTS = {"asc", "desc"}

METRIC_LABELS = {
    "net_revenue": "receita líquida",
    "gross_revenue": "receita bruta",
    "total_discounts": "descontos totais",
    "order_count": "quantidade de pedidos",
    "units_sold": "unidades vendidas",
}
DIMENSION_LABELS = {
    "category": "categoria",
    "region": "região",
    "product": "produto",
}


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def _has_explicit_formula_expression(question: str) -> bool:
    """Recognize explicit equations over physical sales fields."""
    normalized = _normalize(question)
    has_sales_field = bool(
        re.search(
            r"(?<!\w)(?:quantity|unit[_ ]price|discount[_ ]pct)(?!\w)", normalized
        )
    )
    has_operator = bool(re.search(r"[Ã—*=]", question))
    return has_sales_field and has_operator


def is_dimensional_sales_query(question: str) -> bool:
    """Identifica perguntas analíticas que agrupam vendas por dimensão."""
    if not isinstance(question, str) or not question.strip():
        return False

    normalized = _normalize(question)
    if any(marker in normalized for marker in CONCEPTUAL_MARKERS):
        return False

    if _has_explicit_formula_expression(question):
        return False

    has_dimension = any(
        marker in normalized
        for aliases in DIMENSION_MARKERS.values()
        for marker in aliases
    )
    has_sales_intent = any(marker in normalized for marker in SALES_MARKERS)
    return has_dimension and has_sales_intent


def parse_query_plan(response: Any) -> dict[str, Any]:
    """Parseia e valida o JSON do plano; jamais aceita SQL produzido pelo modelo."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise TypeError("O planejador não retornou texto JSON.")

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("O planejador não retornou um objeto JSON válido.")

    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("O plano retornado não contém JSON válido.") from exc

    if not isinstance(plan, dict):
        raise TypeError("O plano precisa ser um objeto JSON.")

    required = {"metric", "dimension", "sort", "limit"}
    if set(plan) != required:
        raise ValueError(
            "O plano deve conter exatamente os campos: metric, dimension, sort e limit."
        )
    if plan["metric"] not in ALLOWED_METRICS:
        raise ValueError("Métrica não permitida no plano de consulta.")
    if plan["dimension"] not in ALLOWED_DIMENSIONS:
        raise ValueError("Dimensão não permitida no plano de consulta.")
    if plan["sort"] not in ALLOWED_SORTS:
        raise ValueError("Ordenação inválida no plano de consulta.")
    if isinstance(plan["limit"], bool) or not isinstance(plan["limit"], int):
        raise TypeError("O limite precisa ser um número inteiro.")
    if not 1 <= plan["limit"] <= 10:
        raise ValueError("O limite deve estar entre 1 e 10.")

    return plan


def _format_brl(value: Any) -> str:
    formatted = f"{float(value):,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def build_query_answer(
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    period: str | None,
) -> str:
    if not rows:
        return "A consulta foi executada, mas não encontrou registros para os critérios informados."

    period_label = (
        f"no período {period}" if period else "em todo o período disponível na base"
    )
    metric = plan["metric"]
    dimension = DIMENSION_LABELS[plan["dimension"]]
    label = METRIC_LABELS[metric]

    parts = []
    for row in rows:
        value = row["value"]
        if metric in {"net_revenue", "gross_revenue", "total_discounts"}:
            rendered = _format_brl(value)
        elif metric == "order_count":
            rendered = f"{int(value)} pedidos"
        else:
            rendered = f"{int(value)} unidades"
        parts.append(f"{row['dimension']}: {rendered}")

    return (
        f"Resultado de {label} por {dimension} {period_label}: "
        + "; ".join(parts)
        + "."
    )


def build_query_prompt(question: str) -> str:
    return f"""Interpret the user's analytical sales question in Portuguese or English.

Do not write SQL. Do not invent tables or columns.
Return only one valid JSON object, without Markdown or explanations,
with exactly these keys:
{{"metric":"...","dimension":"...","sort":"desc|asc","limit":5}}

Allowed values for metric:
- net_revenue: receita líquida, faturamento líquido, net revenue or net sales
- gross_revenue: receita bruta, faturamento bruto, gross revenue or gross sales
- total_discounts: descontos, discounts
- order_count: quantidade de pedidos, number of orders, order count
- units_sold: unidades vendidas, quantidade vendida, units sold, quantity sold

Allowed values for dimension:
- category: categoria, categorias, category, categories
- region: região, regiões, region, regions
- product: produto, produtos, product, products, item, items

Rules:
- Interpret Portuguese and English synonyms equivalently.
- discount_pct is stored as a decimal fraction, not percentage points.
- 0.10 represents 10% and 0.05 represents 5%.
- For net_revenue use quantity * unit_price * (1 - discount_pct).
- Never divide discount_pct by 100.
- For "more", "higher", "top", "best" or ranking requests, use sort="desc".
- For "less" or "lower" ranking requests, use sort="asc".
- For best-selling products without another metric, use units_sold.
- If a ranking by category or region has no explicit metric, use net_revenue.
- Use limit=5 by default; respect a requested number between 1 and 10.
- Do not include filters, tables, fields or additional keys.
- Never return SQL.

Question: {question}
"""


def execute_text_to_query(
    question: str,
    model: Any,
    period: str | None = None,
) -> dict[str, Any]:
    """Gera um plano estruturado, valida-o, compila SQL seguro e executa a consulta."""
    raw_response = model.invoke(build_query_prompt(question))
    plan = parse_query_plan(raw_response)
    rows = get_sales_breakdown(
        metric=plan["metric"],
        dimension=plan["dimension"],
        month=period,
        sort=plan["sort"],
        limit=plan["limit"],
    )
    return {
        "plan": plan,
        "period": period,
        "rows": rows,
        "answer": build_query_answer(plan, rows, period),
    }
