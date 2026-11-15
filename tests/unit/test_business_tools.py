import json

import pytest

from bi_multiagent.tools.business_tools import (
    get_bcb_series,
    get_sales_schema,
    query_sales_database,
    search_business_knowledge,
)


def test_sales_schema_identifies_real_table():
    result = json.loads(get_sales_schema.func())

    assert result["database"] == "sales.db"
    assert result["table_names"] == ["sales"]

    columns = [column["column"] for column in result["tables"]["sales"]]

    assert "sale_id" in columns
    assert "unit_price" in columns


def test_read_only_query_executes_select():
    result = query_sales_database.func("SELECT 1 AS health_check")
    payload = json.loads(result)

    assert payload["read_only"] is True
    assert payload["rows"][0]["health_check"] == 1


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET product = 'x'",
        "SELECT 1; SELECT 2",
        "PRAGMA table_info(sales)",
    ],
)
def test_read_only_query_blocks_unsafe_sql(sql):
    result = query_sales_database.func(sql)

    assert result.startswith("Erro:")


def test_indicator_search_returns_only_documented_catalog():
    result = search_business_knowledge.func(
        "Quais são os principais indicadores de negócio?"
    )

    assert "01_glossario_kpis.md" in result
    assert "Receita bruta" in result
    assert "Receita líquida" in result
    assert "Quantidade vendida" in result
    assert "Número de pedidos" in result
    assert "Receita por categoria" in result
    assert "CAC" not in result
    assert "Lucro bruto" not in result


def test_revenue_search_returns_documented_formulas():
    result = search_business_knowledge.func(
        "Diferença entre receita bruta, descontos e receita líquida"
    )

    assert "01_glossario_kpis.md" in result
    assert "02_dicionario_dados.md" in result
    assert "quantidade × preço unitário" in result
    assert "receita bruta − descontos" in result
    assert "1 - discount_pct" in result


def test_selic_documentation_is_retrieved():
    result = search_business_knowledge.func("fonte da Selic")

    assert "04_fontes_e_limitacoes.md" in result
    assert "série 432" in result
    assert "Banco Central" in result


def test_bcb_tool_validates_date_format():
    result = get_bcb_series.func(
        432,
        "2026-01-01",
        "2026-02-01",
    )

    assert "DD/MM/AAAA" in result


def test_bcb_tool_rejects_reversed_dates():
    result = get_bcb_series.func(
        432,
        "10/02/2026",
        "01/02/2026",
    )

    assert "data inicial" in result.lower()


def test_bcb_tool_rejects_undocumented_series():
    result = get_bcb_series.func(
        433,
        "01/01/2026",
        "31/01/2026",
    )

    assert "única série documentada" in result
    assert "432" in result
