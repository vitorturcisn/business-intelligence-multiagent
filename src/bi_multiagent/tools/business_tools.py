import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import datetime

import requests
from langchain_core.tools import tool

from bi_multiagent.settings import (
    DATABASE_PATH,
    KNOWLEDGE_BASE_DIR,
    MAX_SQL_ROWS,
)

DOCUMENT_ORDER = [
    "01_glossario_kpis.md",
    "02_dicionario_dados.md",
    "03_regras_negocio.md",
    "04_fontes_e_limitacoes.md",
]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _open_database() -> sqlite3.Connection:
    """Abre o banco SQLite em modo somente leitura."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Banco SQLite não encontrado: {DATABASE_PATH}")

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


@tool
def get_sales_schema() -> str:
    """Retorna o schema real do banco de vendas.

    Identifica tabelas, nomes de colunas, tipos e chaves primárias.
    Consulte esta ferramenta antes de construir SQL.
    discount_pct ? fra??o decimal: 0.10 representa 10%.
    """
    with closing(_open_database()) as connection:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        schema = {
            "database": DATABASE_PATH.name,
            "table_names": [],
            "tables": {},
            "column_semantics": {
                "discount_pct": {
                    "storage_format": "decimal_fraction",
                    "meaning": "Fra??o decimal aplicada ao valor bruto do item.",
                    "examples": {
                        "0.10": "10%",
                        "0.05": "5%",
                        "0.00": "0%",
                    },
                    "discount_formula": ("quantity * unit_price * discount_pct"),
                    "net_revenue_formula": (
                        "quantity * unit_price * (1 - discount_pct)"
                    ),
                    "warning": "Never divide discount_pct by 100.",
                }
            },
        }

        for table_row in tables:
            table_name = table_row["name"]
            safe_name = table_name.replace('"', '""')

            columns = connection.execute(f'PRAGMA table_info("{safe_name}")').fetchall()

            schema["table_names"].append(table_name)
            schema["tables"][table_name] = [
                {
                    "column": column["name"],
                    "type": column["type"],
                    "primary_key": bool(column["pk"]),
                }
                for column in columns
            ]

    return json.dumps(
        schema,
        ensure_ascii=False,
        indent=2,
    )


def _select_knowledge_documents(query: str) -> list[str]:
    """Seleciona documentos completos com base na intenção da consulta."""
    normalized = _normalize(query)
    selected = set()

    # Indicadores e glossário de negócio.
    if any(term in normalized for term in ("indicador", "kpi", "glossario")):
        selected.update(
            {
                "01_glossario_kpis.md",
                "03_regras_negocio.md",
                "04_fontes_e_limitacoes.md",
            }
        )

    # Receita, descontos e cálculos comerciais.
    if any(
        term in normalized
        for term in (
            "receita",
            "faturamento",
            "bruta",
            "liquida",
            "desconto",
        )
    ):
        selected.update(
            {
                "01_glossario_kpis.md",
                "02_dicionario_dados.md",
                "03_regras_negocio.md",
                "04_fontes_e_limitacoes.md",
            }
        )

    # Pedidos, unidades, produtos, categorias, colunas e tabelas.
    if any(
        term in normalized
        for term in (
            "pedido",
            "unidade",
            "produto",
            "categoria",
            "coluna",
            "tabela",
            "schema",
            "esquema",
            "venda",
            "sales",
        )
    ):
        selected.update(
            {
                "02_dicionario_dados.md",
                "03_regras_negocio.md",
                "04_fontes_e_limitacoes.md",
            }
        )

    # Fontes externas e Selic.
    if any(
        term in normalized for term in ("selic", "banco central", "fonte", "limitacao")
    ):
        selected.add("04_fontes_e_limitacoes.md")

    # Fallback lexical para perguntas fora das intenções conhecidas.
    if not selected:
        query_tokens = set(re.findall(r"[a-z0-9_]+", normalized))

        scores = []

        for filename in DOCUMENT_ORDER:
            path = KNOWLEDGE_BASE_DIR / filename

            if not path.is_file():
                continue

            content = _normalize(path.read_text(encoding="utf-8"))
            document_tokens = set(re.findall(r"[a-z0-9_]+", content))
            overlap = query_tokens.intersection(document_tokens)

            if overlap:
                scores.append((len(overlap), filename))

        scores.sort(reverse=True)
        selected.update(filename for _, filename in scores[:3])

    return [filename for filename in DOCUMENT_ORDER if filename in selected]


@tool
def search_business_knowledge(query: str) -> str:
    """Recupera documentos completos sobre KPIs e regras de negócio.

    Retorna conteúdo literal dos documentos Markdown, com identificação
    do arquivo de origem. Não apresenta definições que não estejam escritas
    na documentação.
    """
    if not query.strip():
        return "SEM EVIDÊNCIA: informe um assunto para pesquisar."

    if not KNOWLEDGE_BASE_DIR.exists():
        return "SEM EVIDÊNCIA: diretório documental não encontrado."

    selected = _select_knowledge_documents(query)

    if not selected:
        return (
            "SEM EVIDÊNCIA: nenhum documento correspondente foi encontrado. "
            "Não invente definições ou regras."
        )

    evidence = []

    for filename in selected:
        path = KNOWLEDGE_BASE_DIR / filename

        if not path.is_file():
            continue

        content = path.read_text(encoding="utf-8").strip()

        evidence.append(f"[DOCUMENTO: {filename}]\n{content}")

    if not evidence:
        return "SEM EVIDÊNCIA: nenhum documento pôde ser lido."

    return (
        "EVIDÊNCIAS DOCUMENTAIS RECUPERADAS.\n"
        "O conteúdo abaixo foi extraído dos arquivos originais.\n"
        "Preserve as definições, fórmulas e limitações descritas.\n\n"
        + "\n\n".join(evidence)
    )


@tool
def query_sales_database(sql: str) -> str:
    """Executa SQL SELECT somente leitura no banco de vendas.

    São permitidos SELECT e WITH ... SELECT.
    Não são permitidas alterações, múltiplas instruções ou DDL.
    Consulte get_sales_schema antes de criar a consulta.
    discount_pct is a decimal fraction: 0.10 represents 10%.
    Use quantity * unit_price * (1 - discount_pct) for net revenue.
    Never divide discount_pct by 100.
    """
    if not isinstance(sql, str) or not sql.strip():
        return "Erro: a consulta SQL está vazia."

    statement = sql.strip()

    if statement.endswith(";"):
        statement = statement[:-1].strip()

    if ";" in statement:
        return "Erro: múltiplas instruções SQL não são permitidas."

    if "--" in statement or "/*" in statement:
        return "Erro: comentários SQL não são permitidos."

    if not re.match(
        r"^(SELECT|WITH)\b",
        statement,
        flags=re.IGNORECASE,
    ):
        return "Erro: apenas consultas SELECT ou WITH são permitidas."

    forbidden = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|"
        r"ATTACH|DETACH|VACUUM|REINDEX|ANALYZE|PRAGMA)\b",
        flags=re.IGNORECASE,
    )

    if forbidden.search(statement):
        return "Erro: operação SQL não permitida."

    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }

    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed_actions.add(sqlite3.SQLITE_RECURSIVE)

    def read_only_authorizer(
        action: int,
        _arg1: str | None,
        _arg2: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    try:
        with closing(_open_database()) as connection:
            connection.set_authorizer(read_only_authorizer)
            cursor = connection.execute(statement)

            columns = [description[0] for description in (cursor.description or [])]

            rows = cursor.fetchmany(MAX_SQL_ROWS + 1)

        truncated = len(rows) > MAX_SQL_ROWS
        rows = rows[:MAX_SQL_ROWS]

        result = {
            "columns": columns,
            "rows": [dict(row) for row in rows],
            "returned_rows": len(rows),
            "truncated": truncated,
            "read_only": True,
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    except (sqlite3.Error, ValueError) as exc:
        return f"Erro na consulta somente leitura: {exc}"


@tool
def get_bcb_series(
    series_code: int,
    start_date: str,
    end_date: str,
) -> str:
    """Consulta a série 432 do Banco Central: meta Selic.

    Série 432: meta Selic (% ao ano).
    Datas no formato DD/MM/AAAA.

    Outras séries não estão habilitadas neste projeto porque não estão
    documentadas como fontes disponíveis.
    """
    if series_code != 432:
        return (
            "Série não habilitada. A única série documentada e disponível "
            "neste projeto é a 432 (meta Selic, % ao ano). "
            "Não substitua por outra série."
        )

    try:
        start = datetime.strptime(start_date, "%d/%m/%Y").date()
        end = datetime.strptime(end_date, "%d/%m/%Y").date()
    except ValueError:
        return "Erro: utilize datas válidas no formato DD/MM/AAAA."

    if start > end:
        return "Erro: a data inicial deve ser anterior à data final."

    if (end - start).days > 3652:
        return "Erro: limite a consulta a um intervalo de até dez anos."

    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados"

    params = {
        "formato": "json",
        "dataInicial": start.strftime("%d/%m/%Y"),
        "dataFinal": end.strftime("%d/%m/%Y"),
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=(5, 20),
        )
        response.raise_for_status()
        records = response.json()

        if not isinstance(records, list):
            return "Erro: formato inesperado retornado pelo Banco Central."

        result = {
            "source": "Banco Central do Brasil - SGS",
            "series_code": 432,
            "series_name": "Meta Selic",
            "unit": "% ao ano",
            "start_date": start_date,
            "end_date": end_date,
            "records_returned": len(records),
            "observations": records[-30:],
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

    except requests.RequestException as exc:
        return f"Erro na consulta ao Banco Central: {exc}"
    except ValueError as exc:
        return f"Erro ao interpretar a resposta: {exc}"
