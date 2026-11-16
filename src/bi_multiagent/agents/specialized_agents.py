import json
import os
import re
import unicodedata
from datetime import datetime
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from bi_multiagent.schemas.contracts import AgentResult
from bi_multiagent.schemas.multiagent_state import WorkerState
from bi_multiagent.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from bi_multiagent.tools.business_tools import (
    get_bcb_series,
    get_sales_schema,
    query_sales_database,
    search_business_knowledge,
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


# ------------------------------------------------------------
# SALES ANALYST
# ------------------------------------------------------------


def build_sales_sql(question: str) -> str | None:
    """Converte intenções comerciais em templates SQL seguros.

    Reconhece perguntas em português e inglês. O SQL é sempre
    construído por templates fixos, nunca pelo modelo.
    """
    query = normalize_text(question)

    if any(
        term in query
        for term in (
            "receita por categoria",
            "revenue by category",
            "sales by category",
        )
    ):
        return """
            SELECT
                category,
                ROUND(SUM(quantity * unit_price), 2) AS gross_revenue,
                ROUND(
                    SUM(quantity * unit_price * discount_pct),
                    2
                ) AS discounts,
                ROUND(
                    SUM(quantity * unit_price * (1 - discount_pct)),
                    2
                ) AS net_revenue
            FROM sales
            GROUP BY category
            ORDER BY net_revenue DESC
        """

    if any(
        term in query
        for term in (
            "produto mais vendido",
            "produtos mais vendidos",
            "top produtos",
            "best selling product",
            "best-selling product",
            "best selling products",
            "best-selling products",
            "top products",
        )
    ):
        return """
            SELECT
                product,
                SUM(quantity) AS units_sold
            FROM sales
            GROUP BY product
            ORDER BY units_sold DESC
            LIMIT 10
        """

    if any(
        term in query
        for term in (
            "numero de pedidos",
            "quantos pedidos",
            "number of orders",
            "how many orders",
            "order count",
        )
    ):
        return """
            SELECT COUNT(DISTINCT sale_id) AS order_count
            FROM sales
        """

    if any(
        term in query
        for term in (
            "quantidade vendida",
            "unidades vendidas",
            "units sold",
            "quantity sold",
        )
    ):
        return """
            SELECT SUM(quantity) AS units_sold
            FROM sales
        """

    is_discount_question = any(term in query for term in ("desconto", "discount"))
    mentions_revenue = any(
        term in query for term in ("receita", "faturamento", "revenue", "sales")
    )

    if is_discount_question and not mentions_revenue:
        return """
            SELECT
                ROUND(
                    SUM(quantity * unit_price * discount_pct),
                    2
                ) AS total_discounts
            FROM sales
        """

    if any(
        term in query
        for term in (
            "receita bruta",
            "faturamento bruto",
            "gross revenue",
            "gross sales",
        )
    ):
        return """
            SELECT
                ROUND(SUM(quantity * unit_price), 2) AS gross_revenue
            FROM sales
        """

    if any(
        term in query
        for term in (
            "receita liquida",
            "faturamento liquido",
            "net revenue",
            "net sales",
        )
    ):
        return """
            SELECT
                ROUND(
                    SUM(quantity * unit_price * (1 - discount_pct)),
                    2
                ) AS net_revenue
            FROM sales
        """

    if any(term in query for term in ("receita", "faturamento", "revenue", "sales")):
        return """
            SELECT
                ROUND(SUM(quantity * unit_price), 2) AS gross_revenue,
                ROUND(
                    SUM(quantity * unit_price * discount_pct),
                    2
                ) AS discounts,
                ROUND(
                    SUM(quantity * unit_price * (1 - discount_pct)),
                    2
                ) AS net_revenue
            FROM sales
        """

    return None


def _build_sales_planner_model():
    """Cria o modelo local usado somente no planejamento Text-to-Query."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def _extract_sales_period(question: str) -> str | None:
    """Extrai um período explícito no formato YYYY-MM."""
    match = re.search(
        r"\b20\d{2}-(?:0[1-9]|1[0-2])\b",
        question,
    )
    return match.group(0) if match else None


def inspect_sales_schema(state: WorkerState) -> dict:
    schema = json.loads(get_sales_schema.invoke({}))
    return {"schema": schema}


def analyze_sales(
    state: WorkerState,
    planner: Any | None = None,
) -> dict:
    question = state["question"]
    request_id = state["request_id"]
    schema = state["schema"]

    # Text-to-Query original: o LLM escolhe somente valores de um
    # catálogo fechado. O SQL continua sendo montado pelo código
    # original e executado em conexão SQLite read-only.
    try:
        from business_agent.agents.text_to_query import (
            execute_text_to_query,
            is_dimensional_sales_query,
        )

        if is_dimensional_sales_query(question):
            active_planner = (
                planner if planner is not None else _build_sales_planner_model()
            )

            query_result = execute_text_to_query(
                question=question,
                model=active_planner,
                period=_extract_sales_period(question),
            )

            evidence_payload = {
                "method": "Text-to-Query",
                "validated_plan": query_result["plan"],
                "period": query_result["period"],
                "rows": query_result["rows"],
                "answer": query_result["answer"],
                "execution_policy": (
                    "Plano validado por catálogo fechado; "
                    "SQL não é fornecido pelo modelo."
                ),
            }

            result = AgentResult(
                request_id=request_id,
                agent="sales",
                status="completed",
                summary=(
                    "Text-to-Query executado com plano estruturado "
                    "e validado. " + query_result["answer"]
                ),
                evidence=json.dumps(
                    evidence_payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                sources=["data/sample/sales.db"],
            )

            return {"result": result.model_dump(mode="json")}

    except Exception as exc:  # noqa: BLE001
        result = AgentResult(
            request_id=request_id,
            agent="sales",
            status="failed",
            summary=(
                "A consulta dimensional não pôde ser concluída "
                "pelo Text-to-Query validado."
            ),
            evidence=json.dumps(
                {
                    "question": question,
                    "schema": schema,
                    "execution_method": "Text-to-Query",
                },
                ensure_ascii=False,
                indent=2,
            ),
            sources=["data/sample/sales.db"],
            warnings=[
                f"{type(exc).__name__}: {exc}",
                (
                    "A consulta não foi substituída por SQL livre "
                    "ou por um plano não validado."
                ),
            ],
        )

        return {"result": result.model_dump(mode="json")}

    # Consultas não dimensionais continuam usando templates SQL fixos.
    sql = build_sales_sql(question)

    if sql is None:
        result = AgentResult(
            request_id=request_id,
            agent="sales",
            status="completed",
            summary=(
                "O schema comercial foi consultado. "
                "Não foi identificado um indicador numérico específico "
                "para executar automaticamente."
            ),
            evidence=json.dumps(
                schema,
                ensure_ascii=False,
                indent=2,
            ),
            sources=["data/sample/sales.db"],
        )
        return {"result": result.model_dump(mode="json")}

    query_result = query_sales_database.invoke({"sql": sql})
    failed = str(query_result).startswith("Erro")

    result = AgentResult(
        request_id=request_id,
        agent="sales",
        status="failed" if failed else "completed",
        summary=(
            "A consulta SQL parametrizada por template foi executada."
            if not failed
            else "Não foi possível executar a consulta SQL."
        ),
        evidence=(f"SQL:\n{sql.strip()}\n\nRESULTADO:\n{query_result}"),
        sources=["data/sample/sales.db"],
        warnings=(["A consulta retornou erro."] if failed else []),
    )

    return {"result": result.model_dump(mode="json")}


def build_sales_analyst_graph(planner: Any | None = None):
    """Subgrafo Sales Analyst: schema -> análise SQL/Text-to-Query."""

    def analyze_sales_node(state: WorkerState) -> dict:
        return analyze_sales(state, planner=planner)

    builder = StateGraph(WorkerState)

    builder.add_node("inspect_schema", inspect_sales_schema)
    builder.add_node("analyze_sales", analyze_sales_node)

    builder.add_edge(START, "inspect_schema")
    builder.add_edge("inspect_schema", "analyze_sales")
    builder.add_edge("analyze_sales", END)

    return builder.compile()


# ------------------------------------------------------------
# KNOWLEDGE ANALYST
# ------------------------------------------------------------


def _extract_vector_sources(evidence: str) -> list[str]:
    """Extrai nomes de arquivos a partir dos metadados da busca FAISS."""
    raw_sources = re.findall(
        r"(?im)^Fonte:\s*(.+?)\s*$",
        evidence,
    )

    sources = []

    for raw_source in raw_sources:
        normalized_path = raw_source.strip().strip("`")
        filename = re.split(r"[\\/]", normalized_path)[-1].strip()

        if filename and filename not in sources:
            sources.append(filename)

    return sources


def _extract_document_sources(evidence: str) -> list[str]:
    """Extrai os nomes dos documentos da recuperação lexical."""
    sources = re.findall(
        r"\[DOCUMENTO:\s*([^\]]+)\]",
        evidence,
    )

    return list(dict.fromkeys(source.strip() for source in sources))


def retrieve_knowledge(
    state: WorkerState,
    retriever: Any | None = None,
) -> dict:
    question = state["question"]

    backend = os.getenv("KNOWLEDGE_VECTOR_BACKEND", "faiss").strip().casefold()
    backend_label = {"faiss": "FAISS", "supabase": "Supabase PGVector"}.get(
        backend, backend.upper() or "desconhecido"
    )

    warnings = []
    vector_evidence = ""
    lexical_evidence = ""

    # Recuperação semântica pelo backend vetorial configurado.
    try:
        active_retriever = retriever
        if active_retriever is None:
            if backend == "supabase":
                from bi_multiagent.retrieval.supabase_pgvector import (
                    search_supabase_knowledge_base,
                )

                active_retriever = search_supabase_knowledge_base
            elif backend == "faiss":
                from business_agent.tools.rag_tool import search_knowledge_base

                active_retriever = search_knowledge_base
            else:
                raise ValueError(
                    "KNOWLEDGE_VECTOR_BACKEND must be 'faiss' or 'supabase'."
                )

        vector_evidence = active_retriever(question, k=4)

        vector_evidence = str(vector_evidence)

    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"Recuperação vetorial {backend_label} indisponível; "
            f"foi acionado o complemento documental ({type(exc).__name__})."
        )

    # Recuperação documental complementar: preserva a cobertura dos
    # quatro documentos oficiais da base de conhecimento.
    try:
        lexical_evidence = search_business_knowledge.invoke({"query": question})
        lexical_evidence = str(lexical_evidence)

    except Exception as exc:  # noqa: BLE001
        warnings.append(
            "Recuperação documental complementar indisponível: "
            f"{type(exc).__name__}: {exc}"
        )

    sources = []
    sources.extend(_extract_vector_sources(vector_evidence))
    sources.extend(_extract_document_sources(lexical_evidence))
    sources = list(dict.fromkeys(sources))

    if not _extract_vector_sources(vector_evidence) and not warnings:
        warnings.append(
            f"A busca {backend_label} não identificou fontes; "
            "a recuperação documental complementar foi utilizada."
        )

    evidence_parts = []

    if vector_evidence.strip():
        evidence_parts.extend(
            [
                f"RECUPERAÇÃO SEMÂNTICA ({backend_label.upper()}):",
                vector_evidence.strip(),
            ]
        )

    if lexical_evidence.strip():
        evidence_parts.extend(
            [
                "RECUPERAÇÃO DOCUMENTAL COMPLEMENTAR:",
                lexical_evidence.strip(),
            ]
        )

    evidence = "\n\n".join(evidence_parts)

    return {
        "evidence": evidence,
        "sources": sources,
        "warnings": warnings,
    }


def compile_knowledge_result(state: WorkerState) -> dict:
    evidence = state.get("evidence", "")
    sources = state.get("sources", [])
    warnings = state.get("warnings", [])

    # Recalcula o rótulo neste escopo: compile_knowledge_result é uma
    # função separada de retrieve_knowledge e não compartilha variáveis locais.
    backend = os.getenv("KNOWLEDGE_VECTOR_BACKEND", "faiss").strip().casefold()
    backend_label = {"faiss": "FAISS", "supabase": "Supabase PGVector"}.get(
        backend, backend.upper() or "desconhecido"
    )

    if not sources:
        result = AgentResult(
            request_id=state["request_id"],
            agent="knowledge",
            status="failed",
            summary=(
                "Não foram encontradas evidências documentais "
                "suficientes para esta solicitação."
            ),
            evidence=evidence,
            warnings=warnings + ["Ausência de fonte documental correspondente."],
        )
    else:
        result = AgentResult(
            request_id=state["request_id"],
            agent="knowledge",
            status="completed",
            summary=(
                f"Foram identificadas {len(sources)} fontes documentais "
                f"pela recuperação semântica {backend_label} e pela recuperação "
                "documental complementar."
            ),
            evidence=evidence,
            sources=sources,
            warnings=warnings,
        )

    return {"result": result.model_dump(mode="json")}


def build_knowledge_analyst_graph(retriever: Any | None = None):
    """Subgrafo Knowledge Analyst: backend vetorial + recuperação complementar."""

    def retrieve_knowledge_node(state: WorkerState) -> dict:
        return retrieve_knowledge(state, retriever=retriever)

    builder = StateGraph(WorkerState)

    builder.add_node("retrieve_documents", retrieve_knowledge_node)
    builder.add_node("compile_result", compile_knowledge_result)

    builder.add_edge(START, "retrieve_documents")
    builder.add_edge("retrieve_documents", "compile_result")
    builder.add_edge("compile_result", END)

    return builder.compile()


# ------------------------------------------------------------
# ECONOMIC ANALYST
# ------------------------------------------------------------


def parse_economic_period(state: WorkerState) -> dict:
    """Extrai duas datas explícitas DD/MM/AAAA da pergunta."""
    dates = re.findall(
        r"\b\d{2}/\d{2}/\d{4}\b",
        state["question"],
    )

    if len(dates) < 2:
        return {
            "start_date": "",
            "end_date": "",
        }

    try:
        start = datetime.strptime(
            dates[0],
            "%d/%m/%Y",
        )
        end = datetime.strptime(
            dates[1],
            "%d/%m/%Y",
        )
    except ValueError:
        return {
            "start_date": "",
            "end_date": "",
        }

    if start > end:
        return {
            "start_date": "",
            "end_date": "",
        }

    return {
        "start_date": dates[0],
        "end_date": dates[1],
    }


def fetch_economic_series(state: WorkerState) -> dict:
    if not state.get("start_date") or not state.get("end_date"):
        result = AgentResult(
            request_id=state["request_id"],
            agent="economic",
            status="needs_input",
            summary=(
                "Para consultar a meta Selic, informe as datas inicial "
                "e final no formato DD/MM/AAAA."
            ),
            sources=["04_fontes_e_limitacoes.md"],
            warnings=["A API não foi chamada porque o período não foi informado."],
        )

        return {"result": result.model_dump(mode="json")}

    response = get_bcb_series.invoke(
        {
            "series_code": 432,
            "start_date": state["start_date"],
            "end_date": state["end_date"],
        }
    )

    response_text = str(response)
    failed = response_text.startswith(("Erro", "Série não habilitada"))

    result = AgentResult(
        request_id=state["request_id"],
        agent="economic",
        status="failed" if failed else "completed",
        summary=(
            "Consulta à meta Selic realizada no Banco Central."
            if not failed
            else "Não foi possível consultar a série Selic."
        ),
        evidence=response_text,
        sources=(
            ["Banco Central do Brasil - SGS 432 (meta Selic, % ao ano)"]
            if not failed
            else []
        ),
        warnings=(
            ["A API do Banco Central não retornou dados utilizáveis."] if failed else []
        ),
    )

    return {"result": result.model_dump(mode="json")}


def build_economic_analyst_graph():
    """Subgrafo Economic Analyst: período -> consulta SGS."""
    builder = StateGraph(WorkerState)

    builder.add_node("parse_period", parse_economic_period)
    builder.add_node("fetch_selic", fetch_economic_series)

    builder.add_edge(START, "parse_period")
    builder.add_edge("parse_period", "fetch_selic")
    builder.add_edge("fetch_selic", END)

    return builder.compile()
