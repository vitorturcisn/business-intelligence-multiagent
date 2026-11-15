import unicodedata
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import trim_messages
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from bi_multiagent.schemas.state import SingleAgentState
from bi_multiagent.settings import (
    MAX_HISTORY_MESSAGES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from bi_multiagent.tools import (
    TOOLS,
    get_sales_schema,
    search_business_knowledge,
)

SYSTEM_PROMPT = """
Você é um analista de Business Intelligence. Responda em português.

REGRAS OBRIGATÓRIAS DE EVIDÊNCIA:

1. Não invente tabelas, colunas, métricas, fórmulas, períodos ou fontes.

2. O banco comercial contém dados sintéticos e sua tabela principal é
   sales. Respostas numéricas sobre vendas devem usar o SQLite.

3. Antes de criar SQL, consulte get_sales_schema.
   Execute somente consultas SELECT usando query_sales_database.

4. Definições, indicadores e regras de negócio devem vir dos documentos
   retornados por search_business_knowledge.

5. O glossário documenta: receita bruta, descontos, receita líquida,
   quantidade vendida, número de pedidos, receita por categoria
   e comparação entre períodos.

6. Não acrescente CAC, lucro bruto, margem de lucro, OPE ou outros
   indicadores que não estejam presentes nas evidências documentais.

7. Fórmulas documentadas:
   - Receita bruta por item = quantidade x preço unitário.
   - Desconto por item = receita bruta x percentual de desconto.
   - Receita líquida = receita bruta - descontos.

8. Número de pedidos significa identificadores distintos de sale_id.
   Quantidade vendida significa soma da coluna quantity.
   Não confunda pedidos com unidades.

9. Para fontes externas, a única série documentada neste projeto é a
   série 432 do Banco Central: meta Selic, em percentual ao ano.
   Não use a série 433 ou outras séries como fonte de indicadores comerciais.

10. Quando uma informação não estiver nas evidências ou nos resultados
    das ferramentas, diga explicitamente que não está documentada
    ou que não foi possível consultá-la.

11. Ao responder com base em documentos, cite o arquivo entre colchetes,
    por exemplo [Fonte: 01_glossario_kpis.md].

12. Separe fatos documentados de interpretações. Não apresente uma
    hipótese como se fosse uma regra de negócio oficial.

Se houver evidências recuperadas no contexto, priorize essas evidências.
Use ferramentas para complementar a resposta quando necessário.

SEMANTICA DOS DADOS DE VENDAS ? REGRA OBRIGATORIA:
- discount_pct ? armazenado como fra??o decimal, n?o como pontos percentuais.
- 0.10 representa 10%, 0.05 representa 5% e 0.00 representa 0%.
- Receita l?quida: quantity * unit_price * (1 - discount_pct).
- Nunca divida discount_pct por 100.

FORMATACAO MONETARIA BRASILEIRA OBRIGATORIA:
- Ao responder em portugu?s, apresente valores monet?rios no padr?o brasileiro.
- Use ponto para milhares e v?rgula para casas decimais.
- Exemplos: 31255.0 deve ser apresentado como R$ 31.255,00.
- Exemplos: 7650.0 deve ser apresentado como R$ 7.650,00.
- Exemplos: 4365.0 deve ser apresentado como R$ 4.365,00.
- Nunca escreva formatos amb?guos como R$ 31.255.00.
- Preserve os valores retornados pelo banco; n?o invente casas decimais.
"""


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def build_chat_model() -> ChatOllama:
    """Cria o modelo local configurado no .env."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def retrieve_context(state: SingleAgentState) -> dict[str, str]:
    """Recupera schema ou documentos antes da chamada ao LLM.

    Este nó evita depender exclusivamente da decisão do modelo para
    recuperar evidências de negócio.
    """
    messages = state.get("messages", [])

    last_question = next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, HumanMessage) and isinstance(message.content, str)
        ),
        "",
    )

    if not last_question.strip():
        return {"evidence": ""}

    query = _normalize(last_question)
    evidence_parts = []

    schema_terms = (
        "tabela",
        "tabelas",
        "coluna",
        "colunas",
        "schema",
        "esquema",
        "estrutura",
        "receita",
        "venda",
        "faturamento",
        "produto",
        "categoria",
        "pedido",
        "unidade",
        "quantidade",
        "desconto",
    )

    if any(term in query for term in schema_terms):
        schema_result = get_sales_schema.invoke({})
        evidence_parts.append("[SCHEMA DO BANCO SQLITE]\n" + str(schema_result))

    documentation_terms = (
        "indicador",
        "kpi",
        "receita",
        "faturamento",
        "bruta",
        "liquida",
        "desconto",
        "pedido",
        "unidade",
        "categoria",
        "regra",
        "fonte",
        "selic",
        "banco central",
        "limitacao",
        "glossario",
    )

    if any(term in query for term in documentation_terms):
        document_result = search_business_knowledge.invoke({"query": last_question})
        evidence_parts.append("[EVIDÊNCIAS DOCUMENTAIS]\n" + str(document_result))

    return {"evidence": "\n\n".join(evidence_parts)}


def build_single_agent_graph(
    model: Any | None = None,
    checkpointer: Any | None = None,
):
    """Monta StateGraph com recuperação prévia e ciclo ReAct.

    Fluxo:
        START -> retrieve_context -> agent
        agent -> tools -> agent
        agent -> END
    """
    chat_model = model or build_chat_model()
    model_with_tools = chat_model.bind_tools(TOOLS)

    def call_model(state: SingleAgentState) -> dict:
        history = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=MAX_HISTORY_MESSAGES,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )

        system_content = SYSTEM_PROMPT

        evidence = state.get("evidence", "")
        if evidence:
            system_content += (
                "\n\nEVIDÊNCIAS RECUPERADAS PARA ESTA PERGUNTA:\n" + evidence
            )

        response = model_with_tools.invoke(
            [
                SystemMessage(content=system_content),
                *history,
            ]
        )

        return {"messages": [response]}

    builder = StateGraph(SingleAgentState)

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    memory = checkpointer or InMemorySaver()

    return builder.compile(checkpointer=memory)
