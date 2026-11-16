import re
import unicodedata
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from bi_multiagent.agents.specialized_agents import (
    build_economic_analyst_graph,
    build_knowledge_analyst_graph,
    build_sales_analyst_graph,
)
from bi_multiagent.coordination.agent_tools import AgentTool
from bi_multiagent.coordination.weighted_consensus import (
    weighted_evidence_consensus,
)
from bi_multiagent.coordination.instruction_fidelity import audit_instruction_fidelity
from bi_multiagent.schemas.contracts import (
    AgentResult,
    AgentTask,
    SupervisorDecision,
)
from bi_multiagent.schemas.multiagent_state import MultiAgentState
from bi_multiagent.settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ROOT_DIR,
)

SUPERVISOR_PROMPT = """
You are the Supervisor of a multi-agent Business Intelligence system.

Available agents:
- sales: numerical sales analysis, SQL and Text-to-Query.
- knowledge: definitions, formulas, documentation, business rules and FAISS.
- economic: historical Selic queries using BCB SGS series 432.

ROUTING POLICY:

1. Conceptual-only questions about definitions, formulas, meanings,
documentation, business rules or conceptual differences must use
knowledge only.

2. Numerical sales questions requesting values, totals, rankings,
aggregations or quantitative comparisons must use sales.

3. Questions requesting both a numerical result and its explanation,
formula or business rule must use sales and knowledge.

4. Historical Selic values must use economic.
Questions exclusively about the documentation of the Selic source
must use knowledge.

IMPORTANT:
The presence of words such as revenue, sales, discount, category
or calculate does not by itself imply a numerical request.

Examples:
- Formula of net revenue -> knowledge
- Difference between gross and net revenue -> knowledge
- Ranking of net revenue by category -> sales
- Total net revenue -> sales
- Total net revenue and how it is calculated -> sales + knowledge
- Historical Selic in a specified period -> economic

Do not choose unnecessary agents.
Do not execute queries yourself.
Return only the structured decision.
"""


OBJECTIVES = {
    "sales": (
        "Consultar schema e calcular indicadores comerciais "
        "com SQL somente leitura no banco SQLite."
    ),
    "knowledge": ("Recuperar definições, fórmulas, fontes e regras documentadas."),
    "economic": ("Consultar a meta Selic na série SGS 432 quando houver período."),
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _sales_routing_intent(question: str) -> tuple[bool, bool, bool]:
    """Identifica intencao comercial, conceitual e quantitativa."""
    query = normalize_text(question)

    sales_terms = (
        "receita",
        "faturamento",
        "venda",
        "vendas",
        "pedido",
        "pedidos",
        "unidade",
        "unidades",
        "quantidade",
        "produto",
        "categoria",
        "desconto",
    )

    conceptual_terms = (
        "definicao",
        "formula",
        "glossario",
        "documentado",
        "documentacao",
        "regra de negocio",
        "como calcula",
        "como calcular",
        "como e calculada",
        "como e calculado",
        "como ela e calculada",
        "diferenca entre",
        "indicadores",
        "significado",
        "o que e",
    )

    numeric_terms = (
        "quanto",
        "qual foi",
        "valor total",
        " total",
        "ranking",
        "top ",
        "unidades vendidas",
        "quantidade vendida",
        "numero de pedidos",
        "mostre os valores",
        "mostre o ranking",
        "apresente o ranking",
        "calcule o valor",
        "calcule a receita",
        "calcule o total",
        "compare os valores",
        "compare as vendas",
    )

    has_sales = any(term in query for term in sales_terms)
    has_conceptual = any(term in query for term in conceptual_terms)

    explicit_command = query.startswith(("calcule ", "compare ", "comparar "))

    has_numeric = any(term in query for term in numeric_terms) or explicit_command

    return has_sales, has_conceptual, has_numeric


def enforce_routing_policy(
    question: str,
    proposed_agents: list[str],
) -> list[str]:
    """Impede delegacoes comerciais desnecessarias."""
    has_sales, has_conceptual, has_numeric = _sales_routing_intent(question)

    selected = set(proposed_agents)

    if has_sales and has_conceptual:
        if has_numeric:
            selected.update({"sales", "knowledge"})
        else:
            selected.discard("sales")
            selected.add("knowledge")

    elif has_sales and has_numeric:
        selected.add("sales")

    order = ["sales", "knowledge", "economic"]
    return [agent for agent in order if agent in selected]


def fallback_route(question: str) -> list[str]:
    """Roteamento deterministico para contingencia do Supervisor."""
    query = normalize_text(question)
    agents = set()

    asks_selic = "selic" in query or "banco central" in query

    asks_selic_source = (
        "fonte" in query or "documentado" in query or "documentacao" in query
    )

    has_sales, has_conceptual, has_numeric = _sales_routing_intent(question)

    if asks_selic:
        agents.add("knowledge" if asks_selic_source else "economic")

    if has_sales:
        if has_conceptual:
            if has_numeric:
                agents.update({"sales", "knowledge"})
            else:
                agents.add("knowledge")
        elif has_numeric:
            agents.add("sales")

    if not agents:
        agents.add("knowledge")

    order = ["sales", "knowledge", "economic"]
    return [agent for agent in order if agent in agents]


def build_supervisor_model():
    """Supervisor LLM com retorno validado pelo contrato Pydantic."""
    model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    return model.with_structured_output(
        SupervisorDecision,
        method="json_schema",
    )


def _get_question(state: MultiAgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content

    return state.get("question", "")


def _detect_report_export(question: str) -> dict[str, str] | None:
    """Detecta pedidos explícitos para salvar/exportar um relatório."""
    normalized = normalize_text(question)
    tokens = set(re.findall(r"\w+", normalized))

    negative_phrases = (
        ("nao", "salve"),
        ("nao", "salvar"),
        ("nao", "exporte"),
        ("nao", "exportar"),
    )

    for first, second in negative_phrases:
        if first in tokens and second in tokens:
            return None

    has_report = bool(tokens.intersection({"relatorio", "arquivo"}))
    has_save_verb = bool(
        tokens.intersection({"salve", "salvar", "exporte", "exportar"})
    )

    if not (has_report and has_save_verb):
        return None

    return {
        "type": "save_report",
        "description": (
            "Gravar o relatório produzido pelo agente como arquivo Markdown "
            "local. A gravação só ocorrerá após aprovação humana."
        ),
        "target": "outputs/approved_reports/<request_id>.md",
    }


def build_multiagent_graph(
    router: Any | None = None,
    checkpointer: Any | None = None,
    *,
    use_in_memory_saver: bool = True,
):
    """Cria o Supervisor hierárquico com especialistas paralelos.

    Fluxo:
        prepare -> supervisor
        supervisor -> Send(sales, knowledge, economic)
        specialists -> audit -> synthesize -> END
    """
    route_model = router or build_supervisor_model()

    sales_graph = build_sales_analyst_graph()
    knowledge_graph = build_knowledge_analyst_graph()
    economic_graph = build_economic_analyst_graph()

    # Expose specialist subgraphs as LangChain structured tools.
    agent_tools = {
        "sales": AgentTool(
            "sales", OBJECTIVES["sales"], sales_graph
        ),
        "knowledge": AgentTool(
            "knowledge", OBJECTIVES["knowledge"], knowledge_graph
        ),
        "economic": AgentTool(
            "economic", OBJECTIVES["economic"], economic_graph
        ),
    }

    # --------------------------------------------------------
    # Preparação da solicitação
    # --------------------------------------------------------

    def prepare_request(state: MultiAgentState) -> dict:
        question = _get_question(state)

        if not question.strip():
            raise ValueError("A solicitação do usuário está vazia.")

        pending_action = _detect_report_export(question)

        return {
            "question": question,
            "request_id": uuid4().hex,
            "route": [],
            "route_method": "",
            "route_rationale": "",
            "audit": {},
            "weighted_consensus": {},
            "instruction_fidelity": {},
            "approval_required": pending_action is not None,
            "pending_action": pending_action or {},
            "approval": {},
            "action_result": {},
        }

    # --------------------------------------------------------
    # Supervisor: decisão estruturada ou fallback
    # --------------------------------------------------------

    def supervisor_node(state: MultiAgentState) -> dict:
        question = state["question"]

        try:
            decision = route_model.invoke(
                [
                    SystemMessage(content=SUPERVISOR_PROMPT),
                    HumanMessage(content=question),
                ]
            )

            if isinstance(decision, dict):
                decision = SupervisorDecision.model_validate(decision)

            if not isinstance(decision, SupervisorDecision):
                raise TypeError("O Supervisor não retornou SupervisorDecision.")

            raw_selected = list(decision.agents)
            selected = enforce_routing_policy(
                question,
                raw_selected,
            )

            method = "llm_structured_output"
            rationale = decision.rationale

            if selected != raw_selected:
                method += "+policy_guard"
                rationale += (
                    " A politica deterministica corrigiu a delegacao "
                    "para evitar agentes desnecessarios."
                )

        except Exception as exc:  # noqa: BLE001
            # A contingência permite que o sistema continue operando
            # quando o modelo local ou o parser estruturado estiver indisponível.
            selected = fallback_route(question)
            method = "keyword_fallback"
            rationale = (
                f"Roteamento determinístico acionado devido a {type(exc).__name__}."
            )

        decision_payload = {
            "request_id": state["request_id"],
            "agent": "supervisor",
            "selected_agents": selected,
            "method": method,
            "rationale": rationale,
        }

        return {
            "route": selected,
            "route_method": method,
            "route_rationale": rationale,
            "blackboard": [decision_payload],
        }

    # --------------------------------------------------------
    # Fan-out: cada especialista recebe AgentTask estruturada
    # --------------------------------------------------------

    def dispatch_workers(state: MultiAgentState):
        sends = []

        for agent in state["route"]:
            task = AgentTask(
                request_id=state["request_id"],
                agent=agent,
                objective=OBJECTIVES[agent],
                question=state["question"],
            )

            sends.append(
                Send(
                    f"{agent}_worker",
                    {
                        "question": state["question"],
                        "request_id": state["request_id"],
                        "task": task.model_dump(mode="json"),
                    },
                )
            )

        return sends

    # --------------------------------------------------------
    # Wrappers dos subgrafos e publicação no Blackboard
    # --------------------------------------------------------

    def execute_worker(agent: str, state: dict) -> dict:
        agent_tool = agent_tools[agent]
        result = agent_tool.invoke(state["task"])
        result_data = result.model_dump(mode="json")

        artifact = {
            "request_id": state["request_id"],
            "agent": agent,
            "artifact_type": "agent_result",
            "status": result.status,
            "sources": result.sources,
            "evidence": result.evidence,
            "execution_mode": "langchain_structured_tool",
            "tool_name": agent_tool.tool.name,
        }

        return {
            "agent_results": [result_data],
            "blackboard": [artifact],
        }

    def sales_worker(state: dict) -> dict:
        return execute_worker("sales", state)

    def knowledge_worker(state: dict) -> dict:
        return execute_worker("knowledge", state)

    def economic_worker(state: dict) -> dict:
        return execute_worker("economic", state)

    # --------------------------------------------------------
    # Audit Agent: valida presença e coerência das fontes
    # --------------------------------------------------------

    def audit_node(state: MultiAgentState) -> dict:
        current_results = [
            AgentResult.model_validate(item)
            for item in state.get("agent_results", [])
            if item.get("request_id") == state["request_id"]
        ]

        warnings = []

        for result in current_results:
            if result.status == "completed" and not result.evidence.strip():
                warnings.append(f"{result.agent}: resultado sem evidência.")

            if result.agent == "sales":
                if (
                    result.status == "completed"
                    and "data/sample/sales.db" not in result.sources
                ):
                    warnings.append("sales: fonte SQLite não identificada.")

            elif result.agent == "knowledge":
                if result.status == "completed" and not result.sources:
                    warnings.append("knowledge: resultado sem documento de origem.")

            elif (
                result.agent == "economic"
                and result.status == "completed"
                and not any(
                    "sgs 432" in normalize_text(source) for source in result.sources
                )
            ):
                warnings.append("economic: fonte SGS 432 não comprovada.")

        consensus = weighted_evidence_consensus(
            current_results,
            state.get("route", []),
        )
        if consensus["status"] != "accepted":
            warnings.append(
                "Consenso ponderado exige atenção: "
                f"{consensus['status']} "
                f"(suporte={consensus['evidence_support_score']:.2f})."
            )

        audit = {
            "request_id": state["request_id"],
            "results_reviewed": len(current_results),
            "passed": len(warnings) == 0 and consensus["status"] == "accepted",
            "warnings": warnings,
            "weighted_consensus": consensus,
        }

        return {
            "audit": audit,
            "weighted_consensus": consensus,
            "blackboard": [
                {
                    "request_id": state["request_id"],
                    "agent": "audit",
                    "artifact_type": "audit_report",
                    **audit,
                }
            ],
        }

    # --------------------------------------------------------
    # Síntese final
    # --------------------------------------------------------

    def synthesize_node(state: MultiAgentState) -> dict:
        current_results = [
            AgentResult.model_validate(item)
            for item in state.get("agent_results", [])
            if item.get("request_id") == state["request_id"]
        ]

        order = {
            "sales": 0,
            "knowledge": 1,
            "economic": 2,
        }

        current_results.sort(key=lambda result: order[result.agent])

        parts = [
            "SÍNTESE DO SUPERVISOR",
            "",
            f"Agentes acionados: {', '.join(state['route'])}",
            f"Método de roteamento: {state['route_method']}",
            f"Justificativa: {state['route_rationale']}",
            "",
        ]

        for result in current_results:
            parts.extend(
                [
                    f"## {result.agent.upper()}",
                    f"Status: {result.status}",
                    result.summary,
                ]
            )

            if result.sources:
                parts.append("Fontes: " + ", ".join(result.sources))

            if result.evidence:
                parts.extend(
                    [
                        "",
                        "Evidências:",
                        result.evidence[:5000],
                    ]
                )

            if result.warnings:
                parts.append("Alertas: " + "; ".join(result.warnings))

            parts.append("")

        audit = state.get("audit", {})
        consensus = audit.get("weighted_consensus", {})
        consensus_score = consensus.get("evidence_support_score", 0.0)
        parts.extend(
            [
                "## CONSENSO PONDERADO",
                f"Status: {consensus.get('status', 'not_calculated')}",
                f"Índice de suporte por evidências: {consensus_score:.2f}",
                "Método: pesos por especialidade, evidências e fontes;",
                "não é uma probabilidade calibrada.",
                "Agentes ausentes: "
                + (", ".join(consensus.get("missing_agents", [])) or "nenhum"),
                "",
                "## AUDITORIA",
                (
                    "Aprovada."
                    if audit.get("passed", False)
                    else "Atenção: existem avisos de auditoria."
                ),
                f"Resultados revisados: {audit.get('results_reviewed', 0)}",
            ]
        )

        if audit.get("warnings"):
            for warning in audit["warnings"]:
                parts.append(f"- {warning}")

        answer = "\n".join(parts)

        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def instruction_fidelity_node(state: MultiAgentState) -> dict:
        fidelity = audit_instruction_fidelity(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            selected_agents=state.get("route", []),
            agent_results=state.get("agent_results", []),
            request_id=state.get("request_id", ""),
            approval_required=state.get("approval_required", False),
            pending_action=state.get("pending_action", {}),
        )
        failed = fidelity.get("failed_checks", [])
        lines = [
            "## AUDITORIA DE FIDELIDADE",
            f"Status: {fidelity['status']}",
            f"Pontuação: {fidelity['score']:.2f} "
            f"({fidelity['passed_checks']}/{fidelity['total_checks']} verificações)",
            f"Intenção classificada: {fidelity['intent_classification']}",
            "Agentes esperados: "
            + (", ".join(fidelity.get("expected_agents", [])) or "não inferidos"),
            "Agentes acionados: "
            + (", ".join(fidelity.get("selected_agents", [])) or "nenhum"),
            "Verificações pendentes: " + (", ".join(failed) or "nenhuma"),
        ]
        section = "\n".join(lines)
        updated_answer = (state.get("answer", "") + "\n\n" + section).strip()
        return {
            "instruction_fidelity": fidelity,
            "answer": updated_answer,
            "messages": [AIMessage(content=section)],
        }

    # --------------------------------------------------------
    # Construção do StateGraph
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Human-in-the-Loop: decisão humana explícita
    # --------------------------------------------------------

    def human_approval_node(state: MultiAgentState) -> dict:
        pending_action = state.get("pending_action", {})

        # Sem efeitos colaterais antes do interrupt.
        decision = interrupt(
            {
                "type": "human_approval",
                "request_id": state["request_id"],
                "action": pending_action,
                "question": state["question"],
                "report_preview": state.get("answer", "")[:1800],
                "warning": (
                    "A aprovação autoriza somente a gravação local do "
                    "relatório. Nenhum envio externo será realizado."
                ),
            }
        )

        approved = isinstance(decision, dict) and decision.get("approved") is True
        comment = (
            str(decision.get("comment", "")).strip()
            if isinstance(decision, dict)
            else ""
        )

        approval = {
            "approved": approved,
            "decision_by": "human",
            "comment": comment,
            "request_id": state["request_id"],
        }

        return {
            "approval": approval,
            "blackboard": [
                {
                    "request_id": state["request_id"],
                    "agent": "human",
                    "artifact_type": "approval_decision",
                    **approval,
                }
            ],
        }

    def route_after_approval(state: MultiAgentState) -> str:
        if state.get("approval", {}).get("approved") is True:
            return "execute"
        return "reject"

    # --------------------------------------------------------
    # Executor: grava somente após autorização
    # --------------------------------------------------------

    def execute_approved_action_node(state: MultiAgentState) -> dict:
        approval = state.get("approval", {})
        action = state.get("pending_action", {})

        if approval.get("approved") is not True:
            raise PermissionError(
                "Ação bloqueada: não existe autorização humana válida."
            )

        if action.get("type") != "save_report":
            raise ValueError("Tipo de ação não permitido pelo executor.")

        output_dir = ROOT_DIR / "outputs" / "approved_reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{state['request_id']}.md"

        report_content = (
            "# Relatório aprovado\n\n"
            f"**Request ID:** {state['request_id']}\n\n"
            f"**Solicitação:** {state['question']}\n\n"
            "---\n\n"
            f"{state.get('answer', '')}\n"
        )

        # Escrita idempotente: uma repetição não sobrescreve outro conteúdo.
        if output_path.exists():
            existing = output_path.read_text(encoding="utf-8")
            if existing != report_content:
                raise FileExistsError(
                    "J? existe um relatório diferente para este request_id."
                )
        else:
            output_path.write_text(report_content, encoding="utf-8")

        relative_path = output_path.relative_to(ROOT_DIR).as_posix()

        action_result = {
            "status": "executed",
            "action": "save_report",
            "path": relative_path,
            "authorized": True,
            "request_id": state["request_id"],
        }

        message = f"Ação autorizada e executada. Relatório gravado em: {relative_path}"

        return {
            "action_result": action_result,
            "answer": (state.get("answer", "") + "\n\n## AÇÃO AUTORIZADA\n" + message),
            "messages": [AIMessage(content=message)],
            "blackboard": [
                {
                    "request_id": state["request_id"],
                    "agent": "action_executor",
                    "artifact_type": "action_execution",
                    **action_result,
                }
            ],
        }

    # --------------------------------------------------------
    # Rejeição: nenhum arquivo é criado
    # --------------------------------------------------------

    def reject_action_node(state: MultiAgentState) -> dict:
        action_result = {
            "status": "rejected",
            "action": "save_report",
            "authorized": False,
            "request_id": state["request_id"],
        }

        message = "Ação rejeitada pelo usuário. Nenhum relatório foi gravado."

        return {
            "action_result": action_result,
            "answer": (
                state.get("answer", "") + "\n\n## AÇÃO NÃO EXECUTADA\n" + message
            ),
            "messages": [AIMessage(content=message)],
            "blackboard": [
                {
                    "request_id": state["request_id"],
                    "agent": "action_executor",
                    "artifact_type": "action_rejection",
                    **action_result,
                }
            ],
        }

    def route_after_synthesis(state: MultiAgentState) -> str:
        if state.get("approval_required", False):
            return "approval"
        return "end"

    builder = StateGraph(MultiAgentState)

    builder.add_node("prepare_request", prepare_request)
    builder.add_node("supervisor", supervisor_node)

    builder.add_node("sales_worker", sales_worker)
    builder.add_node("knowledge_worker", knowledge_worker)
    builder.add_node("economic_worker", economic_worker)

    builder.add_node("audit", audit_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("instruction_fidelity_audit", instruction_fidelity_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node(
        "execute_approved_action",
        execute_approved_action_node,
    )
    builder.add_node("reject_action", reject_action_node)

    builder.add_edge(START, "prepare_request")
    builder.add_edge("prepare_request", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        dispatch_workers,
        path_map=[
            "sales_worker",
            "knowledge_worker",
            "economic_worker",
        ],
    )

    builder.add_edge("sales_worker", "audit")
    builder.add_edge("knowledge_worker", "audit")
    builder.add_edge("economic_worker", "audit")

    builder.add_edge("audit", "synthesize")
    builder.add_edge("synthesize", "instruction_fidelity_audit")
    builder.add_conditional_edges(
        "instruction_fidelity_audit",
        route_after_synthesis,
        {
            "approval": "human_approval",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "execute": "execute_approved_action",
            "reject": "reject_action",
        },
    )

    builder.add_edge("execute_approved_action", END)
    builder.add_edge("reject_action", END)

    memory = checkpointer if checkpointer is not None else (
        InMemorySaver() if use_in_memory_saver else None
    )

    return builder.compile(checkpointer=memory)
