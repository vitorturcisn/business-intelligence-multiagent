"""Plan-and-Execute reasoning architecture implemented with LangGraph."""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from bi_multiagent.agents.single_agent_graph import (
    SYSTEM_PROMPT,
    retrieve_context,
)
from bi_multiagent.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
from bi_multiagent.tools import TOOLS

MAX_PLAN_STEPS = 5
MAX_OBSERVATION_CHARS = 10000


class PlanExecuteState(MessagesState, total=False):
    """State shared by the planning, execution and synthesis nodes."""

    question: str
    evidence: str
    plan: list[dict[str, Any]]
    step_index: int
    observations: list[dict[str, Any]]
    replan_count: int
    decision: str
    replan_reason: str
    final_answer: str


def _extract_text(response: Any) -> str:
    """Extracts textual content from a LangChain model response."""
    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        content = "\n".join(parts)

    if not isinstance(content, str):
        raise TypeError("The planner must return textual JSON.")

    return content


def _parse_json_object(response: Any) -> dict[str, Any]:
    """Parses one JSON object, tolerating surrounding explanatory text."""
    if isinstance(response, dict):
        return response

    content = _extract_text(response)
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)

    if match is None:
        raise ValueError("The planner did not return a JSON object.")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("The planner returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise TypeError("The planner response must be a JSON object.")

    return payload


def _get_tool_map(tools: list[Any]) -> dict[str, Any]:
    """Creates an explicit allowlist of executable tools."""
    tool_map: dict[str, Any] = {}

    for tool in tools:
        name = getattr(tool, "name", None)

        if not isinstance(name, str) or not name:
            raise ValueError("Every Plan-and-Execute tool must have a name.")

        if name in tool_map:
            raise ValueError(f"Duplicate tool name in allowlist: {name}")

        tool_map[name] = tool

    return tool_map


def _tool_manifest(tool_map: dict[str, Any]) -> str:
    """Returns a compact catalog of allowed tools and arguments."""
    manifest = []

    for name, tool in sorted(tool_map.items()):
        arguments = getattr(tool, "args", {})
        if not isinstance(arguments, dict):
            arguments = {}

        description = str(getattr(tool, "description", "") or "")
        description = " ".join(description.split())[:220]

        manifest.append(
            {
                "name": name,
                "description": description,
                "arguments": sorted(arguments.keys()),
            }
        )

    return json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _can_answer_from_evidence(question: str, evidence: str) -> bool:
    """Checks whether retrieved documents already contain the requested fact."""
    query = _normalize_text(question)
    context = _normalize_text(evidence)

    formula_question = "formula" in query and (
        ("receita" in query and "liquida" in query)
        or ("net" in query and "revenue" in query)
    )

    formula_evidence = (
        "[evidencias documentais]" in context
        and "receita bruta" in context
        and "desconto" in context
        and "receita liquida" in context
    )

    if formula_question and formula_evidence:
        return True

    schema_question = any(
        term in query for term in ("schema", "estrutura", "coluna", "colunas")
    )

    schema_evidence = (
        "[schema do banco sqlite]" in context
        and "sales" in context
        and "table_names" in context
    )

    return schema_question and schema_evidence


def _requires_sales_query(question: str) -> bool:
    """Identifies questions that require actual sales data."""
    query = _normalize_text(question)

    sales_terms = (
        "receita",
        "faturamento",
        "venda",
        "vendas",
        "pedido",
        "pedidos",
        "unidade",
        "quantidade",
        "categoria",
        "produto",
        "regiao",
    )

    quantitative_terms = (
        "total",
        "quanto",
        "quantos",
        "valor",
        "soma",
        "somar",
        "media",
        "ranking",
        "maior",
        "menor",
        "por categoria",
        "por regiao",
        "por produto",
        "por mes",
        "compare",
        "calcule",
        "calcular",
        "mostre",
        "liste",
        "apresente",
    )

    return any(term in query for term in sales_terms) and any(
        term in query for term in quantitative_terms
    )


# DECODE_COMPACTED_SQL_TOOL_RESULT
def _decode_tool_result(value: Any) -> dict[str, Any] | None:
    """Decodes tool results serialized once or multiple times."""
    current = value

    for _ in range(3):
        if not isinstance(current, str):
            break

        try:
            current = json.loads(current)
        except (json.JSONDecodeError, TypeError):
            return None

    return current if isinstance(current, dict) else None


def _fallback_sales_sql(question: str) -> str | None:
    """Builds a deterministic SQL query for total net revenue."""
    query = _normalize_text(question)

    asks_total = any(
        term in query
        for term in ("total", "todas as vendas", "de todas as vendas", "geral")
    )

    if "receita liquida" in query and asks_total:
        return (
            "SELECT COALESCE("
            "SUM(quantity * unit_price * (1 - discount_pct)), 0"
            ") AS net_revenue FROM sales"
        )

    return None


def _invoke_json_object(model: Any, prompt: str) -> dict[str, Any]:
    """Invokes the planner and retries once when the output is invalid."""
    response = model.invoke(prompt)

    try:
        return _parse_json_object(response)
    except (TypeError, ValueError):
        original = _extract_text(response)[:1500]

        repair_prompt = (
            "Return exactly one valid JSON object. "
            "Do not use Markdown, explanations or code fences.\n"
            "Repair this invalid output:\n"
            f"{original}\n\n"
            "Follow the JSON structure requested in the original prompt:\n"
            f"{prompt[:3000]}"
        )

        repaired = model.invoke(repair_prompt)
        return _parse_json_object(repaired)


def _validate_steps(
    raw_steps: Any,
    tool_map: dict[str, Any],
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """Validates a plan against the tool allowlist before execution."""
    if not isinstance(raw_steps, list):
        raise TypeError("The plan must contain a list of steps.")

    if not raw_steps and not allow_empty:
        raise ValueError("A continuation plan must contain at least one step.")

    if len(raw_steps) > MAX_PLAN_STEPS:
        raise ValueError(f"A plan cannot contain more than {MAX_PLAN_STEPS} steps.")

    validated = []

    for index, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise TypeError(f"Plan step {index + 1} must be an object.")

        unexpected = set(step) - {"tool", "args", "reason"}
        if unexpected:
            raise ValueError(
                f"Unexpected fields in plan step {index + 1}: {sorted(unexpected)}"
            )

        tool_name = step.get("tool")
        arguments = step.get("args", {})
        reason = step.get("reason", "")

        if not isinstance(tool_name, str) or tool_name not in tool_map:
            raise ValueError(
                f"Tool '{tool_name}' is not part of the allowed tool catalog."
            )

        if not isinstance(arguments, dict):
            raise TypeError(f"Arguments for step {index + 1} must be a JSON object.")

        if not isinstance(reason, str):
            raise TypeError(f"Reason for step {index + 1} must be text.")

        args_schema = getattr(tool_map[tool_name], "args_schema", None)

        if hasattr(args_schema, "model_validate"):
            args_schema.model_validate(arguments)

        validated.append(
            {
                "tool": tool_name,
                "args": arguments,
                "reason": reason,
            }
        )

    return validated


def _last_user_question(state: PlanExecuteState) -> str:
    """Returns the most recent user question from the conversation."""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content

    question = state.get("question", "")
    return question if isinstance(question, str) else ""


def _compact_result(value: Any) -> str:
    """Serializes tool results into checkpoint-friendly text."""
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            indent=2,
        )
    except (TypeError, ValueError):
        rendered = str(value)

    if len(rendered) > MAX_OBSERVATION_CHARS:
        rendered = (
            rendered[:MAX_OBSERVATION_CHARS]
            + "\n[Result truncated by Plan-and-Execute.]"
        )

    return rendered


def build_plan_execute_graph(
    planner: Any | None = None,
    synthesizer: Any | None = None,
    tools: list[Any] | None = None,
    checkpointer: Any | None = None,
    max_replans: int = 1,
):
    """Builds an explicit Plan-and-Execute graph.

    Flow:
        START -> retrieve_context -> planner
        planner -> execute_step -> execute_step (while steps remain)
        execute_step -> replan
        replan -> execute_step (if more evidence is needed)
        replan -> finalize -> END

    The planner can only select tools from the configured allowlist.
    SQL may only be supplied as the argument of query_sales_database, which enforces read-only execution through SQLite authorization.
    """
    if isinstance(max_replans, bool) or not isinstance(max_replans, int):
        raise TypeError("max_replans must be an integer.")

    if max_replans < 0:
        raise ValueError("max_replans cannot be negative.")

    active_tools = list(TOOLS if tools is None else tools)
    tool_map = _get_tool_map(active_tools)

    planner_model = (
        planner
        if planner is not None
        else ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            reasoning=False,
            format="json",
            num_predict=256,
        )
    )

    synthesizer_model = (
        synthesizer
        if synthesizer is not None
        else ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            reasoning=False,
            num_predict=512,
        )
    )

    manifest = _tool_manifest(tool_map)

    def create_plan(state: PlanExecuteState) -> dict[str, Any]:
        question = _last_user_question(state).strip()

        if not question:
            raise ValueError("A user question is required.")

        evidence = state.get("evidence", "")
        # DETERMINISTIC_TOTAL_NET_REVENUE_PLAN
        fallback_sql = _fallback_sales_sql(question)

        if fallback_sql is not None:
            return {
                "question": question,
                "plan": [
                    {
                        "tool": "query_sales_database",
                        "args": {"sql": fallback_sql},
                        "reason": (
                            "Execute an aggregated SQL calculation of total "
                            "net revenue using the sales database."
                        ),
                    }
                ],
                "step_index": 0,
                "observations": [],
                "replan_count": 0,
                "decision": "execute",
                "replan_reason": ("A deterministic aggregated SQL query is required."),
            }

        if _can_answer_from_evidence(question, evidence):
            return {
                "question": question,
                "plan": [],
                "step_index": 0,
                "observations": [],
                "replan_count": 0,
                "decision": "final",
                "replan_reason": (
                    "Retrieved evidence is sufficient; no tool call is necessary."
                ),
            }

        prompt = f"""You are a concise planning component for Business Intelligence.

        Return exactly one JSON object. No Markdown or explanation.

        Required structure:
        {{"steps":[{{"tool":"tool_name","args":{{}},"reason":"short reason"}}]}}

        If retrieved evidence fully answers a documentary or schema question,
        return exactly:
        {{"steps":[]}}

        Allowed tools and argument names:
        {manifest}

        Rules:
        - Use only listed tools and argument names.
        - Use at most {MAX_PLAN_STEPS} steps; prefer the minimum necessary.
        - For sales calculations, use query_sales_database with one read-only
          SELECT statement in args.sql.
        - Use only table and column names present in the retrieved schema.
        - discount_pct is a decimal fraction: 0.10 means 10%.
        - Net revenue uses quantity * unit_price * (1 - discount_pct).
        - Never divide discount_pct by 100.
        - Never use DDL, writes, PRAGMA or multiple SQL statements.
        - Do not invent values or facts.
        - A documented formula or schema is not a calculated numeric result.
        - For quantitative sales questions, you MUST call query_sales_database.
        - Totals, counts, rankings and grouped sales values require actual SQL.
        - Never return an empty plan for a question asking for a numeric result.
        - Empty steps are allowed only for purely documentary or schema questions.

        Question:
        {question}

        Retrieved evidence:
        {evidence[:4500]}
        """

        payload = _invoke_json_object(planner_model, prompt)

        if set(payload) != {"steps"}:
            raise ValueError("The initial plan must contain only the steps field.")

        steps = _validate_steps(
            payload["steps"],
            tool_map,
            allow_empty=True,
        )
        if _requires_sales_query(question) and not any(
            step["tool"] == "query_sales_database" for step in steps
        ):
            fallback_sql = _fallback_sales_sql(question)

            if fallback_sql is None:
                raise ValueError(
                    "A numeric sales question requires query_sales_database, "
                    "but the planner returned no SQL step and no deterministic "
                    "fallback exists."
                )

            steps.append(
                {
                    "tool": "query_sales_database",
                    "args": {"sql": fallback_sql},
                    "reason": (
                        "The question requires an actual net revenue calculation "
                        "from the sales database."
                    ),
                }
            )

        return {
            "question": question,
            "plan": steps,
            "step_index": 0,
            "observations": [],
            "replan_count": 0,
            "decision": "execute" if steps else "final",
        }

    def execute_step(state: PlanExecuteState) -> dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("step_index", 0)

        if index >= len(plan):
            raise RuntimeError("No pending Plan-and-Execute step exists.")

        step = plan[index]
        tool = tool_map[step["tool"]]

        try:
            result = tool.invoke(step["args"])
            observation = {
                "step": index + 1,
                "tool": step["tool"],
                "reason": step["reason"],
                "status": "completed",
                "result": _compact_result(result),
            }
        except Exception as exc:  # noqa: BLE001
            observation = {
                "step": index + 1,
                "tool": step["tool"],
                "reason": step["reason"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }

        observations = list(state.get("observations", []))
        observations.append(observation)

        return {
            "observations": observations,
            "step_index": index + 1,
        }

    def replan(state: PlanExecuteState) -> dict[str, Any]:
        count = state.get("replan_count", 0)

        if count >= max_replans:
            return {
                "decision": "final",
                "replan_reason": "Maximum replan count reached.",
            }

        question = state.get("question", "")
        observations = state.get("observations", [])
        # DETERMINISTIC_TOTAL_REVENUE_SHORT_CIRCUIT
        if _fallback_sales_sql(question) is not None and any(
            observation.get("tool") == "query_sales_database"
            for observation in observations
        ):
            return {
                "decision": "final",
                "replan_reason": (
                    "The deterministic total net revenue query was attempted; "
                    "the finalizer will verify whether it succeeded."
                ),
            }

        prompt = f"""You are the replanner for a Business Intelligence workflow.

        Return exactly one JSON object and no Markdown.

        Choose one structure:
        {{"decision":"final","steps":[],"reason":"evidence is sufficient"}}
        {{"decision":"continue","steps":[{{"tool":"tool_name","args":{{}},"reason":"short reason"}}],"reason":"specific evidence gap"}}

        Allowed tools and argument names:
        {manifest}

        Original question:
        {question}

        Completed observations:
        {json.dumps(observations, ensure_ascii=False, default=str, separators=(",", ":"))[:3000]}

        Rules:
        - Use only allowed tools and their listed argument names.
        - Choose final when observations answer the question.
        - Continue only when a specific evidence gap remains.
        - For query_sales_database, use one read-only SELECT in args.sql.
        - Never use DDL, writes, PRAGMA or multiple SQL statements.
        - Return no more than {MAX_PLAN_STEPS} additional steps.
        """

        payload = _invoke_json_object(planner_model, prompt)

        decision = payload.get("decision")
        reason = payload.get("reason", "")

        if decision not in {"final", "continue"}:
            raise ValueError("The replanner decision must be final or continue.")

        if not isinstance(reason, str):
            raise TypeError("The replanner reason must be text.")

        if decision == "final":
            return {
                "decision": "final",
                "replan_reason": reason,
            }

        steps = _validate_steps(
            payload.get("steps"),
            tool_map,
            allow_empty=False,
        )

        return {
            "decision": "continue",
            "plan": steps,
            "step_index": 0,
            "replan_count": count + 1,
            "replan_reason": reason,
        }

    def finalize(state: PlanExecuteState) -> dict[str, Any]:
        question = state.get("question", "")
        evidence = state.get("evidence", "")
        observations = state.get("observations", [])
        # DETERMINISTIC_TOTAL_NET_REVENUE_FINALIZATION
        if _fallback_sales_sql(question) is not None:
            query_observation = next(
                (
                    observation
                    for observation in observations
                    if observation.get("tool") == "query_sales_database"
                    and observation.get("status") == "completed"
                ),
                None,
            )

            failure = None

            if query_observation is None:
                failure = "A consulta SQL nao foi concluida com sucesso."
            else:
                payload = _decode_tool_result(query_observation.get("result"))

                rows = payload.get("rows") if payload else None

                if (
                    not isinstance(rows, list)
                    or len(rows) != 1
                    or not isinstance(rows[0], dict)
                    or "net_revenue" not in rows[0]
                ):
                    failure = (
                        "O resultado SQL nao contem uma coluna net_revenue "
                        "valida com o total agregado."
                    )

            if failure is not None:
                answer = (
                    "Nao foi possivel calcular a receita liquida total. "
                    + failure
                    + " Nenhum valor foi estimado ou inventado."
                )

                return {
                    "final_answer": answer,
                    "messages": [AIMessage(content=answer)],
                }

            raw_value = rows[0]["net_revenue"]

            if isinstance(raw_value, bool) or raw_value is None:
                failure = "O valor net_revenue nao e numerico."
            else:
                try:
                    amount = Decimal(str(raw_value)).quantize(
                        Decimal("0.01"),
                        rounding=ROUND_HALF_UP,
                    )
                except (InvalidOperation, ValueError):
                    failure = "O valor net_revenue nao pode ser interpretado."
                else:
                    if not amount.is_finite():
                        failure = "O valor net_revenue nao e finito."

            if failure is not None:
                answer = (
                    "Nao foi possivel apresentar a receita liquida total. "
                    + failure
                    + " Nenhum valor foi estimado ou inventado."
                )

                return {
                    "final_answer": answer,
                    "messages": [AIMessage(content=answer)],
                }

            formatted = f"{amount:,.2f}"
            formatted = (
                formatted.replace(",", "\0").replace(".", ",").replace("\0", ".")
            )

            answer = (
                "A receita liquida total de todas as vendas, calculada "
                "diretamente pelo SQL agregado sobre a tabela `sales`, e "
                f"**R$ {formatted}**.\n\n"
                "Formula aplicada: "
                "`SUM(quantity * unit_price * (1 - discount_pct))`.\n\n"
                "Fonte: banco SQLite `sales.db`. Os registros sao dados "
                "de exemplo do projeto."
            )

            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
            }

        database_succeeded = any(
            observation.get("tool") == "query_sales_database"
            and observation.get("status") == "completed"
            for observation in observations
        )

        if _requires_sales_query(question) and not database_succeeded:
            answer = (
                "Nao foi possivel calcular o indicador solicitado porque "
                "a consulta ao banco de vendas nao foi concluida com sucesso. "
                "No numeric value was inferred or substituted."
            )
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
            }

        system_prompt = (
            SYSTEM_PROMPT
            + "\n\nYou are now synthesizing the result of a Plan-and-Execute "
            "run. Do not claim that a tool was used unless it appears in "
            "the observations. Distinguish missing evidence from a negative "
            "result. Do not invent values, dates or sources."
        )

        user_prompt = f"""Answer the user's original question using only the evidence below.

Original question:
{question}

Retrieved documentary and schema evidence:
{evidence}

Tool execution observations:
{json.dumps(observations, ensure_ascii=False, default=str, indent=2)}

Provide a clear answer. Mention relevant sources and limitations.
"""

        response = synthesizer_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        answer = _extract_text(response).strip()

        if not answer:
            raise ValueError("The synthesizer returned an empty answer.")

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    def after_plan(state: PlanExecuteState) -> str:
        return "execute_step" if state.get("plan") else "finalize"

    def after_step(state: PlanExecuteState) -> str:
        if state.get("step_index", 0) < len(state.get("plan", [])):
            return "execute_step"
        return "replan"

    def after_replan(state: PlanExecuteState) -> str:
        if state.get("decision") == "continue":
            return "execute_step"
        return "finalize"

    builder = StateGraph(PlanExecuteState)

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("planner", create_plan)
    builder.add_node("execute_step", execute_step)
    builder.add_node("replan", replan)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "planner")

    builder.add_conditional_edges(
        "planner",
        after_plan,
        {
            "execute_step": "execute_step",
            "finalize": "finalize",
        },
    )

    builder.add_conditional_edges(
        "execute_step",
        after_step,
        {
            "execute_step": "execute_step",
            "replan": "replan",
        },
    )

    builder.add_conditional_edges(
        "replan",
        after_replan,
        {
            "execute_step": "execute_step",
            "finalize": "finalize",
        },
    )

    builder.add_edge("finalize", END)

    memory = checkpointer if checkpointer is not None else InMemorySaver()

    return builder.compile(checkpointer=memory)
