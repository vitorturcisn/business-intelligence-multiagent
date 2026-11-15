"""Registry local dos agentes especialistas publicados via A2A."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    display_name: str
    description: str
    port: int
    skill_id: str
    skill_name: str
    skill_description: str
    tags: tuple[str, ...]
    examples: tuple[str, ...]


AGENT_SPECS: dict[str, AgentSpec] = {
    "sales": AgentSpec(
        name="sales",
        display_name="BI Sales Analyst",
        description=(
            "Specialist agent for commercial metrics, sales analysis, "
            "read-only SQL and validated Text-to-Query."
        ),
        port=9101,
        skill_id="sales-analysis",
        skill_name="Sales Analysis",
        skill_description=(
            "Calculates sales indicators and analyzes products, categories "
            "and orders using a read-only SQLite database."
        ),
        tags=("sales", "sql", "revenue", "kpi", "text-to-query"),
        examples=(
            "Calculate total net revenue.",
            "Calcule a receita líquida total.",
            "Rank net revenue by category.",
            "Faça um ranking de receita líquida por categoria.",
            "How many orders are in the database?",
            "Quantos pedidos existem na base?",
        ),
    ),
    "knowledge": AgentSpec(
        name="knowledge",
        display_name="BI Knowledge Analyst",
        description=(
            "Specialist agent for business definitions, formulas, "
            "documentation, business rules and retrieval-augmented generation."
        ),
        port=9102,
        skill_id="business-knowledge",
        skill_name="Business Knowledge Retrieval",
        skill_description=(
            "Retrieves documented business definitions, KPI formulas, "
            "data dictionaries, source limitations and business rules."
        ),
        tags=("knowledge", "rag", "faiss", "documentation", "business-rules"),
        examples=(
            "Explain the formula for net revenue.",
            "What is the difference between orders and units?",
            "Which limitations apply to the sample sales database?",
        ),
    ),
    "economic": AgentSpec(
        name="economic",
        display_name="BI Economic Analyst",
        description=(
            "Specialist agent for historical Brazilian Selic target "
            "queries using Banco Central do Brasil SGS series 432."
        ),
        port=9103,
        skill_id="selic-analysis",
        skill_name="Historical Selic Analysis",
        skill_description=(
            "Retrieves historical Selic target observations from "
            "Banco Central do Brasil SGS series 432."
        ),
        tags=("economics", "selic", "bcb", "sgs", "macroeconomics"),
        examples=(
            "What was the Selic target between 2026-10-01 and 2026-10-31?",
            "Retrieve the historical Selic observations for a given period.",
        ),
    ),
}


def get_agent_spec(agent_name: str) -> AgentSpec:
    """Returns the specification for a registered specialist."""
    try:
        return AGENT_SPECS[agent_name]
    except KeyError as exc:
        available = ", ".join(sorted(AGENT_SPECS))
        raise ValueError(
            f"Unknown agent '{agent_name}'. Available agents: {available}"
        ) from exc
