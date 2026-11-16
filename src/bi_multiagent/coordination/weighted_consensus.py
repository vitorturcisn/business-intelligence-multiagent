"""Deterministic, domain-weighted evidence aggregation for specialist agents.

This is an evidence-support score, not a calibrated probability and not a
majority vote over answers. The Sales, Knowledge, and Economic agents provide
complementary evidence, so their outputs must not be treated as interchangeable
votes on the same claim.
"""

from __future__ import annotations

from typing import Any, Iterable

from bi_multiagent.schemas.contracts import AgentResult


# Transparent policy weights reflecting each specialist's primary domain.
# These are configurable policy constants, not weights learned from data.
DEFAULT_AGENT_WEIGHTS: dict[str, float] = {
    "sales": 1.20,
    "knowledge": 1.00,
    "economic": 1.10,
}
DEFAULT_ACCEPTANCE_THRESHOLD = 0.80


def _quality_score(result: AgentResult) -> float:
    """Score evidence completeness for one specialist result in [0, 1]."""
    status_factor = {
        "completed": 1.0,
        "needs_input": 0.5,
        "failed": 0.0,
    }[result.status]

    evidence_factor = 1.0 if result.evidence.strip() else 0.0
    source_factor = 1.0 if result.sources else 0.0
    completeness = 0.50 + 0.25 * evidence_factor + 0.25 * source_factor
    warning_penalty = max(0.50, 1.0 - 0.10 * len(result.warnings))

    return round(status_factor * completeness * warning_penalty, 4)


def weighted_evidence_consensus(
    results: Iterable[AgentResult | dict[str, Any]],
    expected_agents: Iterable[str],
    *,
    weights: dict[str, float] | None = None,
    threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> dict[str, Any]:
    """Aggregate relevant agent evidence using domain weights.

    Missing or failed specialists contribute zero. Agents not selected by the
    Supervisor are ignored. The denominator includes every expected agent, so
    an omitted specialist cannot be hidden by averaging only successful runs.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    selected = list(dict.fromkeys(str(agent) for agent in expected_agents))
    active_weights = dict(DEFAULT_AGENT_WEIGHTS)
    if weights:
        for agent, weight in weights.items():
            if weight <= 0:
                raise ValueError(f"Weight for '{agent}' must be positive.")
            active_weights[agent] = float(weight)

    normalized_results = [AgentResult.model_validate(item) for item in results]
    by_agent: dict[str, AgentResult] = {}
    duplicate_agents: list[str] = []
    for result in normalized_results:
        if result.agent not in selected:
            continue
        if result.agent in by_agent:
            duplicate_agents.append(result.agent)
            if _quality_score(result) < _quality_score(by_agent[result.agent]):
                by_agent[result.agent] = result
        else:
            by_agent[result.agent] = result

    denominator = sum(active_weights.get(agent, 1.0) for agent in selected) or 1.0
    numerator = 0.0
    agent_scores: dict[str, dict[str, Any]] = {}

    for agent in selected:
        weight = active_weights.get(agent, 1.0)
        result = by_agent.get(agent)
        quality = _quality_score(result) if result else 0.0
        weighted_support = round(weight * quality, 4)
        numerator += weighted_support
        agent_scores[agent] = {
            "weight": round(weight, 4),
            "status": result.status if result else "missing",
            "quality_score": quality,
            "weighted_support": weighted_support,
            "evidence_present": bool(result and result.evidence.strip()),
            "source_count": len(result.sources) if result else 0,
            "warning_count": len(result.warnings) if result else 0,
        }

    score = round(numerator / denominator, 4)
    observed_agents = [agent for agent in selected if agent in by_agent]
    missing_agents = [agent for agent in selected if agent not in by_agent]
    statuses = [result.status for result in by_agent.values()]
    reasons: list[str] = []

    if not selected or not observed_agents:
        status = "insufficient_evidence"
        reasons.append("No expected specialist produced a result.")
    elif all(
        entry["quality_score"] == 0.0 for entry in agent_scores.values()
    ):
        status = "insufficient_evidence"
        reasons.append("All expected specialist results have zero evidence support.")
    elif "needs_input" in statuses and "failed" not in statuses:
        status = "needs_input"
        reasons.append("At least one specialist requires additional user input.")
    elif missing_agents or "failed" in statuses or score < threshold:
        status = "review_required"
        if missing_agents:
            reasons.append("Missing specialists: " + ", ".join(missing_agents) + ".")
        if "failed" in statuses:
            reasons.append("At least one specialist failed.")
        if score < threshold:
            reasons.append(
                f"Evidence support {score:.2f} is below threshold {threshold:.2f}."
            )
    else:
        status = "accepted"
        reasons.append("All selected specialists returned sufficient sourced evidence.")

    if duplicate_agents:
        status = "review_required"
        reasons.append(
            "Duplicate specialist results detected: "
            + ", ".join(sorted(set(duplicate_agents)))
            + "."
        )

    return {
        "method": "domain_weighted_evidence_v1",
        "status": status,
        "evidence_support_score": score,
        "acceptance_threshold": threshold,
        "expected_agents": selected,
        "observed_agents": observed_agents,
        "missing_agents": missing_agents,
        "agent_scores": agent_scores,
        "reasons": reasons,
    }
