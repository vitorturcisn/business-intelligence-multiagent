"""Deterministic row-level evaluators for end-to-end Supervisor runs."""

from __future__ import annotations

from typing import Any


def routing_fidelity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = set(reference_outputs.get("expected_agents", []))
    actual = set(outputs.get("selected_agents", []))
    passed = expected == actual
    return {
        "key": "supervisor_routing_fidelity",
        "score": 1.0 if passed else 0.0,
        "comment": f"expected={sorted(expected)} actual={sorted(actual)}",
    }


def hitl_policy_fidelity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = bool(reference_outputs.get("expected_approval_pending", False))
    actual = bool(outputs.get("approval_pending", False))
    passed = expected == actual
    return {
        "key": "supervisor_hitl_policy_fidelity",
        "score": 1.0 if passed else 0.0,
        "comment": f"expected_approval_pending={expected}; actual={actual}",
    }


def instruction_fidelity(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    audit = outputs.get("instruction_fidelity") or {}
    status = str(audit.get("status", "missing"))
    score = float(audit.get("score", 0.0) or 0.0)
    passed = status == "passed"
    return {
        "key": "instruction_fidelity_audit_passed",
        "score": 1.0 if passed else 0.0,
        "comment": f"status={status}; score={score:.4f}; failed_checks={audit.get('failed_checks', [])}",
    }


def weighted_consensus(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    consensus = outputs.get("weighted_consensus") or {}
    status = str(consensus.get("status", "missing"))
    score = float(consensus.get("evidence_support_score", 0.0) or 0.0)
    passed = status == "accepted"
    return {
        "key": "weighted_evidence_consensus_accepted",
        "score": 1.0 if passed else 0.0,
        "comment": f"status={status}; evidence_support_score={score:.4f}; missing_agents={consensus.get('missing_agents', [])}",
    }


def discount_fraction_regression(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    if inputs.get("case_id") != "discount_pct_fraction_regression":
        return {"key": "discount_pct_fraction_regression", "score": 1.0, "comment": "not_applicable"}

    evidence = "\n".join(outputs.get("specialist_evidence", [])).casefold()
    expected_formula = "1 - discount_pct" in evidence or "(1 - discount_pct)" in evidence
    wrong_divisor = "discount_pct / 100" in evidence or "discount_pct/100" in evidence
    passed = expected_formula and not wrong_divisor
    return {
        "key": "discount_pct_fraction_semantics",
        "score": 1.0 if passed else 0.0,
        "comment": f"formula_present={expected_formula}; wrong_division_by_100={wrong_divisor}",
    }
