from bi_multiagent.observability.supervisor_evaluators import (
    discount_fraction_regression,
    hitl_policy_fidelity,
    instruction_fidelity,
    routing_fidelity,
    weighted_consensus,
)


def test_routing_fidelity_exact_match():
    result = routing_fidelity({}, {"selected_agents": ["sales", "knowledge"]}, {"expected_agents": ["sales", "knowledge"]})
    assert result["score"] == 1.0


def test_routing_fidelity_flags_missing_agent():
    result = routing_fidelity({}, {"selected_agents": ["knowledge"]}, {"expected_agents": ["sales", "knowledge"]})
    assert result["score"] == 0.0


def test_hitl_policy_requires_pending_interrupt_for_export():
    result = hitl_policy_fidelity({}, {"approval_pending": True}, {"expected_approval_pending": True})
    assert result["score"] == 1.0


def test_instruction_fidelity_requires_passed_status():
    assert instruction_fidelity({}, {"instruction_fidelity": {"status": "passed", "score": 1.0}}, {})["score"] == 1.0
    assert instruction_fidelity({}, {"instruction_fidelity": {"status": "review_required", "score": 0.8}}, {})["score"] == 0.0


def test_weighted_consensus_requires_accepted():
    assert weighted_consensus({}, {"weighted_consensus": {"status": "accepted", "evidence_support_score": 1.0}}, {})["score"] == 1.0
    assert weighted_consensus({}, {"weighted_consensus": {"status": "review_required", "evidence_support_score": 0.6}}, {})["score"] == 0.0


def test_discount_fraction_evaluator_checks_formula_not_percent_division():
    inputs = {"case_id": "discount_pct_fraction_regression"}
    good = {"specialist_evidence": ["SUM(quantity * unit_price * (1 - discount_pct))"]}
    bad = {"specialist_evidence": ["SUM(quantity * unit_price * (1 - discount_pct / 100))"]}
    assert discount_fraction_regression(inputs, good, {})["score"] == 1.0
    assert discount_fraction_regression(inputs, bad, {})["score"] == 0.0
