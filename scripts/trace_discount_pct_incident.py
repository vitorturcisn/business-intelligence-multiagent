from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

QUESTION = (
    "Calcule a receita líquida por categoria e mostre a fórmula documentada "
    "para discount_pct."
)
EXPECTED_FORMULA = "1 - discount_pct"


def main() -> int:
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        raise RuntimeError("Configure LANGSMITH_API_KEY no .env antes de criar o trace.")

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "business-intelligence-supervisor")

    from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph

    thread_id = f"discount-pct-regression-{uuid.uuid4().hex[:12]}"
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["incident", "regression", "discount_pct", "supervisor"],
        "metadata": {
            "application": "business-intelligence-multiagent",
            "component": "supervisor",
            "scenario": "discount_pct_corrected_regression",
            "expected_formula": EXPECTED_FORMULA,
        },
    }

    graph = build_multiagent_graph(use_in_memory_saver=True)
    result = graph.invoke(
        {"messages": [HumanMessage(content=QUESTION)]},
        config=config,
    )
    snapshot = graph.get_state(config)
    state = dict(snapshot.values or result or {})

    request_id = state.get("request_id")
    specialist_results = [
        item
        for item in state.get("agent_results", [])
        if item.get("request_id") == request_id
    ]
    evidence = "\n\n".join(str(item.get("evidence", "")) for item in specialist_results)
    normalized = evidence.casefold()
    formula_present = EXPECTED_FORMULA in normalized
    wrong_division = (
        "discount_pct / 100" in normalized
        or "discount_pct/100" in normalized
        or "discount_pct /100" in normalized
    )
    selected_agents = list(state.get("route", []))
    consensus = state.get("weighted_consensus") or {}
    fidelity = state.get("instruction_fidelity") or {}

    checks = {
        "sales_selected": "sales" in selected_agents,
        "knowledge_selected": "knowledge" in selected_agents,
        "correct_fraction_formula_present": formula_present,
        "wrong_division_by_100_absent": not wrong_division,
        "weighted_consensus_accepted": consensus.get("status") == "accepted",
        "instruction_fidelity_passed": fidelity.get("status") == "passed",
    }
    passed = all(checks.values())

    output_dir = ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_payload = {
        "created_at": datetime.now().isoformat(),
        "langsmith_project": os.getenv(
            "LANGSMITH_PROJECT", "business-intelligence-supervisor"
        ),
        "thread_id": thread_id,
        "request_id": request_id,
        "question": QUESTION,
        "selected_agents": selected_agents,
        "checks": checks,
        "passed": passed,
        "weighted_consensus": consensus,
        "instruction_fidelity": fidelity,
        "specialist_statuses": {
            item.get("agent"): item.get("status") for item in specialist_results
        },
        "specialist_evidence": [
            str(item.get("evidence", "")) for item in specialist_results
        ],
        "answer": str(state.get("answer", "")),
        "note": (
            "Este é um novo trace da execução corrigida; não recria o trace "
            "histórico de uma eventual execução defeituosa."
        ),
    }
    path = output_dir / "discount_pct_incident_trace_evidence.json"
    path.write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Trace de regressão criado.")
    print("Projeto LangSmith:", evidence_payload["langsmith_project"])
    print("Thread:", thread_id)
    print("Evidência:", path)
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'FALHOU'}: {name}")
    print("Resultado:", "PASSOU" if passed else "REVISÃO NECESSÁRIA")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
