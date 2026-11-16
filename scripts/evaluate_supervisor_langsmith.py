from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DATASET_NAME = "bi-multiagent-supervisor-evaluation-v1"
EXAMPLES = [
    {
        "inputs": {"case_id": "sales_net_revenue_total", "question": "Qual é a receita líquida total?"},
        "outputs": {"expected_agents": ["sales"], "expected_approval_pending": False},
        "metadata": {"domain": "sales", "difficulty": "basic"},
    },
    {
        "inputs": {"case_id": "knowledge_gross_vs_net", "question": "Explique a diferença entre receita bruta e receita líquida."},
        "outputs": {"expected_agents": ["knowledge"], "expected_approval_pending": False},
        "metadata": {"domain": "business_knowledge", "difficulty": "basic"},
    },
    {
        "inputs": {"case_id": "sales_and_knowledge", "question": "Calcule a receita por categoria (líquida) e apresente a fórmula de negócio para discount_pct."},
        "outputs": {"expected_agents": ["sales", "knowledge"], "expected_approval_pending": False},
        "metadata": {"domain": "sales_and_knowledge", "difficulty": "intermediate"},
    },
    {
        "inputs": {"case_id": "selic_period", "question": "Qual foi a meta Selic entre 01/10/2026 e 10/11/2026?"},
        "outputs": {"expected_agents": ["economic"], "expected_approval_pending": False},
        "metadata": {"domain": "economic", "difficulty": "intermediate"},
    },
    {
        "inputs": {"case_id": "discount_pct_fraction_regression", "question": "Calcule a receita por categoria (líquida) e mostre a fórmula do dicionário de dados para discount_pct."},
        "outputs": {"expected_agents": ["sales", "knowledge"], "expected_approval_pending": False},
        "metadata": {"domain": "discount_regression", "difficulty": "critical"},
    },
    {
        "inputs": {"case_id": "explicit_export_requires_hitl", "question": "Analise os indicadores de vendas documentados e salve o relatório em arquivo."},
        "outputs": {"expected_agents": ["knowledge"], "expected_approval_pending": True},
        "metadata": {"domain": "human_approval", "difficulty": "critical"},
    },
]


def configure_client() -> Client:
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not key:
        key = getpass.getpass("LANGSMITH_API_KEY (entrada oculta): ").strip()
    if not key:
        raise RuntimeError("LANGSMITH_API_KEY não foi informada.")
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "business-intelligence-supervisor")
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID", "").strip() or None
    return Client(api_key=key, workspace_id=workspace_id)


def ensure_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id, limit=100))
        by_id = {(getattr(item, "inputs", None) or {}).get("case_id"): item for item in existing}
        expected = {item["inputs"]["case_id"]: item for item in EXAMPLES}
        if set(by_id) != set(expected):
            raise RuntimeError(
                f"Dataset {DATASET_NAME} já existe com conjunto diferente. "
                "Incremente DATASET_NAME no script em vez de sobrescrever histórico."
            )
        for case_id, example in expected.items():
            actual = by_id[case_id]
            if getattr(actual, "inputs", None) != example["inputs"] or getattr(actual, "outputs", None) != example["outputs"]:
                raise RuntimeError(
                    f"Dataset {DATASET_NAME} diverge no caso {case_id}. "
                    "Crie uma nova versão com nome novo."
                )
        return dataset

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Avaliação ponta a ponta do Supervisor BI Multi-Agent: roteamento, "
            "consenso ponderado, Instruction Fidelity Audit, HITL e regressão de discount_pct."
        ),
    )
    client.create_examples(dataset_id=dataset.id, examples=EXAMPLES)
    return dataset


def build_target(graph):
    def target(inputs: dict) -> dict:
        question = str(inputs["question"])
        case_id = str(inputs.get("case_id", "unknown"))
        thread_id = f"langsmith-supervisor-{case_id}-{uuid.uuid4().hex}"
        config = {
            "configurable": {"thread_id": thread_id},
            "tags": ["supervisor", "multi-agent", "phase6", case_id],
            "metadata": {
                "application": "business-intelligence-multiagent",
                "component": "supervisor",
                "evaluation_suite": DATASET_NAME,
                "case_id": case_id,
            },
        }
        result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        snapshot = graph.get_state(config)
        state = dict(snapshot.values or {})
        interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
        approval_pending = bool(interrupts) or "human_approval" in tuple(snapshot.next or ())
        current_id = state.get("request_id")
        agent_results = [
            item for item in state.get("agent_results", [])
            if item.get("request_id") == current_id
        ]
        return {
            "answer": str(state.get("answer", result.get("answer", "") if isinstance(result, dict) else "")),
            "selected_agents": list(state.get("route", [])),
            "route_method": state.get("route_method", ""),
            "audit": state.get("audit", {}),
            "weighted_consensus": state.get("weighted_consensus", {}),
            "instruction_fidelity": state.get("instruction_fidelity", {}),
            "approval_required": bool(state.get("approval_required", False)),
            "approval_pending": approval_pending,
            "specialist_statuses": {item.get("agent"): item.get("status") for item in agent_results},
            "specialist_evidence": [str(item.get("evidence", "")) for item in agent_results],
            "thread_id": thread_id,
            "request_id": current_id,
        }
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia o Supervisor integrado no LangSmith.")
    parser.add_argument("--dry-run", action="store_true", help="Valida os casos e não chama LangSmith/Ollama.")
    args = parser.parse_args()
    print("Dataset:", DATASET_NAME, "| casos:", len(EXAMPLES))
    if args.dry_run:
        print(json.dumps(EXAMPLES, ensure_ascii=False, indent=2))
        print("DRY-RUN OK: nenhum trace foi enviado.")
        return 0

    from bi_multiagent.coordination.supervisor_graph import build_multiagent_graph
    from bi_multiagent.observability.supervisor_evaluators import (
        discount_fraction_regression,
        hitl_policy_fidelity,
        instruction_fidelity,
        routing_fidelity,
        weighted_consensus,
    )

    client = configure_client()
    dataset = ensure_dataset(client)
    graph = build_multiagent_graph(use_in_memory_saver=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    project = os.environ["LANGSMITH_PROJECT"]
    print("LangSmith project:", project)
    print("Dataset ID:", dataset.id)

    results = evaluate(
        build_target(graph),
        data=DATASET_NAME,
        evaluators=[
            routing_fidelity,
            hitl_policy_fidelity,
            instruction_fidelity,
            weighted_consensus,
            discount_fraction_regression,
        ],
        experiment_prefix=f"supervisor-e2e-{timestamp}",
        description="Supervisor E2E: routing, HITL, weighted consensus, instruction fidelity and discount_pct regression.",
        metadata={
            "system": "business-intelligence-multiagent",
            "architecture": "hierarchical-supervisor",
            "dataset_version": "v1",
            "tracing_project": project,
            "model_provider": "Ollama",
        },
        max_concurrency=1,
        num_repetitions=1,
        client=client,
        blocking=True,
    )

    output_dir = ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": DATASET_NAME,
        "dataset_id": str(dataset.id),
        "tracing_project": project,
        "experiment_prefix": f"supervisor-e2e-{timestamp}",
        "experiment_result": str(results),
        "cases": [item["inputs"]["case_id"] for item in EXAMPLES],
        "evaluators": [
            "supervisor_routing_fidelity",
            "supervisor_hitl_policy_fidelity",
            "instruction_fidelity_audit_passed",
            "weighted_evidence_consensus_accepted",
            "discount_pct_fraction_semantics",
        ],
        "note": "Open the LangSmith experiment and inspect root + child runs for trace evidence.",
    }
    path = output_dir / "supervisor_langsmith_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSupervisor evaluation finished.")
    print("Manifest:", path)
    print("Experiment result:", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
