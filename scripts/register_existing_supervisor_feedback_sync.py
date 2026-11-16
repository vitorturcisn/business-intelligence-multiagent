from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DATASET_NAME = "bi-multiagent-supervisor-evaluation-v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verifica e registra feedbacks determinísticos em runs existentes "
            "do experimento LangSmith, sem chamar o modelo."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Nome exato do experimento/projeto LangSmith (ex.: supervisor-e2e-...).",
    )
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        raise RuntimeError("LANGSMITH_API_KEY não está configurada no .env.")

    client = Client(
        api_key=os.environ["LANGSMITH_API_KEY"],
        workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
    )
    project = client.read_project(project_name=args.project)
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    examples = list(client.list_examples(dataset_id=dataset.id, limit=100))
    examples_by_id = {
        (item.inputs or {}).get("case_id"): item for item in examples
    }

    runs = list(client.list_runs(project_id=project.id, is_root=True))
    if not runs:
        raise RuntimeError(f"Nenhuma execução raiz encontrada no projeto {args.project}.")

    matched = {}
    for run in runs:
        reference_id = getattr(run, "reference_example_id", None)
        if reference_id is None:
            continue
        example = next(
            (item for item in examples if str(item.id) == str(reference_id)),
            None,
        )
        if example is not None:
            matched[run.id] = (run, example)

    if len(matched) != len(examples):
        raise RuntimeError(
            f"Esperadas {len(examples)} execuções com referência ao dataset; "
            f"encontradas {len(matched)}. Nenhum feedback foi criado."
        )

    from bi_multiagent.observability.supervisor_evaluators import (
        discount_fraction_regression,
        hitl_policy_fidelity,
        instruction_fidelity,
        routing_fidelity,
        weighted_consensus,
    )

    evaluators = (
        routing_fidelity,
        hitl_policy_fidelity,
        instruction_fidelity,
        weighted_consensus,
        discount_fraction_regression,
    )
    run_ids = list(matched)
    existing = list(client.list_feedback(run_ids=run_ids))
    existing_keys = {
        (str(item.run_id), item.key) for item in existing
    }

    signature = inspect.signature(client.create_feedback)
    if "session_id" in signature.parameters:
        session_arg = {"session_id": project.id}
    elif "project_id" in signature.parameters:
        session_arg = {"project_id": project.id}
    else:
        raise RuntimeError(
            "A versão do SDK LangSmith não expõe session_id nem project_id "
            "em create_feedback."
        )

    created_ids = []
    skipped_not_applicable = 0
    expected_counts = Counter()

    for run_id, (run, example) in matched.items():
        inputs = getattr(run, "inputs", None) or example.inputs or {}
        outputs = getattr(run, "outputs", None) or {}
        references = example.outputs or {}

        if not outputs:
            raise RuntimeError(
                f"Run {run_id} não possui outputs; nenhum feedback foi criado."
            )

        for evaluator in evaluators:
            result = evaluator(inputs, outputs, references)
            key = result["key"]
            if result.get("comment") == "not_applicable":
                skipped_not_applicable += 1
                continue

            expected_counts[key] += 1
            if (str(run_id), key) in existing_keys:
                continue

            # Deliberadamente não passar trace_id: esta chamada é síncrona
            # e permite confirmar o feedback antes de encerrar o processo.
            feedback = client.create_feedback(
                run_id=run_id,
                key=key,
                score=float(result["score"]),
                comment=str(result.get("comment", "")),
                source_info={
                    "type": "model",
                    "metadata": {"evaluator": "deterministic_local"},
                },
                **session_arg,
            )
            feedback_id = getattr(feedback, "id", None)
            if feedback_id is None:
                raise RuntimeError(
                    f"LangSmith não retornou ID para feedback {key} / {run_id}."
                )
            created_ids.append(feedback_id)
            existing_keys.add((str(run_id), key))

    for feedback_id in created_ids:
        client.read_feedback(feedback_id)

    verified = list(client.list_feedback(run_ids=run_ids))
    counts = Counter(item.key for item in verified)
    expected_total = sum(expected_counts.values())
    actual_total = sum(
        1 for item in verified
        if (str(item.run_id), item.key) in existing_keys
    )

    print("=== LANGSMITH SUPERVISOR FEEDBACK ===")
    print("Projeto:", args.project)
    print("Dataset:", DATASET_NAME)
    print("Execuções raiz correspondentes:", len(matched))
    print("Feedbacks já existentes antes:", len(existing))
    print("Feedbacks criados agora:", len(created_ids))
    print("Feedbacks lidos individualmente:", len(created_ids))
    print("Casos não aplicáveis ignorados:", skipped_not_applicable)
    print("Feedbacks retornados pela API:", len(verified))
    for key, expected in sorted(expected_counts.items()):
        print(f"{key}: {counts[key]}/{expected}")

    if actual_total < expected_total or len(verified) < expected_total:
        print("RESULTADO: INCOMPLETO")
        return 2

    print(f"RESULTADO: OK ({expected_total} feedbacks esperados)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
