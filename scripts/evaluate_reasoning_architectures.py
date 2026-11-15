from __future__ import annotations

import argparse

import getpass
import json
import math
import os
import re
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import Client, evaluate

from bi_multiagent.agents.plan_execute_graph import build_plan_execute_graph
from bi_multiagent.agents.single_agent_graph import build_single_agent_graph

DATASET_NAME = "bi-multiagent-reasoning-benchmark-v1"

EXAMPLES = [
    {
        "inputs": {
            "case_id": "sales_total",
            "question": "Qual é a receita líquida total no banco de vendas?",
        },
        "outputs": {
            "expected_terms": ["receita liquida"],
            "expected_numbers": [43270.0],
            "required_tool": "query_sales_database",
            "requires_context": False,
            "max_tool_steps": 4,
        },
        "metadata": {"domain": "sales", "difficulty": "basic"},
    },
    {
        "inputs": {
            "case_id": "sales_by_category",
            "question": (
                "Faça um ranking da receita líquida por categoria, "
                "mostrando as categorias e seus valores."
            ),
        },
        "outputs": {
            "expected_terms": ["eletronicos", "moveis", "acessorios"],
            "expected_numbers": [31255.0, 7650.0, 4365.0],
            "required_tool": "query_sales_database",
            "requires_context": False,
            "max_tool_steps": 4,
        },
        "metadata": {"domain": "sales", "difficulty": "intermediate"},
    },
    {
        "inputs": {
            "case_id": "business_formula",
            "question": (
                "De acordo com a documentação de negócio do projeto, "
                "qual é a fórmula de receita líquida (net revenue)?"
            ),
        },
        "outputs": {
            "expected_terms": ["receita bruta", "descontos"],
            "expected_numbers": [],
            "required_tool": None,
            "requires_context": True,
            "max_tool_steps": 2,
        },
        "metadata": {"domain": "business_knowledge", "difficulty": "basic"},
    },
    {
        "inputs": {
            "case_id": "sales_schema",
            "question": (
                "Consulte o schema real do banco e liste as colunas "
                "disponíveis na tabela sales."
            ),
        },
        "outputs": {
            "expected_terms": [
                "sale_id",
                "sale_date",
                "product",
                "category",
                "region",
                "quantity",
                "unit_price",
                "discount_pct",
            ],
            "expected_numbers": [],
            "required_tool": None,
            "requires_context": True,
            "max_tool_steps": 2,
        },
        "metadata": {"domain": "data_schema", "difficulty": "basic"},
    },
]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def number_is_present(text: str, expected: float) -> bool:
    tokens = re.findall(r"(?<!\w)[+-]?\d[\d.,]*(?!\w)", str(text))

    for token in tokens:
        normalized_candidates = set()
        unsigned = token.lstrip("+-")
        sign = -1 if token.startswith("-") else 1

        if "." in unsigned and "," in unsigned:
            if unsigned.rfind(",") > unsigned.rfind("."):
                normalized_candidates.add(unsigned.replace(".", "").replace(",", "."))
            else:
                normalized_candidates.add(unsigned.replace(",", ""))

        elif unsigned.count(".") > 1:
            groups = unsigned.split(".")
            normalized_candidates.add("".join(groups))

            if len(groups[-1]) in (1, 2):
                normalized_candidates.add("".join(groups[:-1]) + "." + groups[-1])

        elif unsigned.count(",") > 1:
            groups = unsigned.split(",")
            normalized_candidates.add("".join(groups))

            if len(groups[-1]) in (1, 2):
                normalized_candidates.add("".join(groups[:-1]) + "." + groups[-1])

        elif "," in unsigned:
            left, right = unsigned.rsplit(",", 1)

            if len(right) in (1, 2):
                normalized_candidates.add(left.replace(".", "") + "." + right)
            elif len(right) == 3:
                normalized_candidates.add(left + right)
            else:
                normalized_candidates.add(unsigned.replace(",", ""))

        elif "." in unsigned:
            left, right = unsigned.rsplit(".", 1)
            normalized_candidates.add(unsigned)

            if len(right) == 3:
                normalized_candidates.add(left + right)

        else:
            normalized_candidates.add(unsigned)

        candidates = set()

        for candidate in normalized_candidates:
            try:
                candidates.add(sign * float(candidate))
            except ValueError:
                continue

        if any(
            math.isclose(
                value,
                expected,
                rel_tol=1e-5,
                abs_tol=0.01,
            )
            for value in candidates
        ):
            return True

    return False


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content or "")


def term_is_present(term: str, normalized_answer: str) -> bool:
    normalized_term = normalize_text(term)

    if normalized_term in normalized_answer:
        return True

    if normalized_term in {"desconto", "descontos"}:
        words = set(re.findall("[a-z0-9_]+", normalized_answer))
        return bool(words.intersection({"desconto", "descontos"}))

    return False


def response_fidelity(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    answer = normalize_text(outputs.get("answer", ""))
    expected_terms = reference_outputs.get("expected_terms", [])
    expected_numbers = reference_outputs.get("expected_numbers", [])

    checks = []
    missing = []

    for term in expected_terms:
        found = term_is_present(term, answer)
        checks.append(found)
        if not found:
            missing.append(f"termo: {term}")

    original_answer = outputs.get("answer", "")
    for number in expected_numbers:
        found = number_is_present(original_answer, float(number))
        checks.append(found)
        if not found:
            missing.append(f"valor: {number}")

    if not checks:
        score = 1.0
    else:
        score = sum(checks) / len(checks)

    return {
        "key": "response_fidelity",
        "score": score,
        "comment": (
            f"Case={inputs.get('case_id')}; "
            f"checks={sum(checks)}/{len(checks)}; "
            f"missing={missing}"
        ),
    }


def step_execution(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    trajectory = outputs.get("trajectory", {})
    metrics = outputs.get("metrics", {})

    tool_names = trajectory.get("tool_names", [])
    failed_steps = int(metrics.get("failed_step_count", 0))
    required_tool = reference_outputs.get("required_tool")
    requires_context = bool(reference_outputs.get("requires_context"))

    checks = []
    details = []

    if required_tool:
        found = required_tool in tool_names
        checks.append(found)
        details.append(f"required_tool={required_tool}:{found}")

    if requires_context:
        found = int(metrics.get("retrieval_context_chars", 0)) > 0
        checks.append(found)
        details.append(f"context_retrieved={found}")

    no_failures = failed_steps == 0
    checks.append(no_failures)
    details.append(f"failed_steps={failed_steps}")

    tool_count = int(metrics.get("tool_step_count", 0))
    max_steps = int(reference_outputs.get("max_tool_steps", 99))
    within_budget = tool_count <= max_steps
    checks.append(within_budget)
    details.append(f"tool_steps={tool_count}/{max_steps}")

    score = sum(checks) / len(checks) if checks else 1.0

    return {
        "key": "step_execution",
        "score": score,
        "comment": f"Case={inputs.get('case_id')}; " + "; ".join(details),
    }


def trajectory_completeness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    trajectory = outputs.get("trajectory", {})
    metrics = outputs.get("metrics", {})

    steps = trajectory.get("tool_steps", [])
    tool_names = trajectory.get("tool_names", [])
    node_sequence = trajectory.get("node_sequence", [])

    required_tool = reference_outputs.get("required_tool")
    requires_context = bool(reference_outputs.get("requires_context"))

    checks = {
        "answer_nonempty": bool(str(outputs.get("answer", "")).strip()),
        "ends_at_END": bool(node_sequence) and node_sequence[-1] == "END",
        "tool_statuses_ok": all(
            step.get("status") in {"completed", "success"} for step in steps
        ),
        "no_failed_steps": int(metrics.get("failed_step_count", 0)) == 0,
    }

    if required_tool:
        checks["required_tool_used"] = required_tool in tool_names

    if requires_context:
        checks["context_retrieved"] = int(metrics.get("retrieval_context_chars", 0)) > 0

    passed = sum(checks.values())
    total = len(checks)
    score = passed / total if total else 1.0

    return {
        "key": "trajectory_completeness",
        "score": score,
        "comment": (
            f"Case={inputs.get('case_id')}; "
            f"checks={passed}/{total}; "
            f"check_details={checks}; "
            f"tool_sequence={tool_names}; "
            f"nodes={node_sequence}"
        ),
    }


def build_target(architecture: str, graph: Any):
    def target(inputs: dict) -> dict:
        question = str(inputs["question"])
        thread_id = f"benchmark-{architecture}-{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}

        started = time.perf_counter()

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            return {
                "answer": "",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "trajectory": {
                    "tool_steps": [],
                    "tool_names": [],
                    "node_sequence": ["ERROR"],
                },
                "metrics": {
                    "latency_seconds": round(elapsed, 3),
                    "tool_step_count": 0,
                    "failed_step_count": 1,
                    "retrieval_context_chars": 0,
                    "replan_count": 0,
                },
            }

        elapsed = time.perf_counter() - started

        if architecture == "plan_execute":
            answer = str(result.get("final_answer", "") or "")
            observations = result.get("observations", [])

            tool_steps = [
                {
                    "tool": item.get("tool", "unknown_tool"),
                    "status": item.get("status", "unknown"),
                    "reason": item.get("reason", ""),
                }
                for item in observations
            ]

            replan_count = int(result.get("replan_count", 0))
            node_sequence = ["retrieve_context", "planner"]
            node_sequence.extend(["execute_step"] * len(tool_steps))

            if tool_steps:
                node_sequence.extend(["replan"] * (replan_count + 1))

            node_sequence.extend(["finalize", "END"])

        else:
            messages = result.get("messages", [])
            answer = ""

            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    candidate = extract_message_text(message).strip()
                    if candidate:
                        answer = candidate
                        break

            tool_steps = []
            for message in messages:
                if isinstance(message, ToolMessage):
                    tool_steps.append(
                        {
                            "tool": getattr(message, "name", None) or "unknown_tool",
                            "status": getattr(message, "status", None) or "completed",
                            "reason": "",
                        }
                    )

            replan_count = 0
            node_sequence = ["retrieve_context", "agent"]

            for _ in tool_steps:
                node_sequence.extend(["ToolNode", "agent"])

            node_sequence.append("END")

        evidence = result.get("evidence", "")
        context_chars = len(evidence) if isinstance(evidence, str) else 0

        tool_names = [step["tool"] for step in tool_steps]
        failed_steps = sum(
            1 for step in tool_steps if step["status"] not in {"completed", "success"}
        )

        return {
            "answer": answer,
            "error": None,
            "trajectory": {
                "tool_steps": tool_steps,
                "tool_names": tool_names,
                "node_sequence": node_sequence,
            },
            "metrics": {
                "latency_seconds": round(elapsed, 3),
                "tool_step_count": len(tool_steps),
                "failed_step_count": failed_steps,
                "retrieval_context_chars": context_chars,
                "replan_count": replan_count,
            },
        }

    return target


def configure_langsmith() -> Client:
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    if not api_key:
        api_key = getpass.getpass(
            "LANGSMITH_API_KEY ausente. Cole sua chave (entrada oculta): "
        ).strip()

    if not api_key:
        raise RuntimeError(
            "A chave da API LangSmith é obrigatória para publicar o dataset "
            "e os experimentos."
        )

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"

    if not os.getenv("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "bi-multiagent-reasoning"

    return Client()


def get_or_create_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        existing = list(client.list_examples(dataset_id=dataset.id, limit=100))

        expected_ids = {item["inputs"]["case_id"] for item in EXAMPLES}
        actual_ids = {(item.inputs or {}).get("case_id") for item in existing}

        if not existing:
            client.create_examples(
                dataset_id=dataset.id,
                examples=EXAMPLES,
            )
        elif actual_ids != expected_ids:
            raise RuntimeError(
                "O dataset já existe, mas os exemplos não correspondem "
                "à versão esperada. Não será sobrescrito automaticamente."
            )

        return dataset

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Benchmark comparativo ReAct vs Plan-and-Execute. "
            "Quatro casos de vendas, conhecimento de negócio e schema."
        ),
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES,
    )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selected reasoning architectures."
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("react", "plan_execute"),
        default=["react", "plan_execute"],
        help="Architectures to evaluate.",
    )
    args = parser.parse_args()

    print("=== LANGSMITH: CONFIGURACAO ===")
    client = configure_langsmith()

    print("\n=== DATASET ===")
    dataset = get_or_create_dataset(client)
    print(f"Dataset: {dataset.name}")
    print(f"Dataset ID: {dataset.id}")
    print(f"Exemplos: {len(EXAMPLES)}")

    print("\n=== CONSTRUINDO GRAFOS ===")
    react_graph = build_single_agent_graph()
    plan_graph = build_plan_execute_graph()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiments = [
        (
            "react",
            build_target("react", react_graph),
            "ReAct",
        ),
        (
            "plan_execute",
            build_target("plan_execute", plan_graph),
            "Plan-and-Execute",
        ),
    ]

    experiments = [
        experiment
        for experiment in experiments
        if experiment[0] in args.architectures
    ]

    print("Arquiteturas selecionadas:", ", ".join(args.architectures))

    experiment_records = []

    evaluators = [
        response_fidelity,
        step_execution,
        trajectory_completeness,
    ]

    for architecture, target, label in experiments:
        prefix = f"{architecture}-{timestamp}"

        print(f"\n=== AVALIACAO: {label} ===")
        print("Dataset compartilhado:", DATASET_NAME)
        print("Concorrência: 1")

        results = evaluate(
            target,
            data=DATASET_NAME,
            evaluators=evaluators,
            experiment_prefix=prefix,
            description=(
                f"Benchmark {label}; mesmo dataset e mesmo modelo "
                "local configurado no projeto."
            ),
            metadata={
                "architecture": label,
                "benchmark_version": "v1",
                "model_provider": "Ollama",
                "execution_mode": "sequential",
            },
            max_concurrency=1,
            num_repetitions=1,
            client=client,
            blocking=True,
        )

        print("\nResultado do experimento:")
        print(results)

        experiment_records.append(
            {
                "architecture": label,
                "experiment_prefix": prefix,
                "result": str(results),
            }
        )

    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": DATASET_NAME,
        "dataset_id": str(dataset.id),
        "experiment_records": experiment_records,
        "evaluators": [
            "response_fidelity",
            "step_execution",
            "trajectory_completeness",
        ],
    }

    manifest_filename = (
        "reasoning_benchmark_manifest_plan_execute.json"
        if args.architectures == ["plan_execute"]
        else "reasoning_benchmark_manifest.json"
    )
    output_path = output_dir / manifest_filename
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== BENCHMARK CONCLUIDO ===")
    print("Manifesto local:", output_path)
    print("Dataset e experimentos publicados no LangSmith.")
    print("Abra os experimentos para comparar as três métricas.")


if __name__ == "__main__":
    main()
