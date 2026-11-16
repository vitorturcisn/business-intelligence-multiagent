import sqlite3

from bi_multiagent.data.seed import seed_sales_database
from scripts.evaluate_reasoning_architectures import (
    build_benchmark_examples,
    examples_match_reference,
    read_sales_reference,
    response_fidelity,
)


def test_benchmark_references_are_calculated_from_current_database(tmp_path):
    database = tmp_path / "sales.db"
    seed_sales_database(database)

    reference = read_sales_reference(database)
    with sqlite3.connect(database) as connection:
        independent_total = connection.execute(
            """
            SELECT ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 2)
            FROM sales
            """
        ).fetchone()[0]

    assert reference["row_count"] > 0
    assert reference["distinct_orders"] == 500
    assert reference["net_revenue"] == independent_total
    assert reference["fingerprint_sha256"]


def test_benchmark_examples_use_dynamic_total_and_categories(tmp_path):
    database = tmp_path / "sales.db"
    seed_sales_database(database)

    reference = read_sales_reference(database)
    examples = build_benchmark_examples(reference)
    by_case = {example["inputs"]["case_id"]: example for example in examples}

    assert set(by_case) == {
        "sales_total",
        "sales_by_category",
        "business_formula",
        "sales_schema",
    }
    assert by_case["sales_total"]["outputs"]["expected_numbers"] == [
        reference["net_revenue"]
    ]
    assert (
        by_case["sales_by_category"]["outputs"]["expected_category_values"]
        == reference["category_net_revenue"]
    )
    assert all(
        example["metadata"]["database_fingerprint_sha256"]
        == reference["fingerprint_sha256"]
        for example in examples
    )


def test_response_fidelity_scores_dynamic_category_values():
    reference = {
        "expected_terms": ["receita liquida"],
        "expected_numbers": [],
        "expected_category_values": {
            "Acessórios": 1000.0,
            "Eletrônicos": 2000.0,
        },
    }
    result = response_fidelity(
        inputs={"case_id": "sales_by_category"},
        outputs={
            "answer": (
                "Receita líquida: Acessórios R$ 1.000,00; "
                "Eletrônicos R$ 2.000,00."
            )
        },
        reference_outputs=reference,
    )

    assert result["score"] == 1.0
    assert "missing=[]" in result["comment"]


def test_dataset_reference_fingerprint_detects_database_change(tmp_path):
    database = tmp_path / "sales.db"
    seed_sales_database(database)
    original_reference = read_sales_reference(database)
    original_examples = build_benchmark_examples(original_reference)

    class Example:
        def __init__(self, item):
            self.inputs = item["inputs"]
            self.outputs = item["outputs"]
            self.metadata = item["metadata"]

    existing = [Example(item) for item in original_examples]
    assert examples_match_reference(existing, original_examples)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE sales SET unit_price = unit_price + 1 WHERE rowid = 1"
        )

    changed_reference = read_sales_reference(database)
    changed_examples = build_benchmark_examples(changed_reference)
    assert not examples_match_reference(existing, changed_examples)
