
import sqlite3

from bi_multiagent.data.seed import seed_sales_database


def read_sales_rows(path):
    with sqlite3.connect(path) as connection:
        return connection.execute(
            """
            SELECT
                sale_id,
                sale_date,
                product,
                category,
                region,
                quantity,
                unit_price,
                discount_pct
            FROM sales
            ORDER BY sale_id, sale_date, product, quantity
            """
        ).fetchall()


def test_seed_generates_reproducible_database(tmp_path):
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"

    first_result = seed_sales_database(first_db)
    second_result = seed_sales_database(second_db)

    assert first_result["seed"] == second_result["seed"]
    assert first_result["rows"] == second_result["rows"]
    assert first_result["orders"] == 500

    assert read_sales_rows(first_db) == read_sales_rows(second_db)


def test_seed_does_not_overwrite_existing_database(tmp_path):
    database = tmp_path / "sales.db"

    seed_sales_database(database)

    try:
        seed_sales_database(database)
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "O gerador não deveria sobrescrever o banco sem autorização."
        )
