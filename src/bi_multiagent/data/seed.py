
"""Geração determinística do banco SQLite sintético de vendas."""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from bi_multiagent.settings import DATABASE_PATH

SEED = 20261115
ORDER_COUNT = 500

PRODUCTS = [
    ("Notebook Pro", "Eletrônicos", 4500.00),
    ("Monitor 27", "Eletrônicos", 1600.00),
    ("Mouse sem fio", "Acessórios", 120.00),
    ("Teclado mecânico", "Acessórios", 350.00),
    ("Headset", "Acessórios", 280.00),
    ("Cadeira ergonômica", "Móveis", 1300.00),
    ("Mesa de escritório", "Móveis", 900.00),
    ("Webcam", "Acessórios", 220.00),
]

REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
DISCOUNTS = [0.00, 0.05, 0.10, 0.15, 0.20]


def generate_rows() -> list[tuple]:
    """Gera pedidos e itens com resultado sempre reproduzível."""
    rng = random.Random(SEED)
    start_date = date(2025, 1, 1)
    rows = []

    for order_number in range(1, ORDER_COUNT + 1):
        sale_id = f"ORD-{order_number:05d}"
        sale_date = (
            start_date + timedelta(days=rng.randrange(730))
        ).isoformat()

        region = rng.choice(REGIONS)
        item_count = rng.randint(1, 3)

        for _ in range(item_count):
            product, category, base_price = rng.choice(PRODUCTS)

            quantity = rng.randint(1, 8)
            price_factor = rng.choice([0.90, 1.00, 1.05, 1.10])
            unit_price = round(base_price * price_factor, 2)
            discount_pct = rng.choice(DISCOUNTS)

            rows.append(
                (
                    sale_id,
                    sale_date,
                    product,
                    category,
                    region,
                    quantity,
                    unit_price,
                    discount_pct,
                )
            )

    return rows


def has_sales_schema(database_path: Path) -> bool:
    """Verifica se o arquivo já contém a tabela esperada."""
    if not database_path.is_file():
        return False

    try:
        with sqlite3.connect(database_path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(sales)")
            }

        required = {
            "sale_id",
            "sale_date",
            "product",
            "category",
            "region",
            "quantity",
            "unit_price",
            "discount_pct",
        }

        return required.issubset(columns)

    except sqlite3.Error:
        return False


def seed_sales_database(
    database_path: Path = DATABASE_PATH,
    *,
    force: bool = False,
) -> dict:
    """Cria o banco SQLite. Não sobrescreve dados sem autorização."""
    database_path = Path(database_path)

    if database_path.exists():
        if not force:
            raise FileExistsError(
                f"O banco já existe: {database_path}. "
                "Use force=True somente para recriá-lo."
            )

        database_path.unlink()

    database_path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE sales (
                sale_id TEXT NOT NULL,
                sale_date TEXT NOT NULL,
                product TEXT NOT NULL,
                category TEXT NOT NULL,
                region TEXT NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                unit_price REAL NOT NULL CHECK(unit_price >= 0),
                discount_pct REAL NOT NULL
                    CHECK(discount_pct >= 0 AND discount_pct <= 1)
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO sales (
                sale_id,
                sale_date,
                product,
                category,
                region,
                quantity,
                unit_price,
                discount_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        connection.execute(
            "CREATE INDEX idx_sales_date ON sales(sale_date)"
        )
        connection.execute(
            "CREATE INDEX idx_sales_category ON sales(category)"
        )

        connection.commit()

    return {
        "database": str(database_path),
        "rows": len(rows),
        "orders": ORDER_COUNT,
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o banco sintético de vendas."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATABASE_PATH,
        help="Caminho de saída do banco SQLite.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Autoriza substituir um banco existente.",
    )
    args = parser.parse_args()

    result = seed_sales_database(
        args.output,
        force=args.force,
    )

    print("BANCO GERADO COM SUCESSO")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
