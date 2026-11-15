
"""Fixtures compartilhadas por toda a suíte de testes."""

import pytest

from bi_multiagent.data.seed import (
    has_sales_schema,
    seed_sales_database,
)
from bi_multiagent.settings import DATABASE_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_sales_database():
    """Garante que o banco sintético exista antes dos testes."""

    if not DATABASE_PATH.exists():
        seed_sales_database(DATABASE_PATH)

    elif not has_sales_schema(DATABASE_PATH):
        raise RuntimeError(
            "O banco sales.db existe, mas não possui o schema esperado. "
            "Faça backup e recrie-o explicitamente com "
            "python -m bi_multiagent.data.seed --force."
        )
