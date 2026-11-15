"""Gerenciamento do PostgresSaver com conexão direta e ciclo de vida seguro."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from bi_multiagent.settings import POSTGRES_CHECKPOINTER_URI


@contextmanager
def postgres_checkpointer(
    *,
    setup: bool = False,
) -> Iterator[Any]:
    """Mantém uma conexão PostgreSQL durante a utilização do grafo.

    Usa PostgresSaver.from_conn_string, evitando workers de pool.
    O checkpoint continua durável no PostgreSQL.

    Args:
        setup: Inicializa ou atualiza as tabelas do checkpointer.
    """
    if not POSTGRES_CHECKPOINTER_URI:
        raise RuntimeError(
            "POSTGRES_CHECKPOINTER_URI não configurada. Confira o arquivo .env."
        )

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(POSTGRES_CHECKPOINTER_URI) as checkpointer:
        if setup:
            checkpointer.setup()

        yield checkpointer


@contextmanager
def postgres_thread_lock(thread_id: str) -> Iterator[None]:
    """Permite somente uma execução ativa por thread_id.

    O advisory lock pertence a uma conexão PostgreSQL independente,
    portanto coordena processos diferentes da aplicação.
    """
    if not POSTGRES_CHECKPOINTER_URI:
        raise RuntimeError("POSTGRES_CHECKPOINTER_URI não configurada.")

    from psycopg import connect

    with connect(
        POSTGRES_CHECKPOINTER_URI,
        autocommit=True,
        connect_timeout=5,
    ) as connection:
        acquired = connection.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (thread_id,),
        ).fetchone()[0]

        if not acquired:
            raise RuntimeError(
                "DOUBLE_TEXTING_BLOCKED: outra execução está usando "
                f"a thread '{thread_id}'. A nova mensagem foi rejeitada."
            )

        try:
            yield
        finally:
            connection.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                (thread_id,),
            )
